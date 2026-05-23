"""Evaluate Phase 1 bits checkpoints on 1-step downstream metrics (MAPE, DA).

Loads each bits' tokenizer + BaseModel, runs 1-step next-day prediction
on the validation set, then bootstrap-resamples predictions N_ROUNDS times
to compute mean ± std for statistical significance.

Usage:
    python -m hpo.evaluate_phase1_bits
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np
import torch
from tqdm import tqdm

from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT
from reproducibility import set_global_seed

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_DIR = os.path.join(PROJECT_ROOT, "trials", "phase1_bits_search")
SUP_DIR = os.path.join(PROJECT_ROOT, "trials", "phase1_bits_search_sup")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "trials", "phase1_evaluate")
VAL_CACHE = os.path.join(PROJECT_ROOT, "dataset_val.pt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BITS_ALL = list(range(3, 13))  # 3..12
EVAL_BATCH_SIZE = 32
N_BOOTSTRAP = 10
BOOTSTRAP_SEED = 42

# ── DSA config matching Phase 1 ──
BASEMODEL_KWARGS = {
    "dim": 256,
    "depth": 4,
    "heads": 4,
    "num_kv_heads": 2,
    "dsa_windows": [None, 512, 512, None],
    "position_encoding": "rope",
    "rope_base": 10000.0,
    "dropout": 0.08,
}


def _choose_amp_dtype(device):
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_ctx(amp_enabled, amp_dtype):
    if not amp_enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    except Exception:
        return torch.cuda.amp.autocast(dtype=amp_dtype)


def resolve_bits_dir(bits: int):
    for base in (SUP_DIR if bits <= 5 else MAIN_DIR, MAIN_DIR if bits > 5 else SUP_DIR):
        d = os.path.join(base, f"bits_{bits:02d}")
        tok = os.path.join(d, "tokenizer.pt")
        bm = os.path.join(d, "basemodel.pt")
        if os.path.exists(tok) and os.path.exists(bm):
            return d, tok, bm
    raise FileNotFoundError(f"No checkpoints for bits={bits}")


def load_tokenizer(bits: int, device: torch.device):
    _, tok_path, _ = resolve_bits_dir(bits)
    ckpt = torch.load(tok_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    tokenizer = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tokenizer.load_state_dict(ckpt["model_state_dict"], strict=False)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer


def load_model(bits: int, device: torch.device):
    _, _, bm_path = resolve_bits_dir(bits)
    vocab = 1 << bits
    model = KronosReasoningGPT(
        vocab_size_coarse=vocab,
        vocab_size_fine=vocab,
        **BASEMODEL_KWARGS,
    ).to(device)
    ckpt = torch.load(bm_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def load_val_data():
    """Load cached val dataset."""
    print(f"Loading val cache: {VAL_CACHE}")
    payload = torch.load(VAL_CACHE, map_location="cpu", weights_only=False)
    features = payload["features"]
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)

    time_features = {}
    for key in ("minute", "day", "month", "year"):
        t = payload["time_features"][key]
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.long)
        time_features[key] = t

    seq_stats = payload["seq_stats"]
    N = len(seq_stats)
    means = np.zeros((N, 6), dtype=np.float32)
    stds = np.zeros((N, 6), dtype=np.float32)
    for i, s in enumerate(seq_stats):
        means[i] = np.asarray(s["mean"], dtype=np.float32)
        stds[i] = np.asarray(s["std"], dtype=np.float32)

    print(f"  val samples: {features.shape[0]}, seq_len: {features.shape[1]}")
    return features, time_features, torch.from_numpy(means), torch.from_numpy(stds)


@torch.inference_mode()
def evaluate_bits_once(bits: int, features, time_features, means, stds, device: torch.device):
    """Run 1-step evaluation once, return all (pred, actual) pairs."""
    tokenizer = load_tokenizer(bits, device)
    model = load_model(bits, device)

    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"

    N = features.shape[0]
    all_preds = []
    all_actuals = []

    indices = np.arange(N)
    for start in tqdm(range(0, N, EVAL_BATCH_SIZE), desc=f"  bits={bits}", leave=False):
        end = min(start + EVAL_BATCH_SIZE, N)
        idx = indices[start:end]

        batch_feats = features[idx].to(device, non_blocking=True)
        batch_means = means[idx].to(device, non_blocking=True)
        batch_stds = stds[idx].to(device, non_blocking=True)

        input_feats = batch_feats[:, :1023, :]
        actual_norm = batch_feats[:, 1023, 0]

        idx_coarse, idx_fine = tokenizer.encode(input_feats)

        t_min = time_features["minute"][idx][:, :1023].to(device, non_blocking=True).long()
        t_day = time_features["day"][idx][:, :1023].to(device, non_blocking=True).long()
        t_month = time_features["month"][idx][:, :1023].to(device, non_blocking=True).long()
        t_year = time_features["year"][idx][:, :1023].to(device, non_blocking=True).long()

        with _autocast_ctx(use_amp, amp_dtype):
            logits_c, logits_f, _ = model(
                idx_coarse, idx_fine, t_min, t_day, t_month, t_year,
                last_only=True,
            )

        pred_c = logits_c[:, -1, :].float().argmax(dim=-1)
        pred_f = logits_f[:, -1, :].float().argmax(dim=-1)

        decoded = tokenizer.decode(pred_c.unsqueeze(1), pred_f.unsqueeze(1))
        pred_norm = decoded[:, 0, 0]

        pred_log_ret = pred_norm * batch_stds[:, 0] + batch_means[:, 0]
        actual_log_ret = actual_norm * batch_stds[:, 0] + batch_means[:, 0]

        all_preds.append(pred_log_ret.cpu())
        all_actuals.append(actual_log_ret.cpu())

        del batch_feats, batch_means, batch_stds, input_feats
        del idx_coarse, idx_fine, logits_c, logits_f, decoded

    preds = torch.cat(all_preds).numpy().astype(np.float64)
    actuals = torch.cat(all_actuals).numpy().astype(np.float64)

    del tokenizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return preds, actuals


def compute_metrics(pred: np.ndarray, actual: np.ndarray):
    """Compute metrics from prediction arrays (already filtered for finiteness)."""
    # Close-ratio MAPE
    pred_ratio = np.exp(np.clip(pred, -50, 50))
    actual_ratio = np.exp(np.clip(actual, -50, 50))
    mape = float(np.mean(np.abs((pred_ratio - actual_ratio) / np.maximum(np.abs(actual_ratio), 1e-4))) * 100)

    # Direction accuracy
    pred_sign = np.where(pred >= 0, 1, -1)
    actual_sign = np.where(actual >= 0, 1, -1)
    da = float(np.mean(pred_sign == actual_sign) * 100)

    err = pred - actual
    return {
        "mape": mape,
        "da": da,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pred_up_ratio": float(np.mean(pred_sign > 0) * 100),
        "actual_up_ratio": float(np.mean(actual_sign > 0) * 100),
    }


def bootstrap_metrics(preds_all: np.ndarray, actuals_all: np.ndarray, n_rounds: int, seed: int):
    """Bootstrap resample (pred, actual) pairs n_rounds times, return list of metric dicts."""
    rng = np.random.default_rng(seed)
    N = len(preds_all)
    results = []
    for r in range(n_rounds):
        idx = rng.choice(N, size=N, replace=True)
        pred_sample = preds_all[idx]
        actual_sample = actuals_all[idx]
        finite = np.isfinite(pred_sample) & np.isfinite(actual_sample)
        metrics = compute_metrics(pred_sample[finite], actual_sample[finite])
        metrics["round"] = r
        results.append(metrics)
    return results


def summarize_bootstrap(bootstrap_results: list) -> dict:
    """Aggregate bootstrap rounds into mean ± std."""
    keys = ["mape", "da", "mae", "rmse", "pred_up_ratio", "actual_up_ratio"]
    summary = {"n_rounds": len(bootstrap_results)}
    for k in keys:
        vals = [r[k] for r in bootstrap_results]
        summary[k] = round(float(np.mean(vals)), 6)
        summary[f"{k}_std"] = round(float(np.std(vals, ddof=1)), 6)
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 1 — 1-step evaluation on VAL set ({N_BOOTSTRAP}-round bootstrap)")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    set_global_seed(42, deterministic=True)

    features, time_features, means, stds = load_val_data()

    all_summaries = []
    for bits in BITS_ALL:
        t0 = time.time()
        print(f"\n{'='*50}")
        print(f"Evaluating bits={bits} (vocab={1<<bits})")
        print(f"{'='*50}")

        # Single deterministic forward pass
        preds, actuals = evaluate_bits_once(bits, features, time_features, means, stds, device)

        # Bootstrap N rounds for statistical significance
        bootstrap_results = bootstrap_metrics(preds, actuals, N_BOOTSTRAP, BOOTSTRAP_SEED + bits)
        summary = summarize_bootstrap(bootstrap_results)
        summary["bits"] = bits
        summary["vocab_size"] = 1 << bits
        summary["num_samples"] = int(len(preds))
        summary["elapsed_sec"] = round(time.time() - t0, 1)
        summary["bootstrap_rounds"] = bootstrap_results
        all_summaries.append(summary)

        # Save per-bits
        bits_out = os.path.join(OUTPUT_DIR, f"bits_{bits:02d}.json")
        with open(bits_out, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  bits={bits}: MAPE={summary['mape']:.4f}±{summary['mape_std']:.4f}%, "
              f"DA={summary['da']:.2f}±{summary['da_std']:.2f}%, "
              f"MAE={summary['mae']:.6f}±{summary['mae_std']:.6f}, "
              f"RMSE={summary['rmse']:.6f}±{summary['rmse_std']:.6f}, "
              f"elapsed={summary['elapsed_sec']:.0f}s")

        del preds, actuals

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    # Strip bootstrap_rounds detail for summary (keep means + stds)
    slim = [{k: v for k, v in s.items() if k != "bootstrap_rounds"} for s in all_summaries]
    with open(summary_path, "w") as f:
        json.dump(slim, f, indent=2)

    # Print comparison table
    print(f"\n{'='*85}")
    print(f"Phase 1 — 1-step Downstream Evaluation Summary (val, {N_BOOTSTRAP}-round bootstrap)")
    print(f"{'='*85}")
    header = f"{'bits':>5} {'vocab':>6} {'MAPE%':>14} {'DA%':>14} {'MAE':>18} {'RMSE':>18}"
    print(header)
    print("-" * 85)
    for s in all_summaries:
        print(f"{s['bits']:5d} {s['vocab_size']:6d} "
              f"{s['mape']:6.2f}±{s['mape_std']:.2f}%  "
              f"{s['da']:5.2f}±{s['da_std']:.2f}%  "
              f"{s['mae']:.6f}±{s['mae_std']:.6f}  "
              f"{s['rmse']:.6f}±{s['rmse_std']:.6f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
