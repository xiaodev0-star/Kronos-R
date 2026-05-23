"""Evaluate Phase 2 trials on 1-step downstream prediction.

Loads each Optuna trial's tokenizer + BaseModel, runs 1-step next-day
prediction on the validation set, computes MAPE/DA/MAE/RMSE with optional
bootstrap for statistical significance.

Results saved per-trial under trials/phase2_tokenizer/trial_XXX/eval.json
with a summary at trials/phase2_tokenizer/eval_summary.json.

Usage:
    python -m hpo.evaluate_phase2
"""

from __future__ import annotations

import json, os, sys, time
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

# ── Config ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE2_DIR = os.path.join(PROJECT_ROOT, "trials", "phase2_tokenizer")
VAL_CACHE = os.path.join(PROJECT_ROOT, "dataset_val.pt")
EVAL_BATCH_SIZE = 32
N_BOOTSTRAP = 10
BOOTSTRAP_SEED = 42
MAX_TRIALS = 50          # scan trial_000 .. trial_{MAX_TRIALS-1}
SKIP_COMPLETED = True    # skip trials that already have eval.json

# ── Fixed BaseModel config (matching Phase 2 BASEMODEL_PARAMS) ──
BASEMODEL_KWARGS = {
    "dim": 256, "depth": 4, "heads": 4,
    "num_kv_heads": 2,
    "dsa_windows": [None, 512, 512, None],
    "position_encoding": "rope", "rope_base": 10000.0,
    "dropout": 0.08,
}

# ── Helpers ──

def _choose_amp_dtype(device):
    if device.type != "cuda": return None
    if torch.cuda.is_bf16_supported(): return torch.bfloat16
    return torch.float16

def _autocast_ctx(amp_enabled, amp_dtype):
    if not amp_enabled: return nullcontext()
    try: return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    except Exception: return torch.cuda.amp.autocast(dtype=amp_dtype)


def load_val_data():
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
    stds  = np.zeros((N, 6), dtype=np.float32)
    for i, s in enumerate(seq_stats):
        means[i] = np.asarray(s["mean"], dtype=np.float32)
        stds[i]  = np.asarray(s["std"],  dtype=np.float32)
    print(f"  val samples: {features.shape[0]}, seq_len: {features.shape[1]}")
    return features, time_features, torch.from_numpy(means), torch.from_numpy(stds)


@torch.inference_mode()
def evaluate_one_trial(trial_dir: str, features, time_features, means, stds, device):
    """Load tokenizer + BaseModel from trial_dir, run 1-step eval, return (preds, actuals)."""
    tok_path = os.path.join(trial_dir, "tokenizer.pt")
    bm_path  = os.path.join(trial_dir, "basemodel.pt")
    cfg_path = os.path.join(trial_dir, "config.json")

    # ── determine vocab_size from config ──
    bits = 10  # fallback
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            tcfg = json.load(f)
        bits = int(tcfg.get("bits_per_quantizer", 10))
    vocab = 1 << bits

    # ── load tokenizer ──
    tok_ckpt = torch.load(tok_path, map_location=device, weights_only=False)
    tk_cfg = tok_ckpt.get("config", {})
    if not tk_cfg:
        tk_cfg = {"input_dim": 6, "hidden_dim": 128, "embedding_dim": 64,
                  "num_quantizers": 2, "bits_per_quantizer": bits}
    tokenizer = HierarchicalQuantizer(**build_tokenizer_kwargs(tk_cfg)).to(device)
    tokenizer.load_state_dict(tok_ckpt["model_state_dict"], strict=False)
    tokenizer.eval(); tokenizer.requires_grad_(False)

    # ── load BaseModel ──
    model = KronosReasoningGPT(
        vocab_size_coarse=vocab, vocab_size_fine=vocab,
        **BASEMODEL_KWARGS,
    ).to(device)
    bm_ckpt = torch.load(bm_path, map_location=device, weights_only=False)
    model.load_state_dict(bm_ckpt["model_state_dict"], strict=False)
    model.eval()

    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    N = features.shape[0]
    all_preds, all_actuals = [], []
    indices = np.arange(N)

    for start in tqdm(range(0, N, EVAL_BATCH_SIZE), desc=f"  eval", leave=False):
        end = min(start + EVAL_BATCH_SIZE, N)
        idx = indices[start:end]

        batch_feats = features[idx].to(device, non_blocking=True)
        batch_means = means[idx].to(device, non_blocking=True)
        batch_stds  = stds[idx].to(device, non_blocking=True)

        input_feats = batch_feats[:, :1023, :]
        actual_norm = batch_feats[:, 1023, 0]

        idx_coarse, idx_fine = tokenizer.encode(input_feats)

        t_min  = time_features["minute"][idx][:, :1023].to(device, non_blocking=True).long()
        t_day  = time_features["day"][idx][:, :1023].to(device, non_blocking=True).long()
        t_month= time_features["month"][idx][:, :1023].to(device, non_blocking=True).long()
        t_year = time_features["year"][idx][:, :1023].to(device, non_blocking=True).long()

        with _autocast_ctx(use_amp, amp_dtype):
            logits_c, logits_f, _ = model(
                idx_coarse, idx_fine, t_min, t_day, t_month, t_year, last_only=True,
            )

        pred_c = logits_c[:, -1, :].float().argmax(dim=-1)
        pred_f = logits_f[:, -1, :].float().argmax(dim=-1)
        decoded = tokenizer.decode(pred_c.unsqueeze(1), pred_f.unsqueeze(1))
        pred_norm = decoded[:, 0, 0]

        pred_log_ret   = pred_norm * batch_stds[:, 0] + batch_means[:, 0]
        actual_log_ret = actual_norm * batch_stds[:, 0] + batch_means[:, 0]

        all_preds.append(pred_log_ret.cpu())
        all_actuals.append(actual_log_ret.cpu())

        del batch_feats, batch_means, batch_stds, input_feats
        del idx_coarse, idx_fine, logits_c, logits_f, decoded

    preds   = torch.cat(all_preds).numpy().astype(np.float64)
    actuals = torch.cat(all_actuals).numpy().astype(np.float64)

    del tokenizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds, actuals


def compute_metrics(pred: np.ndarray, actual: np.ndarray):
    pred_ratio   = np.exp(np.clip(pred, -50, 50))
    actual_ratio = np.exp(np.clip(actual, -50, 50))
    mape = float(np.mean(np.abs((pred_ratio - actual_ratio) /
                                 np.maximum(np.abs(actual_ratio), 1e-4))) * 100)
    pred_sign   = np.where(pred >= 0, 1, -1)
    actual_sign = np.where(actual >= 0, 1, -1)
    da = float(np.mean(pred_sign == actual_sign) * 100)
    err = pred - actual
    return {
        "mape": mape, "da": da,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pred_up_ratio": float(np.mean(pred_sign > 0) * 100),
        "actual_up_ratio": float(np.mean(actual_sign > 0) * 100),
    }


def bootstrap_metrics(preds_all, actuals_all, n_rounds, seed):
    rng = np.random.default_rng(seed)
    N = len(preds_all)
    results = []
    for r in range(n_rounds):
        idx = rng.choice(N, size=N, replace=True)
        pred_sample   = preds_all[idx]
        actual_sample = actuals_all[idx]
        finite = np.isfinite(pred_sample) & np.isfinite(actual_sample)
        m = compute_metrics(pred_sample[finite], actual_sample[finite])
        m["round"] = r
        results.append(m)
    return results


def summarize_bootstrap(bs_results):
    keys = ["mape", "da", "mae", "rmse", "pred_up_ratio", "actual_up_ratio"]
    s = {"n_rounds": len(bs_results)}
    for k in keys:
        vals = [r[k] for r in bs_results]
        s[k] = round(float(np.mean(vals)), 6)
        s[f"{k}_std"] = round(float(np.std(vals, ddof=1)), 6)
    return s


# ── CSV export ──

def _export_eval_csv(slim: list, path: str):
    """Export all trials to a single CSV with metrics + tokenizer config."""
    import csv

    # Collect all possible keys
    all_keys = set()
    for s in slim:
        all_keys.update(s.keys())
        cfg = s.get("trial_config", {})
        if isinstance(cfg, dict):
            all_keys.update(f"cfg_{k}" for k in cfg.keys())

    # Define column order: trial first, then metrics, then config
    metric_cols = ["trial", "mape", "mape_std", "da", "da_std", "mae", "mae_std",
                   "rmse", "rmse_std", "val_ce", "coarse_utilization",
                   "coarse_dead_tokens", "elapsed_sec", "num_samples"]
    ordered = [c for c in metric_cols if c in all_keys]
    # Then remaining non-cfg columns
    for k in sorted(all_keys):
        if k not in ordered and not k.startswith("cfg_"):
            ordered.append(k)
    # Then config columns
    cfg_keys = sorted(k for k in all_keys if k.startswith("cfg_"))
    ordered.extend(cfg_keys)

    rows = []
    for s in slim:
        row = {}
        for k in ordered:
            if k.startswith("cfg_"):
                cfg = s.get("trial_config", {})
                if isinstance(cfg, dict):
                    row[k] = cfg.get(k[4:], "")  # strip "cfg_" prefix
                else:
                    row[k] = ""
            else:
                row[k] = s.get(k, "")
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nConsolidated CSV: {path}  ({len(rows)} trials)")


# ── Main ──

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 2 — 1-step downstream eval ({N_BOOTSTRAP}-round bootstrap)")
    print(f"  Trials dir: {PHASE2_DIR}")
    print(f"  Max trials:  {MAX_TRIALS}")
    print(f"  Device:      {device}")
    if device.type == "cuda":
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    set_global_seed(42, deterministic=True)
    features, time_features, means, stds = load_val_data()

    all_summaries = []
    skipped = 0
    failed  = 0

    for t_num in range(MAX_TRIALS):
        tdir = os.path.join(PHASE2_DIR, f"trial_{t_num:03d}")
        tok_path = os.path.join(tdir, "tokenizer.pt")
        bm_path  = os.path.join(tdir, "basemodel.pt")

        if not os.path.exists(tok_path) or not os.path.exists(bm_path):
            continue  # trial not trained / failed in Optuna

        eval_path = os.path.join(tdir, "eval.json")
        if SKIP_COMPLETED and os.path.exists(eval_path):
            with open(eval_path) as f:
                s = json.load(f)
            all_summaries.append(s)
            skipped += 1
            continue

        t0 = time.time()
        print(f"\n{'='*50}")
        print(f"Trial {t_num:03d}")
        print(f"{'='*50}")

        try:
            preds, actuals = evaluate_one_trial(tdir, features, time_features, means, stds, device)
            bs = bootstrap_metrics(preds, actuals, N_BOOTSTRAP, BOOTSTRAP_SEED + t_num)
            s = summarize_bootstrap(bs)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
            continue

        s["trial"] = t_num
        s["num_samples"] = int(len(preds))
        s["elapsed_sec"] = round(time.time() - t0, 1)
        s["bootstrap_rounds"] = bs

        # Also read trial config & tokenizer params for later analysis
        cfg_path = os.path.join(tdir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                s["trial_config"] = json.load(f)
        result_path = os.path.join(tdir, "result.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                r = json.load(f)
            s["val_ce"] = r.get("best_val_ce")
            s["coarse_utilization"] = r.get("coarse_utilization")
            s["coarse_dead_tokens"] = r.get("coarse_dead_tokens")

        with open(eval_path, "w") as f:
            json.dump(s, f, indent=2)
        all_summaries.append(s)

        print(f"  MAPE={s['mape']:.4f}±{s['mape_std']:.4f}%  "
              f"DA={s['da']:.2f}±{s['da_std']:.2f}%  "
              f"MAE={s['mae']:.6f}  RMSE={s['rmse']:.6f}  "
              f"time={s['elapsed_sec']:.0f}s")

        del preds, actuals

    # ── Summary JSON ──
    summary_path = os.path.join(PHASE2_DIR, "eval_summary.json")
    slim = [{k: v for k, v in s.items() if k != "bootstrap_rounds"}
            for s in all_summaries]
    with open(summary_path, "w") as f:
        json.dump(slim, f, indent=2)

    # ── Consolidated CSV ──
    csv_path = os.path.join(PHASE2_DIR, "eval.csv")
    _export_eval_csv(slim, csv_path)

    # Rank by MAPE
    ranked = sorted(slim, key=lambda x: x["mape"])

    print(f"\n{'='*80}")
    print(f"Phase 2 Eval — Top-10 by MAPE    (skipped={skipped}, failed={failed})")
    print(f"{'='*80}")
    print(f"{'trial':>6} {'MAPE%':>14} {'DA%':>10} {'MAE':>12} {'RMSE':>12} {'val_ce':>10}")
    print("-" * 70)
    for s in ranked[:10]:
        vc = s.get("val_ce", "?")
        vc_str = f"{vc:.4f}" if isinstance(vc, (int, float)) else str(vc)
        print(f"{s['trial']:6d} {s['mape']:6.2f}±{s['mape_std']:.2f}%  "
              f"{s['da']:5.2f}±{s['da_std']:.2f}%  "
              f"{s['mae']:.6f}  {s['rmse']:.6f}  {vc_str}")

    # Correlation: val_ce vs MAPE
    val_ce_list = [s["val_ce"] for s in slim if isinstance(s.get("val_ce"), (int, float))]
    mape_list   = [s["mape"]   for s in slim if isinstance(s.get("val_ce"), (int, float))]
    if len(val_ce_list) > 2:
        corr = np.corrcoef(val_ce_list, mape_list)[0, 1]
        print(f"\n  val_ce vs MAPE correlation: r = {corr:.4f}")

    print(f"\nResults: {summary_path}")
    print(f"Per-trial: {PHASE2_DIR}/trial_XXX/eval.json")


if __name__ == "__main__":
    main()
