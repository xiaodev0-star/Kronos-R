"""Phase 7b: CI Sampling Grid Search (Idea 1 — temperature sampling CI).

Grid-searches over temperature, number of samples, confidence level, and
feed-mode to find the best sampling-based CI configuration.

Uses the SAME evaluation metrics (interval score, coverage, width) as the
CI post-training HPO, enabling direct head-to-head comparison.

Usage:
    python -m hpo.phase7_ci_sampling
"""

import csv, json, os, sys, time
from argparse import Namespace
from contextlib import nullcontext
from itertools import product

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DataConfig
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT
from posttrain.rollout.data import RolloutWindowDataset, rollout_collate
from posttrain.ci.eval_ci import compute_ci_metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE7_DIR = os.path.join(PROJECT_ROOT, "trials", "phase7_ci")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.pt")
TOKENIZER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer_config.json")
BASEMODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "base_model.pt")
ROLLOUT_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "post_train_rollout", "rollout_scheduled.pt")

TOKENIZER_VOCAB = 1 << 10
PREFIX_LEN = 1023
HORIZON = 10

BACKBONE = {
    "dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
    "dsa_windows": [None, 512, 512],
    "position_encoding": "rope", "rope_base": 10000.0,
    "dropout": 0.1323, "use_revin": False, "num_factor_tokens": 0,
}

# ── Grid search space (expanded for 10-hr budget) ──
GRID_SPACE = {
    "temperature": [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
    "num_samples": [16, 32, 64, 128],
    "confidence_level": [0.68, 0.80, 0.90, 0.95],
    "feed_mode": ["argmax", "median"],
    "model": ["basemodel", "rollout"],
}


def _make_cfg():
    return Namespace(
        prefix_len=PREFIX_LEN, horizon=HORIZON,
        stride_ratio=DataConfig.stride_ratio,
        cache_dir=os.path.join(PROJECT_ROOT, "posttrain", "rollout", "cache"),
        max_stocks=0, cache_rebuild=False,
    )


def _load_tokenizer(device):
    ckpt = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg and os.path.exists(TOKENIZER_CONFIG_PATH):
        with open(TOKENIZER_CONFIG_PATH) as f:
            cfg = json.load(f)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tok.load_state_dict(ckpt["model_state_dict"], strict=False)
    tok.eval()
    tok.requires_grad_(False)
    return tok


def _load_model(device, checkpoint_path):
    bp = BACKBONE
    model = KronosReasoningGPT(
        dim=bp["dim"], depth=bp["depth"], heads=bp["heads"],
        num_kv_heads=bp["num_kv_heads"], dsa_windows=bp["dsa_windows"],
        dropout=bp["dropout"], vocab_size_coarse=TOKENIZER_VOCAB,
        vocab_size_fine=TOKENIZER_VOCAB,
        position_encoding=bp["position_encoding"], rope_base=bp["rope_base"],
        use_revin=bp["use_revin"], num_factor_tokens=bp["num_factor_tokens"],
    ).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _build_val_data(device, max_val_samples=500):
    cfg = _make_cfg()
    val_ds = RolloutWindowDataset("val", cfg=cfg, max_samples=max_val_samples, seed=59)
    print(f"  Val windows: {len(val_ds)}")
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=rollout_collate, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    return val_loader


@torch.no_grad()
def run_ci_sampling(model, tokenizer, val_loader, device, temperature,
                     num_samples, confidence_level, feed_mode):
    """Single CI sampling run. Returns metrics dict."""
    model.eval()
    prefix_len = PREFIX_LEN
    horizon = HORIZON

    all_lower, all_upper, all_actual = [], [], []
    alpha = 1.0 - float(confidence_level)
    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0

    for batch in val_loader:
        feats = batch["features"].to(device=device, dtype=torch.float32)
        means = batch["means"].to(device=device, dtype=torch.float32)
        stds = batch["stds"].to(device=device, dtype=torch.float32)
        actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
        times_f = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}

        B = feats.shape[0]
        if B == 0:
            continue

        idx_c_full, idx_f_full = tokenizer.encode(feats)
        context_c = idx_c_full[:, :prefix_len].clone()
        context_f = idx_f_full[:, :prefix_len].clone()

        step_lower, step_upper = [], []
        temp = max(float(temperature), 1e-5)

        for step in range(horizon):
            cur_len = int(context_c.size(1))
            cur_time = {k: times_f[k][:, :cur_len] for k in ("minute", "day", "month", "year")}

            logits_c, logits_f, _ = model(
                context_c, context_f,
                cur_time["minute"], cur_time["day"],
                cur_time["month"], cur_time["year"],
                last_only=True,
            )

            last_c = logits_c[:, -1, :].float()
            last_f = logits_f[:, -1, :].float()

            probs_c = torch.softmax(last_c / temp, dim=-1)
            probs_f = torch.softmax(last_f / temp, dim=-1)
            sc = torch.multinomial(probs_c, num_samples=int(num_samples), replacement=True)
            sf = torch.multinomial(probs_f, num_samples=int(num_samples), replacement=True)

            decoded = tokenizer.decode(sc, sf)
            pred_norms = decoded[:, :, 0].float()
            pred_rets = pred_norms * stds[:, 0:1] + means[:, 0:1]

            sorted_r = pred_rets.sort(dim=1).values
            idx_low = max(0, min(int(num_samples) - 1, int(low_q * int(num_samples))))
            idx_high = max(0, min(int(num_samples) - 1, int(high_q * int(num_samples))))

            step_lower.append(sorted_r[:, idx_low].cpu())
            step_upper.append(sorted_r[:, idx_high].cpu())

            if step < horizon - 1:
                if feed_mode == "argmax":
                    next_c = last_c.argmax(dim=-1)
                    next_f = last_f.argmax(dim=-1)
                elif feed_mode == "median":
                    median_pos = int(num_samples) // 2
                    sort_idx = pred_rets.argsort(dim=1)
                    next_c = sc[torch.arange(B, device=device), sort_idx[:, median_pos]]
                    next_f = sf[torch.arange(B, device=device), sort_idx[:, median_pos]]
                else:
                    rnd = torch.randint(0, int(num_samples), (B,), device=device)
                    next_c = sc[torch.arange(B, device=device), rnd]
                    next_f = sf[torch.arange(B, device=device), rnd]

                context_c = torch.cat([context_c, next_c.unsqueeze(1)], dim=1)
                context_f = torch.cat([context_f, next_f.unsqueeze(1)], dim=1)

        all_lower.append(torch.stack(step_lower, dim=1))
        all_upper.append(torch.stack(step_upper, dim=1))
        all_actual.append(actual.cpu())

    if not all_lower:
        return {"avg_interval_score": 999.0}

    pl = torch.cat(all_lower, dim=0).numpy()
    pu = torch.cat(all_upper, dim=0).numpy()
    aa = torch.cat(all_actual, dim=0).numpy()

    m = compute_ci_metrics(pl, pu, aa, confidence_level=float(confidence_level))
    return {
        "avg_interval_score": round(m["avg_interval_score"], 6),
        "coverage": round(m["coverage"], 6),
        "avg_width": round(m["avg_width"], 6),
        "path_coverage": round(m["path_coverage"], 6),
        "path_avg_width": round(m["path_avg_width"], 6),
        "path_avg_interval_score": round(m["path_avg_interval_score"], 6),
        "mape_midpoint": round(m["mape_midpoint"], 4),
        "da_midpoint": round(m["da_midpoint"], 4),
        "per_step_coverage": [round(s["coverage"], 4) for s in m["per_step"]],
        "per_step_width": [round(s["avg_width"], 6) for s in m["per_step"]],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 7b — CI Sampling Grid Search")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Check available models
    models_to_test = []
    if os.path.exists(BASEMODEL_PATH):
        models_to_test.append(("basemodel", BASEMODEL_PATH))
    if os.path.exists(ROLLOUT_MODEL_PATH):
        models_to_test.append(("rollout", ROLLOUT_MODEL_PATH))

    print(f"  Models: {[m[0] for m in models_to_test]}")

    tokenizer = _load_tokenizer(device)
    val_loader = _build_val_data(device, max_val_samples=500)

    # Build grid
    grid_keys = ["temperature", "num_samples", "confidence_level", "feed_mode"]
    grid_values = [GRID_SPACE[k] for k in grid_keys]
    total_configs = 1
    for gv in grid_values:
        total_configs *= len(gv)
    total_runs = total_configs * len(models_to_test)
    print(f"  Grid: {len(GRID_SPACE['temperature'])}×{len(GRID_SPACE['num_samples'])}×"
          f"{len(GRID_SPACE['confidence_level'])}×{len(GRID_SPACE['feed_mode'])}×"
          f"{len(models_to_test)} = {total_runs} runs")

    all_rows = []
    run_idx = 0

    for model_name, ckpt_path in models_to_test:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        model = _load_model(device, ckpt_path)

        for temp, ns, conf, fm in product(*grid_values):
            run_idx += 1
            config_id = f"{model_name}_T{temp}_N{ns}_C{conf}_{fm}"
            tdir = os.path.join(PHASE7_DIR, config_id)
            result_path = os.path.join(tdir, "result.json")

            if os.path.exists(result_path):
                with open(result_path) as f:
                    row = json.load(f)
                all_rows.append(row)
                print(f"  [{run_idx}/{total_runs}] {config_id} (cached)  "
                      f"IS={row.get('avg_interval_score','?')}")
                continue

            t0 = time.time()
            try:
                metrics = run_ci_sampling(
                    model=model, tokenizer=tokenizer, val_loader=val_loader,
                    device=device, temperature=temp, num_samples=ns,
                    confidence_level=conf, feed_mode=fm,
                )
            except Exception as e:
                print(f"  [{run_idx}/{total_runs}] {config_id} FAILED: {e}")
                continue

            elapsed = time.time() - t0

            row = {
                "model": model_name,
                "temperature": temp,
                "num_samples": ns,
                "confidence_level": conf,
                "feed_mode": fm,
                **metrics,
                "elapsed_s": round(elapsed, 1),
            }
            all_rows.append(row)

            os.makedirs(tdir, exist_ok=True)
            with open(result_path, "w") as f:
                json.dump(row, f, indent=2)

            print(f"  [{run_idx}/{total_runs}] {config_id}  "
                  f"IS={metrics['avg_interval_score']:.6f}  "
                  f"cov={metrics['coverage']:.4f}  "
                  f"width={metrics['avg_width']:.6f}  "
                  f"({elapsed:.1f}s)")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Save summary ──
    os.makedirs(PHASE7_DIR, exist_ok=True)
    summary_path = os.path.join(PHASE7_DIR, "summary_sampling.csv")
    if all_rows:
        all_keys = set()
        for r in all_rows:
            all_keys.update(r.keys())
        ordered = ["model", "temperature", "num_samples", "confidence_level",
                    "feed_mode", "avg_interval_score", "coverage", "avg_width"]
        ordered += sorted(k for k in all_keys if k not in ordered)
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)

        # Best per model
        for mn in sorted(set(r["model"] for r in all_rows)):
            model_rows = [r for r in all_rows if r["model"] == mn]
            best = min(model_rows, key=lambda r: r["avg_interval_score"])
            print(f"\nBest {mn}: T={best['temperature']} N={best['num_samples']} "
                  f"C={best['confidence_level']} fm={best['feed_mode']}")
            print(f"  IS={best['avg_interval_score']:.6f}  "
                  f"cov={best['coverage']:.4f}  width={best['avg_width']:.6f}")

        print(f"\nSummary: {summary_path}")

    # ── Save cross-method comparison ──
    comparison_path = os.path.join(PHASE7_DIR, "cross_method_summary.json")
    comparison = {
        "description": "CI method comparison: sampling vs training",
        "sampling_best": {},
        "training_best": {},
    }
    if all_rows:
        best_sampling = min(all_rows, key=lambda r: r["avg_interval_score"])
        comparison["sampling_best"] = best_sampling

    training_csv = os.path.join(PHASE7_DIR, "summary_training.csv")
    if os.path.exists(training_csv):
        with open(training_csv, newline="") as f:
            train_rows = list(csv.DictReader(f))
            train_rows = [{k: (float(v) if v.replace('.','').replace('-','').isdigit() else v)
                           for k, v in r.items()} for r in train_rows]
        if train_rows:
            best_training = min(train_rows, key=lambda r: float(r.get("value", 999)))
            comparison["training_best"] = {
                "avg_interval_score": best_training["value"],
                **{k: v for k, v in best_training.items() if k != "value"},
            }

    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"Cross-method comparison: {comparison_path}")


if __name__ == "__main__":
    main()
