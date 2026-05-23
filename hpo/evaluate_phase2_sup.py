"""Evaluate Phase 2 supplement trials on 1-step downstream prediction.

Reuses evaluate_one_trial from evaluate_phase2. Saves to trials/phase2_tokenizer_sup/.

Usage:
    python -m hpo.evaluate_phase2_sup
"""

from __future__ import annotations
import csv, json, os, sys, time
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from hpo.evaluate_phase2 import (
    evaluate_one_trial, bootstrap_metrics, summarize_bootstrap,
    load_val_data,
)
from reproducibility import set_global_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUP_DIR = os.path.join(PROJECT_ROOT, "trials", "phase2_tokenizer_sup")
N_BOOTSTRAP = 10
BOOTSTRAP_SEED = 42
MAX_TRIALS = 20
SKIP_COMPLETED = True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 2 Supplement — 1-step downstream eval")
    print(f"  Device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    set_global_seed(42, deterministic=True)
    features, time_features, means, stds = load_val_data()

    summaries = []
    for t_num in range(MAX_TRIALS):
        tdir = os.path.join(SUP_DIR, f"trial_{t_num:03d}")
        if not os.path.exists(os.path.join(tdir, "basemodel.pt")):
            continue

        eval_path = os.path.join(tdir, "eval.json")
        if SKIP_COMPLETED and os.path.exists(eval_path):
            with open(eval_path) as f:
                summaries.append(json.load(f))
            continue

        t0 = time.time()
        print(f"\nTrial {t_num:03d}")
        try:
            preds, actuals = evaluate_one_trial(tdir, features, time_features, means, stds, device)
            bs = bootstrap_metrics(preds, actuals, N_BOOTSTRAP, BOOTSTRAP_SEED + t_num)
            s = summarize_bootstrap(bs)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        s["trial"] = t_num
        s["num_samples"] = int(len(preds))
        s["elapsed_sec"] = round(time.time() - t0, 1)
        s["bootstrap_rounds"] = bs

        # Attach config + val_ce
        cfg_path = os.path.join(tdir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                s["trial_config"] = json.load(f)
        result_path = os.path.join(tdir, "result.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                r = json.load(f)
            s["val_ce"] = r.get("best_val_ce")

        with open(eval_path, "w") as f:
            json.dump(s, f, indent=2)
        summaries.append(s)
        print(f"  MAPE={s['mape']:.4f}±{s['mape_std']:.4f}%  DA={s['da']:.2f}%  val_ce={s.get('val_ce','?')}")

    # CSV
    if summaries:
        csv_path = os.path.join(SUP_DIR, "eval.csv")
        all_keys = set()
        for s in summaries:
            all_keys.update(k for k in s if k not in ("bootstrap_rounds", "trial_config"))
        ordered = ["trial", "mape", "mape_std", "da", "da_std", "mae", "mae_std",
                   "rmse", "rmse_std", "val_ce", "elapsed_sec", "num_samples"]
        ordered += sorted(k for k in all_keys if k not in ordered)
        rows = [{k: s.get(k, "") for k in ordered} for s in summaries]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        ranked = sorted(summaries, key=lambda x: x["mape"])
        print(f"\nTop-5 by MAPE:")
        for s in ranked[:5]:
            vc = s.get("val_ce", "?")
            print(f"  trial={s['trial']:03d}  MAPE={s['mape']:.4f}±{s['mape_std']:.4f}%  "
                  f"DA={s['da']:.2f}%  val_ce={vc}")
        print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
