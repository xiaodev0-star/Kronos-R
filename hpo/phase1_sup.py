"""Phase 1 Supplement: Bits 3–5 lower-boundary experiment.

Reuses train_tokenizer_fixed / train_basemodel from phase1_bits_search.
Saves results under trials/phase1_bits_search_sup/ (separate from main).

Usage:
    python -m hpo.phase1_sup
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import torch

# Reuse ALL the training infrastructure from phase1_bits_search
from hpo.phase1_bits_search import (
    train_tokenizer_fixed,
    train_basemodel,
    TOKENIZER_PARAMS,
    BASEMODEL_PARAMS,
)

# ── Override output dir ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1_SUP_DIR = os.path.join(PROJECT_ROOT, "trials", "phase1_bits_search_sup")
SUMMARY_PATH = os.path.join(PHASE1_SUP_DIR, "bits_summary.json")

BITS_RANGE = [3, 4, 5]


def bits_dir(bits: int) -> str:
    return os.path.join(PHASE1_SUP_DIR, f"bits_{bits:02d}")


# ──────────────────────────────────────────────────────────
def main():
    os.makedirs(PHASE1_SUP_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Phase 1 Supplement — Bits lower boundary: {BITS_RANGE}")
    print(f"  Output: {PHASE1_SUP_DIR}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print()

    all_results = []

    for bits in BITS_RANGE:
        t0 = time.time()
        bdir = bits_dir(bits)
        os.makedirs(bdir, exist_ok=True)

        print(f"{'='*60}")
        print(f"Bits = {bits}  (vocab_size = {1<<bits})")
        print(f"{'='*60}")

        # 1. Tokenizer
        print(f"\n[1/2] Training tokenizer (bits={bits})...")
        tokenizer = train_tokenizer_fixed(bits, bdir, device)

        # 2. BaseModel
        print(f"\n[2/2] Training BaseModel (DSA, bits={bits})...")
        result = train_basemodel(tokenizer, bits, bdir, device)

        elapsed = time.time() - t0
        result["elapsed_minutes"] = round(elapsed / 60, 1)
        all_results.append(result)

        del tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

        with open(SUMMARY_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  bits={bits} done in {elapsed/60:.1f} min\n")

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"Phase 1 Supplement complete")
    print(f"{'='*60}")
    print(f"{'bits':>5} {'vocab':>6} {'val_ce':>10} {'epoch':>6} "
          f"{'c_util':>8} {'c_low50%':>10} {'f_util':>8} {'f_low50%':>10}")
    print("-" * 70)
    for r in all_results:
        c = r["token_metrics"]["coarse"]
        f = r["token_metrics"]["fine"]
        print(f"{r['bits']:5d} {r['vocab_size']:6d} {r['best_val_ce']:10.4f} {r['epoch_stopped']:6d} "
              f"{c['utilization']:8.3f} {c['low_freq_share']:10.4f} "
              f"{f['utilization']:8.3f} {f['low_freq_share']:10.4f}")

    print(f"\nFull results: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
