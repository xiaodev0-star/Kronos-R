# -*- coding: utf-8 -*-
"""Phase 5 DA — Main comparison runner.

Usage::

    python -m hpo.phase5.run                  # run all methods
    python -m hpo.phase5.run --method grpo    # run GRPO only
    python -m hpo.phase5.run --methods ce,grpo  # run specific methods
    python -m hpo.phase5.run --epochs 10 --lr 5e-5  # custom settings
    python -m hpo.phase5.run --no-lora         # full fine-tuning
    python -m hpo.phase5.run --eval-only       # only re-evaluate from checkpoints
"""

from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime

import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from hpo.phase5.core import OUT_DIR, DEFAULT_CFG, SEED
from hpo.phase5.methods import METHOD_REGISTRY
from hpo.phase5.train import run_method
from reproducibility import set_global_seed


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 DA: Post-training method comparison")
    p.add_argument("--method", type=str, default="all",
                   help="Method to run (ce, expo, dpo, rsft, grpo, or 'all')")
    p.add_argument("--methods", type=str, default="",
                   help="Comma-separated list of methods")
    p.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"],
                   help=f"Training epochs (default: {DEFAULT_CFG['epochs']})")
    p.add_argument("--batch-size", type=int, default=DEFAULT_CFG["batch_size"],
                   help=f"Batch size (default: {DEFAULT_CFG['batch_size']})")
    p.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"],
                   help=f"Learning rate (default: {DEFAULT_CFG['lr']})")
    p.add_argument("--no-lora", action="store_true",
                   help="Full fine-tuning instead of LoRA")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--grpo-group-size", type=int, default=DEFAULT_CFG["grpo_group_size"],
                   help=f"GRPO group size K (default: {DEFAULT_CFG['grpo_group_size']})")
    p.add_argument("--grpo-temperature", type=float, default=DEFAULT_CFG["grpo_temperature"],
                   help=f"GRPO sampling temperature (default: {DEFAULT_CFG['grpo_temperature']})")
    p.add_argument("--grpo-kl-weight", type=float, default=DEFAULT_CFG["grpo_kl_weight"],
                   help=f"GRPO KL penalty weight (default: {DEFAULT_CFG['grpo_kl_weight']})")
    p.add_argument("--output-dir", type=str, default="",
                   help="Override output directory")
    p.add_argument("--eval-only", action="store_true",
                   help="Only re-evaluate from saved checkpoints")
    p.add_argument("--force", action="store_true",
                   help="Force re-run even if completed")
    return p.parse_args()


def main():
    args = parse_args()

    # Parse methods
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip() in METHOD_REGISTRY]
    elif args.method == "all":
        methods = list(METHOD_REGISTRY.keys())
    else:
        methods = [args.method]

    invalid = [m for m in methods if m not in METHOD_REGISTRY]
    if invalid:
        print(f"Unknown methods: {invalid}. Available: {list(METHOD_REGISTRY)}")
        return

    # Setup
    set_global_seed(SEED, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        print("WARNING: CUDA not available, using CPU!")

    print(f"Phase 5 DA Comparison")
    print(f"  Methods: {methods}")
    print(f"  Epochs: {args.epochs}  Batch size: {args.batch_size}")
    print(f"  LoRA: {not args.no_lora}  (rank={args.lora_rank}, alpha={args.lora_alpha})")
    print(f"  Output: {args.output_dir or OUT_DIR}")

    # GRPO-specific overrides
    cfg_override = {}
    if "grpo" in methods:
        cfg_override.update({
            "grpo_group_size": args.grpo_group_size,
            "grpo_temperature": args.grpo_temperature,
            "grpo_kl_weight": args.grpo_kl_weight,
        })
        print(f"  GRPO: K={args.grpo_group_size}  temp={args.grpo_temperature}  "
              f"kl_weight={args.grpo_kl_weight}")

    # Run each method
    all_results = {}
    timings = {}

    for method in methods:
        info = METHOD_REGISTRY[method]
        print(f"\n{'─'*60}")
        print(f"Running: {info['name']} ({method})")
        print(f"{'─'*60}")

        t0 = time.time()

        if args.eval_only:
            # Load existing results
            result_path = os.path.join(args.output_dir or OUT_DIR, method, "result.json")
            if os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as f:
                    all_results[method] = json.load(f)
                print(f"  Loaded existing results.")
            else:
                print(f"  No results found at {result_path}")
            continue

        try:
            result = run_method(
                method=method,
                device=device,
                cfg_override=cfg_override if method == "grpo" else None,
                use_lora=not args.no_lora,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                output_dir=args.output_dir or None,
                resume=not args.force,
            )
            all_results[method] = result
            timings[method] = time.time() - t0
        except Exception as e:
            print(f"\n  !!! {method} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[method] = {"method": method, "error": str(e), "final_val": {}}

    # ── Summary table ──
    print(f"\n{'='*80}")
    print("Phase 5 DA — Cross-Method Summary")
    print(f"{'='*80}")
    header = (f"{'Method':12s}  {'DA':>8s}  {'BalAcc':>8s}  {'MAPE':>8s}  "
              f"{'UpPrec':>8s}  {'DnPrec':>8s}  {'UpRec':>8s}  {'DnRec':>8s}  "
              f"{'Preds':>22s}  {'Time':>8s}")
    print(header)
    print("-" * len(header))

    for method in methods:
        r = all_results.get(method, {})
        fv = r.get("final_val", {})
        preds = fv.get("pred_counts", {})
        pred_str = f"u={preds.get('up',0)} d={preds.get('down',0)} f={preds.get('flat',0)}"
        t = timings.get(method, 0)
        print(f"  {method:10s}  {fv.get('direction_accuracy',0):8.4f}  "
              f"{fv.get('balanced_accuracy',0):8.4f}  {fv.get('mape',0):8.4f}  "
              f"{fv.get('up_precision',0):8.4f}  {fv.get('down_precision',0):8.4f}  "
              f"{fv.get('recall_up',0):8.4f}  {fv.get('recall_down',0):8.4f}  "
              f"{pred_str:22s}  {t:7.1f}s")

    # ── Best method ──
    best_method = max(
        (m for m in methods if m in all_results and "error" not in all_results[m]),
        key=lambda m: all_results[m].get("final_val", {}).get("balanced_accuracy", 0.0),
        default=None,
    )
    if best_method:
        ba = all_results[best_method].get("final_val", {}).get("balanced_accuracy", 0)
        da = all_results[best_method].get("final_val", {}).get("direction_accuracy", 0)
        print(f"\nBest method: {best_method}  (BalAcc={ba:.4f}, DA={da:.4f})")

    # ── Save cross-method summary ──
    cross = {
        "phase": "5_da_integrated",
        "timestamp": datetime.now().isoformat(),
        "methods": methods,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "use_lora": not args.no_lora,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "grpo_config": {
            "group_size": args.grpo_group_size,
            "temperature": args.grpo_temperature,
            "kl_weight": args.grpo_kl_weight,
        } if "grpo" in methods else {},
        "results": {m: {k: v for k, v in r.items() if k != "history"}
                    for m, r in all_results.items()},
        "timings": timings,
    }
    cross_path = os.path.join(args.output_dir or OUT_DIR, "cross_method_summary.json")
    with open(cross_path, "w", encoding="utf-8") as f:
        json.dump(cross, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {cross_path}")

    return cross


if __name__ == "__main__":
    main()
