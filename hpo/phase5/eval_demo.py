# -*- coding: utf-8 -*-
"""Phase 5 DA — Demo set evaluation for all methods + BaseModel.

Evaluates all trained checkpoints on the Demo dataset (held-out last 30 trading days)
and compares with Val set performance to detect overfitting.

Usage::

    python -m hpo.phase5.eval_demo
    python -m hpo.phase5.eval_demo --methods ce,expo,grpo
"""

from __future__ import annotations

import argparse, json, os, sys, time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from hpo.phase5.core import (
    LABEL_DOWN, LABEL_FLAT, LABEL_UP, IGNORE_INDEX,
    SEED, P3_CKPT, TOK_PATH, OUT_DIR, P3, VOCAB, DEFAULT_CFG,
    load_tokenizer, build_model, load_base_weights,
    build_trainable_model, build_ref_model,
    move_batch, prepare_inputs, evaluate, compute_metrics,
    _denorm_last_returns, DirectionDataset, collate_fn,
    _ac,
)
from reproducibility import set_global_seed

METHOD_ORDER = ["basemodel", "ce", "expo", "dpo", "rsft", "grpo"]
LABELS = {"basemodel": "BaseModel (P3)", "ce": "CE", "expo": "ExPO",
          "dpo": "DPO", "rsft": "RSFT", "grpo": "GRPO"}


def load_model_from_checkpoint(method: str, device: torch.device, checkpoint_dir: str | None = None):
    """Load a trained model from its checkpoint. Falls back to base model for 'basemodel'."""
    ckpt_dir = checkpoint_dir or OUT_DIR
    if method == "basemodel":
        model = build_model(device)
        load_base_weights(model, P3_CKPT)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    ckpt_path = os.path.join(ckpt_dir, method, f"phase5_{method}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Check if LoRA or full-FT
    lora_sd = ckpt.get("lora_state_dict", {})
    use_lora = bool(lora_sd)

    model = build_trainable_model(device, use_lora=use_lora)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def evaluate_on_demo(device: torch.device, methods: list[str] | None = None, checkpoint_dir: str | None = None):
    """Evaluate all methods + BaseModel on the Demo dataset."""
    ckpt_dir = checkpoint_dir or OUT_DIR
    if methods is None:
        methods = [m for m in METHOD_ORDER
                   if m == "basemodel" or os.path.exists(os.path.join(ckpt_dir, m, "result.json"))]

    print(f"\n{'='*60}")
    print("Phase 5 DA — Demo Set Evaluation")
    print(f"{'='*60}")
    print(f"  Methods: {methods}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"  Device: {device}")

    # ── Load Demo data ──
    demo_path = os.path.join(_PROJECT_ROOT, "dataset_demo.pt")
    if not os.path.exists(demo_path):
        print(f"Demo dataset not found: {demo_path}")
        return None

    demo_payload = torch.load(demo_path, map_location="cpu", weights_only=False)
    demo_returns = _denorm_last_returns(demo_payload)
    abs_r = np.abs(demo_returns); abs_r = abs_r[np.isfinite(abs_r)]
    demo_eps = max(1e-5, float(np.median(abs_r)) * DEFAULT_CFG["epsilon_scale"])
    print(f"  Demo epsilon = {demo_eps:.6f}  (median |r| = {np.median(abs_r):.6f})")

    # Use "class" mode (count all labels, don't ignore flat) for fair comparison
    demo_ds = DirectionDataset(demo_payload, np.arange(len(demo_returns), dtype=np.int64),
                                demo_returns, demo_eps, "class")
    demo_loader = DataLoader(demo_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    print(f"  Demo samples: {len(demo_ds)}")
    print(f"  Demo class dist: down={demo_ds.class_counts[0]} flat={demo_ds.class_counts[1]} up={demo_ds.class_counts[2]}")

    # ── Load tokenizer ──
    tokenizer = load_tokenizer(device)

    # ── Evaluate each model ──
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    amp_enabled = device.type == "cuda" and amp_dtype is not None

    all_demo_metrics = {}
    all_val_metrics = {}
    timings = {}

    for method in methods:
        print(f"\n  Evaluating {LABELS.get(method, method)} ({method})...")
        t0 = time.time()

        try:
            model = load_model_from_checkpoint(method, device, ckpt_dir)
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
            continue

        demo_metrics = evaluate(model, tokenizer, demo_loader, device, amp_enabled, amp_dtype)
        elapsed = time.time() - t0
        timings[method] = elapsed

        all_demo_metrics[method] = demo_metrics
        print(f"    DA={demo_metrics['direction_accuracy']:.4f}  "
              f"BalAcc={demo_metrics['balanced_accuracy']:.4f}  "
              f"MAPE={demo_metrics['mape']:.4f}  "
              f"preds={demo_metrics['pred_counts']}  "
              f"time={elapsed:.1f}s")

        # Load Val metrics from saved result
        if method == "basemodel":
            # Compute Val metrics for basemodel
            val_payload = torch.load(os.path.join(_PROJECT_ROOT, "dataset_val.pt"),
                                     map_location="cpu", weights_only=False)
            val_returns = _denorm_last_returns(val_payload)
            val_abs = np.abs(val_returns); val_abs = val_abs[np.isfinite(val_abs)]
            val_eps = max(1e-5, float(np.median(val_abs)) * DEFAULT_CFG["epsilon_scale"])
            val_ds = DirectionDataset(val_payload, np.arange(len(val_returns), dtype=np.int64),
                                      val_returns, val_eps, "class")
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
            val_metrics = evaluate(model, tokenizer, val_loader, device, amp_enabled, amp_dtype)
            all_val_metrics[method] = val_metrics
        else:
            result_path = os.path.join(ckpt_dir, method, "result.json")
            if os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as f:
                    r = json.load(f)
                all_val_metrics[method] = r.get("best_metrics", r.get("final_val", {}))

        # Clean up
        del model
        torch.cuda.empty_cache()

    # ── Save results FIRST (before printing, in case of encoding errors) ──
    demo_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methods": methods,
        "demo_metrics": {m: all_demo_metrics.get(m, {}) for m in methods},
        "val_metrics": {m: all_val_metrics.get(m, {}) for m in methods},
        "timings": timings,
    }
    demo_path_out = os.path.join(OUT_DIR, "demo_eval_results.json")
    with open(demo_path_out, "w", encoding="utf-8") as f:
        json.dump(demo_results, f, indent=2, ensure_ascii=False)
    print(f"\nDemo results saved to: {demo_path_out}")

    # ── Print comparison table ──
    _print_comparison(all_val_metrics, all_demo_metrics, timings, methods)

    return demo_results


def _print_comparison(val_metrics, demo_metrics, timings, methods):
    """Print Val vs Demo comparison table."""
    print(f"\n{'='*100}")
    print("Phase 5 DA — Val vs Demo Comparison")
    print(f"{'='*100}")

    # Header
    header = (f"{'Method':14s}  {'Val DA':>8s}  {'Demo DA':>8s}  {'dDA':>8s}  "
              f"{'Val BalAcc':>10s}  {'Demo BalAcc':>10s}  {'dBalAcc':>10s}  "
              f"{'Val MAPE':>8s}  {'Demo MAPE':>8s}  {'dMAPE':>8s}  {'Time':>6s}")
    print(header)
    print("-" * len(header))

    base_demo_da = demo_metrics.get("basemodel", {}).get("direction_accuracy", 0)
    base_demo_ba = demo_metrics.get("basemodel", {}).get("balanced_accuracy", 0)

    for method in methods:
        vm = val_metrics.get(method, {})
        dm = demo_metrics.get(method, {})

        val_da = vm.get("direction_accuracy", 0)
        demo_da = dm.get("direction_accuracy", 0)
        val_ba = vm.get("balanced_accuracy", 0)
        demo_ba = dm.get("balanced_accuracy", 0)
        val_mape = vm.get("mape", 0)
        demo_mape = dm.get("mape", 0)

        delta_da = demo_da - val_da
        delta_ba = demo_ba - val_ba
        delta_mape = demo_mape - val_mape

        # Overfitting indicators
        da_warn = " !" if delta_da < -0.005 else ""
        ba_warn = " !" if delta_ba < -0.005 else ""

        time_str = f"{timings.get(method, 0):.0f}s"

        print(f"  {LABELS.get(method, method):12s}  {val_da:8.4f}  {demo_da:8.4f}  "
              f"{delta_da:+8.4f}{da_warn}  {val_ba:10.4f}  {demo_ba:10.4f}  "
              f"{delta_ba:+10.4f}{ba_warn}  {val_mape:8.4f}  {demo_mape:8.4f}  "
              f"{delta_mape:+8.4f}  {time_str:>6s}")

    # Vs BaseModel improvement
    print(f"\n{'─'*100}")
    print("Improvement vs BaseModel on Demo:")
    print(f"{'Method':14s}  {'dDA vs Base':>12s}  {'dBalAcc vs Base':>16s}  {'dMAPE vs Base':>14s}")
    print("-" * 60)
    for method in methods:
        if method == "basemodel":
            continue
        dm = demo_metrics.get(method, {})
        delta_da = dm.get("direction_accuracy", 0) - base_demo_da
        delta_ba = dm.get("balanced_accuracy", 0) - base_demo_ba
        delta_mape = dm.get("mape", 0) - demo_metrics.get("basemodel", {}).get("mape", 0)
        sig_da = "+" if delta_da > 0.001 else ("~" if abs(delta_da) <= 0.001 else "-")
        sig_ba = "+" if delta_ba > 0.001 else ("~" if abs(delta_ba) <= 0.001 else "-")
        print(f"  {LABELS.get(method, method):12s}  {delta_da:+12.4f} {sig_da}  "
              f"{delta_ba:+16.4f} {sig_ba}  {delta_mape:+14.4f}")

    # Overfitting summary
    print(f"\n{'─'*100}")
    print("Overfitting Check (higher is better for DA/BalAcc, lower for MAPE):")
    print(f"{'Method':14s}  {'Val→Demo DA':>12s}  {'Val→Demo BalAcc':>14s}  {'Val→Demo MAPE':>14s}")
    print("-" * 58)
    for method in methods:
        vm = val_metrics.get(method, {})
        dm = demo_metrics.get(method, {})
        da_gap = dm.get("direction_accuracy", 0) - vm.get("direction_accuracy", 0)
        ba_gap = dm.get("balanced_accuracy", 0) - vm.get("balanced_accuracy", 0)
        mape_gap = dm.get("mape", 0) - vm.get("mape", 0)
        da_status = "OVERFIT" if da_gap < -0.01 else ("ok" if da_gap > -0.005 else "mild")
        ba_status = "OVERFIT" if ba_gap < -0.01 else ("ok" if ba_gap > -0.005 else "mild")
        print(f"  {LABELS.get(method, method):12s}  {da_gap:+10.4f} ({da_status:>7s})  "
              f"{ba_gap:+12.4f} ({ba_status:>7s})  {mape_gap:+12.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 DA — Demo set evaluation")
    p.add_argument("--methods", type=str, default="",
                   help="Comma-separated methods to evaluate (default: all available)")
    p.add_argument("--output-dir", type=str, default="",
                   help="Override output directory for loading checkpoints and saving results")
    p.add_argument("--checkpoint-dir", type=str, default="",
                   help="Directory containing method checkpoints (default: same as output-dir)")
    return p.parse_args()


if __name__ == "__main__":
    global OUT_DIR
    args = parse_args()
    base_dir = args.output_dir or OUT_DIR
    ckpt_dir = args.checkpoint_dir or base_dir

    if args.output_dir:
        OUT_DIR = base_dir
        import hpo.phase5.core as _core
        _core.OUT_DIR = base_dir

    set_global_seed(SEED, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]

    evaluate_on_demo(device, methods, ckpt_dir)
