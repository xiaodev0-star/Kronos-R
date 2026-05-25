# -*- coding: utf-8 -*-
"""Compare Phase 8 STAR-CAST vs Phase 6 Rollout models on validation set.

Evaluates both models on:
  - 10-step autoregressive rollout metrics
  - Step-level and path-level MAPE, DA, MAE, RMSE
  - Prints side-by-side comparison table
"""

import json
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PostTrainRolloutConfig
from evaluate_predictions import load_model
from posttrain.rollout.data import (
    RolloutWindowDataset,
    rollout_collate,
    resolve_project_path,
)
from reproducibility import set_global_seed

# Reuse evaluation from Phase 6
from posttrain.rollout.train_rollout import (
    _amp_dtype,
    _autocast_context,
    _move_batch,
    _encode_features,
    compute_rollout_metrics,
)


@torch.inference_mode()
def predict_autoregressive_returns(model, tokenizer, loader, cfg, device, amp_enabled, amp_dtype):
    """Pure autoregressive 10-step rollout evaluation."""
    model.eval()
    tokenizer.eval()
    prefix_len = int(cfg.prefix_len)
    horizon = int(cfg.horizon)
    pred_parts = []
    actual_parts = []

    for raw_batch in tqdm(loader, desc="Eval AR", leave=False):
        batch = _move_batch(raw_batch, device)
        idx_c_full, idx_f_full = _encode_features(tokenizer, batch["features"])
        context_c = idx_c_full[:, :prefix_len].clone()
        context_f = idx_f_full[:, :prefix_len].clone()
        step_returns = []

        for step in range(horizon):
            cur_len = int(context_c.size(1))
            cur_time = {key: value[:, :cur_len] for key, value in batch["time"].items()}
            with _autocast_context(device, amp_enabled, amp_dtype):
                logits_c, logits_f, _ = model(
                    context_c, context_f,
                    cur_time["minute"], cur_time["day"],
                    cur_time["month"], cur_time["year"],
                    last_only=True,
                )
            pred_c = logits_c[:, -1, :].float().argmax(dim=-1)
            pred_f = logits_f[:, -1, :].float().argmax(dim=-1)
            decoded = tokenizer.decode(pred_c.unsqueeze(1), pred_f.unsqueeze(1))
            pred_norm = decoded[:, 0, 0].float()
            pred_return = pred_norm * batch["stds"][:, 0] + batch["means"][:, 0]
            step_returns.append(pred_return.detach().cpu())
            if step < horizon - 1:
                context_c = torch.cat([context_c, pred_c.unsqueeze(1)], dim=1)
                context_f = torch.cat([context_f, pred_f.unsqueeze(1)], dim=1)

        pred_parts.append(torch.stack(step_returns, dim=1))
        actual_parts.append(batch["actual_returns"].detach().cpu())

    if not pred_parts:
        return np.empty((0, horizon), dtype=np.float32), np.empty((0, horizon), dtype=np.float32)
    return torch.cat(pred_parts, dim=0).numpy(), torch.cat(actual_parts, dim=0).numpy()


def evaluate_checkpoint(checkpoint_path, label, val_loader, cfg, device):
    """Load a checkpoint and evaluate it on the validation set."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    model, tokenizer = load_model(
        device=device, checkpoint_path=checkpoint_path,
        strict_checkpoint_compat=False,
    )
    tokenizer.eval()
    tokenizer.requires_grad_(False)

    amp_dtype = _amp_dtype("bfloat16")
    amp_enabled = device.type == "cuda"

    pred, actual = predict_autoregressive_returns(
        model=model, tokenizer=tokenizer, loader=val_loader,
        cfg=cfg, device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
    )
    metrics = compute_rollout_metrics(pred, actual, mape_eps=float(cfg.mape_eps))
    return metrics


def print_comparison(phase6_metrics, phase8_metrics):
    """Print side-by-side comparison table."""
    print(f"\n{'='*80}")
    print(f"{'STAR-CAST Phase 8 vs Phase 6 Rollout — Validation Comparison':^80}")
    print(f"{'='*80}")

    metrics_to_show = [
        ("num_samples", "Num Samples", "{:.0f}"),
        ("mape", "Step MAPE (%)", "{:.4f}"),
        ("path_mape", "Path MAPE (%)", "{:.4f}"),
        ("da", "Step DA (%)", "{:.2f}"),
        ("mae", "Step MAE", "{:.6f}"),
        ("rmse", "Step RMSE", "{:.6f}"),
        ("path_mae", "Path MAE", "{:.6f}"),
        ("path_rmse", "Path RMSE", "{:.6f}"),
        ("return_mape", "Return MAPE (%)", "{:.4f}"),
        ("path_return_mape", "Path Return MAPE (%)", "{:.4f}"),
        ("pred_up_ratio", "Pred Up Ratio (%)", "{:.2f}"),
        ("actual_up_ratio", "Actual Up Ratio (%)", "{:.2f}"),
    ]

    print(f"{'Metric':<28} {'Phase 6':>16} {'Phase 8':>16} {'Delta':>16}")
    print(f"{'-'*28} {'-'*16} {'-'*16} {'-'*16}")

    for key, name, fmt in metrics_to_show:
        v6 = phase6_metrics.get(key, float("nan"))
        v8 = phase8_metrics.get(key, float("nan"))

        if isinstance(v6, (int, float)) and isinstance(v8, (int, float)) and not (np.isnan(v6) or np.isnan(v8)):
            delta = v8 - v6
            delta_str = f"{delta:+.4f}" if "MAPE" in name or "MAE" in name or "RMSE" in name else f"{delta:+.2f}"
        else:
            delta_str = "N/A"

        v6_str = fmt.format(v6) if not (isinstance(v6, float) and np.isnan(v6)) else "N/A"
        v8_str = fmt.format(v8) if not (isinstance(v8, float) and np.isnan(v8)) else "N/A"

        print(f"{name:<28} {v6_str:>16} {v8_str:>16} {delta_str:>16}")

    print(f"{'='*80}")

    # Determine winner
    p6_path_mape = phase6_metrics.get("path_mape", float("inf"))
    p8_path_mape = phase8_metrics.get("path_mape", float("inf"))

    if p8_path_mape < p6_path_mape:
        improvement = p6_path_mape - p8_path_mape
        rel_improvement = (improvement / p6_path_mape) * 100.0 if p6_path_mape > 0 else 0.0
        print(f"STAR-CAST Phase 8 WINS: Path MAPE {p8_path_mape:.4f}% vs {p6_path_mape:.4f}%")
        print(f"  Absolute improvement: {improvement:.4f}pp")
        print(f"  Relative improvement: {rel_improvement:.2f}%")
    elif p8_path_mape > p6_path_mape:
        degradation = p8_path_mape - p6_path_mape
        print(f"Phase 6 Rollout still better: Path MAPE {p6_path_mape:.4f}% vs {p8_path_mape:.4f}%")
        print(f"  Absolute degradation: {degradation:.4f}pp")
    else:
        print("Tie on Path MAPE.")

    # Per-step comparison
    print(f"\n{'='*80}")
    print(f"{'Per-Step Path MAPE Comparison':^80}")
    print(f"{'='*80}")
    print(f"{'Step':<10} {'Phase 6':>16} {'Phase 8':>16} {'Delta':>16}")
    print(f"{'-'*10} {'-'*16} {'-'*16} {'-'*16}")

    p6_steps = phase6_metrics.get("per_step", [])
    p8_steps = phase8_metrics.get("per_step", [])
    for i in range(max(len(p6_steps), len(p8_steps))):
        p6_path = p6_steps[i]["path_mape"] if i < len(p6_steps) else float("nan")
        p8_path = p8_steps[i]["path_mape"] if i < len(p8_steps) else float("nan")
        if not np.isnan(p6_path) and not np.isnan(p8_path):
            delta = p8_path - p6_path
            print(f"{i+1:<10} {p6_path:>16.4f} {p8_path:>16.4f} {delta:>+16.4f}")

    print(f"{'='*80}")


def main():
    set_global_seed(42, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Paths
    phase6_path = resolve_project_path("checkpoints/post_train_rollout/rollout_scheduled.pt")
    phase8_path = resolve_project_path("checkpoints/post_train_star_cast/star_cast.pt")
    base_model_path = resolve_project_path("checkpoints/base_model.pt")

    # Check which models exist
    models_to_eval = []
    if os.path.exists(phase6_path):
        models_to_eval.append((phase6_path, "Phase 6 Rollout (Oracle-Guided)"))
    else:
        print(f"Phase 6 checkpoint not found: {phase6_path}")

    if os.path.exists(phase8_path):
        models_to_eval.append((phase8_path, "Phase 8 STAR-CAST"))
    else:
        print(f"Phase 8 checkpoint not found: {phase8_path}")

    if os.path.exists(base_model_path) and len(models_to_eval) < 2:
        models_to_eval.append((base_model_path, "Base Model (no post-train)"))

    if len(models_to_eval) < 2:
        print("Need at least 2 models for comparison. Available models:")
        for path in [phase6_path, phase8_path, base_model_path]:
            print(f"  {path}: {'EXISTS' if os.path.exists(path) else 'MISSING'}")
        sys.exit(1)

    # Build val dataset
    cfg = PostTrainRolloutConfig()
    val_dataset = RolloutWindowDataset("val", cfg=cfg, max_samples=0, seed=59)
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=device.type == "cuda",
        collate_fn=rollout_collate,
    )
    print(f"Validation samples: {len(val_dataset)}")

    # Evaluate both models
    results = {}
    for path, label in models_to_eval:
        metrics = evaluate_checkpoint(path, label, val_loader, cfg, device)
        results[label] = metrics
        print(f"\n{label} Summary:")
        print(f"  Path MAPE: {metrics.get('path_mape', 'N/A'):.4f}%" if isinstance(metrics.get('path_mape'), float) else f"  Path MAPE: {metrics.get('path_mape', 'N/A')}")
        print(f"  Step DA:   {metrics.get('da', 'N/A'):.2f}%" if isinstance(metrics.get('da'), float) else f"  Step DA:   {metrics.get('da', 'N/A')}")

    # Print side-by-side comparison
    if len(results) >= 2:
        labels = list(results.keys())
        print_comparison(results[labels[0]], results[labels[1]])

    # Save results
    output_dir = resolve_project_path("outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "phase8_vs_phase6_comparison.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nComparison results saved to: {output_path}")


if __name__ == "__main__":
    main()
