# -*- coding: utf-8 -*-
"""Idea 1 — Sampling-based Confidence Interval inference (no training).

Load a trained Kronos-R model (base or rollout-finetuned) and construct
multi-step prediction intervals via temperature sampling:

  1. Feed the 1023-token observed prefix.
  2. At each future step, sample N alternative tokens from the model's
     temperature-scaled output distribution.
  3. Decode all N samples to log-returns; sort them; take α/2 and 1-α/2
     empirical quantiles as the prediction interval.
  4. Feed the *argmax* (or median) token as context for the next step.
  5. Repeat for H steps → H-step CI trajectories.

This is a pure inference method — no further training is required.
"""

import argparse
import json
import os
from contextlib import nullcontext
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PostTrainRolloutConfig
from evaluate_predictions import load_model
from posttrain.ci.data import RolloutWindowDataset, rollout_collate
from posttrain.ci.eval_ci import compute_ci_metrics
from reproducibility import set_global_seed


def _amp_dtype(name):
    name = str(name).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return None


def _autocast_context(device, enabled, dtype):
    if device.type != "cuda" or not enabled or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def predict_ci_sampling(
    model,
    tokenizer,
    loader,
    cfg,
    device,
    amp_enabled,
    amp_dtype,
    num_samples=64,
    temperature=1.0,
    confidence_level=0.80,
    feed_mode="argmax",  # "argmax" | "median" | "random"
):
    """Autoregressive multi-step CI prediction via temperature sampling.

    At each step the model produces a distribution over next tokens.
    We sample *num_samples* tokens, decode them, and take empirical
    quantiles for the prediction interval.  A single "context" token is
    selected (by *feed_mode*) and appended for the next step.

    Returns
    -------
    pred_lower, pred_upper, pred_median, actual_returns : ndarray [N, H]
    """
    model.eval()
    tokenizer.eval()
    prefix_len = int(cfg.prefix_len)
    horizon = int(cfg.horizon)

    pred_lower_parts = []
    pred_upper_parts = []
    pred_median_parts = []
    actual_parts = []

    alpha = 1.0 - float(confidence_level)
    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0

    for raw_batch in tqdm(loader, desc="CI sampling", leave=False):
        batch = {
            "features": raw_batch["features"].to(device=device, dtype=torch.float32, non_blocking=True),
            "time": {
                key: value.to(device=device, dtype=torch.long, non_blocking=True)
                for key, value in raw_batch["time"].items()
            },
            "means": raw_batch["means"].to(device=device, dtype=torch.float32, non_blocking=True),
            "stds": raw_batch["stds"].to(device=device, dtype=torch.float32, non_blocking=True),
        }

        features = batch["features"]
        B = int(features.size(0))

        # ── encode prefix ──
        idx_coarse_full, idx_fine_full = tokenizer.encode(features)
        context_c = idx_coarse_full[:, :prefix_len].clone()
        context_f = idx_fine_full[:, :prefix_len].clone()

        step_lower = []
        step_upper = []
        step_median = []

        for step in range(horizon):
            cur_len = int(context_c.size(1))
            cur_time = {
                key: value[:, :cur_len]
                for key, value in batch["time"].items()
            }

            with _autocast_context(device, amp_enabled, amp_dtype):
                logits_c, logits_f, _ = model(
                    context_c, context_f,
                    cur_time["minute"], cur_time["day"],
                    cur_time["month"], cur_time["year"],
                    last_only=True,
                )

            last_logits_c = logits_c[:, -1, :].float()
            last_logits_f = logits_f[:, -1, :].float()

            # ── sample N tokens ──
            temp = max(float(temperature), 1e-5)
            probs_c = torch.softmax(last_logits_c / temp, dim=-1)
            probs_f = torch.softmax(last_logits_f / temp, dim=-1)

            samples_c = torch.multinomial(probs_c, num_samples=int(num_samples), replacement=True)
            samples_f = torch.multinomial(probs_f, num_samples=int(num_samples), replacement=True)

            # ── decode all samples at once ──
            decoded = tokenizer.decode(samples_c, samples_f)  # [B, N, 6]
            pred_norms = decoded[:, :, 0].float()
            pred_returns = pred_norms * batch["stds"][:, 0:1] + batch["means"][:, 0:1]

            # ── empirical quantiles ──
            sorted_returns = pred_returns.sort(dim=1).values  # [B, N]
            idx_low = max(0, min(int(num_samples) - 1, int(low_q * int(num_samples))))
            idx_high = max(0, min(int(num_samples) - 1, int(high_q * int(num_samples))))
            lower = sorted_returns[:, idx_low]
            upper = sorted_returns[:, idx_high]
            median = sorted_returns[:, int(num_samples) // 2]

            step_lower.append(lower.detach().cpu())
            step_upper.append(upper.detach().cpu())
            step_median.append(median.detach().cpu())

            # ── select context token for next step ──
            if step < horizon - 1:
                if feed_mode == "argmax":
                    next_c = last_logits_c.argmax(dim=-1)
                    next_f = last_logits_f.argmax(dim=-1)
                elif feed_mode == "median":
                    median_pos = int(num_samples) // 2
                    sorted_idx = pred_returns.argsort(dim=1)
                    next_c = samples_c[torch.arange(B, device=device), sorted_idx[:, median_pos]]
                    next_f = samples_f[torch.arange(B, device=device), sorted_idx[:, median_pos]]
                else:  # random
                    rand_idx = torch.randint(0, int(num_samples), (B,), device=device)
                    next_c = samples_c[torch.arange(B, device=device), rand_idx]
                    next_f = samples_f[torch.arange(B, device=device), rand_idx]

                context_c = torch.cat([context_c, next_c.unsqueeze(1)], dim=1)
                context_f = torch.cat([context_f, next_f.unsqueeze(1)], dim=1)

        pred_lower_parts.append(torch.stack(step_lower, dim=1))
        pred_upper_parts.append(torch.stack(step_upper, dim=1))
        pred_median_parts.append(torch.stack(step_median, dim=1))
        actual_parts.append(raw_batch["actual_returns"].detach().cpu())

    if not pred_lower_parts:
        empty = np.empty((0, horizon), dtype=np.float32)
        return empty, empty, empty, empty

    pred_lower = torch.cat(pred_lower_parts, dim=0).numpy()
    pred_upper = torch.cat(pred_upper_parts, dim=0).numpy()
    pred_median = torch.cat(pred_median_parts, dim=0).numpy()
    actual = torch.cat(actual_parts, dim=0).numpy()
    return pred_lower, pred_upper, pred_median, actual


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Sampling-based CI prediction")
    parser.add_argument("--checkpoint-path", default=PostTrainRolloutConfig.checkpoint_path)
    parser.add_argument("--output-dir", default="outputs/ci_sampling")
    parser.add_argument("--prefix-len", type=int, default=PostTrainRolloutConfig.prefix_len)
    parser.add_argument("--horizon", type=int, default=PostTrainRolloutConfig.horizon)
    parser.add_argument("--num-samples", type=int, default=64,
                        help="Number of temperature samples per step")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--confidence-level", type=float, default=0.80,
                        help="Nominal confidence level (e.g. 0.80)")
    parser.add_argument("--feed-mode", choices=["argmax", "median", "random"], default="argmax")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--use-amp", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--amp-dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-tag", default="")
    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    set_global_seed(int(args.seed), deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    amp_dtype = _amp_dtype(args.amp_dtype)
    amp_enabled = bool(args.use_amp) and device.type == "cuda" and amp_dtype is not None

    # ── build config namespace ──
    cfg = argparse.Namespace(
        checkpoint_path=args.checkpoint_path,
        prefix_len=int(args.prefix_len),
        horizon=int(args.horizon),
        output_dir=args.output_dir,
        batch_size=max(1, int(args.batch_size)),
        max_stocks=int(args.max_stocks),
        max_samples=int(args.max_samples),
        cache_dir=PostTrainRolloutConfig.cache_dir,
        stride_ratio=PostTrainRolloutConfig.stride_ratio,
        random_seed=int(args.seed),
        mape_eps=float(PostTrainRolloutConfig.mape_eps),
    )

    model, tokenizer = load_model(device=device, checkpoint_path=cfg.checkpoint_path,
                                   strict_checkpoint_compat=False)
    tokenizer.eval()
    tokenizer.requires_grad_(False)

    val_dataset = RolloutWindowDataset(
        "val", cfg=cfg,
        max_samples=int(args.max_samples) if int(args.max_samples) > 0 else 0,
        seed=int(args.seed) + 17,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        drop_last=False,
        collate_fn=rollout_collate,
    )

    print(f"Model: {args.checkpoint_path}")
    print(f"Val windows: {len(val_dataset)}")
    print(f"CI params: samples={args.num_samples}, temp={args.temperature}, "
          f"confidence={args.confidence_level}, feed={args.feed_mode}")

    pred_lower, pred_upper, pred_median, actual = predict_ci_sampling(
        model=model,
        tokenizer=tokenizer,
        loader=val_loader,
        cfg=cfg,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        num_samples=int(args.num_samples),
        temperature=float(args.temperature),
        confidence_level=float(args.confidence_level),
        feed_mode=str(args.feed_mode),
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    npz_path = os.path.join(cfg.output_dir, f"ci_samples{tag}.npz")
    np.savez_compressed(npz_path,
                         pred_lower=pred_lower, pred_upper=pred_upper,
                         pred_median=pred_median, actual=actual)
    print(f"Saved predictions: {npz_path}")

    metrics = compute_ci_metrics(
        pred_lower=pred_lower,
        pred_upper=pred_upper,
        actual_returns=actual,
        confidence_level=float(args.confidence_level),
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    metrics_path = os.path.join(cfg.output_dir, f"ci_metrics{tag}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics: {metrics_path}")

    return metrics


if __name__ == "__main__":
    main()
