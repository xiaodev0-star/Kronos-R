# -*- coding: utf-8 -*-
"""v1: Batch Autoregressive Inference

核心优化: 将多个stock的AR推理合并为批次处理,提高GPU利用率。

原始方法: 每个stock独立AR循环, 每次只处理batch_size=1, GPU利用率低
v1方法:   N个stock一起经历AR循环, 每次forward处理batch_size=N

关键洞察: 模型太小 (dim=384, depth=3), 单token/单stock推理GPU利用率极低。
         批处理是解决此问题的唯一有效方法。

预期加速:
  - batch_size=4:  ~3x
  - batch_size=8:  ~5x
  - batch_size=16: ~7x (受8GB VRAM限制)

用法:
    python -m Inference.v1_batch_ar --horizon 10 --batch-size 8 --num-stocks 100
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PARENT)

from Inference.utils import (
    setup_device, autocast_ctx, load_model_and_tokenizer,
    load_rollout_data, prepare_inference_batch, compute_metrics, Timer,
    baseline_ar_predict, report_memory,
)


# ─── v1: Batch AR Inference ───────────────────────────────────────────────────

@torch.no_grad()
def v1_batch_ar_predict(model, tokenizer, features, means, stds, times, horizon, device,
                        use_amp=True, amp_dtype="bf16"):
    """Batch autoregressive prediction.

    All stocks in the batch share the same AR loop.
    Each forward pass processes the entire batch, maximizing GPU utilization.

    Args:
        features: [B, prefix_len, 6] normalized features
        means, stds: [B, 6] per-stock normalization stats
        times: dict of [B, total_len] time features
        horizon: number of future steps

    Returns:
        pred_returns: [B, horizon] predicted log returns
    """
    B = features.size(0)
    prefix_len = features.size(1)

    # Step 1: Tokenize entire batch at once
    with autocast_ctx(device, use_amp, amp_dtype):
        idx_c, idx_f = tokenizer.encode(features)

    # Step 2: Initialize sequence with prefix tokens
    cur_c = idx_c[:, :prefix_len].clone()
    cur_f = idx_f[:, :prefix_len].clone()

    # Pre-allocate output
    pred_indices_c = torch.empty(B, horizon, device=device, dtype=torch.long)
    pred_indices_f = torch.empty(B, horizon, device=device, dtype=torch.long)

    # Step 3: AR loop — each iteration processes the entire batch
    for step in range(horizon):
        sl = cur_c.size(1)

        # Slice time features for current sequence length
        step_time = {
            k: times[k][:, :sl].contiguous()
            for k in ("minute", "day", "month", "year")
        }

        with autocast_ctx(device, use_amp, amp_dtype):
            logits_c, logits_f, _ = model(
                cur_c, cur_f,
                step_time["minute"], step_time["day"],
                step_time["month"], step_time["year"],
                last_only=True,
            )

        # Argmax prediction
        pc = logits_c[:, -1, :].float().argmax(dim=-1)
        pf = logits_f[:, -1, :].float().argmax(dim=-1)

        pred_indices_c[:, step] = pc
        pred_indices_f[:, step] = pf

        # Append predicted tokens for next step
        if step < horizon - 1:
            cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
            cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)

    # Step 4: Decode all predictions at once
    pred_returns = torch.empty(B, horizon, device=device, dtype=torch.float32)
    for step in range(horizon):
        decoded = tokenizer.decode(
            pred_indices_c[:, step:step + 1],
            pred_indices_f[:, step:step + 1],
        )
        pred_returns[:, step] = (
            decoded[:, 0, 0].float() * stds[:, 0] + means[:, 0]
        )

    return pred_returns


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="v1: Batch AR Inference")
    parser.add_argument("--mode", default="demo", choices=["demo", "val", "train"])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-stocks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of stocks per batch (higher = faster, limited by VRAM)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    args = parser.parse_args(argv)

    device = setup_device()
    print(f"\n{'='*60}")
    print(f"v1: Batch Autoregressive Inference")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Horizon: {args.horizon}")
    print(f"Batch size: {args.batch_size}")

    # Load model
    print("\n[1/4] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        device, args.checkpoint, args.tokenizer_path
    )
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load data
    print(f"\n[2/4] Loading {args.mode} data...")
    payload = load_rollout_data(args.mode)
    total = payload["features"].size(0)
    num_stocks = min(args.num_stocks, total)
    print(f"  {total} windows available, using {num_stocks}")

    # Select stocks
    rng = np.random.default_rng(42)
    indices = rng.choice(total, size=num_stocks, replace=False).tolist()

    # ─── Benchmark ───
    if args.benchmark:
        print(f"\n[3/4] Benchmarking...")

        # Baseline: process one stock at a time
        print("  Baseline (batch_size=1)...")
        base_times = []
        for run in tqdm(range(args.benchmark_runs), desc="  Baseline", leave=False):
            idx = indices[run % len(indices)]
            b = prepare_inference_batch(payload, torch.tensor([idx]), device)
            with Timer(sync_cuda=True) as t:
                _ = baseline_ar_predict(
                    model, tokenizer, b["features"], b["means"],
                    b["stds"], b["time"], args.horizon, device,
                    use_amp=args.use_amp, amp_dtype=args.amp_dtype,
                )
            base_times.append(t.ms)

        base_mean = np.mean(base_times)
        print(f"  Baseline per-stock: {base_mean:.1f} ms")

        # v1: batch processing
        print(f"  v1 (batch_size={args.batch_size})...")
        v1_times = []
        for run in tqdm(range(args.benchmark_runs), desc="  v1 Batch", leave=False):
            start_idx = (run * args.batch_size) % max(1, num_stocks - args.batch_size)
            batch_indices = indices[start_idx:start_idx + args.batch_size]
            b = prepare_inference_batch(
                payload, torch.tensor(batch_indices, dtype=torch.long), device
            )
            with Timer(sync_cuda=True) as t:
                _ = v1_batch_ar_predict(
                    model, tokenizer, b["features"], b["means"],
                    b["stds"], b["time"], args.horizon, device,
                    use_amp=args.use_amp, amp_dtype=args.amp_dtype,
                )
            v1_times.append(t.ms / len(batch_indices))  # per-stock time

        v1_mean = np.mean(v1_times)
        speedup = base_mean / v1_mean
        print(f"  v1 per-stock: {v1_mean:.1f} ms")
        print(f"  Speedup: {speedup:.2f}x")
    else:
        base_mean = v1_mean = speedup = 0

    # ─── Full prediction ───
    print(f"\n[4/4] Running full prediction on {num_stocks} stocks...")

    all_preds = []
    all_actuals = []
    total_time = 0.0
    batch_count = 0

    for start in tqdm(range(0, num_stocks, args.batch_size), desc="  Predicting"):
        end = min(start + args.batch_size, num_stocks)
        batch_indices = indices[start:end]
        b = prepare_inference_batch(
            payload, torch.tensor(batch_indices, dtype=torch.long), device
        )

        with Timer(sync_cuda=True) as t:
            pred = v1_batch_ar_predict(
                model, tokenizer, b["features"], b["means"],
                b["stds"], b["time"], args.horizon, device,
                use_amp=args.use_amp, amp_dtype=args.amp_dtype,
            )

        total_time += t.ms
        batch_count += 1
        all_preds.append(pred.cpu())
        all_actuals.append(b["actual_returns"].cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    actuals = torch.cat(all_actuals, dim=0).numpy()

    # Metrics
    per_step = {}
    for step in range(args.horizon):
        per_step[f"step_{step + 1}"] = compute_metrics(preds[:, step], actuals[:, step])

    overall = compute_metrics(preds.reshape(-1), actuals.reshape(-1))

    print(f"\n{'='*60}")
    print(f"v1 Results Summary")
    print(f"{'='*60}")
    print(f"  Stocks: {num_stocks}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total time: {total_time:.0f} ms")
    print(f"  Per stock: {total_time / num_stocks:.1f} ms")
    print(f"  Throughput: {num_stocks / (total_time / 1000):.1f} stocks/sec")
    print(f"  Overall MAPE: {overall['mape']:.4f}%  DA: {overall['da']:.2f}%")
    print(f"  Overall MAE: {overall['mae']:.6f}  RMSE: {overall['rmse']:.6f}")
    if args.benchmark:
        print(f"  Speedup vs baseline: {speedup:.2f}x")

    report_memory(device)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results = {
            "version": "v1_batch_ar",
            "device": str(device),
            "horizon": args.horizon,
            "batch_size": args.batch_size,
            "num_stocks": num_stocks,
            "total_time_ms": total_time,
            "per_stock_ms": total_time / num_stocks,
            "throughput_stocks_per_sec": num_stocks / (total_time / 1000),
            "overall_metrics": overall,
            "per_step_metrics": per_step,
        }
        if args.benchmark:
            results["benchmark"] = {
                "baseline_per_stock_ms": base_mean,
                "v1_per_stock_ms": v1_mean,
                "speedup": speedup,
            }
        path = os.path.join(args.output_dir, "v1_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved: {path}")

    return preds, actuals


if __name__ == "__main__":
    main()
