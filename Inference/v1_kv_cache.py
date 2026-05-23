# -*- coding: utf-8 -*-
"""v1: KV-Cache Autoregressive Inference

核心优化: 使用 KV-cache 避免每次 AR 步骤重新计算所有历史 token 的注意力。

原始方法: 每个 AR 步骤执行完整前向传播 (1024+ tokens) → 10步=10次完整前向
v1 方法:   Prefix 一次完整前向 + 缓存 KV → 每步仅处理1个新token

预期加速: 5-7x (对于10步自回归)

用法:
    python -m Inference.v1_kv_cache --mode demo --horizon 10 --num-samples 100
"""

import argparse
import gc
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PARENT)

from Inference.utils import (
    setup_device, autocast_ctx, load_model_and_tokenizer,
    load_rollout_data, prepare_inference_batch, compute_metrics, Timer,
    baseline_ar_predict, report_memory,
)


# ─── v1: KV-Cache AR Inference ────────────────────────────────────────────────

@torch.no_grad()
def v1_ar_predict(model, tokenizer, features, means, stds, times, horizon, device,
                  use_amp=True, amp_dtype="bf16"):
    """v1 optimized AR prediction using KV-cache.

    Process:
      1. Tokenizer encode → coarse + fine indices
      2. Single full backbone forward with KV cache collection
      3. For each step: incremental block forward (K,V cached) → post-backbone → decode
    """
    B = features.size(0)
    prefix_len = features.size(1)

    # Step 1: Tokenize
    idx_c, idx_f = tokenizer.encode(features)
    t_min = times["minute"][:, :prefix_len]
    t_day = times["day"][:, :prefix_len]
    t_month = times["month"][:, :prefix_len]
    t_year = times["year"][:, :prefix_len]

    # Step 2: KV-cache AR predict
    with autocast_ctx(device, use_amp, amp_dtype):
        pred_c, pred_f = model.predict_ar_kv_cache(
            idx_c, idx_f, t_min, t_day, t_month, t_year,
            horizon=horizon,
            temperature=1.0,
            use_sampling=False,
        )

    # Step 3: Decode all predictions
    pred_returns = []
    for step in range(horizon):
        pc = pred_c[:, step:step + 1]
        pf = pred_f[:, step:step + 1]
        decoded = tokenizer.decode(pc, pf)
        pred_ret = decoded[:, 0, 0].float() * stds[:, 0] + means[:, 0]
        pred_returns.append(pred_ret)

    return torch.stack(pred_returns, dim=1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="v1: KV-Cache AR Inference")
    parser.add_argument("--mode", default="demo", choices=["demo", "val", "train"])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of samples to predict (0=all)")
    parser.add_argument("--num-stocks", type=int, default=10,
                        help="Number of stocks to benchmark")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per inference call")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path")
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--benchmark", action="store_true", default=True,
                        help="Run benchmark comparison vs baseline")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    args = parser.parse_args(argv)

    device = setup_device()
    print(f"\n{'='*60}")
    print(f"v1: KV-Cache Optimized Inference")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Horizon: {args.horizon}")
    print(f"Mode: {args.mode}")

    # Load model and tokenizer
    print("\n[1/4] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        device, args.checkpoint, args.tokenizer_path
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: {param_count:,} parameters")

    # Load data
    print(f"\n[2/4] Loading {args.mode} data...")
    payload = load_rollout_data(args.mode)

    # Prepare samples
    print(f"\n[3/4] Preparing inference batches...")
    total_windows = payload["features"].size(0)
    num_samples = min(args.num_samples, total_windows) if args.num_samples > 0 else total_windows
    print(f"  Total windows: {total_windows}, Using: {num_samples}")

    if args.benchmark:
        # Select random stocks for benchmark
        rng = np.random.default_rng(42)
        available = min(num_samples, args.num_stocks)
        stock_indices = rng.choice(num_samples, size=available, replace=False).tolist()
    else:
        stock_indices = list(range(min(num_samples, args.num_stocks)))

    # Warmup
    print(f"\n[4/4] Warmup ({args.warmup_runs} runs)...")
    for i in range(args.warmup_runs):
        idx = stock_indices[i % len(stock_indices)]
        batch = prepare_inference_batch(payload, torch.tensor([idx]), device)
        _ = v1_ar_predict(
            model, tokenizer, batch["features"], batch["means"], batch["stds"],
            batch["time"], args.horizon, device,
            use_amp=args.use_amp, amp_dtype=args.amp_dtype,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ─── Benchmark: v1 vs baseline ───
    if args.benchmark:
        print(f"\n{'='*60}")
        print(f"Benchmark: v1 (KV-Cache) vs Baseline (Full Forward)")
        print(f"{'='*60}")

        baseline_times = []
        v1_times = []

        for run in tqdm(range(args.benchmark_runs), desc="Benchmarking"):
            idx = stock_indices[run % len(stock_indices)]
            batch = prepare_inference_batch(payload, torch.tensor([idx]), device)
            feats = batch["features"]
            means = batch["means"]
            stds = batch["stds"]
            times_b = batch["time"]
            h = args.horizon

            # Baseline timing
            with Timer(sync_cuda=True) as t_base:
                _ = baseline_ar_predict(
                    model, tokenizer, feats, means, stds, times_b, h, device,
                    use_amp=args.use_amp, amp_dtype=args.amp_dtype,
                )
            baseline_times.append(t_base.ms)

            # v1 timing
            with Timer(sync_cuda=True) as t_v1:
                _ = v1_ar_predict(
                    model, tokenizer, feats, means, stds, times_b, h, device,
                    use_amp=args.use_amp, amp_dtype=args.amp_dtype,
                )
            v1_times.append(t_v1.ms)

        baseline_mean = np.mean(baseline_times)
        baseline_std = np.std(baseline_times)
        v1_mean = np.mean(v1_times)
        v1_std = np.std(v1_times)
        speedup = baseline_mean / v1_mean if v1_mean > 0 else float("inf")

        print(f"\n{'─'*50}")
        print(f"  Baseline (Full Forward): {baseline_mean:.2f} ± {baseline_std:.2f} ms")
        print(f"  v1 (KV-Cache):          {v1_mean:.2f} ± {v1_std:.2f} ms")
        print(f"  Speedup:                {speedup:.2f}x")
        print(f"{'─'*50}")

    # ─── Full prediction on all selected stocks ───
    print(f"\n{'='*60}")
    print(f"Running full prediction on {len(stock_indices)} stocks...")
    print(f"{'='*60}")

    all_predictions = []
    all_actuals = []
    total_time = 0.0

    for i, idx in enumerate(tqdm(stock_indices, desc="Predicting")):
        batch = prepare_inference_batch(payload, torch.tensor([idx]), device)
        actual = batch["actual_returns"].cpu()

        with Timer(sync_cuda=True) as t:
            pred = v1_ar_predict(
                model, tokenizer, batch["features"], batch["means"], batch["stds"],
                batch["time"], args.horizon, device,
                use_amp=args.use_amp, amp_dtype=args.amp_dtype,
            )

        total_time += t.ms
        all_predictions.append(pred.cpu())
        all_actuals.append(actual)

    # ─── Compute metrics ───
    preds = torch.cat(all_predictions, dim=0).numpy()
    actuals = torch.cat(all_actuals, dim=0).numpy()

    per_step_metrics = {}
    for step in range(args.horizon):
        m = compute_metrics(preds[:, step], actuals[:, step])
        per_step_metrics[f"step_{step + 1}"] = m

    overall = compute_metrics(preds.reshape(-1), actuals.reshape(-1))

    print(f"\n{'='*60}")
    print(f"Results Summary (v1 KV-Cache)")
    print(f"{'='*60}")
    print(f"  Stocks: {len(stock_indices)}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Total time: {total_time:.0f} ms")
    print(f"  Per stock: {total_time / len(stock_indices):.1f} ms")
    print(f"\n  Overall Metrics:")
    print(f"    MAPE: {overall['mape']:.4f}%")
    print(f"    DA:   {overall['da']:.2f}%")
    print(f"    MAE:  {overall['mae']:.6f}")
    print(f"    RMSE: {overall['rmse']:.6f}")

    print(f"\n  Per-Step Metrics:")
    for step in range(args.horizon):
        m = per_step_metrics[f"step_{step + 1}"]
        print(f"    Day {step + 1:2d}: MAPE={m['mape']:.4f}% DA={m['da']:.2f}% MAE={m['mae']:.6f}")

    report_memory(device)

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results = {
            "version": "v1_kv_cache",
            "device": str(device),
            "horizon": args.horizon,
            "num_stocks": len(stock_indices),
            "total_time_ms": total_time,
            "per_stock_ms": total_time / len(stock_indices),
            "overall_metrics": overall,
            "per_step_metrics": per_step_metrics,
        }
        if args.benchmark:
            results["benchmark"] = {
                "baseline_mean_ms": baseline_mean,
                "baseline_std_ms": baseline_std,
                "v1_mean_ms": v1_mean,
                "v1_std_ms": v1_std,
                "speedup": speedup,
            }
        path = os.path.join(args.output_dir, "v1_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved: {path}")

    return preds, actuals


if __name__ == "__main__":
    main()
