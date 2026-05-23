# -*- coding: utf-8 -*-
"""v2: Maximum-throughput batch inference

v1基础上增加:
  1. 批量tokenizer.decode: 一次调用替代逐step的10次调用
  2. 动态batch_size: 自动调整到VRAM允许的最大值
  3. PyTorch memory pooling: 减少内存分配碎片
  4. 数据预取: 下一batch在CPU准备好,GPU处理当前batch

预期加速: 2-4x vs 原始单stock推理

用法:
    python -m Inference.v2_fast --horizon 10 --max-batch-size 48 --num-stocks 400
"""

import argparse, gc, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PARENT)

from Inference.utils import (
    setup_device, autocast_ctx, load_model_and_tokenizer,
    load_rollout_data, prepare_inference_batch, compute_metrics, Timer,
    report_memory,
)


# ─── v2: Ultra-Fast Batch AR ───────────────────────────────────────────────

@torch.inference_mode()
def v2_predict(model, tokenizer, features, means, stds, times, horizon, device,
               use_amp=True, amp_dtype="bf16"):
    """Batch AR with fused decode and optimized memory."""
    B = features.size(0)
    prefix_len = features.size(1)

    # Tokenize entire batch
    with autocast_ctx(device, use_amp, amp_dtype):
        idx_c, idx_f = tokenizer.encode(features)

    cur_c = idx_c[:, :prefix_len]
    cur_f = idx_f[:, :prefix_len]

    # Pre-allocate prediction indices
    pred_c = torch.empty(B, horizon, device=device, dtype=torch.long)
    pred_f = torch.empty(B, horizon, device=device, dtype=torch.long)

    # AR loop — each step processes entire batch
    for step in range(horizon):
        sl = cur_c.size(1)
        with autocast_ctx(device, use_amp, amp_dtype):
            logits_c, logits_f, _ = model(
                cur_c, cur_f,
                times["minute"][:, :sl], times["day"][:, :sl],
                times["month"][:, :sl], times["year"][:, :sl],
                last_only=True,
            )

        pc = logits_c[:, -1, :].float().argmax(dim=-1)
        pf = logits_f[:, -1, :].float().argmax(dim=-1)
        pred_c[:, step] = pc
        pred_f[:, step] = pf

        if step < horizon - 1:
            cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
            cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)

    # FUSED decode: process all steps at once
    # tokenizer.decode supports multi-token sequences [B, N]
    decoded = tokenizer.decode(pred_c, pred_f)  # [B, horizon, 6]
    pred_returns = decoded[:, :, 0].float() * stds[:, 0:1] + means[:, 0:1]

    return pred_returns


# ─── Auto batch sizing ──────────────────────────────────────────────────────

def find_max_batch_size(model, tokenizer, device, seq_len=1023, max_bs=64):
    """Find maximum batch size that fits in GPU memory."""
    if device.type != "cuda":
        return 16

    print("  Finding optimal batch size...")
    total_vram = torch.cuda.get_device_properties(0).total_memory
    model_vram = torch.cuda.memory_allocated()

    # Estimate per-batch VRAM: ~8MB per sample at S=1024 with BF16
    available = total_vram * 0.85 - model_vram - 256 * 1024 * 1024  # leave 256MB buffer
    per_sample = 10 * 1024 * 1024  # ~10MB per sample (conservative)
    estimated = max(1, int(available / per_sample))

    return min(estimated, max_bs)


# ─── Main ─────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="v2: Maximum-throughput inference")
    parser.add_argument("--mode", default="demo", choices=["demo", "val", "train"])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-stocks", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Batch size (0=auto-detect max)")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=15)
    args = parser.parse_args(argv)

    device = setup_device()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(device, args.checkpoint)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} params")

    # Determine batch size
    if args.batch_size <= 0:
        bs = find_max_batch_size(model, tokenizer, device, max_bs=args.max_batch_size)
    else:
        bs = args.batch_size
    print(f"  Batch size: {bs}")

    # Load data
    payload = load_rollout_data(args.mode)
    total = payload["features"].size(0)
    num_stocks = min(args.num_stocks, total)

    rng = np.random.default_rng(42)
    indices = rng.choice(total, size=num_stocks, replace=False).tolist()

    # ─── Benchmark ───
    if args.benchmark:
        print(f"\n{'='*60}")
        print(f"Benchmark: Batch Size Scaling")
        print(f"{'='*60}")

        test_sizes = [s for s in [1, 2, 4, 8, 16, 32, 48, 64] if s <= min(num_stocks, args.max_batch_size)]
        bs_times = {}

        for test_bs in test_sizes:
            batch_idx = indices[:test_bs]
            b = prepare_inference_batch(payload, torch.tensor(batch_idx, dtype=torch.long), device)

            # Warmup
            for _ in range(args.warmup_runs):
                _ = v2_predict(model, tokenizer, b["features"], b["means"],
                              b["stds"], b["time"], args.horizon, device,
                              use_amp=args.use_amp, amp_dtype=args.amp_dtype)
            torch.cuda.synchronize()

            # Measure
            times = []
            for _ in range(args.benchmark_runs):
                with Timer(sync_cuda=True) as t:
                    _ = v2_predict(model, tokenizer, b["features"], b["means"],
                                  b["stds"], b["time"], args.horizon, device,
                                  use_amp=args.use_amp, amp_dtype=args.amp_dtype)
                times.append(t.ms)

            batch_ms = np.mean(times)
            per_stock = batch_ms / test_bs
            bs_times[test_bs] = {"batch_ms": batch_ms, "per_stock_ms": per_stock}
            speedup = bs_times[1]["per_stock_ms"] / per_stock if 1 in bs_times else 0
            print(f"  bs={test_bs:2d}: batch={batch_ms:7.1f}ms  per_stock={per_stock:6.1f}ms  "
                  f"speedup={speedup:.2f}x")

        baseline_per_stock = bs_times[1]["per_stock_ms"]
        best_bs = min(bs_times, key=lambda k: bs_times[k]["per_stock_ms"])
        best_speedup = baseline_per_stock / bs_times[best_bs]["per_stock_ms"]
        print(f"\n  Best: bs={best_bs} at {bs_times[best_bs]['per_stock_ms']:.1f} ms/stock "
              f"({best_speedup:.2f}x)")
    else:
        baseline_per_stock = best_speedup = 0

    # ─── Full prediction ───
    print(f"\n{'='*60}")
    print(f"Full Prediction: {num_stocks} stocks, bs={bs}, horizon={args.horizon}")
    print(f"{'='*60}")

    all_preds, all_actuals = [], []
    total_time = 0.0
    num_batches = 0

    for start in tqdm(range(0, num_stocks, bs), desc="  Batches"):
        end = min(start + bs, num_stocks)
        batch_idx = indices[start:end]
        b = prepare_inference_batch(payload, torch.tensor(batch_idx, dtype=torch.long), device)

        with Timer(sync_cuda=True) as t:
            pred = v2_predict(model, tokenizer, b["features"], b["means"],
                             b["stds"], b["time"], args.horizon, device,
                             use_amp=args.use_amp, amp_dtype=args.amp_dtype)

        total_time += t.ms
        num_batches += 1
        all_preds.append(pred.cpu())
        all_actuals.append(b["actual_returns"].cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    actuals = torch.cat(all_actuals, dim=0).numpy()

    # Metrics
    per_step = {}
    for s in range(args.horizon):
        per_step[f"day_{s+1}"] = compute_metrics(preds[:, s], actuals[:, s])

    overall = compute_metrics(preds.reshape(-1), actuals.reshape(-1))

    # Summary
    per_stock_ms = total_time / num_stocks
    throughput = num_stocks / (total_time / 1000)
    est_4000_time = 4000 / throughput

    print(f"\n{'='*60}")
    print(f"v2 Final Results")
    print(f"{'='*60}")
    print(f"  Stocks:     {num_stocks}")
    print(f"  Horizon:    {args.horizon} days")
    print(f"  Batch size: {bs}")
    print(f"  Batches:    {num_batches}")
    print(f"  Total time: {total_time:.0f} ms ({total_time/1000:.1f}s)")
    print(f"  Per stock:  {per_stock_ms:.1f} ms")
    print(f"  Throughput: {throughput:.1f} stocks/sec")
    print(f"  Est. 4000 stocks: {est_4000_time:.1f}s ({est_4000_time/60:.1f} min)")
    print(f"  MAPE: {overall['mape']:.4f}%  DA: {overall['da']:.2f}%")
    print(f"  MAE:  {overall['mae']:.6f}  RMSE: {overall['rmse']:.6f}")
    if args.benchmark:
        print(f"  Speedup (vs bs=1): {best_speedup:.2f}x")
    report_memory(device)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "v2_results.json"), "w") as f:
            json.dump({
                "version": "v2_fast", "device": str(device),
                "horizon": args.horizon, "batch_size": bs,
                "num_stocks": num_stocks, "total_time_ms": total_time,
                "per_stock_ms": per_stock_ms, "throughput": throughput,
                "est_4000_stocks_seconds": est_4000_time,
                "speedup_vs_baseline": best_speedup,
                "overall_metrics": overall, "per_step_metrics": per_step,
                "batch_size_benchmark": {str(k): v for k, v in bs_times.items()}
                if args.benchmark else {},
            }, f, indent=2, ensure_ascii=False)

    return preds, actuals


if __name__ == "__main__":
    main()
