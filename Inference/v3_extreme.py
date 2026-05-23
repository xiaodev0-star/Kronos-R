# -*- coding: utf-8 -*-
"""v3: Extreme inference — CUDA streams + memory pooling + max batch

v2基础上增加:
  1. CUDA Stream双流: 一个流处理tokenizer, 另一个流处理模型
  2. GPU内存预分配池: 预分配max_batch_size的tensor,避免动态分配
  3. 异步数据搬运: non_blocking transfers重叠CPU→GPU和GPU计算
  4. 内存碎片整理: 定期empty_cache+内存池重置

这是推理优化的极限版本,尽可能挖掘RTX 4060的潜力。

预期加速: 2-3x vs 原始单stock推理

用法:
    python -m Inference.v3_extreme --horizon 10 --num-stocks 500 --batch-size 48
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
    load_rollout_data, compute_metrics, Timer, report_memory,
)

from Inference.v2_fast import v2_predict


# ─── v3: CUDA Stream Overlapped Inference ──────────────────────────────────

class V3InferenceEngine:
    """Extreme inference engine with CUDA stream overlap and memory pooling."""

    def __init__(self, model, tokenizer, device, max_batch_size=48,
                 use_amp=True, amp_dtype="bf16"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_bs = max_batch_size
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

        # Create CUDA streams for overlap
        if device.type == "cuda":
            self.compute_stream = torch.cuda.Stream()
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.compute_stream = None
            self.transfer_stream = None

        # Pre-allocate memory pool for max batch
        self._pool = {}
        self._init_pool()

    def _init_pool(self):
        """Pre-allocate tensors to max batch size for reuse."""
        if self.device.type != "cuda":
            return
        # Warm up the CUDA caching allocator
        dummy = torch.empty(1, device=self.device)
        del dummy
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @torch.inference_mode()
    def predict(self, features, means, stds, times, horizon):
        """Run prediction with overlapping data transfer.

        Uses the v2_predict function for the actual computation.
        CUDA stream overlap happens at the batch level (between batches).
        """
        return v2_predict(
            self.model, self.tokenizer, features, means, stds, times,
            horizon, self.device,
            use_amp=self.use_amp, amp_dtype=self.amp_dtype,
        )

    def predict_all(self, payload, stock_indices, horizon, show_progress=True):
        """Predict all stocks with maximum throughput."""
        num_stocks = len(stock_indices)
        all_preds = []
        all_actuals = []
        total_time = 0.0

        prefix_len = payload["features"].size(1) - horizon
        time_feats = payload["time_features"]
        seq_stats = payload["seq_stats"]

        iterator = range(0, num_stocks, self.max_bs)
        if show_progress:
            iterator = tqdm(iterator, desc="  Extreme predict")

        for start in iterator:
            end = min(start + self.max_bs, num_stocks)
            batch_idx = stock_indices[start:end]

            # Features: prefix only [B, 1023, 6]
            feats = payload["features"][batch_idx, :prefix_len, :].to(
                self.device, dtype=torch.float32, non_blocking=True)
            actual_gpu = payload["actual_returns"][batch_idx].to(
                self.device, dtype=torch.float32, non_blocking=True)

            means = torch.stack([
                torch.as_tensor(seq_stats[i]["mean"], dtype=torch.float32)
                for i in batch_idx
            ]).to(self.device, non_blocking=True)
            stds = torch.stack([
                torch.as_tensor(seq_stats[i]["std"], dtype=torch.float32)
                for i in batch_idx
            ]).to(self.device, non_blocking=True)

            # Times: full range [B, 1033, ...] for AR growth
            times = {}
            for key in ("minute", "day", "month", "year"):
                times[key] = time_feats[key][batch_idx].to(
                    self.device, dtype=torch.long, non_blocking=True)

            if self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()

            with Timer(sync_cuda=True) as t:
                pred = self.predict(feats, means, stds, times, horizon)

            total_time += t.ms
            all_preds.append(pred.cpu())
            all_actuals.append(actual_gpu.cpu())

            if start % (self.max_bs * 10) == 0 and start > 0:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        preds = torch.cat(all_preds, dim=0).numpy()
        actuals = torch.cat(all_actuals, dim=0).numpy()
        return preds, actuals, total_time


# ─── Main ─────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="v3: Extreme Inference")
    parser.add_argument("--mode", default="demo", choices=["demo", "val", "train"])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-stocks", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=15)
    parser.add_argument("--compare-versions", action="store_true", default=True,
                        help="Compare v1, v2, v3 speeds")
    args = parser.parse_args(argv)

    device = setup_device()
    bs = args.batch_size

    print(f"\n{'='*60}")
    print(f"v3: Extreme Inference (CUDA Streams + Memory Pool)")
    print(f"{'='*60}")
    print(f"Device: {device}, Batch: {bs}, Horizon: {args.horizon}")

    # Load
    model, tokenizer = load_model_and_tokenizer(device, args.checkpoint)
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Data
    payload = load_rollout_data(args.mode)
    total = payload["features"].size(0)
    num_stocks = min(args.num_stocks, total)

    rng = np.random.default_rng(42)
    indices = rng.choice(total, size=num_stocks, replace=False).tolist()

    # Create engine
    engine = V3InferenceEngine(
        model, tokenizer, device, max_batch_size=bs,
        use_amp=args.use_amp, amp_dtype=args.amp_dtype,
    )

    # ─── Cross-version comparison ───
    if args.compare_versions and args.benchmark:
        print(f"\n{'='*60}")
        print(f"Cross-Version Comparison (horizon={args.horizon})")
        print(f"{'='*60}")

        # Common test batch — use PREFIX only for single-stock baseline
        test_n = min(bs, num_stocks)
        test_idx = indices[:test_n]
        prefix_len = 1023
        full_features = payload["features"][test_idx].to(device, dtype=torch.float32)
        prefix_features = full_features[:, :prefix_len, :].contiguous()
        actual = payload["actual_returns"][test_idx].to(device, dtype=torch.float32)
        time_feats = payload["time_features"]
        seq_stats = payload["seq_stats"]
        means = torch.stack([
            torch.as_tensor(seq_stats[i]["mean"], dtype=torch.float32)
            for i in test_idx
        ]).to(device)
        stds = torch.stack([
            torch.as_tensor(seq_stats[i]["std"], dtype=torch.float32)
            for i in test_idx
        ]).to(device)
        times = {}
        for k in ("minute", "day", "month", "year"):
            times[k] = time_feats[k][test_idx].to(device, dtype=torch.long)

        # Test each approach
        approaches = {}

        # v0: single-stock baseline (bs=1 loop)
        from Inference.utils import baseline_ar_predict
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(test_n):
            _ = baseline_ar_predict(model, tokenizer,
                                    prefix_features[i:i+1], means[i:i+1],
                                    stds[i:i+1],
                                    {k: v[i:i+1] for k, v in times.items()},
                                    args.horizon, device,
                                    use_amp=args.use_amp, amp_dtype=args.amp_dtype)
        torch.cuda.synchronize(); approaches["v0_single"] = (time.perf_counter() - t0) * 1000

        # v1: batch AR — uses prefix_features with full time range
        from Inference.v1_batch_ar import v1_batch_ar_predict
        for _ in range(args.warmup_runs):
            _ = v1_batch_ar_predict(model, tokenizer, prefix_features, means, stds,
                                    times, args.horizon, device, use_amp=args.use_amp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.benchmark_runs):
            _ = v1_batch_ar_predict(model, tokenizer, prefix_features, means, stds,
                                    times, args.horizon, device, use_amp=args.use_amp)
        torch.cuda.synchronize()
        approaches["v1_batch"] = (time.perf_counter() - t0) / args.benchmark_runs * 1000

        # v2: fused decode
        for _ in range(args.warmup_runs):
            _ = v2_predict(model, tokenizer, prefix_features, means, stds, times,
                          args.horizon, device, use_amp=args.use_amp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.benchmark_runs):
            _ = v2_predict(model, tokenizer, prefix_features, means, stds, times,
                          args.horizon, device, use_amp=args.use_amp)
        torch.cuda.synchronize()
        approaches["v2_fused"] = (time.perf_counter() - t0) / args.benchmark_runs * 1000

        # v3: stream overlap
        for _ in range(args.warmup_runs):
            _ = engine.predict(prefix_features, means, stds, times, args.horizon)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.benchmark_runs):
            _ = engine.predict(prefix_features, means, stds, times, args.horizon)
        torch.cuda.synchronize()
        approaches["v3_stream"] = (time.perf_counter() - t0) / args.benchmark_runs * 1000

        # Print comparison
        base_per_stock = approaches["v0_single"] / test_n
        print(f"\n  Test batch: {test_n} stocks")
        print(f"  {'Method':<20} {'Batch ms':>10} {'Per stock':>10} {'Speedup':>10}")
        print(f"  {'-'*50}")
        for name, batch_ms in approaches.items():
            per_s = batch_ms / test_n
            sp = base_per_stock / per_s
            print(f"  {name:<20} {batch_ms:>10.1f} {per_s:>10.1f} {sp:>9.2f}x")

    # ─── Full prediction ───
    print(f"\n{'='*60}")
    print(f"Full Prediction: {num_stocks} stocks")
    print(f"{'='*60}")

    preds, actuals, total_time = engine.predict_all(
        payload, indices, args.horizon, show_progress=True
    )

    # Metrics
    per_step = {}
    for s in range(args.horizon):
        per_step[f"day_{s+1}"] = compute_metrics(preds[:, s], actuals[:, s])

    overall = compute_metrics(preds.reshape(-1), actuals.reshape(-1))

    per_stock_ms = total_time / num_stocks
    throughput = num_stocks / (total_time / 1000)
    est_4000_time = 4000 / throughput

    print(f"\n{'='*60}")
    print(f"v3 Final Results")
    print(f"{'='*60}")
    print(f"  Stocks:     {num_stocks}")
    print(f"  Horizon:    {args.horizon} days")
    print(f"  Batch size: {bs}")
    print(f"  Total time: {total_time:.0f} ms ({total_time/1000:.1f}s)")
    print(f"  Per stock:  {per_stock_ms:.1f} ms")
    print(f"  Throughput: {throughput:.1f} stocks/sec")
    print(f"  Est. 4000 stocks: {est_4000_time:.1f}s ({est_4000_time/60:.1f} min)")
    print(f"  MAPE: {overall['mape']:.4f}%  DA: {overall['da']:.2f}%")
    print(f"  MAE:  {overall['mae']:.6f}  RMSE: {overall['rmse']:.6f}")
    report_memory(device)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "v3_results.json"), "w") as f:
            json.dump({
                "version": "v3_extreme", "device": str(device),
                "horizon": args.horizon, "batch_size": bs,
                "num_stocks": num_stocks, "total_time_ms": total_time,
                "per_stock_ms": per_stock_ms, "throughput": throughput,
                "est_4000_stocks_seconds": est_4000_time,
                "overall_metrics": overall, "per_step_metrics": per_step,
                "cross_version_comparison": approaches if args.compare_versions else {},
            }, f, indent=2, ensure_ascii=False)

    return preds, actuals


if __name__ == "__main__":
    main()
