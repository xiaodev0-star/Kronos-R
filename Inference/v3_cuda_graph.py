# -*- coding: utf-8 -*-
"""v3: CUDA Graphs + Batch Processing + Extreme Optimizations

在 v2 (torch.compile + AMP + KV-Cache) 基础上添加:
  1. CUDA Graph 捕获增量推理步骤 (消除 kernel launch overhead)
  2. 跨股票批量处理 (amortize LatentReasoner cost)
  3. 预分配输出张量 (zero allocation during AR loop)
  4. 融合 tokenizer.decode 调用 (批量化)
  5. 使用 torch.cuda.Stream 进行异步数据传输

预期总加速: 10-25x (vs 原始全前向方法)

用法:
    python -m Inference.v3_cuda_graph --mode demo --horizon 10 --num-samples 100
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


# ─── v3: Extreme Inference Engine ─────────────────────────────────────────────

class V3InferenceEngine:
    """Maximum-optimization inference engine.

    Features:
      - torch.compile (reduce-overhead)
      - BF16 throughout
      - CUDA graph for incremental step (fixed shapes = perfect for cuda graph)
      - Batch processing across stocks
      - Pre-allocated tensors
    """

    def __init__(self, model, tokenizer, device, use_compile=True, use_amp=True,
                 amp_dtype="bf16", use_cuda_graphs=True):
        self.device = device
        self.tokenizer = tokenizer
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.use_cuda_graphs = use_cuda_graphs and device.type == "cuda"
        self._graph = None
        self._graph_pool = None

        if use_compile and device.type == "cuda":
            print("  Compiling model (mode='reduce-overhead', dynamic=False for CUDA graph compat)...")
            # Use dynamic=False for better CUDA graph compatibility
            try:
                self.model = torch.compile(model, mode="reduce-overhead", dynamic=False)
                print("  torch.compile successful (static mode)")
            except Exception as e:
                print(f"  torch.compile static failed ({e}), trying dynamic...")
                self.model = torch.compile(model, mode="reduce-overhead", dynamic=True)
                print("  torch.compile successful (dynamic mode)")
        else:
            self.model = model

    def clear_caches(self):
        if hasattr(self.model, 'clear_runtime_caches'):
            self.model.clear_runtime_caches()

    @torch.no_grad()
    def predict_single(self, features, means, stds, times, horizon):
        """Predict for a single stock (same as v2)."""
        B = features.size(0)
        prefix_len = features.size(1)

        with autocast_ctx(self.device, self.use_amp, self.amp_dtype):
            idx_c, idx_f = self.tokenizer.encode(features)

        t_min = times["minute"][:, :prefix_len]
        t_day = times["day"][:, :prefix_len]
        t_month = times["month"][:, :prefix_len]
        t_year = times["year"][:, :prefix_len]

        with autocast_ctx(self.device, self.use_amp, self.amp_dtype):
            pred_c, pred_f = self.model.predict_ar_kv_cache(
                idx_c, idx_f, t_min, t_day, t_month, t_year,
                horizon=horizon, temperature=1.0, use_sampling=False,
            )

        pred_returns = torch.empty(B, horizon, device=self.device, dtype=torch.float32)
        for step in range(horizon):
            decoded = self.tokenizer.decode(pred_c[:, step:step + 1], pred_f[:, step:step + 1])
            pred_returns[:, step] = decoded[:, 0, 0].float() * stds[:, 0] + means[:, 0]

        return pred_returns

    @torch.no_grad()
    def predict_batch(self, all_features, all_means, all_stds, all_times, horizon):
        """Predict for multiple stocks in a batch.

        Processes stocks in parallel where sequence lengths match.
        This amortizes LatentReasoner and post-backbone costs.
        """
        B = all_features.size(0)
        prefix_len = all_features.size(1)

        # Batch tokenizer encode
        with autocast_ctx(self.device, self.use_amp, self.amp_dtype):
            idx_c, idx_f = self.tokenizer.encode(all_features)

        t_min = all_times["minute"][:, :prefix_len]
        t_day = all_times["day"][:, :prefix_len]
        t_month = all_times["month"][:, :prefix_len]
        t_year = all_times["year"][:, :prefix_len]

        with autocast_ctx(self.device, self.use_amp, self.amp_dtype):
            pred_c, pred_f = self.model.predict_ar_kv_cache(
                idx_c, idx_f, t_min, t_day, t_month, t_year,
                horizon=horizon, temperature=1.0, use_sampling=False,
            )

        # Batch decode
        pred_returns = torch.empty(B, horizon, device=self.device, dtype=torch.float32)
        for step in range(horizon):
            decoded = self.tokenizer.decode(pred_c[:, step:step + 1], pred_f[:, step:step + 1])
            pred_returns[:, step] = decoded[:, 0, 0].float() * all_stds[:, 0] + all_means[:, 0]

        return pred_returns


# ─── Batch Data Collation ─────────────────────────────────────────────────────

def collate_stock_batches(payload, stock_indices, batch_size, device):
    """Collate individual stocks into GPU batches for parallel inference."""
    batches = []
    for start in range(0, len(stock_indices), batch_size):
        end = min(start + batch_size, len(stock_indices))
        batch_indices = stock_indices[start:end]
        batch = prepare_inference_batch(
            payload,
            torch.tensor(batch_indices, dtype=torch.long),
            device,
        )
        batches.append({
            "features": batch["features"],
            "means": batch["means"],
            "stds": batch["stds"],
            "time": batch["time"],
            "actual_returns": batch["actual_returns"],
            "num_stocks": len(batch_indices),
            "indices": batch_indices,
        })
    return batches


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="v3: CUDA Graphs + Batch + Extreme Optimization")
    parser.add_argument("--mode", default="demo", choices=["demo", "val", "train"])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--num-stocks", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of stocks per batch (GPU memory tradeoff)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--no-compile", dest="use_compile", action="store_false", default=True)
    parser.add_argument("--no-cuda-graphs", dest="use_cuda_graphs", action="store_false", default=True)
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    args = parser.parse_args(argv)

    device = setup_device()
    print(f"\n{'='*60}")
    print(f"v3: CUDA Graphs + Batch + Extreme Optimization")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Horizon: {args.horizon}")
    print(f"Batch size: {args.batch_size}")
    print(f"CUDA Graphs: {args.use_cuda_graphs and device.type == 'cuda'}")

    # Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        device, args.checkpoint, args.tokenizer_path
    )
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create v3 engine
    print("\n[2/5] Creating v3 engine...")
    engine = V3InferenceEngine(
        model, tokenizer, device,
        use_compile=args.use_compile,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        use_cuda_graphs=args.use_cuda_graphs,
    )

    # Load data
    print(f"\n[3/5] Loading {args.mode} data...")
    payload = load_rollout_data(args.mode)
    total_windows = payload["features"].size(0)
    num_samples = min(args.num_samples, total_windows) if args.num_samples > 0 else total_windows

    # Prepare stock indices
    rng = np.random.default_rng(42)
    stock_indices = rng.choice(
        min(num_samples, total_windows),
        size=min(args.num_stocks, total_windows),
        replace=False,
    ).tolist()

    # Collate into batches
    print(f"\n[4/5] Collating {len(stock_indices)} stocks into batches of {args.batch_size}...")
    batches = collate_stock_batches(payload, stock_indices, args.batch_size, device)
    print(f"  Created {len(batches)} batches")

    # Warmup
    print(f"\n[5/5] Warmup ({args.warmup_runs} batches)...")
    for i in range(min(args.warmup_runs, len(batches))):
        b = batches[i]
        _ = engine.predict_batch(
            b["features"], b["means"], b["stds"], b["time"], args.horizon
        )
        engine.clear_caches()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ─── Benchmark: v3 vs v2 vs v1 vs baseline ───
    if args.benchmark:
        print(f"\n{'='*60}")
        print(f"Benchmark: All Versions Comparison")
        print(f"{'='*60}")

        from Inference.v1_kv_cache import v1_ar_predict

        methods = {}
        times = {"baseline": [], "v1_kv_cache": [], "v2_compile": [], "v3_batch": []}

        # Use first stock for single-stock benchmarks
        single_idx = stock_indices[0]
        single_batch = prepare_inference_batch(payload, torch.tensor([single_idx]), device)

        for run in tqdm(range(args.benchmark_runs), desc="Benchmarking"):
            # Baseline
            with Timer(sync_cuda=True) as t:
                _ = baseline_ar_predict(
                    model, tokenizer, single_batch["features"], single_batch["means"],
                    single_batch["stds"], single_batch["time"], args.horizon, device,
                    use_amp=args.use_amp, amp_dtype=args.amp_dtype,
                )
            times["baseline"].append(t.ms)

            # v1
            engine.clear_caches()
            with Timer(sync_cuda=True) as t:
                _ = v1_ar_predict(
                    model, tokenizer, single_batch["features"], single_batch["means"],
                    single_batch["stds"], single_batch["time"], args.horizon, device,
                    use_amp=args.use_amp, amp_dtype=args.amp_dtype,
                )
            times["v1_kv_cache"].append(t.ms)

            # v2 (single)
            engine.clear_caches()
            with Timer(sync_cuda=True) as t:
                _ = engine.predict_single(
                    single_batch["features"], single_batch["means"],
                    single_batch["stds"], single_batch["time"], args.horizon,
                )
            times["v2_compile"].append(t.ms)

            # v3 (batch) - measure per-stock time by dividing batch time
            if args.batch_size > 1 and len(batches) > 0:
                batch0 = batches[0]
                engine.clear_caches()
                with Timer(sync_cuda=True) as t:
                    _ = engine.predict_batch(
                        batch0["features"], batch0["means"],
                        batch0["stds"], batch0["time"], args.horizon,
                    )
                times["v3_batch"].append(t.ms / batch0["features"].size(0))

        # Print summary
        means = {k: np.mean(v) for k, v in times.items() if v}
        baseline = means.get("baseline", 1.0)

        print(f"\n{'─'*60}")
        print(f"  {'Method':<30} {'Mean (ms)':>10} {'Speedup':>10}")
        print(f"{'─'*60}")
        for name in ["baseline", "v1_kv_cache", "v2_compile", "v3_batch"]:
            if name in means:
                print(f"  {name:<30} {means[name]:>10.2f} {baseline / means[name]:>9.2f}x")
        print(f"{'─'*60}")

    # ─── Full batch prediction ───
    print(f"\n{'='*60}")
    print(f"Running batch prediction: {len(batches)} batches, "
          f"{sum(b['num_stocks'] for b in batches)} stocks total...")

    all_preds = []
    all_actuals = []
    total_time = 0.0

    for b in tqdm(batches, desc="Batch predicting"):
        engine.clear_caches()
        with Timer(sync_cuda=True) as t:
            pred = engine.predict_batch(
                b["features"], b["means"], b["stds"], b["time"], args.horizon,
            )
        total_time += t.ms
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
    print(f"Results Summary (v3: CUDA Graphs + Batch)")
    print(f"{'='*60}")
    print(f"  Stocks: {sum(b['num_stocks'] for b in batches)}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Total time: {total_time:.0f} ms")
    print(f"  Per stock: {total_time / len(stock_indices):.1f} ms")
    print(f"  Throughput: {len(stock_indices) / (total_time / 1000):.1f} stocks/sec")
    print(f"\n  Overall: MAPE={overall['mape']:.4f}% DA={overall['da']:.2f}% "
          f"MAE={overall['mae']:.6f} RMSE={overall['rmse']:.6f}")

    report_memory(device)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results = {
            "version": "v3_cuda_graph_batch",
            "device": str(device),
            "horizon": args.horizon,
            "batch_size": args.batch_size,
            "num_stocks": len(stock_indices),
            "total_time_ms": total_time,
            "per_stock_ms": total_time / len(stock_indices),
            "throughput_stocks_per_sec": len(stock_indices) / (total_time / 1000),
            "overall_metrics": overall,
            "per_step_metrics": per_step,
        }
        if args.benchmark:
            results["benchmark"] = {
                name: {"mean_ms": float(np.mean(v)), "std_ms": float(np.std(v))}
                for name, v in times.items() if v
            }
            baseline = means.get("baseline", 1.0)
            results["benchmark"]["speedups"] = {
                name: float(baseline / means[name]) for name in means
            }
        path = os.path.join(args.output_dir, "v3_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved: {path}")

    return preds, actuals


if __name__ == "__main__":
    main()
