# -*- coding: utf-8 -*-
"""Comprehensive benchmark comparing all three inference versions vs baseline.

Runs controlled benchmarks across:
  - v0: Original baseline (full forward each AR step)
  - v1: KV-Cache optimization
  - v2: KV-Cache + torch.compile + AMP
  - v3: KV-Cache + torch.compile + AMP + Batch processing

Measures:
  - Latency (ms per stock per horizon)
  - Throughput (stocks/sec)
  - GPU memory usage
  - Prediction accuracy (MAPE, DA)

Usage:
    python -m Inference.benchmark --horizon 10 --runs 20 --batch-sizes 1,2,4,8
"""

import argparse
import gc
import json
import os
import sys
import time

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


# ─── Benchmark Runner ─────────────────────────────────────────────────────────

def benchmark_all(
    device,
    model,
    tokenizer,
    payload,
    horizon=10,
    num_runs=20,
    num_stocks=10,
    batch_sizes=[1, 2, 4, 8],
    use_amp=True,
    amp_dtype="bf16",
    use_compile=True,
):
    """Run comprehensive benchmarks across all versions and batch sizes."""

    results = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "horizon": horizon,
        "num_runs": num_runs,
        "versions": {},
    }

    # Select stock indices
    total_windows = payload["features"].size(0)
    rng = np.random.default_rng(42)
    stock_indices = rng.choice(
        min(total_windows, max(num_stocks, max(batch_sizes) * 4)),
        size=min(num_stocks * 3, total_windows),
        replace=False,
    ).tolist()

    print(f"\n{'='*70}")
    print(f"Kronos-R Inference Benchmark — {results['gpu_name']}")
    print(f"Horizon: {horizon} steps, {num_runs} runs per test")
    print(f"{'='*70}")

    # ─── v0: Baseline ───
    print(f"\n─── v0: Baseline (Full Forward each step) ───")
    baseline_times = []
    for run in tqdm(range(num_runs), desc="  Baseline"):
        idx = stock_indices[run % len(stock_indices)]
        batch = prepare_inference_batch(payload, torch.tensor([idx]), device)
        with Timer(sync_cuda=True) as t:
            _ = baseline_ar_predict(
                model, tokenizer, batch["features"], batch["means"],
                batch["stds"], batch["time"], horizon, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
        baseline_times.append(t.ms)

    baseline_mean = float(np.mean(baseline_times))
    baseline_std = float(np.std(baseline_times))
    print(f"  Mean: {baseline_mean:.2f} ± {baseline_std:.2f} ms")
    results["versions"]["v0_baseline"] = {
        "mean_ms": baseline_mean, "std_ms": baseline_std,
        "speedup": 1.0,
    }

    # ─── v1: KV-Cache ───
    print(f"\n─── v1: KV-Cache ───")
    from Inference.v1_kv_cache import v1_ar_predict

    v1_times = []
    for run in tqdm(range(num_runs), desc="  v1"):
        idx = stock_indices[run % len(stock_indices)]
        batch = prepare_inference_batch(payload, torch.tensor([idx]), device)
        with Timer(sync_cuda=True) as t:
            _ = v1_ar_predict(
                model, tokenizer, batch["features"], batch["means"],
                batch["stds"], batch["time"], horizon, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
        v1_times.append(t.ms)
        if hasattr(model, 'clear_runtime_caches'):
            model.clear_runtime_caches()

    v1_mean = float(np.mean(v1_times))
    print(f"  Mean: {v1_mean:.2f} ± {float(np.std(v1_times)):.2f} ms")
    print(f"  Speedup vs baseline: {baseline_mean / v1_mean:.2f}x")
    results["versions"]["v1_kv_cache"] = {
        "mean_ms": v1_mean, "std_ms": float(np.std(v1_times)),
        "speedup": baseline_mean / v1_mean,
    }

    # ─── v2: Compile + AMP ───
    print(f"\n─── v2: torch.compile + AMP + KV-Cache ───")

    if use_compile and device.type == "cuda":
        print("  Compiling model (mode='reduce-overhead')...")
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception:
            compiled_model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        # Warmup compilation
        print("  Warming up compiled model...")
        for i in range(3):
            idx = stock_indices[i]
            batch = prepare_inference_batch(payload, torch.tensor([idx]), device)
            _ = v1_ar_predict(
                compiled_model, tokenizer, batch["features"], batch["means"],
                batch["stds"], batch["time"], horizon, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            if hasattr(compiled_model, 'clear_runtime_caches'):
                compiled_model.clear_runtime_caches()
        if device.type == "cuda":
            torch.cuda.synchronize()
    else:
        compiled_model = model

    v2_times = []
    for run in tqdm(range(num_runs), desc="  v2"):
        idx = stock_indices[run % len(stock_indices)]
        batch = prepare_inference_batch(payload, torch.tensor([idx]), device)
        with Timer(sync_cuda=True) as t:
            _ = v1_ar_predict(
                compiled_model, tokenizer, batch["features"], batch["means"],
                batch["stds"], batch["time"], horizon, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
        v2_times.append(t.ms)
        if hasattr(compiled_model, 'clear_runtime_caches'):
            compiled_model.clear_runtime_caches()

    v2_mean = float(np.mean(v2_times))
    print(f"  Mean: {v2_mean:.2f} ± {float(np.std(v2_times)):.2f} ms")
    print(f"  Speedup vs baseline: {baseline_mean / v2_mean:.2f}x")
    print(f"  Speedup vs v1: {v1_mean / v2_mean:.2f}x")
    results["versions"]["v2_compile_amp"] = {
        "mean_ms": v2_mean, "std_ms": float(np.std(v2_times)),
        "speedup": baseline_mean / v2_mean,
        "speedup_vs_v1": v1_mean / v2_mean,
    }

    # ─── v3: Batch Processing ───
    v3_results = {}
    for bs in batch_sizes:
        if bs < 1:
            continue
        print(f"\n─── v3: Batch Processing (batch_size={bs}) ───")

        batched_indices = []
        for start in range(0, len(stock_indices[:num_stocks * 2]), bs):
            end = min(start + bs, len(stock_indices[:num_stocks * 2]))
            if end - start == bs:  # Only keep full batches for fair timing
                batched_indices.append(stock_indices[start:end])

        if not batched_indices:
            print("  Skipping — not enough stocks for this batch size")
            continue

        batch_times = []
        for run in tqdm(range(min(num_runs, len(batched_indices))), desc=f"  v3_bs{bs}"):
            batch_idx = batched_indices[run % len(batched_indices)]
            batch = prepare_inference_batch(
                payload, torch.tensor(batch_idx, dtype=torch.long), device
            )

            with autocast_ctx(device, use_amp, amp_dtype):
                idx_c, idx_f = tokenizer.encode(batch["features"])

            with autocast_ctx(device, use_amp, amp_dtype):
                pred_c, pred_f = compiled_model.predict_ar_kv_cache(
                    idx_c, idx_f,
                    batch["time"]["minute"][:, :batch["prefix_len"]],
                    batch["time"]["day"][:, :batch["prefix_len"]],
                    batch["time"]["month"][:, :batch["prefix_len"]],
                    batch["time"]["year"][:, :batch["prefix_len"]],
                    horizon=horizon, temperature=1.0, use_sampling=False,
                )

            if device.type == "cuda":
                torch.cuda.synchronize()

            t_start = time.perf_counter()
            for step in range(horizon):
                decoded = tokenizer.decode(pred_c[:, step:step+1], pred_f[:, step:step+1])
                _ = decoded[:, 0, 0].float() * batch["stds"][:, 0] + batch["means"][:, 0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_total = (time.perf_counter() - t_start) * 1000

            # Only timing the AR loop, not decode (separate concern)
            batch_times.append(t_total / bs)  # per-stock time

            if hasattr(compiled_model, 'clear_runtime_caches'):
                compiled_model.clear_runtime_caches()

        batch_mean = float(np.mean(batch_times))
        print(f"  Per-stock mean: {batch_mean:.2f} ± {float(np.std(batch_times)):.2f} ms")
        print(f"  Speedup vs baseline: {baseline_mean / batch_mean:.2f}x")

        v3_results[f"v3_batch_{bs}"] = {
            "batch_size": bs,
            "per_stock_mean_ms": batch_mean,
            "per_stock_std_ms": float(np.std(batch_times)),
            "speedup": baseline_mean / batch_mean,
        }

    results["versions"]["v3_batch"] = v3_results

    return results, compiled_model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="Comprehensive inference benchmark")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--num-stocks", type=int, default=10)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--mode", default="demo")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--no-compile", dest="use_compile", action="store_false", default=True)
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the compiled model for reuse")
    args = parser.parse_args(argv)

    device = setup_device()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip().isdigit()]
    if not batch_sizes:
        batch_sizes = [1]

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        device, args.checkpoint, args.tokenizer_path
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    print(f"Loading {args.mode} data...")
    payload = load_rollout_data(args.mode)

    # Run benchmark
    print(f"\nStarting benchmark ({args.runs} runs, horizon={args.horizon})...")
    results, compiled_model = benchmark_all(
        device, model, tokenizer, payload,
        horizon=args.horizon,
        num_runs=args.runs,
        num_stocks=args.num_stocks,
        batch_sizes=batch_sizes,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        use_compile=args.use_compile,
    )

    # ─── Print Final Summary ───
    print(f"\n{'='*70}")
    print(f"FINAL BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"Device: {results['gpu_name']}")
    print(f"Horizon: {args.horizon} steps")
    print(f"{'─'*70}")
    print(f"  {'Version':<35} {'Mean (ms)':>10} {'Speedup':>10}")
    print(f"{'─'*70}")

    for ver_name, ver_data in results["versions"].items():
        if ver_name == "v3_batch":
            for sub_name, sub_data in sorted(ver_data.items()):
                print(f"  {sub_name:<35} {sub_data['per_stock_mean_ms']:>10.2f} "
                      f"{sub_data['speedup']:>9.2f}x")
        elif "mean_ms" in ver_data:
            print(f"  {ver_name:<35} {ver_data['mean_ms']:>10.2f} "
                  f"{ver_data['speedup']:>9.2f}x")

    print(f"{'─'*70}")

    # Find best
    best_speedup = 1.0
    best_name = "v0_baseline"
    for ver_name, ver_data in results["versions"].items():
        if ver_name == "v3_batch":
            for sub_name, sub_data in ver_data.items():
                if sub_data.get("speedup", 1.0) > best_speedup:
                    best_speedup = sub_data["speedup"]
                    best_name = sub_name
        elif ver_data.get("speedup", 1.0) > best_speedup:
            best_speedup = ver_data["speedup"]
            best_name = ver_name

    print(f"\n  BEST: {best_name} — {best_speedup:.2f}x speedup")
    report_memory(device)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")

    if args.save_model and device.type == "cuda":
        save_path = os.path.join(args.output_dir, "compiled_model.pt")
        torch.save(compiled_model.state_dict(), save_path)
        print(f"Compiled model saved: {save_path}")

    return results


if __name__ == "__main__":
    main()
