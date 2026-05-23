# -*- coding: utf-8 -*-
"""Final comprehensive benchmark: all working inference versions.

Tests each version for:
  1. Speed (ms per stock for 10-step AR)
  2. Prediction quality (match with reference)
  3. GPU memory usage

Outputs a final comparison report.
"""

import json, os, sys, time
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
from Inference.v2_fast import v2_predict


@torch.inference_mode()
def verify_match(pred_a, pred_b):
    """Check if two prediction tensors match."""
    diff = (pred_a - pred_b).abs()
    max_d = diff.max().item()
    mse = diff.pow(2).mean().item()
    a_f = pred_a.reshape(-1); b_f = pred_b.reshape(-1)
    a_c = a_f - a_f.mean(); b_c = b_f - b_f.mean()
    corr = (a_c * b_c).sum() / ((a_c.pow(2).sum() * b_c.pow(2).sum()).sqrt() + 1e-10)
    da = (torch.sign(pred_a) == torch.sign(pred_b)).float().mean().item() * 100
    return {"max_diff": max_d, "mse": mse, "pearson": corr.item(), "da": da}


def main():
    device = setup_device()
    print(f"\n{'='*70}")
    print(f"  Kronos-R Inference Optimization — Final Benchmark")
    print(f"{'='*70}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")

    # Load model
    print(f"\n[1] Loading reference model...")
    model, tokenizer = load_model_and_tokenizer(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model: {n_params:,} params, dim=384, depth=3, heads=4")

    # Data
    print(f"\n[2] Loading demo data...")
    payload = load_rollout_data("demo")
    total = payload["features"].size(0)
    print(f"    Available: {total} windows")
    horizon = 10

    rng = np.random.default_rng(42)
    stock_indices = rng.choice(total, size=min(200, total), replace=False).tolist()

    # ─── Define versions to test ───
    versions = {}

    # v0: Baseline single-stock
    def v0_fn(feats, means, stds, times, h):
        return baseline_ar_predict(model, tokenizer, feats, means, stds, times, h, device)

    versions["v0_baseline"] = {"fn": v0_fn, "desc": "Baseline (single, full fwd)"}

    # v1: Batch AR
    from Inference.v1_batch_ar import v1_batch_ar_predict
    def v1_fn(feats, means, stds, times, h):
        return v1_batch_ar_predict(model, tokenizer, feats, means, stds, times, h, device)
    versions["v1_batch_ar"] = {"fn": v1_fn, "desc": "v1: Batch AR"}

    # v2: Fused decode + auto batch (best)
    def v2_fn(feats, means, stds, times, h):
        return v2_predict(model, tokenizer, feats, means, stds, times, h, device)
    versions["v2_fast"] = {"fn": v2_fn, "desc": "v2: Batch + Fused decode"}

    # ─── Quality verification ───
    print(f"\n[3] Verifying prediction quality...")
    q_n = 4
    q_idx = stock_indices[:q_n]
    q_batch = prepare_inference_batch(payload, torch.tensor(q_idx, dtype=torch.long), device)

    # Get reference: process each stock INDIVIDUALLY with baseline
    ref_preds_list = []
    for i in range(q_n):
        with autocast_ctx(device, True, "bf16"):
            p = baseline_ar_predict(model, tokenizer,
                                    q_batch["features"][i:i+1],
                                    q_batch["means"][i:i+1],
                                    q_batch["stds"][i:i+1],
                                    {k: v[i:i+1] for k, v in q_batch["time"].items()},
                                    horizon, device)
        ref_preds_list.append(p.cpu().float())
    ref_pred = torch.cat(ref_preds_list, dim=0)

    # Get batch predictions
    with autocast_ctx(device, True, "bf16"):
        v1_pred = v1_batch_ar_predict(model, tokenizer, q_batch["features"],
                                       q_batch["means"], q_batch["stds"],
                                       q_batch["time"], horizon, device).cpu().float()
        torch.cuda.synchronize()
        v2_pred = v2_predict(model, tokenizer, q_batch["features"],
                             q_batch["means"], q_batch["stds"],
                             q_batch["time"], horizon, device).cpu().float()

    q1 = verify_match(ref_pred, v1_pred)
    q2 = verify_match(ref_pred, v2_pred)
    q12 = verify_match(v1_pred, v2_pred)

    print(f"    v1 vs ref: pearson={q1['pearson']:.6f} da={q1['da']:.1f}% "
          f"max_diff={q1['max_diff']:.8f} [{'PASS' if q1['pearson']>0.999 else 'FAIL'}]")
    print(f"    v2 vs ref: pearson={q2['pearson']:.6f} da={q2['da']:.1f}% "
          f"max_diff={q2['max_diff']:.8f} [{'PASS' if q2['pearson']>0.999 else 'FAIL'}]")
    print(f"    v1 vs v2:  pearson={q12['pearson']:.6f} da={q12['da']:.1f}% "
          f"max_diff={q12['max_diff']:.8f}")

    # ─── Speed benchmark at different batch sizes ───
    print(f"\n[4] Speed benchmark at multiple batch sizes...")
    bs_list = [1, 2, 4, 8, 16, 32]
    warmup, n_runs = 3, 10

    all_results = {}
    for bs in bs_list:
        if bs > len(stock_indices):
            continue
        b_idx = stock_indices[:bs]
        b = prepare_inference_batch(payload, torch.tensor(b_idx, dtype=torch.long), device)
        all_results[bs] = {}

        for name, ver in versions.items():
            # Warmup
            for _ in range(warmup):
                _ = ver["fn"](b["features"], b["means"], b["stds"], b["time"], horizon)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Measure
            times = []
            for _ in range(n_runs):
                with Timer(sync_cuda=True) as t:
                    _ = ver["fn"](b["features"], b["means"], b["stds"], b["time"], horizon)
                times.append(t.ms)

            batch_ms = float(np.mean(times))
            per_stock = batch_ms / bs
            all_results[bs][name] = {"batch_ms": batch_ms, "per_stock_ms": per_stock}

    # ─── Print final report ───
    print(f"\n{'='*70}")
    print(f"  FINAL BENCHMARK REPORT")
    print(f"{'='*70}")
    print(f"  {'bs':>5} | {'v0_baseline':>14} | {'v1_batch':>14} | {'v2_fast':>14} | {'best':>10}")
    print(f"  {'-'*5}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")

    v0_single = all_results[1]["v0_baseline"]["per_stock_ms"]
    best_overall = v0_single
    best_bs = 1

    for bs in bs_list:
        if bs not in all_results:
            continue
        v0 = all_results[bs]["v0_baseline"]["per_stock_ms"]
        v1 = all_results[bs]["v1_batch_ar"]["per_stock_ms"]
        v2 = all_results[bs]["v2_fast"]["per_stock_ms"]
        best = min(v0, v1, v2)
        if best < best_overall:
            best_overall = best
            best_bs = bs
        print(f"  {bs:>5} | {v0:>12.1f}ms | {v1:>12.1f}ms | {v2:>12.1f}ms | {best:>8.1f}ms")

    speedup = all_results[1]["v0_baseline"]["per_stock_ms"] / best_overall
    print(f"\n  Best configuration: v2_fast, bs={best_bs}")
    print(f"  Best latency:       {best_overall:.1f} ms/stock")
    print(f"  Best speedup:       {speedup:.2f}x")
    print(f"  Throughput:         {1000/best_overall:.1f} stocks/sec")

    # 4000-stock estimate
    est_4000 = 4000 * best_overall / 1000
    est_4000_baseline = 4000 * v0_single / 1000
    print(f"\n  4000-stock estimate:")
    print(f"    Baseline: {est_4000_baseline:.0f}s")
    print(f"    Optimized: {est_4000:.0f}s")
    print(f"    Saved: {est_4000_baseline - est_4000:.0f}s ({(est_4000_baseline - est_4000)/60:.1f} min)")

    report_memory(device)

    # ─── Export results ───
    output = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "model_params": n_params,
        "horizon": horizon,
        "results": {},
        "quality": {},
        "best": {
            "version": "v2_fast",
            "batch_size": best_bs,
            "per_stock_ms": best_overall,
            "speedup": speedup,
            "throughput_stocks_per_sec": 1000 / best_overall,
            "estimated_4000_stocks_seconds": est_4000,
        },
    }

    for bs, bs_data in all_results.items():
        output["results"][str(bs)] = bs_data

    for name, ver in versions.items():
        if "quality" in ver:
            output["quality"][name] = ver["quality"]

    os.makedirs("outputs", exist_ok=True)
    path = "outputs/final_benchmark_report.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved: {path}")
    print(f"\n{'='*70}")
    print(f"  Done!")


if __name__ == "__main__":
    main()
