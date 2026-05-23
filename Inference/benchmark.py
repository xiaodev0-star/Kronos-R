# -*- coding: utf-8 -*-
"""Kronos-R 推理基准测试 + 质量验证

测试内容:
  1. 速度基准: 不同batch_size下的吞吐量
  2. 质量验证: 优化版与原版推理结果对比
  3. 全量估算: 4000 stocks预测时间预估

用法:
    python -m Inference.benchmark --horizon 10 --num-stocks 200
"""

import argparse, json, os, sys, time
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
def verify_quality(model, tokenizer, payload, num_stocks=100, bs=32):
    """验证优化版(v2)与原版推理结果的一致性。"""
    rng = np.random.default_rng(42)
    total = payload["features"].size(0)
    indices = rng.choice(total, size=min(num_stocks, total), replace=False).tolist()
    horizon = 10

    # 参考: 逐stock bs=1
    ref_preds = []
    for i in tqdm(indices, desc="  参考(bs=1)"):
        b = prepare_inference_batch(payload, torch.tensor([i]), device)
        with autocast_ctx(device, True, "bf16"):
            p = v2_predict(model, tokenizer, b["features"], b["means"],
                          b["stds"], b["time"], horizon, device)
        ref_preds.append(p.cpu())
    ref = torch.cat(ref_preds, dim=0).numpy()

    # 优化版: 批次处理
    batch_preds = []
    for start in tqdm(range(0, len(indices), bs), desc=f"  批次(bs={bs})"):
        end = min(start + bs, len(indices))
        b = prepare_inference_batch(payload, torch.tensor(indices[start:end]), device)
        with autocast_ctx(device, True, "bf16"):
            p = v2_predict(model, tokenizer, b["features"], b["means"],
                          b["stds"], b["time"], horizon, device)
        batch_preds.append(p.cpu())
    batch = torch.cat(batch_preds, dim=0).numpy()

    # 逐stock对比
    max_diffs = np.abs(ref - batch).max(axis=1)
    da_per_stock = (np.sign(ref) == np.sign(batch)).mean(axis=1) * 100
    exact = (max_diffs == 0).sum()

    # 聚合指标
    actual = payload["actual_returns"][indices].numpy()
    ref_m = compute_metrics(ref.reshape(-1), actual.reshape(-1))
    batch_m = compute_metrics(batch.reshape(-1), actual.reshape(-1))

    return {
        "num_stocks": len(indices),
        "exact_match": int(exact),
        "exact_pct": float(exact / len(indices) * 100),
        "direction_agree_mean": float(da_per_stock.mean()),
        "max_diff_mean": float(max_diffs.mean()),
        "max_diff_max": float(max_diffs.max()),
        "mape_ref": ref_m["mape"],
        "mape_batch": batch_m["mape"],
        "mape_delta": abs(ref_m["mape"] - batch_m["mape"]),
        "da_ref": ref_m["da"],
        "da_batch": batch_m["da"],
        "da_delta": abs(ref_m["da"] - batch_m["da"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kronos-R 推理基准测试")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-stocks", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv)

    device = setup_device()
    model, tokenizer = load_model_and_tokenizer(device)
    model.eval()
    payload = load_rollout_data("demo")

    print(f"\n{'='*60}")
    print(f"Kronos-R Inference Benchmark")
    print(f"{'='*60}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model:  {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Data:   {payload['features'].size(0)} demo windows")

    # ─── 质量验证 ───
    if args.verify:
        print(f"\n[1] 质量验证 (优化版 vs 原版)...")
        quality = verify_quality(model, tokenizer, payload, num_stocks=100, bs=args.batch_size)

        print(f"  Stock数量:     {quality['num_stocks']}")
        print(f"  完全一致:      {quality['exact_match']}/{quality['num_stocks']} "
              f"({quality['exact_pct']:.1f}%)")
        print(f"  方向一致性:    {quality['direction_agree_mean']:.1f}%")
        print(f"  平均最大差异:  {quality['max_diff_mean']:.8f}")
        print(f"  最大差异:      {quality['max_diff_max']:.8f}")
        print(f"  MAPE: ref={quality['mape_ref']:.4f}% batch={quality['mape_batch']:.4f}% "
              f"(delta={quality['mape_delta']:.4f}%)")
        print(f"  DA:   ref={quality['da_ref']:.2f}% batch={quality['da_batch']:.2f}% "
              f"(delta={quality['da_delta']:.2f}%)")

    # ─── 速度基准 ───
    print(f"\n[2] 速度基准测试...")
    total = payload["features"].size(0)
    rng = np.random.default_rng(42)
    stock_idx = rng.choice(total, size=min(args.num_stocks, total), replace=False).tolist()

    bs_list = [1, 4, 8, 16, 32]
    results = {}
    for bs in bs_list:
        if bs > len(stock_idx):
            continue
        b_idx = stock_idx[:bs]
        b = prepare_inference_batch(payload, torch.tensor(b_idx, dtype=torch.long), device)

        for _ in range(3):
            _ = v2_predict(model, tokenizer, b["features"], b["means"],
                          b["stds"], b["time"], args.horizon, device)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        times = []
        for _ in range(10):
            with Timer(sync_cuda=True) as t:
                _ = v2_predict(model, tokenizer, b["features"], b["means"],
                              b["stds"], b["time"], args.horizon, device)
            times.append(t.ms)

        batch_ms = float(np.mean(times))
        per_stock = batch_ms / bs
        results[bs] = {"batch_ms": batch_ms, "per_stock_ms": per_stock}

        if bs == 1:
            baseline = per_stock

        speedup = baseline / per_stock if baseline else 1.0
        print(f"  bs={bs:2d}: {batch_ms:8.1f}ms total  {per_stock:6.1f}ms/stock  "
              f"{speedup:5.2f}x  {1000/per_stock:.0f} stocks/s")

    best_bs = min(results, key=lambda k: results[k]["per_stock_ms"])
    best_ps = results[best_bs]["per_stock_ms"]
    best_sp = baseline / best_ps
    est_4000 = 4000 * best_ps / 1000
    est_base = 4000 * baseline / 1000

    print(f"\n  最佳: bs={best_bs}, {best_ps:.1f}ms/stock, {best_sp:.2f}x")
    print(f"  4000 stocks: {est_4000:.0f}s (原版 {est_base:.0f}s, 节省 {est_base-est_4000:.0f}s)")
    report_memory(device)

    # ─── 保存报告 ───
    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "model_params": sum(p.numel() for p in model.parameters()),
        "horizon": args.horizon,
        "batch_size_results": {str(k): v for k, v in results.items()},
        "best": {"batch_size": best_bs, "per_stock_ms": best_ps, "speedup": best_sp,
                 "throughput": 1000 / best_ps, "est_4000_seconds": est_4000},
    }
    if args.verify:
        report["quality"] = quality

    path = os.path.join(args.output_dir, "benchmark_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {path}")


if __name__ == "__main__":
    main()
