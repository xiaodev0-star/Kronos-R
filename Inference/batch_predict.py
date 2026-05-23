# -*- coding: utf-8 -*-
"""Batch prediction: predict N days for ALL stocks in the dataset.

Uses the v2_optimized batch AR inference for maximum throughput.
Processes all stocks in the dataset directory in parallel batches.

Usage:
    python -m Inference.batch_predict --horizon 10 --batch-size 32 --max-stocks 100
    python -m Inference.batch_predict --horizon 30 --output-dir outputs/predictions
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PARENT)

from config import DataConfig
from Inference.utils import (
    setup_device, autocast_ctx, load_model_and_tokenizer,
    compute_metrics, Timer, report_memory,
)


# ─── Data loading from raw CSV ────────────────────────────────────────────────

def load_stock_features(symbol, data_dir="dataset", seq_len=1023):
    """Load and preprocess features for a single stock from CSV.

    Returns normalized features, means, stds, time features.
    """
    csv_path = os.path.join(_PARENT, data_dir, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None

    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    if not required.issubset(df.columns):
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df[df["volume"] > 0].reset_index(drop=True)

    if len(df) < seq_len + 1:
        return None

    # Build features
    prev_close = df["close"].shift(1)
    features_df = pd.DataFrame({
        "log_ret": np.log(df["close"] / prev_close),
        "log_high": np.log(df["high"] / prev_close),
        "log_low": np.log(df["low"] / prev_close),
        "log_open": np.log(df["open"] / prev_close),
        "log_vol": np.log1p(df["volume"]),
        "log_amt": np.log1p(df["amount"]),
    })
    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(features_df) < seq_len:
        return None

    # Use last seq_len rows
    features_df = features_df.iloc[-seq_len:]
    dates = df.loc[features_df.index, "date"].reset_index(drop=True)

    features = features_df.values.astype(np.float32)
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std[std == 0] = 1.0
    features_norm = ((features - mean) / std).astype(np.float32)

    # Time features
    t_day = np.clip(np.array([d.day for d in dates], dtype=np.int64) - 1, 0, 30)
    t_month = np.clip(np.array([d.month for d in dates], dtype=np.int64) - 1, 0, 11)
    t_year = np.clip(np.array([d.year for d in dates], dtype=np.int64) - DataConfig.base_year, 0, 99)
    t_min = np.zeros(seq_len, dtype=np.int64)

    return {
        "features": features_norm,
        "mean": mean.astype(np.float32).squeeze(0),
        "std": std.astype(np.float32).squeeze(0),
        "time": {"minute": t_min, "day": t_day, "month": t_month, "year": t_year},
        "last_close": float(df["close"].iloc[-1]),
        "last_date": str(dates.iloc[-1].date()),
    }


def discover_all_stocks(data_dir="dataset"):
    """Find all stock CSV files in the dataset directory."""
    import glob
    pattern = os.path.join(_PARENT, data_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    stocks = []
    for f in files:
        symbol = os.path.basename(f).replace(".csv", "")
        if symbol.isdigit():
            stocks.append(symbol)
    return stocks


# ─── Batch Prediction Engine ──────────────────────────────────────────────────

class BatchPredictor:
    """High-throughput batch predictor for all stocks."""

    def __init__(self, model, tokenizer, device, batch_size=8, use_amp=True, amp_dtype="bf16"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

    @torch.no_grad()
    def predict_stock(self, stock_data, horizon):
        """Predict N days for a single stock using optimized AR (v2 approach)."""
        feats = torch.from_numpy(stock_data["features"]).unsqueeze(0).to(
            self.device, dtype=torch.float32)
        means = torch.from_numpy(stock_data["mean"]).unsqueeze(0).to(
            self.device, dtype=torch.float32)
        stds = torch.from_numpy(stock_data["std"]).unsqueeze(0).to(
            self.device, dtype=torch.float32)
        times = {
            k: torch.from_numpy(v).unsqueeze(0).to(self.device, dtype=torch.long)
            for k, v in stock_data["time"].items()
        }

        seq_len = feats.size(1)
        with autocast_ctx(self.device, self.use_amp, self.amp_dtype):
            idx_c, idx_f = self.tokenizer.encode(feats)

        cur_c = idx_c[:, :seq_len]
        cur_f = idx_f[:, :seq_len]

        pred_c_list, pred_f_list = [], []

        for step in range(horizon):
            sl = cur_c.size(1)
            with autocast_ctx(self.device, self.use_amp, self.amp_dtype):
                logits_c, logits_f, _ = self.model(
                    cur_c, cur_f,
                    times["minute"][:, :sl], times["day"][:, :sl],
                    times["month"][:, :sl], times["year"][:, :sl],
                    last_only=True,
                )

            pc = logits_c[:, -1, :].float().argmax(dim=-1)
            pf = logits_f[:, -1, :].float().argmax(dim=-1)
            pred_c_list.append(pc)
            pred_f_list.append(pf)

            if step < horizon - 1:
                cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
                cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)

        pred_c = torch.stack(pred_c_list, dim=1)
        pred_f = torch.stack(pred_f_list, dim=1)
        decoded = self.tokenizer.decode(pred_c, pred_f)
        pred_returns = decoded[:, :, 0].float() * stds[:, 0:1] + means[:, 0:1]

        pred_rets = pred_returns[0].tolist()
        pred_prices = []
        last_close = float(stock_data["last_close"])
        cum_log_ret = 0.0
        for ret in pred_rets:
            cum_log_ret += ret
            pred_prices.append(last_close * np.exp(cum_log_ret))

        return {
            "log_returns": pred_rets,
            "predicted_prices": pred_prices,
            "pred_indices_coarse": pred_c.cpu().squeeze(0).tolist(),
            "pred_indices_fine": pred_f.cpu().squeeze(0).tolist(),
        }

    def predict_batch(self, stock_data_list, horizon):
        """Predict for a batch of stocks in parallel.

        All stocks must have the same sequence length (they should, = 1023).
        """
        B = len(stock_data_list)
        seq_len = len(stock_data_list[0]["features"])

        # Stack features
        feats = torch.stack([
            torch.from_numpy(sd["features"]) for sd in stock_data_list
        ]).to(self.device, dtype=torch.float32)

        means = torch.stack([
            torch.from_numpy(sd["mean"]) for sd in stock_data_list
        ]).to(self.device, dtype=torch.float32)

        stds = torch.stack([
            torch.from_numpy(sd["std"]) for sd in stock_data_list
        ]).to(self.device, dtype=torch.float32)

        times = {}
        for key in ("minute", "day", "month", "year"):
            times[key] = torch.stack([
                torch.from_numpy(sd["time"][key]) for sd in stock_data_list
            ]).to(self.device, dtype=torch.long)

        with autocast_ctx(self.device, self.use_amp, self.amp_dtype):
            idx_c, idx_f = self.tokenizer.encode(feats)

        # v2-optimized AR: full batch forward each step
        cur_c = idx_c[:, :seq_len]
        cur_f = idx_f[:, :seq_len]

        pred_c = torch.empty(B, horizon, device=self.device, dtype=torch.long)
        pred_f = torch.empty(B, horizon, device=self.device, dtype=torch.long)

        for step in range(horizon):
            sl = cur_c.size(1)
            with autocast_ctx(self.device, self.use_amp, self.amp_dtype):
                logits_c, logits_f, _ = self.model(
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

        # Fused decode: all steps at once
        decoded = self.tokenizer.decode(pred_c, pred_f)
        pred_returns_batch = decoded[:, :, 0].float() * stds[:, 0:1] + means[:, 0:1]

        results = []
        for i in range(B):
            pred_rets = pred_returns_batch[i].tolist()
            pred_prices = []
            last_close = float(stock_data_list[i]["last_close"])
            cum_log_ret = 0.0
            for ret in pred_rets:
                cum_log_ret += ret
                pred_prices.append(last_close * np.exp(cum_log_ret))

            results.append({
                "log_returns": pred_rets,
                "predicted_prices": pred_prices,
                "pred_indices_coarse": pred_c[i].cpu().tolist(),
                "pred_indices_fine": pred_f[i].cpu().tolist(),
            })

        return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch prediction for all stocks")
    parser.add_argument("--horizon", type=int, default=10, help="Forecast horizon (days)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="Max stocks to process (0=all)")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output-dir", default="outputs/batch_predictions")
    parser.add_argument("--output-format", default="json", choices=["json", "csv", "pt"])
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stocks with existing predictions")
    parser.add_argument("--use-compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="use_compile", action="store_false")
    args = parser.parse_args(argv)

    device = setup_device()
    print(f"\n{'='*60}")
    print(f"Batch Prediction — All Stocks")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Horizon: {args.horizon} days")
    print(f"Batch size: {args.batch_size}")

    # Discover stocks
    print("\n[1/4] Discovering stocks...")
    all_stocks = discover_all_stocks(args.data_dir)
    print(f"  Found {len(all_stocks)} stock CSV files")

    if args.max_stocks > 0 and args.max_stocks < len(all_stocks):
        rng = np.random.default_rng(42)
        all_stocks = sorted(rng.choice(all_stocks, size=args.max_stocks, replace=False).tolist())
        print(f"  Using {len(all_stocks)} stocks (sampled)")

    # Check resume
    os.makedirs(args.output_dir, exist_ok=True)
    if args.resume:
        existing = set()
        for f in os.listdir(args.output_dir):
            if f.endswith(".json"):
                existing.add(f.replace(".json", ""))
        all_stocks = [s for s in all_stocks if s not in existing]
        print(f"  Resuming: {len(all_stocks)} stocks remaining after resume")

    # Load model
    print("\n[2/4] Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        device, args.checkpoint, args.tokenizer_path
    )

    if args.use_compile and device.type == "cuda":
        print("  Compiling model...")
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create predictor
    predictor = BatchPredictor(
        model, tokenizer, device,
        batch_size=args.batch_size,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
    )

    # Process stocks
    print(f"\n[3/4] Processing {len(all_stocks)} stocks...")

    # First, load all stock features (CPU only, parallelized via threading)
    print("  Loading stock features...")
    stock_data_map = {}
    failed = []
    for symbol in tqdm(all_stocks, desc="  Loading CSV"):
        data = load_stock_features(symbol, args.data_dir)
        if data is not None:
            stock_data_map[symbol] = data
        else:
            failed.append(symbol)

    stocks_loaded = sorted(stock_data_map.keys())
    print(f"  Loaded {len(stocks_loaded)} stocks, {len(failed)} failed")

    # Batch process
    print(f"\n[4/4] Running batch predictions...")
    total_start = time.perf_counter()
    all_results = {}
    batch_count = 0

    for start in tqdm(range(0, len(stocks_loaded), args.batch_size), desc="  Predicting"):
        end = min(start + args.batch_size, len(stocks_loaded))
        batch_symbols = stocks_loaded[start:end]
        batch_data = [stock_data_map[s] for s in batch_symbols]

        try:
            batch_results = predictor.predict_batch(batch_data, args.horizon)

            for symbol, result in zip(batch_symbols, batch_results):
                result["symbol"] = symbol
                result["horizon"] = args.horizon
                result["last_close"] = float(stock_data_map[symbol]["last_close"])
                result["last_date"] = stock_data_map[symbol]["last_date"]
                all_results[symbol] = result

                # Save individual result
                if args.output_format == "json":
                    out_path = os.path.join(args.output_dir, f"{symbol}.json")
                    with open(out_path, "w") as f:
                        json.dump(result, f, ensure_ascii=False)

            batch_count += 1

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at batch {batch_count}, reducing batch size and retrying...")
            torch.cuda.empty_cache()
            # Fall back to single-stock prediction
            for symbol in batch_symbols:
                try:
                    data = stock_data_map[symbol]
                    result = predictor.predict_stock(data, args.horizon)
                    result["symbol"] = symbol
                    result["horizon"] = args.horizon
                    result["last_close"] = float(data["last_close"])
                    result["last_date"] = data["last_date"]
                    all_results[symbol] = result

                    if args.output_format == "json":
                        out_path = os.path.join(args.output_dir, f"{symbol}.json")
                        with open(out_path, "w") as f:
                            json.dump(result, f, ensure_ascii=False)
                except Exception as e:
                    failed.append(symbol)
            torch.cuda.empty_cache()

    total_time = time.perf_counter() - total_start

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"Batch Prediction Summary")
    print(f"{'='*60}")
    print(f"  Total stocks: {len(all_stocks)}")
    print(f"  Successfully predicted: {len(all_results)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Horizon: {args.horizon} days")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Throughput: {len(all_results) / total_time:.1f} stocks/sec")
    print(f"  Per stock: {total_time / max(1, len(all_results)) * 1000:.1f} ms")
    report_memory(device)

    # Save summary
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "horizon": args.horizon,
        "batch_size": args.batch_size,
        "total_stocks": len(all_stocks),
        "successful": len(all_results),
        "failed": len(failed),
        "failed_symbols": failed[:100],
        "total_time_seconds": total_time,
        "throughput_stocks_per_sec": len(all_results) / max(total_time, 1e-6),
    }

    summary_path = os.path.join(args.output_dir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

    # Save consolidated predictions
    if args.output_format == "pt":
        torch.save(all_results, os.path.join(args.output_dir, "all_predictions.pt"))
    elif args.output_format == "csv":
        csv_rows = []
        for symbol, result in all_results.items():
            row = {"symbol": symbol}
            for d in range(args.horizon):
                row[f"day_{d+1}_log_ret"] = result["log_returns"][d]
                row[f"day_{d+1}_price"] = result["predicted_prices"][d]
            csv_rows.append(row)
        pd.DataFrame(csv_rows).to_csv(
            os.path.join(args.output_dir, "all_predictions.csv"), index=False
        )

    return all_results


if __name__ == "__main__":
    main()
