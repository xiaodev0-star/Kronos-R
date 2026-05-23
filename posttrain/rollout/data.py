# -*- coding: utf-8 -*-
"""Strict train/val rollout windows for autoregressive post-training.

The rollout task needs 1023 observed tokens plus 10 hidden future tokens.  The
existing 1024-token caches are single-step caches, so this module builds an
independent cache from raw CSV files and excludes the demo date range entirely.
Normalization statistics are computed from the observed prefix only.
"""

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import DataConfig, PostTrainRolloutConfig
from data_processor import _dataset_source_fingerprint


TIME_KEYS = ("minute", "day", "month", "year")
FEATURE_COLS = tuple(DataConfig.feature_cols)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_project_path(path_like):
    text = str(path_like or "").strip()
    if not text:
        return ""
    expanded = os.path.expanduser(text)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(_PROJECT_ROOT, expanded))


@dataclass(frozen=True)
class RolloutSplitInfo:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    demo_start: str
    demo_end: str
    demo_days: int
    total_dates: int


def _decode_sector_id(symbol):
    sector_banking = 0
    sector_securities = 1
    sector_machinery = 11
    sector_electronics = 23
    sector_semiconductor = 24
    sector_new_energy = 41
    sector_other = 50
    try:
        code = int(symbol) if str(symbol).isdigit() else 0
    except Exception:
        code = 0
    if 601288 <= code <= 601398:
        return sector_banking
    if 600030 <= code <= 600999:
        return sector_securities
    if 600000 <= code <= 600029:
        return sector_banking
    if 300000 <= code <= 300749:
        return sector_electronics
    if 300750 <= code <= 300999:
        return sector_new_energy
    if 688000 <= code <= 688999:
        return sector_semiconductor
    if 2000 <= code <= 2999:
        return sector_machinery
    return sector_other


def _cache_signature(mode, cfg):
    data_dir = resolve_project_path(DataConfig.data_dir)
    return {
        "mode": str(mode),
        "prefix_len": int(cfg.prefix_len),
        "horizon": int(cfg.horizon),
        "stride_ratio": float(cfg.stride_ratio),
        "feature_cols": tuple(FEATURE_COLS),
        "random_seed": int(getattr(DataConfig, "random_seed", 42)),
        "train_val_split": float(DataConfig.train_val_split),
        "demo_days": int(max(1, int(getattr(DataConfig, "demo_days", 30)))),
        "demo_ratio": float(DataConfig.demo_ratio),
        "max_stocks": int(cfg.max_stocks) if int(cfg.max_stocks) > 0 else None,
        "normalization": "prefix_only",
        "source_fingerprint": _dataset_source_fingerprint(data_dir),
    }


def rollout_cache_path(mode, cfg=PostTrainRolloutConfig):
    cache_dir = resolve_project_path(cfg.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"rollout_{mode}.pt")


def _read_stock_frames(cfg):
    data_dir = resolve_project_path(DataConfig.data_dir)
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    max_stocks = int(getattr(cfg, "max_stocks", 0) or 0)
    if max_stocks > 0 and len(files) > max_stocks:
        rng = np.random.default_rng(int(getattr(DataConfig, "random_seed", 42)))
        files = sorted(rng.choice(files, size=max_stocks, replace=False).tolist())

    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    frames = []
    for path in tqdm(files, desc="Loading rollout CSV"):
        try:
            df = pd.read_csv(
                path,
                usecols=lambda col: col in required or col in {"symbol", "sector_id"},
                low_memory=False,
                dtype={"symbol": str},
            )
            if not required.issubset(df.columns):
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            df = df[df["volume"] > 0].reset_index(drop=True)
            if len(df) < int(cfg.prefix_len) + int(cfg.horizon):
                continue

            if "symbol" in df.columns and len(df["symbol"]) > 0 and pd.notna(df["symbol"].iloc[0]):
                symbol = str(df["symbol"].iloc[0])
            else:
                symbol = os.path.basename(path).split(".")[0]
            if "sector_id" in df.columns and len(df["sector_id"]) > 0 and pd.notna(df["sector_id"].iloc[0]):
                sector_id = int(df["sector_id"].iloc[0])
            else:
                sector_id = _decode_sector_id(symbol)
            frames.append({"symbol": symbol, "sector_id": sector_id, "df": df})
        except Exception:
            continue
    return frames


def _split_info(frames):
    all_dates = sorted(set(d for item in frames for d in item["df"]["date"].tolist()))
    total_dates = len(all_dates)
    if total_dates < 3:
        raise RuntimeError("Date range is too small for rollout split.")
    demo_days = min(max(1, int(getattr(DataConfig, "demo_days", 30))), total_dates - 1)
    split_demo_idx = max(2, total_dates - demo_days)
    split_train_val_idx = int(split_demo_idx * float(DataConfig.train_val_split))
    split_train_val_idx = min(max(split_train_val_idx, 1), split_demo_idx - 1)

    def iso(idx):
        return pd.Timestamp(all_dates[idx]).date().isoformat()

    return RolloutSplitInfo(
        train_start=iso(0),
        train_end=iso(split_train_val_idx - 1),
        val_start=iso(split_train_val_idx),
        val_end=iso(split_demo_idx - 1),
        demo_start=iso(split_demo_idx),
        demo_end=iso(total_dates - 1),
        demo_days=int(demo_days),
        total_dates=int(total_dates),
    )


def _time_features(dates):
    return {
        "minute": np.zeros(len(dates), dtype=np.int16),
        "day": np.clip(np.asarray([d.day for d in dates], dtype=np.int16) - 1, 0, 30),
        "month": np.clip(np.asarray([d.month for d in dates], dtype=np.int16) - 1, 0, 11),
        "year": np.clip(
            np.asarray([d.year - DataConfig.base_year for d in dates], dtype=np.int16),
            0,
            99,
        ),
    }


def _mode_accepts_window(mode, target_start, target_end, split):
    start = pd.Timestamp(target_start).date().isoformat()
    end = pd.Timestamp(target_end).date().isoformat()
    if mode == "train":
        return split.train_start <= start and end <= split.train_end
    if mode == "val":
        return split.val_start <= start and end <= split.val_end
    if mode == "demo":
        return split.demo_start <= start and end <= split.demo_end
    raise ValueError(f"Unsupported rollout mode: {mode}")


def _process_features(df):
    out = df.sort_values("date").reset_index(drop=True).copy()
    prev_close = out["close"].shift(1)
    out["log_ret"] = np.log(out["close"] / prev_close)
    out["log_high"] = np.log(out["high"] / prev_close)
    out["log_low"] = np.log(out["low"] / prev_close)
    out["log_open"] = np.log(out["open"] / prev_close)
    out["log_vol"] = np.log1p(out["volume"])
    out["log_amt"] = np.log1p(out["amount"])
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=list(FEATURE_COLS)).reset_index(drop=True)
    return out


def build_rollout_cache(mode, cfg=PostTrainRolloutConfig):
    prefix_len = int(cfg.prefix_len)
    horizon = int(cfg.horizon)
    total_len = prefix_len + horizon
    stride = max(1, int(prefix_len * float(cfg.stride_ratio)))

    frames = _read_stock_frames(cfg)
    if not frames:
        raise RuntimeError("No valid stock CSV files found for rollout cache.")
    split = _split_info(frames)

    features: List[np.ndarray] = []
    sector_ids: List[int] = []
    seq_stats: List[Dict[str, np.ndarray]] = []
    time_parts = {key: [] for key in TIME_KEYS}
    actual_returns: List[np.ndarray] = []
    symbols: List[str] = []
    target_dates: List[List[str]] = []

    for item in tqdm(frames, desc=f"Building rollout {mode} windows"):
        try:
            processed = _process_features(item["df"])
            if len(processed) < total_len:
                continue
            raw = processed[list(FEATURE_COLS)].values.astype(np.float32)
            dates = processed["date"].tolist()
            num_windows = (len(raw) - total_len) // stride + 1
            for window_idx in range(num_windows):
                start = window_idx * stride
                prefix_end = start + prefix_len
                end = start + total_len
                target_start_date = dates[prefix_end]
                target_end_date = dates[end - 1]
                if not _mode_accepts_window(mode, target_start_date, target_end_date, split):
                    continue

                prefix = raw[start:prefix_end]
                mean = np.mean(prefix, axis=0).astype(np.float32)
                std = np.std(prefix, axis=0).astype(np.float32)
                std[std == 0] = 1.0

                seq = raw[start:end]
                seq_norm = ((seq - mean) / std).astype(np.float32)
                seq_dates = dates[start:end]
                tf = _time_features(seq_dates)

                features.append(seq_norm)
                sector_ids.append(int(item["sector_id"]))
                seq_stats.append({"mean": mean, "std": std})
                for key in TIME_KEYS:
                    time_parts[key].append(tf[key])
                actual_returns.append(seq[prefix_len:, 0].astype(np.float32))
                symbols.append(str(item["symbol"]))
                target_dates.append([pd.Timestamp(d).date().isoformat() for d in dates[prefix_end:end]])
        except Exception:
            continue

    if features:
        features_tensor = torch.from_numpy(np.stack(features, axis=0).astype(np.float32))
        actual_tensor = torch.from_numpy(np.stack(actual_returns, axis=0).astype(np.float32))
        time_features = {
            key: torch.from_numpy(np.stack(parts, axis=0).astype(np.int16))
            for key, parts in time_parts.items()
        }
    else:
        features_tensor = torch.empty((0, total_len, len(FEATURE_COLS)), dtype=torch.float32)
        actual_tensor = torch.empty((0, horizon), dtype=torch.float32)
        time_features = {
            key: torch.empty((0, total_len), dtype=torch.int16)
            for key in TIME_KEYS
        }

    payload = {
        "features": features_tensor,
        "sector_ids": torch.as_tensor(sector_ids, dtype=torch.long),
        "time_features": time_features,
        "seq_stats": seq_stats,
        "actual_returns": actual_tensor,
        "symbols": symbols,
        "target_dates": target_dates,
        "split_info": split.__dict__,
        "_rollout_cache_signature": _cache_signature(mode, cfg),
    }
    path = rollout_cache_path(mode, cfg)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    print(f"Saved rollout {mode} cache: {path} ({features_tensor.size(0)} windows)")
    return payload


def load_rollout_cache(mode, cfg=PostTrainRolloutConfig):
    path = rollout_cache_path(mode, cfg)
    rebuild = bool(getattr(cfg, "cache_rebuild", False))
    expected = _cache_signature(mode, cfg)
    if os.path.exists(path) and not rebuild:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("_rollout_cache_signature") == expected:
            return payload
        print(f"Rollout {mode} cache signature mismatch; rebuilding: {path}")
    return build_rollout_cache(mode, cfg)


class RolloutWindowDataset(Dataset):
    def __init__(self, mode, cfg=PostTrainRolloutConfig, max_samples=0, seed=42):
        self.mode = str(mode)
        self.cfg = cfg
        self.payload = load_rollout_cache(self.mode, cfg)
        self.features = self.payload["features"].to(dtype=torch.float32)
        self.sector_ids = self.payload["sector_ids"].to(dtype=torch.long)
        self.time_features = {
            key: value.to(dtype=torch.long)
            for key, value in self.payload["time_features"].items()
        }
        self.seq_stats = self.payload["seq_stats"]
        self.actual_returns = self.payload["actual_returns"].to(dtype=torch.float32)
        self.symbols = self.payload.get("symbols", [""] * int(self.features.size(0)))
        self.target_dates = self.payload.get("target_dates", [[] for _ in range(int(self.features.size(0)))])

        n = int(self.features.size(0))
        indices = np.arange(n, dtype=np.int64)
        max_samples = int(max_samples or 0)
        if max_samples > 0 and max_samples < n:
            rng = np.random.default_rng(int(seed))
            indices = np.sort(rng.choice(indices, size=max_samples, replace=False))
        self.indices = indices

    def __len__(self):
        return int(len(self.indices))

    def __getitem__(self, idx):
        row = int(self.indices[int(idx)])
        stat = self.seq_stats[row]
        return {
            "features": self.features[row],
            "sector_id": self.sector_ids[row],
            "time": {key: value[row] for key, value in self.time_features.items()},
            "mean": torch.as_tensor(stat["mean"], dtype=torch.float32),
            "std": torch.as_tensor(stat["std"], dtype=torch.float32),
            "actual_returns": self.actual_returns[row],
            "sample_id": f"{self.mode}:{row}",
            "symbol": self.symbols[row] if row < len(self.symbols) else "",
            "target_dates": self.target_dates[row] if row < len(self.target_dates) else [],
        }


def rollout_collate(batch):
    return {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "sector_ids": torch.as_tensor([int(item["sector_id"]) for item in batch], dtype=torch.long),
        "time": {
            key: torch.stack([item["time"][key] for item in batch], dim=0)
            for key in TIME_KEYS
        },
        "means": torch.stack([item["mean"] for item in batch], dim=0),
        "stds": torch.stack([item["std"] for item in batch], dim=0),
        "actual_returns": torch.stack([item["actual_returns"] for item in batch], dim=0),
        "sample_ids": [item["sample_id"] for item in batch],
        "symbols": [item["symbol"] for item in batch],
        "target_dates": [item["target_dates"] for item in batch],
    }

