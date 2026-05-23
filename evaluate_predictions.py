# -*- coding: utf-8 -*-
"""评估与预测入口：加载模型、构建 demo 滚动预测项。被 posttrain/direction/train_da.py 引用。"""

import numpy as np
import torch

from config import DataConfig, ModelConfig, PathConfig, TrainingConfig
from data_processor import AShareDataset
from model.kronos_reasoning import KronosReasoningGPT
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs


def load_model(device, checkpoint_path, strict_checkpoint_compat=True):
    """从 base checkpoint 加载 KronosReasoningGPT + tokenizer。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_config = checkpoint.get("model_config", {})
    if not model_config:
        model_config = _export_runtime_model_config()

    # 从 checkpoint 或 tokenizer.pt 加载 tokenizer。
    tokenizer_sd = checkpoint.get("tokenizer_state_dict")
    if tokenizer_sd is not None and "config" in checkpoint.get("config", {}):
        tk_config = checkpoint["config"]
        tokenizer = HierarchicalQuantizer(**build_tokenizer_kwargs(tk_config)).to(device)
    else:
        tokenizer = _load_tokenizer(device)
    if tokenizer_sd is not None:
        tokenizer.load_state_dict(tokenizer_sd, strict=False)
    tokenizer.eval()
    tokenizer.requires_grad_(False)

    # 构建 KronosReasoningGPT (DSA + GQA, no sector).
    from config import ModelConfig as MC
    model = KronosReasoningGPT(
        dim=model_config.get("dim", MC.dim),
        depth=model_config.get("depth", MC.depth),
        heads=model_config.get("heads", MC.heads),
        num_kv_heads=model_config.get("num_kv_heads", 2),
        dsa_windows=model_config.get("dsa_windows", None),
        dropout=model_config.get("dropout", MC.dropout),
        vocab_size_coarse=model_config.get("vocab_size_coarse", MC.vocab_size_coarse),
        vocab_size_fine=model_config.get("vocab_size_fine", MC.vocab_size_fine),
        num_latent_tokens=model_config.get("num_latent_tokens", MC.num_latent_tokens),
        latent_reasoner_depth=model_config.get("latent_reasoner_depth", MC.latent_reasoner_depth),
        latent_cross_heads=model_config.get("latent_cross_heads", MC.latent_cross_heads),
        position_encoding=model_config.get("position_encoding", MC.position_encoding),
        rope_base=model_config.get("rope_base", MC.rope_base),
        alibi_decay_base=model_config.get("alibi_decay_base", MC.alibi_decay_base),
        max_len=model_config.get("max_len", MC.max_len),
        horizon_tokens=model_config.get("horizon_tokens", MC.horizon_tokens),
        horizon_decoder_depth=model_config.get("horizon_decoder_depth", MC.horizon_decoder_depth),
        horizon_decoder_heads=model_config.get("horizon_decoder_heads", MC.horizon_decoder_heads),
        use_revin=model_config.get("use_revin", MC.use_revin),
        revin_affine=model_config.get("revin_affine", MC.revin_affine),
        revin_eps=model_config.get("revin_eps", MC.revin_eps),
        num_factor_tokens=model_config.get("num_factor_tokens", MC.num_factor_tokens),
    ).to(device)

    sd = checkpoint.get("model_state_dict")
    if sd is not None:
        if strict_checkpoint_compat:
            model.load_state_dict(sd, strict=True)
        else:
            model.load_state_dict(sd, strict=False)
    else:
        print("Warning: no model_state_dict in checkpoint, using fresh model weights.")

    model.eval()
    return model, tokenizer


def _load_tokenizer(device):
    """从 TokenizerConfig.save_path 加载 tokenizer。"""
    path = TrainingConfig.tokenizer_path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    tokenizer = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tokenizer.load_state_dict(ckpt["model_state_dict"], strict=False)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    print(f"Tokenizer loaded: {path}")
    return tokenizer


def _export_runtime_model_config():
    """Export model config for DSA+GQA checkpoint (no sector, no LinearAttention)."""
    keys = (
        "dim", "depth", "heads", "dropout",
        "vocab_size_coarse", "vocab_size_fine",
        "num_latent_tokens", "latent_reasoner_depth", "latent_cross_heads",
        "position_encoding", "rope_base", "alibi_decay_base",
        "max_len", "horizon_tokens", "horizon_decoder_depth", "horizon_decoder_heads",
        "use_revin", "revin_affine", "revin_eps", "num_factor_tokens",
        "num_kv_heads", "dsa_windows",
    )
    cfg = {}
    for key in keys:
        if hasattr(ModelConfig, key):
            cfg[key] = getattr(ModelConfig, key)
    # DSA defaults if not in ModelConfig
    cfg.setdefault("num_kv_heads", 2)
    cfg.setdefault("dsa_windows", [None, 512, 512, None])
    return cfg


def _decode_sector_id(symbol):
    sector_banking = 0
    sector_securities = 1
    sector_machinery = 11
    sector_electronics = 23
    sector_semiconductor = 24
    sector_new_energy = 41
    sector_other = 50
    try:
        code = int(symbol) if symbol.isdigit() else 0
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


def build_rolling_1d_eval_items(demo_dataset, symbols=None):
    """从 demo_dataset 构建滚动 1 天预测评估项。

    每项包含一个长度为 seq_len 的归一化 prompt，
    对应预测次日涨跌的滚动任务。
    """
    seq_len = DataConfig.seq_len
    if symbols is None:
        symbols = sorted(demo_dataset.raw_data.keys())

    eval_items = []
    skip_reasons = {}

    for sym in symbols:
        raw = demo_dataset.raw_data.get(sym)
        if raw is None:
            skip_reasons[sym] = "no raw data"
            continue
        dates = raw.get("dates", [])
        closes = raw.get("close", [])

        sector_id = raw.get("sector_id")
        if sector_id is None:
            sector_id = _decode_sector_id(sym)

        available_days = len(dates)
        if available_days < seq_len + 1:
            skip_reasons[sym] = f"too few days ({available_days} < {seq_len + 1})"
            continue

        df = None
        try:
            import pandas as pd
            df = pd.DataFrame(raw)
            df["date"] = pd.to_datetime(df["dates"])
            df = df.sort_values("date").reset_index(drop=True)
        except Exception:
            skip_reasons[sym] = "failed to build DataFrame"
            continue

        prev_close = df["close"].shift(1)
        amt_col = df["amount"] if "amount" in df.columns else df["volume"] * 0
        features_df = pd.DataFrame({
            "log_ret": np.log(df["close"] / prev_close),
            "log_high": np.log(df["high"] / prev_close),
            "log_low": np.log(df["low"] / prev_close),
            "log_open": np.log(df["open"] / prev_close),
            "log_vol": np.log1p(df["volume"]),
            "log_amt": np.log1p(amt_col),
        })
        features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(features_df) < seq_len + 1:
            skip_reasons[sym] = f"too few valid feature rows ({len(features_df)} < {seq_len + 1})"
            continue

        aligned_dates = df.loc[features_df.index, "date"].reset_index(drop=True)

        stride = max(1, int(seq_len * DataConfig.stride_ratio))
        for i in range(0, len(features_df) - seq_len, stride):
            seq_features = features_df.iloc[i:i + seq_len].values.astype(np.float64)
            seq_dates = aligned_dates.iloc[i:i + seq_len]

            mean = np.mean(seq_features, axis=0)
            std = np.std(seq_features, axis=0)
            std[std == 0] = 1.0
            seq_norm = ((seq_features - mean) / std).astype(np.float32)

            time_day = np.clip(np.array([d.day for d in seq_dates], dtype=np.int16) - 1, 0, 30)
            time_month = np.clip(np.array([d.month for d in seq_dates], dtype=np.int16) - 1, 0, 11)
            time_year = np.clip(np.array([d.year for d in seq_dates], dtype=np.int16) - DataConfig.base_year, 0, 99)

            eval_items.append({
                "prompt_norm": seq_norm,
                "prompt_mean": mean.astype(np.float32),
                "prompt_std": std.astype(np.float32),
                "sector_id": sector_id,
                "prompt_time": {
                    "minute": np.zeros(seq_len, dtype=np.int64),
                    "day": time_day.astype(np.int64),
                    "month": time_month.astype(np.int64),
                    "year": time_year.astype(np.int64),
                },
                "hist_closes": closes[max(0, i):i + seq_len],
                "actual_future": closes[i + seq_len:i + seq_len + 1],
                "future_dates": seq_dates.iloc[-1:].tolist(),
            })

    return eval_items, skip_reasons
