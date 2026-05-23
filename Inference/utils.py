# -*- coding: utf-8 -*-
"""Shared utilities for Kronos-R inference optimization scripts."""

import gc
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add parent to path for config import
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from config import DataConfig, ModelConfig, TrainingConfig, TokenizerConfig


# ─── Device & precision setup ────────────────────────────────────────────────

def setup_device(precision="bf16"):
    """Configure device and precision for optimal inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    return device


def autocast_ctx(device, enabled=True, dtype_name="bf16"):
    """Get autocast context for inference."""
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if str(dtype_name).strip().lower() in ("bf16", "bfloat16") else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


# ─── Model loading ────────────────────────────────────────────────────────────

def load_tokenizer(device, tokenizer_path=None):
    """Load tokenizer from checkpoint."""
    from Inference.models.tokenizer import HierarchicalQuantizer
    from Inference.models.tokenizer_config import build_tokenizer_kwargs

    if tokenizer_path is None:
        tokenizer_path = os.path.join(_PARENT, TrainingConfig.tokenizer_path)
    ckpt = torch.load(tokenizer_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    tokenizer = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tokenizer.load_state_dict(ckpt["model_state_dict"], strict=False)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer


def load_base_model(device, checkpoint_path=None, strict_compat=False):
    """Load KronosReasoningGPT model from checkpoint."""
    from Inference.models.kronos_reasoning import KronosReasoningGPT

    if checkpoint_path is None:
        checkpoint_path = os.path.join(_PARENT, TrainingConfig.base_model_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", {})

    model = KronosReasoningGPT(
        dim=model_config.get("dim", ModelConfig.dim),
        depth=model_config.get("depth", ModelConfig.depth),
        heads=model_config.get("heads", ModelConfig.heads),
        num_kv_heads=model_config.get("num_kv_heads", 2),
        dsa_windows=model_config.get("dsa_windows", None),
        dropout=model_config.get("dropout", ModelConfig.dropout),
        vocab_size_coarse=model_config.get("vocab_size_coarse", ModelConfig.vocab_size_coarse),
        vocab_size_fine=model_config.get("vocab_size_fine", ModelConfig.vocab_size_fine),
        num_latent_tokens=model_config.get("num_latent_tokens", ModelConfig.num_latent_tokens),
        latent_reasoner_depth=model_config.get("latent_reasoner_depth", ModelConfig.latent_reasoner_depth),
        latent_cross_heads=model_config.get("latent_cross_heads", ModelConfig.latent_cross_heads),
        position_encoding=model_config.get("position_encoding", ModelConfig.position_encoding),
        rope_base=model_config.get("rope_base", ModelConfig.rope_base),
        alibi_decay_base=model_config.get("alibi_decay_base", ModelConfig.alibi_decay_base),
        max_len=model_config.get("max_len", ModelConfig.max_len),
        use_revin=model_config.get("use_revin", ModelConfig.use_revin),
        num_factor_tokens=model_config.get("num_factor_tokens", ModelConfig.num_factor_tokens),
    ).to(device)

    sd = checkpoint.get("model_state_dict")
    if sd is not None:
        if strict_compat:
            model.load_state_dict(sd, strict=True)
        else:
            model.load_state_dict(sd, strict=False)

    model.eval()
    return model


def load_model_and_tokenizer(device, checkpoint_path=None, tokenizer_path=None):
    """Load both model and tokenizer."""
    tokenizer = load_tokenizer(device, tokenizer_path)
    model = load_base_model(device, checkpoint_path)
    return model, tokenizer


# ─── Data loading for inference ───────────────────────────────────────────────

def load_rollout_data(mode="demo"):
    """Load rollout cache data for inference."""
    from posttrain.rollout.data import load_rollout_cache, RolloutWindowDataset, rollout_collate

    try:
        from posttrain.rollout.data import PostTrainRolloutConfig
    except Exception:
        from config import PostTrainRolloutConfig

    payload = load_rollout_cache(mode)
    features = payload["features"].to(dtype=torch.float32)
    print(f"Loaded {mode} rollout data: {features.size(0)} windows, "
          f"prefix_len={features.size(1) - 10}, horizon=10")
    return payload


def prepare_inference_batch(payload, indices, device):
    """Extract a batch from rollout payload for inference."""
    features = payload["features"][indices].to(device=device, dtype=torch.float32, non_blocking=True)
    time_feats = payload["time_features"]
    seq_stats = payload["seq_stats"]
    prefix_len = int(payload["features"].size(1) - 10)

    B = features.size(0)
    means = torch.stack([
        torch.as_tensor(seq_stats[i]["mean"], dtype=torch.float32)
        for i in indices.tolist()
    ], dim=0).to(device=device, non_blocking=True)
    stds = torch.stack([
        torch.as_tensor(seq_stats[i]["std"], dtype=torch.float32)
        for i in indices.tolist()
    ], dim=0).to(device=device, non_blocking=True)

    times = {}
    for key in ("minute", "day", "month", "year"):
        times[key] = time_feats[key][indices].to(device=device, dtype=torch.long, non_blocking=True)

    actual = features[:, prefix_len:, 0].to(device=device, dtype=torch.float32)
    return {
        "features": features[:, :prefix_len, :],
        "means": means,
        "stds": stds,
        "time": times,
        "actual_returns": actual,
        "prefix_len": prefix_len,
    }


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(pred_returns, actual_returns, mape_eps=1e-4):
    """Compute MAPE, DA, MAE, RMSE metrics."""
    pred = np.asarray(pred_returns, dtype=np.float64)
    actual = np.asarray(actual_returns, dtype=np.float64)

    finite = np.isfinite(pred) & np.isfinite(actual)
    if finite.sum() == 0:
        return {"mape": float("nan"), "da": float("nan"), "mae": float("nan"),
                "rmse": float("nan"), "n": 0}

    p = pred[finite]
    a = actual[finite]

    # Ratio-space MAPE
    pr = np.exp(np.clip(p, -50.0, 50.0))
    ar = np.exp(np.clip(a, -50.0, 50.0))
    denom = np.maximum(np.abs(ar), mape_eps)
    mape = float(np.mean(np.abs(pr - ar) / denom) * 100.0)

    # Direction accuracy
    da = float(np.mean((np.sign(p) == np.sign(a))) * 100.0)

    # MAE, RMSE
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))

    return {"mape": mape, "da": da, "mae": mae, "rmse": rmse, "n": int(finite.sum())}


# ─── Benchmark timer ──────────────────────────────────────────────────────────

class Timer:
    """Context manager for timing with CUDA synchronization."""

    def __init__(self, name="", sync_cuda=True):
        self.name = name
        self.sync_cuda = sync_cuda
        self.start = None
        self.end = None
        self.elapsed = None

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

    @property
    def ms(self):
        return self.elapsed * 1000 if self.elapsed else None


# ─── Baseline inference (original method, for comparison) ────────────────────

@torch.no_grad()
def baseline_ar_predict(model, tokenizer, features, means, stds, times, horizon, device,
                        use_amp=True, amp_dtype="bf16"):
    """Original AR inference — full forward pass each step (no KV cache)."""
    B = features.size(0)
    prefix_len = features.size(1)

    # Encode prefix
    idx_c, idx_f = tokenizer.encode(features)
    cur_c = idx_c.clone()
    cur_f = idx_f.clone()

    pred_returns = []
    for step in range(horizon):
        sl = cur_c.size(1)
        step_time = {k: times[k][:, :sl] for k in ("minute", "day", "month", "year")}

        with autocast_ctx(device, use_amp, amp_dtype):
            logits_c, logits_f, _ = model(
                cur_c, cur_f,
                step_time["minute"], step_time["day"],
                step_time["month"], step_time["year"],
                last_only=True,
            )

        pc = logits_c[:, -1, :].argmax(dim=-1)
        pf = logits_f[:, -1, :].argmax(dim=-1)

        decoded = tokenizer.decode(pc.unsqueeze(1), pf.unsqueeze(1))
        pred_ret = decoded[:, 0, 0].float() * stds[:, 0] + means[:, 0]
        pred_returns.append(pred_ret)

        if step < horizon - 1:
            cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
            cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)

    return torch.stack(pred_returns, dim=1)


# ─── Memory reporting ─────────────────────────────────────────────────────────

def report_memory(device):
    """Report GPU memory usage."""
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"  GPU Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
