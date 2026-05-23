# -*- coding: utf-8 -*-
"""Scheduled self-rollout post-training for Kronos-R.

This module is intentionally independent from posttrain.direction.  It trains
for the deployment scenario where only the first 1023 tokens are real and every
future step must condition on previously generated, imperfect tokens.
"""

import argparse
import copy
import json
import math
import os
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PostTrainRolloutConfig, TrainingConfig
from evaluate_predictions import load_model
from model.lora import trainable_parameter_summary
from posttrain.rollout.data import (
    RolloutWindowDataset,
    resolve_project_path,
    rollout_cache_path,
    rollout_collate,
)
from reproducibility import set_global_seed


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _amp_dtype(name):
    value = str(name).strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    return None


def _autocast_context(device, enabled, dtype):
    if device.type != "cuda" or not enabled or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _cfg_to_dict(cfg):
    result = {}
    for key, value in vars(cfg).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result


def _cuda_peak_memory_stats(device):
    if device.type != "cuda":
        return {}
    return {
        "cuda_peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3),
        "cuda_peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / (1024 ** 3), 3),
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Post_Train_Rollout scheduled self-rollout")
    parser.add_argument("--checkpoint-path", default=PostTrainRolloutConfig.checkpoint_path)
    parser.add_argument("--output-dir", default=PostTrainRolloutConfig.output_dir)
    parser.add_argument("--save-name", default=PostTrainRolloutConfig.save_name)
    parser.add_argument("--save-epoch-checkpoints", type=_as_bool, default=PostTrainRolloutConfig.save_epoch_checkpoints)
    parser.add_argument("--prefix-len", type=int, default=PostTrainRolloutConfig.prefix_len)
    parser.add_argument("--horizon", type=int, default=PostTrainRolloutConfig.horizon)
    parser.add_argument("--stride-ratio", type=float, default=PostTrainRolloutConfig.stride_ratio)
    parser.add_argument("--cache-dir", default=PostTrainRolloutConfig.cache_dir)
    parser.add_argument("--cache-rebuild", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=PostTrainRolloutConfig.max_stocks)
    parser.add_argument("--max-train-samples", type=int, default=PostTrainRolloutConfig.max_train_samples)
    parser.add_argument("--max-val-samples", type=int, default=PostTrainRolloutConfig.max_val_samples)
    parser.add_argument("--epochs", type=int, default=PostTrainRolloutConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=PostTrainRolloutConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=PostTrainRolloutConfig.eval_batch_size)
    parser.add_argument("--accumulation-steps", type=int, default=PostTrainRolloutConfig.accumulation_steps)
    parser.add_argument("--num-workers", type=int, default=PostTrainRolloutConfig.num_workers)
    parser.add_argument("--lr", type=float, default=PostTrainRolloutConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=PostTrainRolloutConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=PostTrainRolloutConfig.grad_clip)
    parser.add_argument("--max-train-updates", type=int, default=PostTrainRolloutConfig.max_train_updates)
    parser.add_argument("--progress-interval", type=int, default=PostTrainRolloutConfig.progress_interval)
    parser.add_argument("--rollout-ratio-start", type=float, default=PostTrainRolloutConfig.rollout_ratio_start)
    parser.add_argument("--rollout-ratio-end", type=float, default=PostTrainRolloutConfig.rollout_ratio_end)
    parser.add_argument("--anchor-weight", type=float, default=PostTrainRolloutConfig.anchor_weight)
    parser.add_argument("--kl-weight", type=float, default=PostTrainRolloutConfig.kl_weight)
    parser.add_argument("--numeric-mape-weight", type=float, default=PostTrainRolloutConfig.numeric_mape_weight)
    parser.add_argument("--numeric-top-k", type=int, default=PostTrainRolloutConfig.numeric_top_k)
    parser.add_argument("--numeric-soft-ce-weight", type=float, default=PostTrainRolloutConfig.numeric_soft_ce_weight)
    parser.add_argument("--numeric-soft-ce-top-k", type=int, default=PostTrainRolloutConfig.numeric_soft_ce_top_k)
    parser.add_argument("--numeric-soft-ce-temp", type=float, default=PostTrainRolloutConfig.numeric_soft_ce_temp)
    parser.add_argument("--step-weight-gamma", type=float, default=PostTrainRolloutConfig.step_weight_gamma)
    parser.add_argument("--use-sampling", type=_as_bool, default=PostTrainRolloutConfig.use_sampling)
    parser.add_argument("--sampling-temperature", type=float, default=PostTrainRolloutConfig.sampling_temperature)
    parser.add_argument("--freeze-backbone", type=_as_bool, default=PostTrainRolloutConfig.freeze_backbone)
    parser.add_argument("--trainable-scope", choices=["all", "heads"], default=PostTrainRolloutConfig.trainable_scope)
    parser.add_argument("--use-gradient-checkpointing", type=_as_bool, default=PostTrainRolloutConfig.use_gradient_checkpointing)
    parser.add_argument("--use-amp", type=_as_bool, default=PostTrainRolloutConfig.use_amp)
    parser.add_argument("--amp-dtype", default=PostTrainRolloutConfig.amp_dtype)
    parser.add_argument("--use-tf32", type=_as_bool, default=PostTrainRolloutConfig.use_tf32)
    parser.add_argument("--mape-eps", type=float, default=PostTrainRolloutConfig.mape_eps)
    parser.add_argument("--deterministic", type=_as_bool, default=PostTrainRolloutConfig.deterministic)
    parser.add_argument("--seed", type=int, default=PostTrainRolloutConfig.random_seed)
    parser.add_argument("--eval-only", action="store_true")
    # ── Experiment N: Curriculum ──
    parser.add_argument("--curriculum-horizons", default="")
    parser.add_argument("--curriculum-updates", default="")
    # ── OpenAI-style Verifier Loop: Oracle-Guided Rollout ──
    parser.add_argument("--oracle-guided", type=_as_bool, default=False)
    parser.add_argument("--oracle-top-k", type=int, default=8)
    parser.add_argument("--oracle-temp", type=float, default=1.5)
    return parser


def _namespace_from_args(args):
    return argparse.Namespace(
        checkpoint_path=resolve_project_path(args.checkpoint_path),
        output_dir=resolve_project_path(args.output_dir),
        save_name=str(args.save_name),
        save_epoch_checkpoints=bool(args.save_epoch_checkpoints),
        prefix_len=int(args.prefix_len),
        horizon=int(args.horizon),
        stride_ratio=float(args.stride_ratio),
        cache_dir=resolve_project_path(args.cache_dir),
        cache_rebuild=bool(args.cache_rebuild),
        max_stocks=int(args.max_stocks),
        max_train_samples=int(args.max_train_samples),
        max_val_samples=int(args.max_val_samples),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        accumulation_steps=max(1, int(args.accumulation_steps)),
        num_workers=max(0, int(args.num_workers)),
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        max_train_updates=int(args.max_train_updates),
        progress_interval=max(1, int(args.progress_interval)),
        rollout_ratio_start=float(args.rollout_ratio_start),
        rollout_ratio_end=float(args.rollout_ratio_end),
        anchor_weight=float(args.anchor_weight),
        kl_weight=float(args.kl_weight),
        numeric_mape_weight=float(args.numeric_mape_weight),
        numeric_top_k=int(args.numeric_top_k),
        numeric_soft_ce_weight=float(args.numeric_soft_ce_weight),
        numeric_soft_ce_top_k=int(args.numeric_soft_ce_top_k),
        numeric_soft_ce_temp=float(args.numeric_soft_ce_temp),
        step_weight_gamma=float(args.step_weight_gamma),
        use_sampling=bool(args.use_sampling),
        sampling_temperature=float(args.sampling_temperature),
        freeze_backbone=bool(args.freeze_backbone),
        trainable_scope=str(args.trainable_scope),
        use_gradient_checkpointing=bool(args.use_gradient_checkpointing),
        use_amp=bool(args.use_amp),
        amp_dtype=str(args.amp_dtype),
        use_tf32=bool(args.use_tf32),
        mape_eps=float(args.mape_eps),
        deterministic=bool(args.deterministic),
        random_seed=int(args.seed),
        eval_only=bool(args.eval_only),
        curriculum_horizons=str(getattr(args, "curriculum_horizons", "")),
        curriculum_updates=str(getattr(args, "curriculum_updates", "")),
        oracle_guided=bool(getattr(args, "oracle_guided", False)),
        oracle_top_k=int(getattr(args, "oracle_top_k", 8)),
        oracle_temp=float(getattr(args, "oracle_temp", 1.5)),
    )


def _move_batch(batch, device):
    return {
        "features": batch["features"].to(device=device, dtype=torch.float32, non_blocking=True),
        "time": {
            key: value.to(device=device, dtype=torch.long, non_blocking=True)
            for key, value in batch["time"].items()
        },
        "means": batch["means"].to(device=device, dtype=torch.float32, non_blocking=True),
        "stds": batch["stds"].to(device=device, dtype=torch.float32, non_blocking=True),
        "actual_returns": batch["actual_returns"].to(device=device, dtype=torch.float32, non_blocking=True),
        "sample_ids": batch["sample_ids"],
        "symbols": batch["symbols"],
        "target_dates": batch["target_dates"],
    }


def _encode_features(tokenizer, features):
    with torch.no_grad():
        idx_coarse, idx_fine = tokenizer.encode(features)
    return idx_coarse.long(), idx_fine.long()


def _sample_or_argmax(logits, use_sampling, temperature):
    logits = logits.float()
    if not bool(use_sampling):
        return logits.argmax(dim=-1)
    temp = max(1e-4, float(temperature))
    probs = torch.softmax(logits / temp, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


# ═══════════════════════════════════════════════════════════════════
# OpenAI-style Verifier Loop: Oracle-Guided Step-Level Rollout
# ═══════════════════════════════════════════════════════════════════


class _ErrorTrajectoryBank:
    """Fixed-size memory bank of high-error trajectories for distillation."""

    def __init__(self, max_size=64):
        self.max_size = int(max_size)
        self.trajectories = []  # list of {tokens_c, tokens_f, per_step_error, ...}

    def add(self, tokens_c, tokens_f, per_step_error, prefix_hidden):
        """Add a trajectory; keep only the worst (highest error) ones."""
        avg_err = float(per_step_error.mean().item())
        entry = {
            "tokens_c": tokens_c.detach().cpu(),
            "tokens_f": tokens_f.detach().cpu(),
            "per_step_error": per_step_error.detach().cpu(),
            "prefix_hidden": prefix_hidden.detach().cpu() if prefix_hidden is not None else None,
            "avg_error": avg_err,
        }
        self.trajectories.append(entry)
        if len(self.trajectories) > self.max_size:
            self.trajectories.sort(key=lambda e: e["avg_error"], reverse=True)
            self.trajectories = self.trajectories[: self.max_size]

    def sample_batch(self, batch_size, device):
        """Sample a batch of error trajectories for distillation."""
        if not self.trajectories:
            return None
        import random
        sampled = random.choices(self.trajectories, k=min(batch_size, len(self.trajectories)))
        return {
            "tokens_c": torch.stack([s["tokens_c"] for s in sampled]).to(device=device),
            "tokens_f": torch.stack([s["tokens_f"] for s in sampled]).to(device=device),
            "per_step_error": torch.stack([s["per_step_error"] for s in sampled]).to(device=device),
        }

    def __len__(self):
        return len(self.trajectories)


@torch.no_grad()
def _oracle_guided_scheduled_inputs(
    model,
    tokenizer,
    idx_coarse_full,
    idx_fine_full,
    time_features,
    prefix_len,
    horizon,
    means,
    stds,
    actual_returns,
    top_k,
    temperature,
    device,
    amp_enabled,
    amp_dtype,
):
    """Oracle-guided step-level rollout with joint coarse+fine candidate search.

    At each step: ONE model forward → sample K (coarse, fine) pairs from
    temperature-scaled logits → decode all at once → pick best per sample
    via Oracle (actual return).  Append best token to context for next step.
    """
    if int(horizon) <= 1:
        return idx_coarse_full[:, :prefix_len], idx_fine_full[:, :prefix_len], 0.0

    was_training = model.training
    model.eval()

    context_c = idx_coarse_full[:, :prefix_len].clone()
    context_f = idx_fine_full[:, :prefix_len].clone()
    K = max(2, int(top_k))
    temp = max(1e-4, float(temperature))
    B = int(idx_coarse_full.size(0))

    for step in range(int(horizon) - 1):
        cur_len = int(context_c.size(1))
        cur_time = {key: value[:, :cur_len] for key, value in time_features.items()}

        # ONE forward pass (model is eval, same input → same output)
        with _autocast_context(device, amp_enabled, amp_dtype):
            logits_c, logits_f, _ = model(
                context_c, context_f,
                cur_time["minute"], cur_time["day"],
                cur_time["month"], cur_time["year"],
                last_only=True,
            )

        # Sample K (coarse, fine) pairs at once
        probs_c = torch.softmax(logits_c[:, -1, :].float() / temp, dim=-1)
        probs_f = torch.softmax(logits_f[:, -1, :].float() / temp, dim=-1)
        samples_c = torch.multinomial(probs_c, num_samples=K, replacement=True)  # [B, K]
        samples_f = torch.multinomial(probs_f, num_samples=K, replacement=True)

        # Decode all K candidates in one call
        decoded = tokenizer.decode(samples_c, samples_f)  # [B, K, 6]
        pred_norms = decoded[:, :, 0].float()
        pred_returns = pred_norms * stds[:, 0:1] + means[:, 0:1]  # [B, K]
        errs = (pred_returns - actual_returns[:, step:step + 1]).abs()

        # Pick best pair for each sample in batch
        best_k = errs.argmin(dim=1)
        rows = torch.arange(B, device=device)
        best_c = samples_c[rows, best_k]
        best_f = samples_f[rows, best_k]

        context_c = torch.cat([context_c, best_c.unsqueeze(1)], dim=1)
        context_f = torch.cat([context_f, best_f.unsqueeze(1)], dim=1)

    if was_training:
        model.train()

    return context_c, context_f, 1.0


# ═══════════════════════════════════════════════════════════════════
# Standard Scheduled Self-Rollout (non-Oracle path)
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def _build_scheduled_inputs(
    model,
    idx_coarse_full,
    idx_fine_full,
    time_features,
    prefix_len,
    horizon,
    rollout_ratio,
    use_sampling,
    sampling_temperature,
    device,
    amp_enabled,
    amp_dtype,
):
    """Build context by feeding model's own predictions back autoregressively.

    At each step, with probability (1 - rollout_ratio), use the ground-truth
    token (teacher forcing); with probability rollout_ratio, use the model's
    own argmax/sampled prediction.  Returns context of length prefix_len+horizon-1.
    """
    if int(horizon) <= 1:
        return idx_coarse_full[:, :prefix_len], idx_fine_full[:, :prefix_len], 0.0

    was_training = model.training
    model.eval()

    context_c = idx_coarse_full[:, :prefix_len].clone()
    context_f = idx_fine_full[:, :prefix_len].clone()
    B = int(idx_coarse_full.size(0))
    used_pred = 0.0

    for step in range(int(horizon) - 1):
        cur_len = int(context_c.size(1))
        cur_time = {key: value[:, :cur_len] for key, value in time_features.items()}

        with _autocast_context(device, amp_enabled, amp_dtype):
            logits_c, logits_f, _ = model(
                context_c, context_f,
                cur_time["minute"], cur_time["day"],
                cur_time["month"], cur_time["year"],
                last_only=True,
            )

        # Decide: use ground-truth or model prediction
        use_self = float(rollout_ratio) if float(rollout_ratio) >= 1.0 else (
            torch.rand(1, device=device).item() < float(rollout_ratio))

        if use_self:
            pred_c = _sample_or_argmax(logits_c[:, -1, :], use_sampling, sampling_temperature)
            pred_f = _sample_or_argmax(logits_f[:, -1, :], use_sampling, sampling_temperature)
            used_pred += 1.0
        else:
            gt_pos = prefix_len + step
            pred_c = idx_coarse_full[:, gt_pos]
            pred_f = idx_fine_full[:, gt_pos]

        context_c = torch.cat([context_c, pred_c.unsqueeze(1)], dim=1)
        context_f = torch.cat([context_f, pred_f.unsqueeze(1)], dim=1)

    if was_training:
        model.train()

    return context_c, context_f, used_pred / max(1, horizon - 1)


def _selected_logits(logits_c, logits_f, prefix_len, horizon):
    start = int(prefix_len) - 1
    end = start + int(horizon)
    return logits_c[:, start:end, :], logits_f[:, start:end, :]


def _step_weights(horizon, gamma, device):
    horizon = int(horizon)
    if horizon <= 1:
        return torch.ones(1, device=device, dtype=torch.float32)
    steps = torch.arange(horizon, device=device, dtype=torch.float32)
    return 1.0 + float(gamma) * steps / float(horizon - 1)


def _weighted_token_ce(logits_c, logits_f, target_c, target_f, weights):
    vocab_c = int(logits_c.size(-1))
    vocab_f = int(logits_f.size(-1))
    loss_c = F.cross_entropy(
        logits_c.reshape(-1, vocab_c).float(),
        target_c.reshape(-1),
        reduction="none",
    ).view_as(target_c)
    loss_f = F.cross_entropy(
        logits_f.reshape(-1, vocab_f).float(),
        target_f.reshape(-1),
        reduction="none",
    ).view_as(target_f)
    per_step = loss_c + loss_f
    weights = weights.to(device=per_step.device, dtype=per_step.dtype).view(1, -1)
    return (per_step * weights).sum() / weights.sum().clamp_min(1.0) / per_step.size(0)


def _kl_to_reference(logits_c, logits_f, ref_c, ref_f, weights):
    logp_c = F.log_softmax(logits_c.float(), dim=-1)
    logp_f = F.log_softmax(logits_f.float(), dim=-1)
    refp_c = F.softmax(ref_c.float(), dim=-1)
    refp_f = F.softmax(ref_f.float(), dim=-1)
    kl_c = F.kl_div(logp_c, refp_c, reduction="none").sum(dim=-1)
    kl_f = F.kl_div(logp_f, refp_f, reduction="none").sum(dim=-1)
    weights = weights.to(device=kl_c.device, dtype=kl_c.dtype).view(1, -1)
    return ((kl_c + kl_f) * weights).sum() / weights.sum().clamp_min(1.0) / kl_c.size(0)


def _expected_return_from_topk(tokenizer, logits_c, logits_f, means, stds, top_k):
    top_k_c = min(max(1, int(top_k)), int(logits_c.size(-1)))
    top_k_f = min(max(1, int(top_k)), int(logits_f.size(-1)))
    B, H, _ = logits_c.shape
    top_logits_c, top_idx_c = torch.topk(logits_c.float(), k=top_k_c, dim=-1)
    top_logits_f, top_idx_f = torch.topk(logits_f.float(), k=top_k_f, dim=-1)
    prob_c = torch.softmax(top_logits_c, dim=-1)
    prob_f = torch.softmax(top_logits_f, dim=-1)
    pair_prob = prob_c.unsqueeze(-1) * prob_f.unsqueeze(-2)
    pair_c = top_idx_c.unsqueeze(-1).expand(B, H, top_k_c, top_k_f).reshape(B * H, top_k_c * top_k_f)
    pair_f = top_idx_f.unsqueeze(-2).expand(B, H, top_k_c, top_k_f).reshape(B * H, top_k_c * top_k_f)
    with torch.no_grad():
        decoded = tokenizer.decode(pair_c, pair_f)[..., 0].float()
        decoded = decoded.view(B, H, top_k_c, top_k_f)
        returns = decoded * stds[:, 0].view(B, 1, 1, 1) + means[:, 0].view(B, 1, 1, 1)
    return (pair_prob * returns).sum(dim=(-1, -2))


def _numeric_mape_surrogate(tokenizer, logits_c, logits_f, actual_returns, means, stds, weights, top_k, mape_eps):
    expected_return = _expected_return_from_topk(
        tokenizer=tokenizer, logits_c=logits_c, logits_f=logits_f,
        means=means, stds=stds, top_k=top_k,
    )
    pred_ratio = torch.exp(torch.clamp(expected_return, -20.0, 20.0))
    actual_ratio = torch.exp(torch.clamp(actual_returns.float(), -20.0, 20.0))
    per_step = torch.abs(pred_ratio - actual_ratio) / actual_ratio.abs().clamp_min(float(mape_eps))
    weights = weights.to(device=per_step.device, dtype=per_step.dtype).view(1, -1)
    return (per_step * weights).sum() / weights.sum().clamp_min(1.0) / per_step.size(0)


def _candidate_indices_with_gold(logits, target, top_k):
    top_k = max(1, int(top_k))
    if top_k == 1:
        return target.unsqueeze(-1)
    base_k = min(top_k - 1, int(logits.size(-1)))
    _, top_idx = torch.topk(logits.float(), k=base_k, dim=-1)
    return torch.cat([top_idx, target.unsqueeze(-1)], dim=-1)


def _numeric_soft_pair_ce(
    tokenizer, logits_c, logits_f, target_c, target_f,
    actual_returns, means, stds, weights, top_k, temperature, mape_eps,
):
    cand_c = _candidate_indices_with_gold(logits_c, target_c, top_k)
    cand_f = _candidate_indices_with_gold(logits_f, target_f, top_k)
    B, H, Kc = cand_c.shape
    Kf = int(cand_f.size(-1))
    pair_c = cand_c.unsqueeze(-1).expand(B, H, Kc, Kf).reshape(B * H, Kc * Kf)
    pair_f = cand_f.unsqueeze(-2).expand(B, H, Kc, Kf).reshape(B * H, Kc * Kf)
    with torch.no_grad():
        decoded = tokenizer.decode(pair_c, pair_f)[..., 0].float()
        decoded = decoded.view(B, H, Kc, Kf)
        returns = decoded * stds[:, 0].view(B, 1, 1, 1) + means[:, 0].view(B, 1, 1, 1)
        pred_ratio = torch.exp(torch.clamp(returns, -20.0, 20.0))
        actual_ratio = torch.exp(torch.clamp(actual_returns.float(), -20.0, 20.0)).view(B, H, 1, 1)
        numeric_error = torch.abs(pred_ratio - actual_ratio) / actual_ratio.abs().clamp_min(float(mape_eps))
        temp = max(1e-6, float(temperature))
        soft_target = torch.softmax((-numeric_error / temp).reshape(B, H, -1), dim=-1).view(B, H, Kc, Kf)
    logp_c = F.log_softmax(logits_c.float(), dim=-1)
    logp_f = F.log_softmax(logits_f.float(), dim=-1)
    cand_logp_c = torch.gather(logp_c, dim=-1, index=cand_c)
    cand_logp_f = torch.gather(logp_f, dim=-1, index=cand_f)
    pair_logp = cand_logp_c.unsqueeze(-1) + cand_logp_f.unsqueeze(-2)
    per_step = -(soft_target * pair_logp).sum(dim=(-1, -2))
    weights = weights.to(device=per_step.device, dtype=per_step.dtype).view(1, -1)
    return (per_step * weights).sum() / weights.sum().clamp_min(1.0) / per_step.size(0)


def rollout_training_loss(
    model,
    reference_model,
    tokenizer,
    batch,
    cfg,
    device,
    amp_enabled,
    amp_dtype,
    rollout_ratio,
    effective_horizon=None,
):
    prefix_len = int(cfg.prefix_len)
    horizon = int(effective_horizon) if effective_horizon is not None else int(cfg.horizon)
    data_horizon = int(cfg.horizon)  # actual data has this many future steps
    idx_c_full, idx_f_full = _encode_features(tokenizer, batch["features"])
    target_c = idx_c_full[:, prefix_len:prefix_len + horizon]
    target_f = idx_f_full[:, prefix_len:prefix_len + horizon]
    weights = _step_weights(horizon, cfg.step_weight_gamma, device)
    actual_returns_h = batch["actual_returns"][:, :horizon]

    oracle_guided = bool(getattr(cfg, "oracle_guided", False)) or bool(getattr(cfg, "openai_style", False))
    if oracle_guided:
        input_c, input_f, used_ratio = _oracle_guided_scheduled_inputs(
            model=model,
            tokenizer=tokenizer,
            idx_coarse_full=idx_c_full,
            idx_fine_full=idx_f_full,
            time_features=batch["time"],
            prefix_len=prefix_len,
            horizon=horizon,
            means=batch["means"],
            stds=batch["stds"],
            actual_returns=actual_returns_h,
            top_k=int(getattr(cfg, "oracle_top_k", 8)),
            temperature=float(getattr(cfg, "oracle_temp", 1.5)),
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
    else:
        input_c, input_f, used_ratio = _build_scheduled_inputs(
            model=model,
            idx_coarse_full=idx_c_full,
            idx_fine_full=idx_f_full,
            time_features=batch["time"],
            prefix_len=prefix_len,
            horizon=horizon,
            rollout_ratio=rollout_ratio,
            use_sampling=bool(cfg.use_sampling),
            sampling_temperature=float(cfg.sampling_temperature),
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
    train_len = int(input_c.size(1))
    train_time = {key: value[:, :train_len] for key, value in batch["time"].items()}

    with _autocast_context(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _, hidden = model(
            input_c,
            input_f,
            train_time["minute"],
            train_time["day"],
            train_time["month"],
            train_time["year"],
            return_hidden=True,
        )
    rollout_c, rollout_f = _selected_logits(logits_c, logits_f, prefix_len, horizon)
    rollout_loss = _weighted_token_ce(rollout_c, rollout_f, target_c, target_f, weights)
    total = rollout_loss

    numeric_loss = rollout_loss.detach() * 0.0
    if float(getattr(cfg, "numeric_mape_weight", 0.0)) > 0.0:
        numeric_loss = _numeric_mape_surrogate(
            tokenizer=tokenizer,
            logits_c=rollout_c,
            logits_f=rollout_f,
            actual_returns=actual_returns_h,
            means=batch["means"],
            stds=batch["stds"],
            weights=weights,
            top_k=int(getattr(cfg, "numeric_top_k", 16)),
            mape_eps=float(cfg.mape_eps),
        )
        total = total + float(cfg.numeric_mape_weight) * numeric_loss

    numeric_soft_ce = rollout_loss.detach() * 0.0
    if float(getattr(cfg, "numeric_soft_ce_weight", 0.0)) > 0.0:
        numeric_soft_ce = _numeric_soft_pair_ce(
            tokenizer=tokenizer,
            logits_c=rollout_c,
            logits_f=rollout_f,
            target_c=target_c,
            target_f=target_f,
            actual_returns=actual_returns_h,
            means=batch["means"],
            stds=batch["stds"],
            weights=weights,
            top_k=int(getattr(cfg, "numeric_soft_ce_top_k", 8)),
            temperature=float(getattr(cfg, "numeric_soft_ce_temp", 0.005)),
            mape_eps=float(cfg.mape_eps),
        )
        total = total + float(cfg.numeric_soft_ce_weight) * numeric_soft_ce

    anchor_loss = rollout_loss.detach() * 0.0
    if float(cfg.anchor_weight) > 0.0:
        anchor_c = idx_c_full[:, :prefix_len + horizon - 1]
        anchor_f = idx_f_full[:, :prefix_len + horizon - 1]
        anchor_time = {key: value[:, :anchor_c.size(1)] for key, value in batch["time"].items()}
        with _autocast_context(device, amp_enabled, amp_dtype):
            anchor_logits_c, anchor_logits_f, _ = model(
                anchor_c,
                anchor_f,
                anchor_time["minute"],
                anchor_time["day"],
                anchor_time["month"],
                anchor_time["year"],
            )
        anchor_sel_c, anchor_sel_f = _selected_logits(anchor_logits_c, anchor_logits_f, prefix_len, horizon)
        anchor_loss = _weighted_token_ce(anchor_sel_c, anchor_sel_f, target_c, target_f, weights)
        total = total + float(cfg.anchor_weight) * anchor_loss

    kl_loss = rollout_loss.detach() * 0.0
    if reference_model is not None and float(cfg.kl_weight) > 0.0:
        with torch.no_grad():
            with _autocast_context(device, amp_enabled, amp_dtype):
                ref_logits_c, ref_logits_f, _ = reference_model(
                    input_c,
                    input_f,
                    train_time["minute"],
                    train_time["day"],
                    train_time["month"],
                    train_time["year"],
                )
            ref_sel_c, ref_sel_f = _selected_logits(ref_logits_c, ref_logits_f, prefix_len, horizon)
        kl_loss = _kl_to_reference(rollout_c, rollout_f, ref_sel_c, ref_sel_f, weights)
        total = total + float(cfg.kl_weight) * kl_loss

    return total, {
        "rollout_loss": float(rollout_loss.detach().item()),
        "numeric_mape": float(numeric_loss.detach().item()),
        "numeric_soft_ce": float(numeric_soft_ce.detach().item()),
        "anchor_loss": float(anchor_loss.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
        "used_pred_ratio": float(used_ratio),
    }


def compute_rollout_metrics(pred_returns, actual_returns, mape_eps=1e-4):
    pred = np.asarray(pred_returns, dtype=np.float64)
    actual = np.asarray(actual_returns, dtype=np.float64)
    if pred.size == 0:
        return {"num_samples": 0}
    if pred.shape != actual.shape:
        raise ValueError(f"Prediction/actual shape mismatch: {pred.shape} vs {actual.shape}")

    def one(p, a):
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        finite = np.isfinite(p) & np.isfinite(a)
        if finite.sum() == 0:
            return {
                "num_samples": 0,
                "mape": math.nan,
                "return_mape": math.nan,
                "da": math.nan,
                "mae": math.nan,
                "rmse": math.nan,
                "pred_up_ratio": math.nan,
                "actual_up_ratio": math.nan,
            }
        p = p[finite]
        a = a[finite]
        pred_ratio = np.exp(np.clip(p, -50.0, 50.0))
        actual_ratio = np.exp(np.clip(a, -50.0, 50.0))
        ratio_denom = np.maximum(np.abs(actual_ratio), float(mape_eps))
        return_denom = np.maximum(np.abs(a), float(mape_eps))
        err = p - a
        pred_sign = np.where(p >= 0.0, 1, -1)
        actual_sign = np.where(a >= 0.0, 1, -1)
        return {
            "num_samples": int(len(a)),
            "mape": float(np.mean(np.abs((pred_ratio - actual_ratio) / ratio_denom)) * 100.0),
            "return_mape": float(np.mean(np.abs((p - a) / return_denom)) * 100.0),
            "da": float(np.mean(pred_sign == actual_sign) * 100.0),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "pred_up_ratio": float(np.mean(pred_sign > 0) * 100.0),
            "actual_up_ratio": float(np.mean(actual_sign > 0) * 100.0),
        }

    overall = one(pred, actual)
    pred_cum = np.cumsum(pred, axis=1)
    actual_cum = np.cumsum(actual, axis=1)
    path_metrics = one(pred_cum, actual_cum)
    overall["path_mape"] = path_metrics["mape"]
    overall["path_return_mape"] = path_metrics["return_mape"]
    overall["path_mae"] = path_metrics["mae"]
    overall["path_rmse"] = path_metrics["rmse"]
    per_step = []
    for step in range(pred.shape[1]):
        row = one(pred[:, step], actual[:, step])
        path_row = one(pred_cum[:, step], actual_cum[:, step])
        row["path_mape"] = path_row["mape"]
        row["path_return_mape"] = path_row["return_mape"]
        row["path_mae"] = path_row["mae"]
        row["path_rmse"] = path_row["rmse"]
        row["step"] = int(step + 1)
        per_step.append(row)
    overall["per_step"] = per_step
    overall["horizon"] = int(pred.shape[1])
    overall["num_sequences"] = int(pred.shape[0])
    return overall


@torch.inference_mode()
def predict_autoregressive_returns(
    model,
    tokenizer,
    loader,
    cfg,
    device,
    amp_enabled,
    amp_dtype,
    sample_temp_start=0.0,
    sample_temp_end=0.0,
    sample_temp_steps=0,
):
    model.eval()
    tokenizer.eval()
    prefix_len = int(cfg.prefix_len)
    horizon = int(cfg.horizon)
    pred_parts = []
    actual_parts = []

    for raw_batch in tqdm(loader, desc="Eval rollout AR", leave=False):
        batch = _move_batch(raw_batch, device)
        idx_c_full, idx_f_full = _encode_features(tokenizer, batch["features"])
        context_c = idx_c_full[:, :prefix_len].clone()
        context_f = idx_f_full[:, :prefix_len].clone()
        step_returns = []

        for step in range(horizon):
            cur_len = int(context_c.size(1))
            cur_time = {key: value[:, :cur_len] for key, value in batch["time"].items()}
            with _autocast_context(device, amp_enabled, amp_dtype):
                logits_c, logits_f, _ = model(
                    context_c,
                    context_f,
                    cur_time["minute"],
                    cur_time["day"],
                    cur_time["month"],
                    cur_time["year"],
                    last_only=True,
                )
            anneal_steps = int(sample_temp_steps)
            if anneal_steps > 0 and step < anneal_steps:
                frac = float(step) / max(1, anneal_steps - 1) if anneal_steps > 1 else 0.0
                temp = float(sample_temp_start) + frac * (float(sample_temp_end) - float(sample_temp_start))
                temp = max(1e-4, temp)
                probs_c = torch.softmax(logits_c[:, -1, :].float() / temp, dim=-1)
                probs_f = torch.softmax(logits_f[:, -1, :].float() / temp, dim=-1)
                pred_c = torch.multinomial(probs_c, num_samples=1).squeeze(1)
                pred_f = torch.multinomial(probs_f, num_samples=1).squeeze(1)
            else:
                pred_c = logits_c[:, -1, :].float().argmax(dim=-1)
                pred_f = logits_f[:, -1, :].float().argmax(dim=-1)
            decoded = tokenizer.decode(pred_c.unsqueeze(1), pred_f.unsqueeze(1))
            pred_norm = decoded[:, 0, 0].float()
            pred_return = pred_norm * batch["stds"][:, 0] + batch["means"][:, 0]
            step_returns.append(pred_return.detach().cpu())
            if step < horizon - 1:
                context_c = torch.cat([context_c, pred_c.unsqueeze(1)], dim=1)
                context_f = torch.cat([context_f, pred_f.unsqueeze(1)], dim=1)

        pred_parts.append(torch.stack(step_returns, dim=1))
        actual_parts.append(batch["actual_returns"].detach().cpu())

    if not pred_parts:
        return np.empty((0, horizon), dtype=np.float32), np.empty((0, horizon), dtype=np.float32)
    return torch.cat(pred_parts, dim=0).numpy(), torch.cat(actual_parts, dim=0).numpy()


def evaluate_model(model, tokenizer, val_loader, cfg, device, amp_enabled, amp_dtype,
                   sample_temp_start=0.0, sample_temp_end=0.0, sample_temp_steps=0):
    pred, actual = predict_autoregressive_returns(
        model=model,
        tokenizer=tokenizer,
        loader=val_loader,
        cfg=cfg,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        sample_temp_start=sample_temp_start,
        sample_temp_end=sample_temp_end,
        sample_temp_steps=sample_temp_steps,
    )
    return compute_rollout_metrics(pred, actual, mape_eps=float(cfg.mape_eps))


def _configure_trainable(model, cfg):
    scope = str(getattr(cfg, "trainable_scope", "all")).strip().lower()
    if bool(cfg.freeze_backbone):
        for param in model.parameters():
            param.requires_grad = False
    elif scope == "heads":
        for param in model.parameters():
            param.requires_grad = False
        trainable_prefixes = (
            "norm.",
            "head_coarse.",
            "coarse_to_fine.",
            "fine_gate.",
            "fine_norm.",
            "head_fine.",
        )
        for name, param in model.named_parameters():
            if name.startswith(trainable_prefixes):
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters for rollout post-training.")
    return [{"params": params, "lr": float(cfg.learning_rate)}]


def _build_optimizer(param_groups, cfg, device):
    base = {"weight_decay": float(cfg.weight_decay)}
    candidates = []
    if device.type == "cuda":
        candidates.extend([{**base, "fused": True}, {**base, "foreach": True}])
    candidates.append(base)
    last_exc = None
    for kwargs in candidates:
        try:
            return torch.optim.AdamW(param_groups, **kwargs), kwargs
        except (TypeError, RuntimeError, ValueError) as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    return torch.optim.AdamW(param_groups, **base), base


def _save_checkpoint(path, model, tokenizer, cfg, metrics, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    payload = {
        "stage": "rollout_scheduled_self_ce",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_state_dict": raw_model.state_dict(),
        "tokenizer_state_dict": tokenizer.state_dict(),
        "model_config": getattr(raw_model, "model_config", None),
        "post_train_rollout_config": _cfg_to_dict(cfg),
        "metrics": metrics,
        "history": history,
    }
    torch.save(payload, path)
    return path


def _write_history(path, cfg, history, best_metrics, base_metrics=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": _cfg_to_dict(cfg),
        "train_cache": rollout_cache_path("train", cfg),
        "val_cache": rollout_cache_path("val", cfg),
        "base_metrics": base_metrics,
        "best_metrics": best_metrics,
        "history": history,
        "data_policy": {
            "split": "train/val only; demo date range excluded from cache construction",
            "normalization": "mean/std computed from the 1023-token observed prefix only",
            "evaluation": "pure autoregressive 10-step rollout; no future ground-truth token is fed back",
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def train(cfg):
    set_global_seed(int(cfg.random_seed), deterministic=bool(cfg.deterministic))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.use_tf32)
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    amp_dtype = _amp_dtype(cfg.amp_dtype)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda" and amp_dtype is not None)

    train_dataset = RolloutWindowDataset(
        "train",
        cfg=cfg,
        max_samples=int(cfg.max_train_samples),
        seed=int(cfg.random_seed),
    )
    val_dataset = RolloutWindowDataset(
        "val",
        cfg=cfg,
        max_samples=int(cfg.max_val_samples),
        seed=int(cfg.random_seed) + 17,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"Rollout dataset is empty: train={len(train_dataset)}, val={len(val_dataset)}")

    loader_kwargs = {
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
        "collate_fn": rollout_collate,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(cfg.eval_batch_size)),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    model, tokenizer = load_model(
        device=device,
        checkpoint_path=cfg.checkpoint_path,
        strict_checkpoint_compat=False,
    )
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    if bool(cfg.use_gradient_checkpointing):
        model.enable_gradient_checkpointing(True)

    reference_model = None
    needs_ref = (float(cfg.kl_weight) > 0.0)
    if needs_ref:
        reference_model = copy.deepcopy(model).to(device)
        reference_model.eval()
        reference_model.requires_grad_(False)

    base_metrics = None
    if bool(cfg.eval_only):
        metrics = evaluate_model(model, tokenizer, val_loader, cfg, device, amp_enabled, amp_dtype)
        print(json.dumps({"eval_only_metrics": metrics}, indent=2, ensure_ascii=False))
        return {"metrics": metrics}

    param_groups = _configure_trainable(model, cfg)
    optimizer, optimizer_kwargs = _build_optimizer(param_groups, cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    summary = trainable_parameter_summary(model)
    print(f"Device: {device}, amp={amp_enabled}, amp_dtype={cfg.amp_dtype}")
    print(f"Train windows={len(train_dataset)}, val windows={len(val_dataset)}")
    print(f"Trainable parameters: {summary}")
    print(f"Optimizer: AdamW {optimizer_kwargs}")

    total_steps = max(1, math.ceil(len(train_loader) / int(cfg.accumulation_steps)) * int(cfg.epochs))
    warmup = max(1, total_steps // 10)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup),
        eta_min=float(cfg.learning_rate) * 0.05,
    )

    history = []
    best_score = -float("inf")
    best_metrics = None
    best_path = os.path.join(cfg.output_dir, cfg.save_name)
    updates = 0

    # ── Experiment N: Curriculum Horizons ──
    curriculum_horizons_str = str(getattr(cfg, "curriculum_horizons", "")).strip()
    curriculum_updates_str = str(getattr(cfg, "curriculum_updates", "")).strip()
    curriculum_horizons = None
    curriculum_updates_list = None
    if curriculum_horizons_str:
        curriculum_horizons = [int(x.strip()) for x in curriculum_horizons_str.split(",") if x.strip()]
        if curriculum_updates_str:
            curriculum_updates_list = [int(x.strip()) for x in curriculum_updates_str.split(",") if x.strip()]
        else:
            per_stage = max(1, int(cfg.max_train_updates) // len(curriculum_horizons)) if int(cfg.max_train_updates) > 0 else 12
            curriculum_updates_list = [per_stage] * len(curriculum_horizons)
        print(f"Curriculum: horizons={curriculum_horizons}, updates_per_stage={curriculum_updates_list}")


    if curriculum_horizons is not None and len(curriculum_horizons) > 0:
        # Curriculum training: loop over stages
        effective_horizon = curriculum_horizons[0]
        for stage_idx, (stage_horizon, stage_updates) in enumerate(
            zip(curriculum_horizons, curriculum_updates_list)
        ):
            effective_horizon = int(stage_horizon)
            stage_max_updates = int(stage_updates)
            print(f"\n=== Curriculum stage {stage_idx + 1}/{len(curriculum_horizons)}: horizon={effective_horizon}, updates={stage_max_updates} ===")
            stage_updates_done = 0

            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_totals = {
                "loss": 0.0, "rollout_loss": 0.0, "numeric_mape": 0.0,
                "numeric_soft_ce": 0.0, "anchor_loss": 0.0, "kl_loss": 0.0,
                "path_aware_loss": 0.0, "beam_distill_loss": 0.0,
                "contrastive_loss": 0.0, "grpo_loss": 0.0,
                "iterative_dpo_loss": 0.0, "reinforcepp_policy_loss": 0.0,
                "expert_iter_sft_loss": 0.0, "orpo_nll": 0.0,
                "used_pred_ratio": 0.0,
            }
            batches = 0
            pbar = tqdm(train_loader, desc=f"Curriculum h={effective_horizon}")
            for batch_idx, raw_batch in enumerate(pbar, start=1):
                batch = _move_batch(raw_batch, device)
                rollout_ratio = 1.0

                with _autocast_context(device, amp_enabled, amp_dtype):
                    loss, stats = rollout_training_loss(
                        model=model, reference_model=reference_model,
                        tokenizer=tokenizer, batch=batch, cfg=cfg,
                        device=device, amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype, rollout_ratio=rollout_ratio,
                        effective_horizon=effective_horizon,
                    )
                    scaled_loss = loss / int(cfg.accumulation_steps)

                scaler.scale(scaled_loss).backward()
                if batch_idx % int(cfg.accumulation_steps) == 0 or batch_idx == len(train_loader):
                    if float(cfg.grad_clip) > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for group in param_groups for p in group["params"]],
                            max_norm=float(cfg.grad_clip),
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if updates < warmup:
                        warmup_sched.step()
                    else:
                        cosine_sched.step()
                    updates += 1
                    stage_updates_done += 1

                value = float(loss.detach().item())
                epoch_totals["loss"] += value
                for key in ("rollout_loss", "numeric_mape", "numeric_soft_ce", "anchor_loss", "kl_loss",
                            "used_pred_ratio"):
                    epoch_totals[key] += float(stats.get(key, 0.0))
                batches += 1

                if batch_idx % int(cfg.progress_interval) == 0:
                    pbar.set_postfix({
                        "loss": f"{value:.4f}",
                        "self": f"{stats.get('used_pred_ratio', 0):.2f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    })

                if stage_updates_done >= stage_max_updates:
                    break

            train_row = {key: value / max(1, batches) for key, value in epoch_totals.items()}
            val_metrics = evaluate_model(model, tokenizer, val_loader, cfg, device, amp_enabled, amp_dtype)
            score = -float(val_metrics.get("path_mape", val_metrics.get("mape", float("inf"))))
            row = {
                "stage": int(stage_idx + 1),
                "horizon": effective_horizon,
                "updates": int(updates),
                "train": train_row,
                "val": val_metrics,
                "memory": _cuda_peak_memory_stats(device),
            }
            history.append(row)
            print(json.dumps(row, indent=2, ensure_ascii=False))

            if score > best_score:
                best_score = score
                best_metrics = val_metrics
                _save_checkpoint(best_path, model, tokenizer, cfg, val_metrics, history)

            if updates >= int(cfg.max_train_updates):
                break
    else:
        # Standard (non-curriculum) training loop
        for epoch in range(int(cfg.epochs)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_totals = {
                "loss": 0.0, "rollout_loss": 0.0, "numeric_mape": 0.0,
                "numeric_soft_ce": 0.0, "anchor_loss": 0.0, "kl_loss": 0.0,
                "used_pred_ratio": 0.0,
            }
            batches = 0
            pbar = tqdm(train_loader, desc=f"Rollout train epoch {epoch + 1}/{cfg.epochs}")
            for batch_idx, raw_batch in enumerate(pbar, start=1):
                progress = 0.0
                if int(cfg.epochs) > 0:
                    progress = (epoch + (batch_idx - 1) / max(1, len(train_loader))) / max(1, int(cfg.epochs))
                rollout_ratio = float(cfg.rollout_ratio_start) + (
                    float(cfg.rollout_ratio_end) - float(cfg.rollout_ratio_start)
                ) * progress
                batch = _move_batch(raw_batch, device)
                with _autocast_context(device, amp_enabled, amp_dtype):
                    loss, stats = rollout_training_loss(
                        model=model,
                        reference_model=reference_model,
                        tokenizer=tokenizer,
                        batch=batch,
                        cfg=cfg,
                        device=device,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        rollout_ratio=rollout_ratio,
                    )
                    scaled_loss = loss / int(cfg.accumulation_steps)

                scaler.scale(scaled_loss).backward()
                if batch_idx % int(cfg.accumulation_steps) == 0 or batch_idx == len(train_loader):
                    if float(cfg.grad_clip) > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for group in param_groups for p in group["params"]],
                            max_norm=float(cfg.grad_clip),
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if updates < warmup:
                        warmup_sched.step()
                    else:
                        cosine_sched.step()
                    updates += 1

                value = float(loss.detach().item())
                epoch_totals["loss"] += value
                for key in ("rollout_loss", "numeric_mape", "numeric_soft_ce", "anchor_loss", "kl_loss",
                            "used_pred_ratio"):
                    epoch_totals[key] += float(stats.get(key, 0.0))
                batches += 1
                if batch_idx % int(cfg.progress_interval) == 0:
                    pbar.set_postfix({
                        "loss": f"{value:.4f}",
                        "self": f"{stats['used_pred_ratio']:.2f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    })

                if int(cfg.max_train_updates) > 0 and updates >= int(cfg.max_train_updates):
                    break

            train_row = {key: value / max(1, batches) for key, value in epoch_totals.items()}
            val_metrics = evaluate_model(model, tokenizer, val_loader, cfg, device, amp_enabled, amp_dtype)
            score = -float(val_metrics.get("path_mape", val_metrics.get("mape", float("inf"))))
            row = {
                "epoch": int(epoch + 1),
                "updates": int(updates),
                "rollout_ratio": float(rollout_ratio),
                "train": train_row,
                "val": val_metrics,
                "memory": _cuda_peak_memory_stats(device),
            }
            history.append(row)
            print(json.dumps(row, indent=2, ensure_ascii=False))

            if bool(cfg.save_epoch_checkpoints):
                epoch_path = os.path.join(
                    cfg.output_dir,
                    f"{os.path.splitext(cfg.save_name)[0]}-epoch{epoch + 1}.pt",
                )
                _save_checkpoint(epoch_path, model, tokenizer, cfg, val_metrics, history)

            if score > best_score:
                best_score = score
                best_metrics = val_metrics
                _save_checkpoint(best_path, model, tokenizer, cfg, val_metrics, history)

            if int(cfg.max_train_updates) > 0 and updates >= int(cfg.max_train_updates):
                break

    history_path = os.path.join(cfg.output_dir, "rollout_scheduled_history.json")
    _write_history(history_path, cfg, history, best_metrics, base_metrics=base_metrics)
    print(f"Best rollout checkpoint: {best_path}")
    print(f"History: {history_path}")
    return {"best_path": best_path, "history_path": history_path, "best_metrics": best_metrics, "history": history}


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = _namespace_from_args(args)
    os.makedirs(cfg.output_dir, exist_ok=True)
    return train(cfg)


if __name__ == "__main__":
    main()
