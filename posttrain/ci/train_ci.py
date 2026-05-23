# -*- coding: utf-8 -*-
"""Idea 2 — Confidence-Interval-aware post-training for Kronos-R.

Trains the model to produce well-calibrated, sharp prediction distributions
that yield narrow confidence intervals with correct coverage.

Core idea
---------
The model outputs a probability distribution over next-token pairs.
Each pair decodes to a scalar log-return.  We optimise two properties:

  Sharpness   – probability mass should concentrate near the true value
                → narrow prediction intervals.
  Coverage    – the interval constructed from the distribution should
                contain the true value with the nominal probability.

These are realised through two complementary loss terms:

  Concentration loss  (differentiable)
      L_conc = E_{p}[ |r - y| ]  — expected absolute error under the
      model's predicted distribution.  Directly encourages sharpness.

  Interval-score surrogate  (differentiable via soft-quantile approximation)
      A smooth approximation of the Gneiting-Raftery interval score that
      penalises wide intervals and missed coverage.

Training follows the scheduled-self-rollout pattern from the rollout
post-training: at each step the model conditions on its own previous
predictions (with a curriculum that increases the self-feedback ratio).
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

from config import PostTrainCIConfig, PostTrainRolloutConfig, TrainingConfig
from evaluate_predictions import load_model
from model.lora import trainable_parameter_summary
from posttrain.ci.data import (
    RolloutWindowDataset,
    resolve_project_path,
    rollout_cache_path,
    rollout_collate,
)
from posttrain.ci.eval_ci import compute_ci_metrics
from reproducibility import set_global_seed


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════

def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _amp_dtype(name):
    name = str(name).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
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


# ═══════════════════════════════════════════════════════════════════
# token-level helpers
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _encode_features(tokenizer, features):
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
# Scheduled self-rollout input construction
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _build_scheduled_inputs(
    model, idx_coarse_full, idx_fine_full, time_features,
    prefix_len, horizon, rollout_ratio, device,
    amp_enabled, amp_dtype,
):
    """Identical to rollout._build_scheduled_inputs."""
    if int(horizon) <= 1:
        return idx_coarse_full[:, :prefix_len], idx_fine_full[:, :prefix_len], 0.0

    was_training = model.training
    model.eval()

    context_c = idx_coarse_full[:, :prefix_len].clone()
    context_f = idx_fine_full[:, :prefix_len].clone()
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

        use_self = (
            float(rollout_ratio) >= 1.0
            or torch.rand(1, device=device).item() < float(rollout_ratio)
        )

        if use_self:
            pred_c = logits_c[:, -1, :].float().argmax(dim=-1)
            pred_f = logits_f[:, -1, :].float().argmax(dim=-1)
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


# ═══════════════════════════════════════════════════════════════════
# Anchor losses (standard token CE + KL to reference)
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# CI-specific loss: concentration + interval-score surrogate
# ═══════════════════════════════════════════════════════════════════

def _topk_pair_distribution(logits_c, logits_f, top_k):
    """Extract top-K token-pair probabilities and decoded returns.

    Returns
    -------
    pair_probs : [B, H, K, K]  — renormalised joint probabilities.
    returns    : [B, H, K, K]  — decoded log-returns (detached).
    """
    B, H, _ = logits_c.shape
    K = min(int(top_k), int(logits_c.size(-1)), int(logits_f.size(-1)))

    probs_c = F.softmax(logits_c.float(), dim=-1)
    probs_f = F.softmax(logits_f.float(), dim=-1)

    top_prob_c, top_idx_c = torch.topk(probs_c, k=K, dim=-1)
    top_prob_f, top_idx_f = torch.topk(probs_f, k=K, dim=-1)

    # renormalise
    top_prob_c = top_prob_c / top_prob_c.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    top_prob_f = top_prob_f / top_prob_f.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    pair_probs = top_prob_c.unsqueeze(-1) * top_prob_f.unsqueeze(-2)  # [B, H, K, K]

    pair_c = top_idx_c.unsqueeze(-1).expand(B, H, K, K).reshape(B * H, K * K)
    pair_f = top_idx_f.unsqueeze(-2).expand(B, H, K, K).reshape(B * H, K * K)

    # decode is non-differentiable → detach returns
    with torch.no_grad():
        decoded = tokenizer.decode(pair_c, pair_f)[..., 0].float()
        returns = decoded.view(B, H, K, K)
        # returns are already normalised log_ret; denormalisation happens
        # per-sample with means/stds if needed, but concentration loss
        # works in normalised space for stability.

    return pair_probs, returns, (top_idx_c, top_idx_f)


# Module-level tokenizer reference set in train().
_tokenizer = None


def tokenizer():
    return _tokenizer


def concentration_loss(logits_c, logits_f, actual_returns, top_k):
    """Expected absolute error under the model's predicted distribution.

    L_conc = E_{p(c,f)}[ |r(c,f) - y| ]

    Minimising this sharpens the distribution around the true value,
    which directly translates to narrower prediction intervals.
    Fully differentiable — gradients flow through the token probabilities.
    """
    pair_probs, returns, _ = _topk_pair_distribution(logits_c, logits_f, top_k)
    B, H, K, _ = pair_probs.shape
    y = actual_returns.float().view(B, H, 1, 1)
    abs_error = (returns - y).abs()
    expected_error = (pair_probs * abs_error).sum(dim=(-1, -2))  # [B, H]
    return expected_error.mean()


def interval_score_from_distribution(logits_c, logits_f, actual_returns,
                                      confidence_level, top_k):
    """Compute the interval score from the predicted return distribution.

    Constructs a prediction interval [L, U] from the α/2 and 1-α/2
    weighted quantiles of the top-K return distribution, then evaluates
    the Gneiting-Raftery interval score.

    The quantile computation is non-differentiable (involves sorting),
    so this is primarily a *monitoring* metric.  Use concentration_loss
    for the actual training gradient.
    """
    pair_probs, returns, _ = _topk_pair_distribution(logits_c, logits_f, top_k)
    B, H, K, _ = pair_probs.shape
    alpha = 1.0 - float(confidence_level)
    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0

    # flatten pair dimension
    ret_flat = returns.view(B, H, K * K)
    prob_flat = pair_probs.view(B, H, K * K)

    # sort by return value
    sort_idx = ret_flat.argsort(dim=-1)
    sorted_ret = ret_flat.gather(-1, sort_idx)
    sorted_prob = prob_flat.gather(-1, sort_idx)
    cum_prob = sorted_prob.cumsum(dim=-1)
    cum_prob = cum_prob / cum_prob[..., -1:].clamp_min(1e-8)

    # weighted quantile by interpolation
    def _weighted_quantile(values, cumprobs, q):
        """Find the value at quantile q in each row."""
        # cumprobs: [B, H, N]  sorted cumulative probabilities
        # values: [B, H, N]    sorted return values
        # Find first index where cumprob >= q
        N = cumprobs.shape[-1]
        mask = cumprobs >= q  # [B, H, N]
        # argmax on boolean gives first True
        idx = mask.float().argmax(dim=-1)  # [B, H]
        # clamp to valid range for gather
        idx_clamped = idx.clamp(0, N - 1)
        # gather values at those indices
        row_idx = torch.arange(B, device=values.device).view(B, 1).expand(B, H)
        col_idx = torch.arange(H, device=values.device).view(1, H).expand(B, H)
        return values[row_idx, col_idx, idx_clamped]

    L = _weighted_quantile(sorted_ret, cum_prob, low_q)
    U = _weighted_quantile(sorted_ret, cum_prob, high_q)
    y = actual_returns.float()

    # interval score (proper scoring rule)
    width = U - L
    penalty_low = torch.clamp(L - y, min=0.0)
    penalty_high = torch.clamp(y - U, min=0.0)
    iscore = width + (2.0 / max(float(alpha), 1e-8)) * (penalty_low + penalty_high)

    # also check coverage for monitoring
    covered = ((y >= L) & (y <= U)).float()

    return iscore.mean(), {
        "avg_width": width.mean().detach().item(),
        "coverage": covered.mean().detach().item(),
        "avg_iscore": iscore.mean().detach().item(),
    }


def ci_training_loss(
    model, reference_model, batch, cfg, device,
    amp_enabled, amp_dtype, rollout_ratio,
):
    """Composite CI training loss.

    Returns (total_loss, stats_dict).
    """
    global _tokenizer

    prefix_len = int(cfg.prefix_len)
    horizon = int(cfg.horizon)
    idx_c_full, idx_f_full = _encode_features(_tokenizer, batch["features"])
    target_c = idx_c_full[:, prefix_len:prefix_len + horizon]
    target_f = idx_f_full[:, prefix_len:prefix_len + horizon]
    weights = _step_weights(horizon, getattr(cfg, "step_weight_gamma", 0.5), device)
    actual_returns_h = batch["actual_returns"][:, :horizon]

    # build scheduled inputs
    input_c, input_f, used_ratio = _build_scheduled_inputs(
        model=model,
        idx_coarse_full=idx_c_full,
        idx_fine_full=idx_f_full,
        time_features=batch["time"],
        prefix_len=prefix_len,
        horizon=horizon,
        rollout_ratio=rollout_ratio,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    train_len = int(input_c.size(1))
    train_time = {
        key: value[:, :train_len]
        for key, value in batch["time"].items()
    }

    with _autocast_context(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _, hidden = model(
            input_c, input_f,
            train_time["minute"], train_time["day"],
            train_time["month"], train_time["year"],
            return_hidden=True,
        )

    rollout_c, rollout_f = _selected_logits(logits_c, logits_f, prefix_len, horizon)

    # ── 1. Anchor CE loss ──
    anchor_loss = _weighted_token_ce(rollout_c, rollout_f, target_c, target_f, weights)
    total = anchor_loss

    # ── 2. Concentration loss ──
    conc_weight = float(getattr(cfg, "concentration_weight", 1.0))
    conc_loss = anchor_loss.new_zeros(())
    conc_stats = {}
    if conc_weight > 0.0:
        # Denormalise actual returns for concentration loss
        means = batch["means"][:, 0].view(-1, 1)
        stds = batch["stds"][:, 0].view(-1, 1)
        actual_denorm = actual_returns_h * stds + means
        conc_loss = concentration_loss(
            logits_c=rollout_c,
            logits_f=rollout_f,
            actual_returns=actual_denorm,
            top_k=int(getattr(cfg, "ci_top_k", 32)),
        )
        total = total + conc_weight * conc_loss

    # ── 3. Interval score (monitoring + optional gradient) ──
    iscore_weight = float(getattr(cfg, "interval_score_weight", 0.0))
    iscore_loss = anchor_loss.new_zeros(())
    iscore_stats = {"avg_width": 0.0, "coverage": 0.0, "avg_iscore": 0.0}
    if iscore_weight > 0.0:
        means = batch["means"][:, 0].view(-1, 1)
        stds = batch["stds"][:, 0].view(-1, 1)
        actual_denorm = actual_returns_h * stds + means
        iscore_loss, iscore_stats = interval_score_from_distribution(
            logits_c=rollout_c,
            logits_f=rollout_f,
            actual_returns=actual_denorm,
            confidence_level=float(getattr(cfg, "ci_confidence_level", 0.80)),
            top_k=int(getattr(cfg, "ci_top_k", 32)),
        )
        # The interval score quantile computation is non-differentiable.
        # We add it as a scalar reward-like signal: if the interval score
        # is small, we slightly reinforce the current token distribution;
        # if it is large, we penalise.  This is done by scaling the CE loss.
        iscore_detached = iscore_loss.detach()
        total = total + iscore_weight * iscore_detached * anchor_loss

    # ── 4. KL to reference ──
    kl_weight = float(getattr(cfg, "kl_weight", 0.02))
    kl_loss = anchor_loss.new_zeros(())
    if reference_model is not None and kl_weight > 0.0:
        with torch.no_grad():
            with _autocast_context(device, amp_enabled, amp_dtype):
                ref_logits_c, ref_logits_f, _ = reference_model(
                    input_c, input_f,
                    train_time["minute"], train_time["day"],
                    train_time["month"], train_time["year"],
                )
            ref_sel_c, ref_sel_f = _selected_logits(
                ref_logits_c, ref_logits_f, prefix_len, horizon,
            )
        kl_loss = _kl_to_reference(rollout_c, rollout_f, ref_sel_c, ref_sel_f, weights)
        total = total + kl_weight * kl_loss

    return total, {
        "anchor_loss": float(anchor_loss.detach().item()),
        "concentration_loss": float(conc_loss.detach().item()),
        "iscore_loss": float(iscore_loss.detach().item()) if isinstance(iscore_loss, torch.Tensor) else float(iscore_loss),
        "kl_loss": float(kl_loss.detach().item()),
        "used_pred_ratio": float(used_ratio),
        "avg_width": float(iscore_stats.get("avg_width", 0.0)),
        "coverage": float(iscore_stats.get("coverage", 0.0)),
        "avg_iscore": float(iscore_stats.get("avg_iscore", 0.0)),
    }


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_ci_from_distribution(
    model, loader, cfg, device, amp_enabled, amp_dtype,
    confidence_level=0.80, top_k=32,
):
    """Construct CIs from the model's predicted distribution quantiles.

    Unlike ci_sampling.py (temperature sampling), this uses the model's
    softmax probabilities directly — faster and deterministic.
    """
    global _tokenizer

    model.eval()
    _tokenizer.eval()
    prefix_len = int(cfg.prefix_len)
    horizon = int(cfg.horizon)

    pred_lower = []
    pred_upper = []
    actual_all = []

    alpha = 1.0 - float(confidence_level)
    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0

    for raw_batch in tqdm(loader, desc="Eval CI", leave=False):
        batch = {
            "features": raw_batch["features"].to(device=device, dtype=torch.float32, non_blocking=True),
            "time": {
                key: value.to(device=device, dtype=torch.long, non_blocking=True)
                for key, value in raw_batch["time"].items()
            },
            "means": raw_batch["means"].to(device=device, dtype=torch.float32, non_blocking=True),
            "stds": raw_batch["stds"].to(device=device, dtype=torch.float32, non_blocking=True),
        }

        idx_c_full, idx_f_full = _encode_features(_tokenizer, batch["features"])

        context_c = idx_c_full[:, :prefix_len].clone()
        context_f = idx_f_full[:, :prefix_len].clone()

        step_lower = []
        step_upper = []

        for step in range(horizon):
            cur_len = int(context_c.size(1))
            cur_time = {
                key: value[:, :cur_len]
                for key, value in batch["time"].items()
            }

            with _autocast_context(device, amp_enabled, amp_dtype):
                logits_c, logits_f, _ = model(
                    context_c, context_f,
                    cur_time["minute"], cur_time["day"],
                    cur_time["month"], cur_time["year"],
                    last_only=True,
                )

            last_logits_c = logits_c[:, -1, :].float().unsqueeze(1)  # [B, 1, V]
            last_logits_f = logits_f[:, -1, :].float().unsqueeze(1)

            pair_probs, returns, _ = _topk_pair_distribution(
                last_logits_c, last_logits_f, int(top_k),
            )
            # squeeze H=1 dimension
            pair_probs = pair_probs.squeeze(1)  # [B, K, K]
            returns = returns.squeeze(1)

            B, K, _ = pair_probs.shape
            ret_flat = returns.view(B, K * K)
            prob_flat = pair_probs.view(B, K * K)

            sort_idx = ret_flat.argsort(dim=-1)
            sorted_ret = ret_flat.gather(-1, sort_idx)
            sorted_prob = prob_flat.gather(-1, sort_idx)
            cum_prob = sorted_prob.cumsum(dim=-1)
            cum_prob = cum_prob / cum_prob[..., -1:].clamp_min(1e-8)

            # weighted quantiles
            N = K * K
            mask_low = cum_prob >= low_q
            idx_low = mask_low.float().argmax(dim=-1).clamp(0, N - 1)
            mask_high = cum_prob >= high_q
            idx_high = mask_high.float().argmax(dim=-1).clamp(0, N - 1)
            rows = torch.arange(B, device=returns.device)

            L = sorted_ret[rows, idx_low]
            U = sorted_ret[rows, idx_high]

            # denormalise
            L = L * batch["stds"][:, 0] + batch["means"][:, 0]
            U = U * batch["stds"][:, 0] + batch["means"][:, 0]

            step_lower.append(L.detach().cpu())
            step_upper.append(U.detach().cpu())

            # feed argmax for next step
            if step < horizon - 1:
                next_c = last_logits_c[:, -1, :].argmax(dim=-1)
                next_f = last_logits_f[:, -1, :].argmax(dim=-1)
                context_c = torch.cat([context_c, next_c.unsqueeze(1)], dim=1)
                context_f = torch.cat([context_f, next_f.unsqueeze(1)], dim=1)

        pred_lower.append(torch.stack(step_lower, dim=1))
        pred_upper.append(torch.stack(step_upper, dim=1))
        actual_all.append(raw_batch["actual_returns"].detach().cpu())

    if not pred_lower:
        empty = np.empty((0, horizon), dtype=np.float32)
        return empty, empty, empty

    pred_lower_np = torch.cat(pred_lower, dim=0).numpy()
    pred_upper_np = torch.cat(pred_upper, dim=0).numpy()
    actual_np = torch.cat(actual_all, dim=0).numpy()
    return pred_lower_np, pred_upper_np, actual_np


def evaluate_ci_model(model, val_loader, cfg, device, amp_enabled, amp_dtype):
    pred_lower, pred_upper, actual = predict_ci_from_distribution(
        model=model,
        loader=val_loader,
        cfg=cfg,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        confidence_level=float(getattr(cfg, "ci_confidence_level", 0.80)),
        top_k=int(getattr(cfg, "ci_top_k", 32)),
    )
    return compute_ci_metrics(
        pred_lower=pred_lower,
        pred_upper=pred_upper,
        actual_returns=actual,
        confidence_level=float(getattr(cfg, "ci_confidence_level", 0.80)),
        mape_eps=float(getattr(cfg, "mape_eps", 1e-4)),
    )


# ═══════════════════════════════════════════════════════════════════
# Training orchestration
# ═══════════════════════════════════════════════════════════════════

def _configure_trainable(model, cfg):
    scope = str(getattr(cfg, "trainable_scope", "all")).strip().lower()
    if bool(cfg.freeze_backbone):
        for param in model.parameters():
            param.requires_grad = False
    elif scope == "heads":
        for param in model.parameters():
            param.requires_grad = False
        trainable_prefixes = (
            "norm.", "head_coarse.", "coarse_to_fine.",
            "fine_gate.", "fine_norm.", "head_fine.",
        )
        for name, param in model.named_parameters():
            if name.startswith(trainable_prefixes):
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters for CI post-training.")
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
        "stage": "ci_post_train",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_state_dict": raw_model.state_dict(),
        "tokenizer_state_dict": tokenizer.state_dict(),
        "model_config": getattr(raw_model, "model_config", None),
        "post_train_ci_config": _cfg_to_dict(cfg),
        "metrics": metrics,
        "history": history,
    }
    torch.save(payload, path)
    return path


def _write_history(path, cfg, history, best_metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": _cfg_to_dict(cfg),
        "best_metrics": best_metrics,
        "history": history,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def _move_batch(raw_batch, device):
    return {
        "features": raw_batch["features"].to(device=device, dtype=torch.float32, non_blocking=True),
        "time": {
            key: value.to(device=device, dtype=torch.long, non_blocking=True)
            for key, value in raw_batch["time"].items()
        },
        "means": raw_batch["means"].to(device=device, dtype=torch.float32, non_blocking=True),
        "stds": raw_batch["stds"].to(device=device, dtype=torch.float32, non_blocking=True),
        "actual_returns": raw_batch["actual_returns"].to(device=device, dtype=torch.float32, non_blocking=True),
        "sample_ids": raw_batch["sample_ids"],
        "symbols": raw_batch["symbols"],
        "target_dates": raw_batch["target_dates"],
    }


def train(cfg):
    global _tokenizer

    set_global_seed(int(cfg.random_seed), deterministic=bool(cfg.deterministic))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.use_tf32)
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    amp_dtype = _amp_dtype(cfg.amp_dtype)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda" and amp_dtype is not None)

    # ── datasets ──
    train_dataset = RolloutWindowDataset(
        "train", cfg=cfg,
        max_samples=int(cfg.max_train_samples),
        seed=int(cfg.random_seed),
    )
    val_dataset = RolloutWindowDataset(
        "val", cfg=cfg,
        max_samples=int(cfg.max_val_samples),
        seed=int(cfg.random_seed) + 17,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"CI dataset empty: train={len(train_dataset)}, val={len(val_dataset)}"
        )

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

    # ── model ──
    model, _tokenizer = load_model(
        device=device,
        checkpoint_path=cfg.checkpoint_path,
        strict_checkpoint_compat=False,
    )
    _tokenizer.eval()
    _tokenizer.requires_grad_(False)
    if bool(cfg.use_gradient_checkpointing):
        model.enable_gradient_checkpointing(True)

    reference_model = None
    if float(cfg.kl_weight) > 0.0:
        reference_model = copy.deepcopy(model).to(device)
        reference_model.eval()
        reference_model.requires_grad_(False)

    # ── optimiser & scheduler ──
    param_groups = _configure_trainable(model, cfg)
    optimizer, opt_kwargs = _build_optimizer(param_groups, cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    summary = trainable_parameter_summary(model)

    print(f"Device: {device}, amp={amp_enabled}, amp_dtype={cfg.amp_dtype}")
    print(f"Train windows={len(train_dataset)}, val windows={len(val_dataset)}")
    print(f"Trainable parameters: {summary}")
    print(f"Optimizer: AdamW {opt_kwargs}")

    total_steps = max(1, math.ceil(len(train_loader) / int(cfg.accumulation_steps)) * int(cfg.epochs))
    warmup = max(1, total_steps // 10)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup),
        eta_min=float(cfg.learning_rate) * 0.05,
    )

    history = []
    best_score = float("inf")  # lower interval score is better
    best_metrics = None
    best_path = os.path.join(cfg.output_dir, cfg.save_name)
    updates = 0

    for epoch in range(int(cfg.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_totals = {
            "loss": 0.0, "anchor_loss": 0.0, "concentration_loss": 0.0,
            "iscore_loss": 0.0, "kl_loss": 0.0,
            "avg_width": 0.0, "coverage": 0.0, "avg_iscore": 0.0,
            "used_pred_ratio": 0.0,
        }
        batches = 0
        pbar = tqdm(train_loader, desc=f"CI train epoch {epoch + 1}/{cfg.epochs}")

        for batch_idx, raw_batch in enumerate(pbar, start=1):
            progress = 0.0
            if int(cfg.epochs) > 0:
                progress = (epoch + (batch_idx - 1) / max(1, len(train_loader))) / max(1, int(cfg.epochs))
            rollout_ratio = float(cfg.rollout_ratio_start) + (
                float(cfg.rollout_ratio_end) - float(cfg.rollout_ratio_start)
            ) * progress

            batch = _move_batch(raw_batch, device)

            with _autocast_context(device, amp_enabled, amp_dtype):
                loss, stats = ci_training_loss(
                    model=model,
                    reference_model=reference_model,
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
            for key in ("anchor_loss", "concentration_loss", "iscore_loss",
                         "kl_loss", "avg_width", "coverage", "avg_iscore",
                         "used_pred_ratio"):
                epoch_totals[key] += float(stats.get(key, 0.0))
            batches += 1

            if batch_idx % int(cfg.progress_interval) == 0:
                pbar.set_postfix({
                    "loss": f"{value:.4f}",
                    "conc": f"{stats.get('concentration_loss', 0):.4f}",
                    "cov": f"{stats.get('coverage', 0):.3f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

            if int(cfg.max_train_updates) > 0 and updates >= int(cfg.max_train_updates):
                break

        train_row = {key: value / max(1, batches) for key, value in epoch_totals.items()}

        # ── validation ──
        val_metrics = evaluate_ci_model(
            model=model,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        score = float(val_metrics.get("avg_interval_score", float("inf")))

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
            _save_checkpoint(epoch_path, model, _tokenizer, cfg, val_metrics, history)

        if score < best_score:
            best_score = score
            best_metrics = val_metrics
            _save_checkpoint(best_path, model, _tokenizer, cfg, val_metrics, history)

        if int(cfg.max_train_updates) > 0 and updates >= int(cfg.max_train_updates):
            break

    history_path = os.path.join(cfg.output_dir, "ci_training_history.json")
    _write_history(history_path, cfg, history, best_metrics)
    print(f"Best CI checkpoint: {best_path}  (interval_score={best_score:.6f})")
    print(f"History: {history_path}")
    return {
        "best_path": best_path,
        "history_path": history_path,
        "best_metrics": best_metrics,
        "history": history,
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Post_Train_CI confidence-interval post-training")
    parser.add_argument("--checkpoint-path", default=PostTrainCIConfig.checkpoint_path)
    parser.add_argument("--output-dir", default=PostTrainCIConfig.output_dir)
    parser.add_argument("--save-name", default=PostTrainCIConfig.save_name)
    parser.add_argument("--save-epoch-checkpoints", type=_as_bool, default=PostTrainCIConfig.save_epoch_checkpoints)
    parser.add_argument("--prefix-len", type=int, default=PostTrainCIConfig.prefix_len)
    parser.add_argument("--horizon", type=int, default=PostTrainCIConfig.horizon)
    parser.add_argument("--max-stocks", type=int, default=PostTrainCIConfig.max_stocks)
    parser.add_argument("--max-train-samples", type=int, default=PostTrainCIConfig.max_train_samples)
    parser.add_argument("--max-val-samples", type=int, default=PostTrainCIConfig.max_val_samples)
    parser.add_argument("--epochs", type=int, default=PostTrainCIConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=PostTrainCIConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=PostTrainCIConfig.eval_batch_size)
    parser.add_argument("--accumulation-steps", type=int, default=PostTrainCIConfig.accumulation_steps)
    parser.add_argument("--num-workers", type=int, default=PostTrainCIConfig.num_workers)
    parser.add_argument("--lr", type=float, default=PostTrainCIConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=PostTrainCIConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=PostTrainCIConfig.grad_clip)
    parser.add_argument("--max-train-updates", type=int, default=PostTrainCIConfig.max_train_updates)
    parser.add_argument("--progress-interval", type=int, default=PostTrainCIConfig.progress_interval)
    parser.add_argument("--rollout-ratio-start", type=float, default=PostTrainCIConfig.rollout_ratio_start)
    parser.add_argument("--rollout-ratio-end", type=float, default=PostTrainCIConfig.rollout_ratio_end)
    parser.add_argument("--concentration-weight", type=float, default=PostTrainCIConfig.concentration_weight)
    parser.add_argument("--interval-score-weight", type=float, default=PostTrainCIConfig.interval_score_weight)
    parser.add_argument("--kl-weight", type=float, default=PostTrainCIConfig.kl_weight)
    parser.add_argument("--ci-confidence-level", type=float, default=PostTrainCIConfig.ci_confidence_level)
    parser.add_argument("--ci-top-k", type=int, default=PostTrainCIConfig.ci_top_k)
    parser.add_argument("--step-weight-gamma", type=float, default=PostTrainCIConfig.step_weight_gamma)
    parser.add_argument("--freeze-backbone", type=_as_bool, default=PostTrainCIConfig.freeze_backbone)
    parser.add_argument("--trainable-scope", choices=["all", "heads"], default=PostTrainCIConfig.trainable_scope)
    parser.add_argument("--use-gradient-checkpointing", type=_as_bool, default=PostTrainCIConfig.use_gradient_checkpointing)
    parser.add_argument("--use-amp", type=_as_bool, default=PostTrainCIConfig.use_amp)
    parser.add_argument("--amp-dtype", default=PostTrainCIConfig.amp_dtype)
    parser.add_argument("--use-tf32", type=_as_bool, default=PostTrainCIConfig.use_tf32)
    parser.add_argument("--mape-eps", type=float, default=PostTrainCIConfig.mape_eps)
    parser.add_argument("--deterministic", type=_as_bool, default=PostTrainCIConfig.deterministic)
    parser.add_argument("--seed", type=int, default=PostTrainCIConfig.random_seed)
    parser.add_argument("--eval-only", action="store_true")
    return parser


def _namespace_from_args(args):
    return argparse.Namespace(
        checkpoint_path=resolve_project_path(args.checkpoint_path),
        output_dir=resolve_project_path(args.output_dir),
        save_name=str(args.save_name),
        save_epoch_checkpoints=bool(args.save_epoch_checkpoints),
        prefix_len=int(args.prefix_len),
        horizon=int(args.horizon),
        stride_ratio=float(PostTrainCIConfig.stride_ratio),
        cache_dir=resolve_project_path(PostTrainCIConfig.cache_dir),
        cache_rebuild=False,
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
        concentration_weight=float(args.concentration_weight),
        interval_score_weight=float(args.interval_score_weight),
        kl_weight=float(args.kl_weight),
        ci_confidence_level=float(args.ci_confidence_level),
        ci_top_k=int(args.ci_top_k),
        step_weight_gamma=float(args.step_weight_gamma),
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
    )


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = _namespace_from_args(args)
    os.makedirs(cfg.output_dir, exist_ok=True)
    return train(cfg)


if __name__ == "__main__":
    main()
