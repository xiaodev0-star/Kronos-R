# -*- coding: utf-8 -*-
"""Phase 8: STAR-CAST Engine — Self-Training with Asymmetric Reward and
Continuous-Asymmetric Dual-Engine Fine-Tuning.

STAR-CAST = Noisy Exploration + Oracle Filter + Dual-Engine Update

At each training step:
  1. Noisy Exploration: rollout N=4 trajectories with NEFTune noise + temperature sampling
  2. Oracle Filter: select the best trajectory with correct direction sign
  3. Dual-Engine Update:
     - Continuous layer: asymmetric direction-aware loss on expected returns
     - Discrete layer: STaR-style CE loss on golden trajectories only
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

from config import PostTrainStarCastConfig
from evaluate_predictions import load_model
from model.lora import trainable_parameter_summary
from posttrain.rollout.data import (
    RolloutWindowDataset,
    resolve_project_path,
    rollout_cache_path,
    rollout_collate,
)
from reproducibility import set_global_seed

# ── Reuse evaluation utilities from Phase 6 ──
from posttrain.rollout.train_rollout import (
    _autocast_context,
    _amp_dtype,
    _as_bool,
    _cfg_to_dict,
    _cuda_peak_memory_stats,
    _move_batch,
    _encode_features,
    _configure_trainable,
    _build_optimizer,
    _write_history,
    compute_rollout_metrics,
    predict_autoregressive_returns,
    evaluate_model,
)


def _save_star_cast_checkpoint(path, model, tokenizer, cfg, metrics, history):
    """Save STAR-CAST checkpoint with correct stage identifier."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    payload = {
        "stage": "star_cast_phase8",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_state_dict": raw_model.state_dict(),
        "tokenizer_state_dict": tokenizer.state_dict(),
        "model_config": getattr(raw_model, "model_config", None),
        "post_train_star_cast_config": _cfg_to_dict(cfg),
        "metrics": metrics,
        "history": history,
    }
    torch.save(payload, path)
    return path


# ═══════════════════════════════════════════════════════════════════════
# Module 1: NEFTune Noise Injection (for financial time-series generalization)
# ═══════════════════════════════════════════════════════════════════════

def inject_neftune_noise(hidden_states, noise_alpha=5.0):
    """Inject scaled uniform noise into hidden states after embedding.

    Formula: noise = Uniform(-1, 1) * alpha / sqrt(L * D)

    This forces the model to learn directionality from "imperfect history",
    improving robustness to financial time-series noise.
    """
    if not hidden_states.requires_grad:
        return hidden_states
    B, L, D = hidden_states.shape
    noise = (torch.rand_like(hidden_states) * 2 - 1) * (float(noise_alpha) / math.sqrt(L * D))
    return hidden_states + noise


# ═══════════════════════════════════════════════════════════════════════
# Module 2: Differentiable Expected Returns & Asymmetric Direction Loss
# ═══════════════════════════════════════════════════════════════════════

def get_differentiable_expected_returns(tokenizer, logits_c, logits_f, means, stds, top_k=16,
                                         sharpening_temp=1.0):
    """Bridge the discrete-token to continuous-return gradient gap.

    Computes the fully-differentiable expected log-return at each step by:
      1. Taking top-K coarse and fine tokens
      2. Computing joint (coarse, fine) probabilities (with optional sharpening)
      3. Decoding all K*K pairs to continuous returns
      4. Computing the probability-weighted expected return

    Probability sharpening (sharpening_temp < 1.0) forces the token distribution
    to be sharper/peakier, preventing the weighted average from collapsing to ~0
    when the distribution is flat (high entropy). This ensures the expected return
    reflects the model's true top-1 intent.

    Args:
        tokenizer: HierarchicalQuantizer with decode(idx_c, idx_f) -> [B, N, 6]
        logits_c: [B, H, V_c] coarse logits
        logits_f: [B, H, V_f] fine logits
        means: [B, 6] per-feature means from prefix normalization
        stds:  [B, 6] per-feature stds from prefix normalization
        top_k: number of top tokens to consider
        sharpening_temp: temperature for softmax (< 1.0 = sharper, 1.0 = unchanged)

    Returns:
        expected_returns: [B, H] soft expected log-return at each step
    """
    B, H, V_c = logits_c.shape
    K = min(int(top_k), V_c)

    top_logits_c, top_idx_c = torch.topk(logits_c.float(), k=K, dim=-1)  # [B, H, K]
    top_logits_f, top_idx_f = torch.topk(logits_f.float(), k=K, dim=-1)

    # Temperature sharpening: lower temp -> sharper distribution -> less averaging to 0
    prob_c = F.softmax(top_logits_c / sharpening_temp, dim=-1)  # [B, H, K]
    prob_f = F.softmax(top_logits_f / sharpening_temp, dim=-1)
    prob_c = prob_c / prob_c.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prob_f = prob_f / prob_f.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # Joint probability: [B, H, K, K]
    joint_prob = prob_c.unsqueeze(-1) * prob_f.unsqueeze(-2)

    # Build all K*K pairs for batched decoding
    pair_c = top_idx_c.unsqueeze(-1).expand(B, H, K, K).reshape(B * H, K * K)
    pair_f = top_idx_f.unsqueeze(-2).expand(B, H, K, K).reshape(B * H, K * K)

    with torch.no_grad():
        decoded = tokenizer.decode(pair_c, pair_f)[..., 0].float()  # log_ret column
        returns_grid = decoded.view(B, H, K, K) * stds[:, 0].view(B, 1, 1, 1) + means[:, 0].view(B, 1, 1, 1)

    expected_returns = (joint_prob * returns_grid).sum(dim=(-1, -2))  # [B, H]
    return expected_returns


def compute_asymmetric_direction_loss(expected_returns, actual_returns, eps=1e-4,
                                       alpha=3.0, beta=10.0, timidity_weight=2.0,
                                       timidity_ratio=0.5):
    """Asymmetric direction-aware penalty with push-forward for timid predictions.

    Three penalty regimes:
      - Wrong direction: amplified by (alpha + beta * |expected|) — devastating
      - Correct direction but too conservative (|pred| < |actual| * timidity_ratio):
        amplified by timidity_weight — forces model to commit to larger magnitudes
      - Correct direction & adequate magnitude: weight ~1.0 (standard L1)

    The push-forward penalty breaks the "zero-collapse" trap where the model
    learns to predict ~0 to avoid triggering the asymmetric wrong-direction penalty.

    Args:
        expected_returns: [B, H] model's expected log-returns
        actual_returns:   [B, H] ground-truth log-returns
        eps: threshold for "flat/directional" classification
        alpha: base penalty multiplier for wrong direction
        beta: magnitude-scaled penalty for wrong direction
        timidity_weight: penalty multiplier for correct-but-timid predictions
        timidity_ratio: threshold ratio — |pred| < |actual| * ratio counts as timid

    Returns:
        per_step_loss: [B, H] weighted absolute error per step
    """
    abs_error = torch.abs(expected_returns - actual_returns)
    direction_product = expected_returns * actual_returns

    is_directional = torch.abs(actual_returns) > eps
    is_wrong_direction = (direction_product < 0) & is_directional
    # Push-forward: correct direction but predicted magnitude < ratio * actual
    is_correct_but_timid = (
        (direction_product > 0)
        & (torch.abs(expected_returns) < torch.abs(actual_returns) * timidity_ratio)
        & is_directional
    )

    # Dynamic penalty: default 1.0, amplified for wrong direction or timidity
    penalty_weight = torch.ones_like(abs_error)
    wrong_penalty = alpha + beta * torch.abs(expected_returns)
    penalty_weight = torch.where(is_wrong_direction, wrong_penalty, penalty_weight)
    penalty_weight = torch.where(is_correct_but_timid, timidity_weight, penalty_weight)

    return abs_error * penalty_weight


# ═══════════════════════════════════════════════════════════════════════
# Module 3: STAR-CAST Single Training Step
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _star_cast_exploration(
    model, tokenizer, batch, cfg, device, amp_enabled, amp_dtype,
):
    """Phase A: Noisy Exploration — rollout N trajectories per sample.

    Each trajectory is generated autoregressively with:
      - NEFTune noise injected into embeddings
      - Temperature sampling for token selection

    Returns:
        golden_c, golden_f: [B, prefix_len + horizon] best trajectory tokens
        has_golden: [B] bool mask — True if a valid golden trajectory was found
        path_returns: [B, N] total path returns for each trajectory
    """
    prefix_len = int(cfg.prefix_len)
    horizon = int(cfg.horizon)
    N = int(cfg.num_trajectories)
    temp = max(1e-4, float(cfg.exploration_temperature))
    neftune_alpha = float(cfg.neftune_alpha)
    B = int(batch["features"].size(0))

    idx_c_full, idx_f_full = _encode_features(tokenizer, batch["features"])
    means = batch["means"]
    stds = batch["stds"]
    actual_returns_h = batch["actual_returns"][:, :horizon]
    actual_path_return = torch.sum(actual_returns_h, dim=1)  # [B]

    was_training = model.training
    model.eval()

    # ── Expand batch for N parallel trajectories ──
    # Each sample gets N independent rollouts
    context_c = idx_c_full[:, :prefix_len].unsqueeze(1).expand(B, N, prefix_len).reshape(B * N, prefix_len)
    context_f = idx_f_full[:, :prefix_len].unsqueeze(1).expand(B, N, prefix_len).reshape(B * N, prefix_len)

    # Expand time features
    time_expanded = {}
    for key in ("minute", "day", "month", "year"):
        t = batch["time"][key][:, :prefix_len]
        time_expanded[key] = t.unsqueeze(1).expand(B, N, prefix_len).reshape(B * N, prefix_len)

    # Track per-step returns for each trajectory
    all_step_returns = []  # list of [B*N, 1] tensors

    for step in range(horizon):
        cur_len = int(context_c.size(1))
        cur_time = {key: value[:, :cur_len] for key, value in time_expanded.items()}

        with _autocast_context(device, amp_enabled, amp_dtype):
            logits_c, logits_f, _ = model(
                context_c, context_f,
                cur_time["minute"], cur_time["day"],
                cur_time["month"], cur_time["year"],
                last_only=True,
                neftune_alpha=neftune_alpha,
            )

        # Temperature sampling
        probs_c = F.softmax(logits_c[:, -1, :].float() / temp, dim=-1)
        probs_f = F.softmax(logits_f[:, -1, :].float() / temp, dim=-1)
        pred_c = torch.multinomial(probs_c, num_samples=1)  # [B*N, 1]
        pred_f = torch.multinomial(probs_f, num_samples=1)

        # Decode to continuous returns
        decoded = tokenizer.decode(pred_c, pred_f)[..., 0].float()  # [B*N, 1]
        step_ret = decoded * stds.repeat_interleave(N, dim=0)[:, 0:1] + means.repeat_interleave(N, dim=0)[:, 0:1]
        all_step_returns.append(step_ret)

        # Append predicted token to context
        context_c = torch.cat([context_c, pred_c], dim=1)
        context_f = torch.cat([context_f, pred_f], dim=1)

        # Update time features for next step
        for key in time_expanded:
            next_t = batch["time"][key][:, prefix_len + step:prefix_len + step + 1]
            next_t_expanded = next_t.unsqueeze(1).expand(B, N, 1).reshape(B * N, 1)
            time_expanded[key] = torch.cat([time_expanded[key], next_t_expanded], dim=1)

    # ── Reshape results to [B, N, H] ──
    step_returns_tensor = torch.cat(all_step_returns, dim=1)  # [B*N, H]
    step_returns_tensor = step_returns_tensor.view(B, N, horizon)
    path_returns = torch.sum(step_returns_tensor, dim=2)  # [B, N]

    # ── Oracle Filter: select best trajectory per sample ──
    best_paths_c_list = []
    best_paths_f_list = []
    has_golden_list = []

    context_c_reshaped = context_c.view(B, N, prefix_len + horizon)
    context_f_reshaped = context_f.view(B, N, prefix_len + horizon)

    for b in range(B):
        real_total = actual_path_return[b].item()
        path_ret_b = path_returns[b]  # [N]

        is_correct_dir = (path_ret_b * real_total) > 0

        if is_correct_dir.any() and abs(real_total) > float(cfg.mape_eps):
            valid_indices = torch.where(is_correct_dir)[0]
            # Base error: absolute path-return error
            errors = torch.abs(path_ret_b[valid_indices] - real_total)
            # Volatility-matching penalty: penalize trajectories whose predicted
            # magnitude is much smaller than the actual magnitude.
            # This breaks the Oracle's preference for conservative near-zero predictions
            # and encourages selection of trajectories with realistic volatility.
            mag_penalty = torch.clamp(
                torch.abs(torch.tensor(real_total, device=path_ret_b.device))
                - torch.abs(path_ret_b[valid_indices]), min=0,
            )
            errors = errors + float(cfg.oracle_magnitude_penalty) * mag_penalty
            best_idx = valid_indices[errors.argmin()]

            best_paths_c_list.append(context_c_reshaped[b, best_idx])
            best_paths_f_list.append(context_f_reshaped[b, best_idx])
            has_golden_list.append(True)
        else:
            # Fallback: use ground-truth tokens (teacher forcing)
            best_paths_c_list.append(idx_c_full[b, :prefix_len + horizon])
            best_paths_f_list.append(idx_f_full[b, :prefix_len + horizon])
            has_golden_list.append(False)

    golden_c = torch.stack(best_paths_c_list).to(device)
    golden_f = torch.stack(best_paths_f_list).to(device)
    has_golden = torch.tensor(has_golden_list, device=device)

    if was_training:
        model.train()

    return golden_c, golden_f, has_golden, path_returns.view(B, N)


def train_star_cast_step(model, tokenizer, batch, cfg, device, amp_enabled, amp_dtype):
    """Single STAR-CAST training step.

    Returns:
        loss: scalar total loss
        stats: dict with component losses and golden_rate
    """
    prefix_len = int(cfg.prefix_len)
    horizon = int(cfg.horizon)
    B = int(batch["features"].size(0))

    # ── Encode features for later use ──
    with torch.no_grad():
        idx_c_full, idx_f_full = _encode_features(tokenizer, batch["features"])

    actual_returns_h = batch["actual_returns"][:, :horizon]

    # ── Phase A: Noisy Exploration + Oracle Filter ──
    golden_c, golden_f, has_golden, _ = _star_cast_exploration(
        model=model, tokenizer=tokenizer, batch=batch, cfg=cfg,
        device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
    )

    # ── Phase B: Dual-Engine Update ──
    model.train()

    # Forward pass on golden trajectories (length = prefix_len + horizon)
    # Use input[:, :-1] to predict tokens from position prefix_len onwards
    train_len = int(golden_c.size(1))
    train_time = {key: value[:, :train_len] for key, value in batch["time"].items()}

    with _autocast_context(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _, hidden = model(
            golden_c[:, :-1],
            golden_f[:, :-1],
            train_time["minute"][:, :train_len - 1],
            train_time["day"][:, :train_len - 1],
            train_time["month"][:, :train_len - 1],
            train_time["year"][:, :train_len - 1],
            return_hidden=True,
            neftune_alpha=0.0,  # no NEFTune during training forward
        )

    # Extract rollout region logits: positions [prefix_len-1, prefix_len+H-2]
    start = prefix_len - 1
    end = start + horizon
    rollout_c = logits_c[:, start:end, :]  # [B, H, V_c]
    rollout_f = logits_f[:, start:end, :]  # [B, H, V_f]

    # ═══════════════════════════════════════════════════════════════
    # Engine 1: Continuous Layer — Asymmetric Direction Loss
    # ═══════════════════════════════════════════════════════════════

    expected_traj = get_differentiable_expected_returns(
        tokenizer, rollout_c, rollout_f,
        batch["means"], batch["stds"],
        top_k=int(cfg.top_k_expected_return),
        sharpening_temp=float(cfg.prob_sharpening_temp),
    )

    # Step-level asymmetric loss (with push-forward)
    step_loss_matrix = compute_asymmetric_direction_loss(
        expected_traj, actual_returns_h,
        alpha=float(cfg.asymmetric_alpha), beta=float(cfg.asymmetric_beta),
        timidity_weight=float(cfg.timidity_penalty_weight),
        timidity_ratio=float(cfg.timidity_ratio_threshold),
    )
    step_asym_loss = step_loss_matrix.mean()

    # Path-level asymmetric loss (cumulative returns, with push-forward)
    expected_path = torch.cumsum(expected_traj, dim=1)
    actual_path = torch.cumsum(actual_returns_h, dim=1)
    path_loss_matrix = compute_asymmetric_direction_loss(
        expected_path, actual_path,
        alpha=float(cfg.path_asymmetric_alpha), beta=float(cfg.path_asymmetric_beta),
        timidity_weight=float(cfg.timidity_penalty_weight),
        timidity_ratio=float(cfg.timidity_ratio_threshold),
    )
    path_asym_loss = path_loss_matrix.mean()

    # ═══════════════════════════════════════════════════════════════
    # Engine 2: Discrete Layer — STaR CE Reinforcement
    # ═══════════════════════════════════════════════════════════════
    if has_golden.any():
        target_c = golden_c[has_golden, prefix_len:prefix_len + horizon]
        target_f = golden_f[has_golden, prefix_len:prefix_len + horizon]
        act_logits_c = rollout_c[has_golden]
        act_logits_f = rollout_f[has_golden]

        ce_loss_c = F.cross_entropy(
            act_logits_c.reshape(-1, act_logits_c.size(-1)).float(),
            target_c.reshape(-1),
        )
        ce_loss_f = F.cross_entropy(
            act_logits_f.reshape(-1, act_logits_f.size(-1)).float(),
            target_f.reshape(-1),
        )
        star_ce_loss = ce_loss_c + ce_loss_f
    else:
        star_ce_loss = torch.tensor(0.0, device=device)

    # ═══════════════════════════════════════════════════════════════
    # Total Loss
    # ═══════════════════════════════════════════════════════════════
    total_loss = (
        float(cfg.step_asym_weight) * step_asym_loss +
        float(cfg.path_asym_weight) * path_asym_loss +
        float(cfg.star_ce_weight) * star_ce_loss
    )

    golden_rate = has_golden.float().mean()

    return total_loss, {
        "total_loss": float(total_loss.detach().item()),
        "step_asym": float(step_asym_loss.detach().item()),
        "path_asym": float(path_asym_loss.detach().item()),
        "star_ce": float(star_ce_loss.detach().item()) if has_golden.any() else 0.0,
        "golden_rate": float(golden_rate.item()),
    }


# ═══════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 8: STAR-CAST self-training")
    parser.add_argument("--checkpoint-path", default=PostTrainStarCastConfig.checkpoint_path)
    parser.add_argument("--output-dir", default=PostTrainStarCastConfig.output_dir)
    parser.add_argument("--save-name", default=PostTrainStarCastConfig.save_name)
    parser.add_argument("--save-epoch-checkpoints", type=_as_bool, default=PostTrainStarCastConfig.save_epoch_checkpoints)
    parser.add_argument("--prefix-len", type=int, default=PostTrainStarCastConfig.prefix_len)
    parser.add_argument("--horizon", type=int, default=PostTrainStarCastConfig.horizon)
    parser.add_argument("--stride-ratio", type=float, default=PostTrainStarCastConfig.stride_ratio)
    parser.add_argument("--cache-dir", default=PostTrainStarCastConfig.cache_dir)
    parser.add_argument("--cache-rebuild", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=PostTrainStarCastConfig.max_stocks)
    parser.add_argument("--max-train-samples", type=int, default=PostTrainStarCastConfig.max_train_samples)
    parser.add_argument("--max-val-samples", type=int, default=PostTrainStarCastConfig.max_val_samples)
    parser.add_argument("--epochs", type=int, default=PostTrainStarCastConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=PostTrainStarCastConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=PostTrainStarCastConfig.eval_batch_size)
    parser.add_argument("--accumulation-steps", type=int, default=PostTrainStarCastConfig.accumulation_steps)
    parser.add_argument("--num-workers", type=int, default=PostTrainStarCastConfig.num_workers)
    parser.add_argument("--lr", type=float, default=PostTrainStarCastConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=PostTrainStarCastConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=PostTrainStarCastConfig.grad_clip)
    parser.add_argument("--max-train-updates", type=int, default=PostTrainStarCastConfig.max_train_updates)
    parser.add_argument("--progress-interval", type=int, default=PostTrainStarCastConfig.progress_interval)
    parser.add_argument("--checkpoint-interval", type=int, default=PostTrainStarCastConfig.checkpoint_interval)

    # STAR-CAST hyperparameters
    parser.add_argument("--neftune-alpha", type=float, default=PostTrainStarCastConfig.neftune_alpha)
    parser.add_argument("--num-trajectories", type=int, default=PostTrainStarCastConfig.num_trajectories)
    parser.add_argument("--exploration-temperature", type=float, default=PostTrainStarCastConfig.exploration_temperature)
    parser.add_argument("--top-k-expected-return", type=int, default=PostTrainStarCastConfig.top_k_expected_return)
    parser.add_argument("--asymmetric-alpha", type=float, default=PostTrainStarCastConfig.asymmetric_alpha)
    parser.add_argument("--asymmetric-beta", type=float, default=PostTrainStarCastConfig.asymmetric_beta)
    parser.add_argument("--path-asymmetric-alpha", type=float, default=PostTrainStarCastConfig.path_asymmetric_alpha)
    parser.add_argument("--path-asymmetric-beta", type=float, default=PostTrainStarCastConfig.path_asymmetric_beta)
    parser.add_argument("--step-asym-weight", type=float, default=PostTrainStarCastConfig.step_asym_weight)
    parser.add_argument("--path-asym-weight", type=float, default=PostTrainStarCastConfig.path_asym_weight)
    parser.add_argument("--star-ce-weight", type=float, default=PostTrainStarCastConfig.star_ce_weight)
    parser.add_argument("--timidity-penalty-weight", type=float, default=PostTrainStarCastConfig.timidity_penalty_weight)
    parser.add_argument("--timidity-ratio-threshold", type=float, default=PostTrainStarCastConfig.timidity_ratio_threshold)
    parser.add_argument("--oracle-magnitude-penalty", type=float, default=PostTrainStarCastConfig.oracle_magnitude_penalty)
    parser.add_argument("--prob-sharpening-temp", type=float, default=PostTrainStarCastConfig.prob_sharpening_temp)
    parser.add_argument("--actionable-da-threshold", type=float, default=PostTrainStarCastConfig.actionable_da_threshold)

    parser.add_argument("--freeze-backbone", type=_as_bool, default=PostTrainStarCastConfig.freeze_backbone)
    parser.add_argument("--trainable-scope", choices=["all", "heads"], default=PostTrainStarCastConfig.trainable_scope)
    parser.add_argument("--use-gradient-checkpointing", type=_as_bool, default=PostTrainStarCastConfig.use_gradient_checkpointing)
    parser.add_argument("--use-amp", type=_as_bool, default=PostTrainStarCastConfig.use_amp)
    parser.add_argument("--amp-dtype", default=PostTrainStarCastConfig.amp_dtype)
    parser.add_argument("--use-tf32", type=_as_bool, default=PostTrainStarCastConfig.use_tf32)
    parser.add_argument("--mape-eps", type=float, default=PostTrainStarCastConfig.mape_eps)
    parser.add_argument("--deterministic", type=_as_bool, default=PostTrainStarCastConfig.deterministic)
    parser.add_argument("--seed", type=int, default=PostTrainStarCastConfig.random_seed)
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
        checkpoint_interval=int(args.checkpoint_interval),
        neftune_alpha=float(args.neftune_alpha),
        num_trajectories=int(args.num_trajectories),
        exploration_temperature=float(args.exploration_temperature),
        top_k_expected_return=int(args.top_k_expected_return),
        asymmetric_alpha=float(args.asymmetric_alpha),
        asymmetric_beta=float(args.asymmetric_beta),
        path_asymmetric_alpha=float(args.path_asymmetric_alpha),
        path_asymmetric_beta=float(args.path_asymmetric_beta),
        step_asym_weight=float(args.step_asym_weight),
        path_asym_weight=float(args.path_asym_weight),
        star_ce_weight=float(args.star_ce_weight),
        timidity_penalty_weight=float(args.timidity_penalty_weight),
        timidity_ratio_threshold=float(args.timidity_ratio_threshold),
        oracle_magnitude_penalty=float(args.oracle_magnitude_penalty),
        prob_sharpening_temp=float(args.prob_sharpening_temp),
        actionable_da_threshold=float(args.actionable_da_threshold),
        freeze_backbone=bool(args.freeze_backbone),
        trainable_scope=str(args.trainable_scope),
        use_gradient_checkpointing=bool(args.use_gradient_checkpointing),
        use_amp=bool(args.use_amp),
        amp_dtype=str(args.amp_dtype),
        use_tf32=bool(args.use_tf32),
        mape_eps=float(args.mape_eps),
        deterministic=bool(args.deterministic),
        random_seed=int(args.seed),
    )


# ═══════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════

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
        raise RuntimeError(f"STAR-CAST dataset empty: train={len(train_dataset)}, val={len(val_dataset)}")

    loader_kwargs = {
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
        "collate_fn": rollout_collate,
    }
    train_loader = DataLoader(
        train_dataset, batch_size=max(1, int(cfg.batch_size)),
        shuffle=True, drop_last=False, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=max(1, int(cfg.eval_batch_size)),
        shuffle=False, drop_last=False, **loader_kwargs,
    )

    model, tokenizer = load_model(
        device=device, checkpoint_path=cfg.checkpoint_path,
        strict_checkpoint_compat=False,
    )
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    if bool(cfg.use_gradient_checkpointing):
        model.enable_gradient_checkpointing(True)

    param_groups = _configure_trainable(model, cfg)
    optimizer, optimizer_kwargs = _build_optimizer(param_groups, cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    summary = trainable_parameter_summary(model)

    print(f"=== STAR-CAST Phase 8 Training ===")
    print(f"Device: {device}, amp={amp_enabled}, amp_dtype={cfg.amp_dtype}")
    print(f"Train windows={len(train_dataset)}, val windows={len(val_dataset)}")
    print(f"Trainable parameters: {summary}")
    print(f"Optimizer: AdamW {optimizer_kwargs}")
    print(f"NEFTune alpha={cfg.neftune_alpha}, trajectories={cfg.num_trajectories}")
    print(f"Exploration temp={cfg.exploration_temperature}, top_k={cfg.top_k_expected_return}")

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
    best_score = -float("inf")
    best_metrics = None
    best_path = os.path.join(cfg.output_dir, cfg.save_name)
    updates = 0

    for epoch in range(int(cfg.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_totals = {
            "total_loss": 0.0, "step_asym": 0.0, "path_asym": 0.0,
            "star_ce": 0.0, "golden_rate": 0.0,
        }
        batches = 0
        pbar = tqdm(train_loader, desc=f"STAR-CAST epoch {epoch + 1}/{cfg.epochs}")

        for batch_idx, raw_batch in enumerate(pbar, start=1):
            batch = _move_batch(raw_batch, device)

            loss, stats = train_star_cast_step(
                model=model, tokenizer=tokenizer, batch=batch, cfg=cfg,
                device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
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

                # ── Step-interval checkpoint ──
                ci = int(getattr(cfg, "checkpoint_interval", 0))
                if ci > 0 and updates % ci == 0:
                    step_path = os.path.join(
                        cfg.output_dir,
                        f"{os.path.splitext(cfg.save_name)[0]}-step{updates}.pt",
                    )
                    _save_star_cast_checkpoint(step_path, model, tokenizer, cfg, {"updates": updates}, history)

            for key in epoch_totals:
                epoch_totals[key] += float(stats.get(key, 0.0))
            batches += 1

            if batch_idx % int(cfg.progress_interval) == 0:
                pbar.set_postfix({
                    "loss": f"{stats['total_loss']:.4f}",
                    "golden": f"{stats['golden_rate']:.2f}",
                    "step_asym": f"{stats['step_asym']:.4f}",
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
            _save_star_cast_checkpoint(
                epoch_path, model, tokenizer, cfg, val_metrics, history,
            )

        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            _save_star_cast_checkpoint(best_path, model, tokenizer, cfg, val_metrics, history)

        if int(cfg.max_train_updates) > 0 and updates >= int(cfg.max_train_updates):
            break

    history_path = os.path.join(cfg.output_dir, "star_cast_history.json")
    _write_history(history_path, cfg, history, best_metrics)
    print(f"Best STAR-CAST checkpoint: {best_path}")
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
