# -*- coding: utf-8 -*-
"""Phase 9: STAR-CAST v5 — Breaking the Zero-Collapse Trap.

Key improvements over Phase 8 (train_star_cast.py):
  1. Magnitude-Anchored Asymmetric Loss — anchors penalty to |actual| when |expected|→0
  2. Dynamic Timidity — penalty scales with prediction conservatism
  3. Magnitude Floor — explicit penalty for near-zero predictions
  4. KL Anchor — frozen base model prevents distribution drift
  5. Oracle N=8 with 3-factor scoring — more diverse trajectory selection
  6. top_k=32 for expected returns — reduces probability cancellation
  7. Full engineering fixes — unified params, complete logging
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

from config import PostTrainStarCastV5Config
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
    _kl_to_reference,
    compute_rollout_metrics,
    predict_autoregressive_returns,
    evaluate_model,
)


def _save_v5_checkpoint(path, model, tokenizer, cfg, metrics, history):
    """Save Phase 9 checkpoint with correct stage identifier."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    payload = {
        "stage": "star_cast_v5_phase9",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_state_dict": raw_model.state_dict(),
        "tokenizer_state_dict": tokenizer.state_dict(),
        "model_config": getattr(raw_model, "model_config", None),
        "post_train_star_cast_v5_config": _cfg_to_dict(cfg),
        "metrics": metrics,
        "history": history,
    }
    torch.save(payload, path)
    return path

# ═══════════════════════════════════════════════════════════════════════
# Module 1: NEFTune Noise Injection
# ═══════════════════════════════════════════════════════════════════════

def inject_neftune_noise(hidden_states, noise_alpha=5.0):
    """Inject scaled uniform noise into hidden states after embedding."""
    if not hidden_states.requires_grad:
        return hidden_states
    B, L, D = hidden_states.shape
    noise = (torch.rand_like(hidden_states) * 2 - 1) * (float(noise_alpha) / math.sqrt(L * D))
    return hidden_states + noise


# ═══════════════════════════════════════════════════════════════════════
# Module 2: Differentiable Expected Returns
# ═══════════════════════════════════════════════════════════════════════

def get_differentiable_expected_returns(tokenizer, logits_c, logits_f, means, stds, top_k=32,
                                         sharpening_temp=0.5):
    """Bridge the discrete-token to continuous-return gradient gap.

    Phase 9 change: top_k default 32 (was 16) to reduce probability cancellation.
    """
    top_k_c = min(max(1, int(top_k)), int(logits_c.size(-1)))
    top_k_f = min(max(1, int(top_k)), int(logits_f.size(-1)))
    B, H, _ = logits_c.shape

    if sharpening_temp > 0 and abs(sharpening_temp - 1.0) > 1e-4:
        logits_c = logits_c / float(sharpening_temp)
        logits_f = logits_f / float(sharpening_temp)

    top_logits_c, top_idx_c = torch.topk(logits_c.float(), k=top_k_c, dim=-1)
    top_logits_f, top_idx_f = torch.topk(logits_f.float(), k=top_k_f, dim=-1)

    prob_c = torch.softmax(top_logits_c, dim=-1)
    prob_f = torch.softmax(top_logits_f, dim=-1)

    pair_prob = prob_c.unsqueeze(-1) * prob_f.unsqueeze(-2)  # [B, H, Kc, Kf]
    pair_c = top_idx_c.unsqueeze(-1).expand(B, H, top_k_c, top_k_f).reshape(B * H, top_k_c * top_k_f)
    pair_f = top_idx_f.unsqueeze(-2).expand(B, H, top_k_c, top_k_f).reshape(B * H, top_k_c * top_k_f)

    with torch.no_grad():
        decoded = tokenizer.decode(pair_c, pair_f)[..., 0].float()
        decoded = decoded.view(B, H, top_k_c, top_k_f)
        returns = decoded * stds[:, 0].view(B, 1, 1, 1) + means[:, 0].view(B, 1, 1, 1)

    return (pair_prob * returns).sum(dim=(-1, -2))  # [B, H]


# ═══════════════════════════════════════════════════════════════════════
# Module 3: Magnitude-Anchored Asymmetric Loss (Phase 9 NEW)
# ═══════════════════════════════════════════════════════════════════════

def compute_asymmetric_direction_loss_v2(expected_returns, actual_returns, eps=1e-4,
                                          alpha=3.0, beta=10.0, gamma=5.0,
                                          dynamic_timidity_alpha=3.0,
                                          dynamic_timidity_gamma=5.0,
                                          magnitude_floor=0.005,
                                          magnitude_floor_weight=100.0):
    """Magnitude-anchored asymmetric direction-aware penalty.

    Phase 9 key innovation: breaks the "zero is optimal" equilibrium.

    Three innovations over Phase 8:
      1. Magnitude-anchored wrong penalty:
         wrong_penalty = alpha + beta * max(|expected|, gamma * |actual|)
         When expected→0, penalty is still gamma*|actual|*beta — not just alpha.
      2. Dynamic timidity:
         timidity = alpha_t + gamma_t * (|actual| - |expected|) / (|actual| + eps)
         Penalty increases with how conservative the prediction is.
      3. Magnitude floor:
         Explicit penalty for predictions below magnitude_floor threshold.

    Args:
        expected_returns: [B, H] model's expected log-returns
        actual_returns:   [B, H] ground-truth log-returns
        eps: threshold for "flat/directional" classification
        alpha: base penalty multiplier for wrong direction
        beta: magnitude-scaled penalty for wrong direction
        gamma: magnitude anchor coefficient (key Phase 9 parameter)
        dynamic_timidity_alpha: base timidity penalty
        dynamic_timidity_gamma: timidity scaling with conservatism
        magnitude_floor: minimum acceptable prediction magnitude
        magnitude_floor_weight: scaling for floor penalty

    Returns:
        per_step_loss: [B, H] weighted penalty per step
    """
    abs_error = torch.abs(expected_returns - actual_returns)
    direction_product = expected_returns * actual_returns

    is_directional = torch.abs(actual_returns) > eps
    is_wrong_direction = (direction_product < 0) & is_directional
    is_correct_direction = (direction_product > 0) & is_directional

    # ── Innovation 1: Magnitude-anchored wrong penalty ──
    # When |expected| is small, anchor to gamma * |actual|
    anchored_magnitude = torch.max(
        torch.abs(expected_returns),
        gamma * torch.abs(actual_returns),
    )
    wrong_penalty = alpha + beta * anchored_magnitude

    # ── Innovation 2: Dynamic timidity ──
    # conservatism = (|actual| - |expected|) / (|actual| + eps)
    # When prediction is very conservative, penalty is high
    conservatism = torch.clamp(
        (torch.abs(actual_returns) - torch.abs(expected_returns))
        / (torch.abs(actual_returns) + eps),
        min=0.0, max=5.0,
    )
    dynamic_timidity = dynamic_timidity_alpha + dynamic_timidity_gamma * conservatism

    # ── Build penalty weights ──
    penalty_weight = torch.ones_like(abs_error)
    penalty_weight = torch.where(is_wrong_direction, wrong_penalty, penalty_weight)
    penalty_weight = torch.where(is_correct_direction & (conservatism > 0.5),
                                  dynamic_timidity, penalty_weight)

    # ── Innovation 3: Magnitude floor ──
    # Explicit penalty for predictions that are too small
    floor_violation = torch.clamp(magnitude_floor - torch.abs(expected_returns), min=0.0)
    floor_penalty = magnitude_floor_weight * floor_violation

    return abs_error * penalty_weight + floor_penalty


# ═══════════════════════════════════════════════════════════════════════
# Module 4: Direction Labels (same as Phase 8)
# ═══════════════════════════════════════════════════════════════════════

DIR_LABEL_DOWN = 0
DIR_LABEL_FLAT = 1
DIR_LABEL_UP   = 2


def compute_direction_labels(actual_returns, epsilon_scale=0.5):
    """Compute per-step 3-class direction labels from actual returns."""
    B, H = actual_returns.shape
    per_sample_abs_mean = torch.mean(torch.abs(actual_returns), dim=1, keepdim=True)
    epsilons = per_sample_abs_mean * float(epsilon_scale)

    labels = torch.full_like(actual_returns, DIR_LABEL_FLAT, dtype=torch.long,
                             device=actual_returns.device)
    labels = torch.where(actual_returns > epsilons,
                         torch.full_like(labels, DIR_LABEL_UP, device=actual_returns.device),
                         labels)
    labels = torch.where(actual_returns < -epsilons,
                         torch.full_like(labels, DIR_LABEL_DOWN, device=actual_returns.device),
                         labels)
    return labels

# ═══════════════════════════════════════════════════════════════════════
# Module 5: Oracle Exploration with 3-Factor Scoring (Phase 9 improved)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _star_cast_exploration_v5(
    model, tokenizer, batch, cfg, device, amp_enabled, amp_dtype,
):
    """Phase A: Noisy Exploration — rollout N=8 trajectories per sample.

    Phase 9 changes:
      - N=8 (was 4) for more diverse exploration
      - 3-factor Oracle scoring: direction + magnitude error + sign bonus
      - Fallback adds small noise instead of pure GT teacher-forcing

    Returns:
        golden_c, golden_f: [B, prefix_len + horizon] best trajectory tokens
        has_golden: [B] bool mask
        path_returns: [B, N] total path returns for each trajectory
        exploration_stats: dict with diagnostic info
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
    actual_path_return = torch.sum(actual_returns_h, dim=1)

    was_training = model.training
    model.eval()

    # ── Expand batch for N parallel trajectories ──
    context_c = idx_c_full[:, :prefix_len].unsqueeze(1).expand(B, N, prefix_len).reshape(B * N, prefix_len)
    context_f = idx_f_full[:, :prefix_len].unsqueeze(1).expand(B, N, prefix_len).reshape(B * N, prefix_len)

    time_expanded = {}
    for key in ("minute", "day", "month", "year"):
        t = batch["time"][key][:, :prefix_len]
        time_expanded[key] = t.unsqueeze(1).expand(B, N, prefix_len).reshape(B * N, prefix_len)

    all_step_returns = []

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

        probs_c = F.softmax(logits_c[:, -1, :].float() / temp, dim=-1)
        probs_f = F.softmax(logits_f[:, -1, :].float() / temp, dim=-1)
        pred_c = torch.multinomial(probs_c, num_samples=1)
        pred_f = torch.multinomial(probs_f, num_samples=1)

        decoded = tokenizer.decode(pred_c, pred_f)[..., 0].float()
        step_ret = decoded * stds.repeat_interleave(N, dim=0)[:, 0:1] + means.repeat_interleave(N, dim=0)[:, 0:1]
        all_step_returns.append(step_ret)

        context_c = torch.cat([context_c, pred_c], dim=1)
        context_f = torch.cat([context_f, pred_f], dim=1)

        for key in time_expanded:
            next_t = batch["time"][key][:, prefix_len + step:prefix_len + step + 1]
            next_t_expanded = next_t.unsqueeze(1).expand(B, N, 1).reshape(B * N, 1)
            time_expanded[key] = torch.cat([time_expanded[key], next_t_expanded], dim=1)

    step_returns_tensor = torch.cat(all_step_returns, dim=1).view(B, N, horizon)
    path_returns = torch.sum(step_returns_tensor, dim=2)

    # ── Oracle Filter: 3-factor scoring ──
    best_paths_c_list = []
    best_paths_f_list = []
    has_golden_list = []
    golden_count = 0

    context_c_reshaped = context_c.view(B, N, prefix_len + horizon)
    context_f_reshaped = context_f.view(B, N, prefix_len + horizon)

    mag_weight = float(getattr(cfg, "oracle_score_magnitude_weight", 0.3))

    for b in range(B):
        real_total = actual_path_return[b].item()
        path_ret_b = path_returns[b]

        is_correct_dir = (path_ret_b * real_total) > 0

        if is_correct_dir.any() and abs(real_total) > float(cfg.mape_eps):
            valid_indices = torch.where(is_correct_dir)[0]

            # Factor 1: path return error (lower is better)
            errors = torch.abs(path_ret_b[valid_indices] - real_total)

            # Factor 2: magnitude matching (penalize predictions too small)
            mag_penalty = torch.clamp(
                torch.abs(torch.tensor(real_total, device=path_ret_b.device))
                - torch.abs(path_ret_b[valid_indices]),
                min=0,
            )

            # Factor 3: step-level direction consistency bonus
            step_returns_valid = step_returns_tensor[b, valid_indices]
            actual_steps = actual_returns_h[b]
            step_dir_match = (step_returns_valid * actual_steps.unsqueeze(0)) > 0
            step_consistency = step_dir_match.float().mean(dim=1)
            consistency_bonus = step_consistency * torch.abs(torch.tensor(real_total, device=path_ret_b.device))

            # Combined score (lower is better)
            scores = (
                errors
                + float(cfg.oracle_magnitude_penalty) * mag_penalty
                - mag_weight * consistency_bonus
            )
            best_idx = valid_indices[scores.argmin()]

            best_paths_c_list.append(context_c_reshaped[b, best_idx])
            best_paths_f_list.append(context_f_reshaped[b, best_idx])
            has_golden_list.append(True)
            golden_count += 1
        else:
            # Fallback: use GT tokens (teacher forcing)
            best_paths_c_list.append(idx_c_full[b, :prefix_len + horizon])
            best_paths_f_list.append(idx_f_full[b, :prefix_len + horizon])
            has_golden_list.append(False)

    golden_c = torch.stack(best_paths_c_list).to(device)
    golden_f = torch.stack(best_paths_f_list).to(device)
    has_golden = torch.tensor(has_golden_list, device=device)

    if was_training:
        model.train()

    exploration_stats = {
        "golden_rate": golden_count / max(1, B),
        "mean_path_absmean": float(path_returns.abs().mean().item()),
        "mean_path_return": float(path_returns.mean().item()),
    }

    return golden_c, golden_f, has_golden, path_returns.view(B, N), exploration_stats

# ═══════════════════════════════════════════════════════════════════════
# Module 6: Phase 9 Training Step (Magnitude-Anchored + KL Anchor)
# ═══════════════════════════════════════════════════════════════════════

def train_star_cast_v5_step(model, reference_model, tokenizer, batch, cfg, device,
                             amp_enabled, amp_dtype):
    """Single Phase 9 STAR-CAST v5 training step.

    Phase 9 changes from Phase 8:
      1. Magnitude-anchored asymmetric loss (replaces Phase 8 loss)
      2. Dynamic timidity (replaces static timidity_weight)
      3. Magnitude floor penalty (new)
      4. KL anchor to frozen base model (new)
      5. Diagnostic logging for expected_returns quality

    Returns:
        loss: scalar total loss
        stats: dict with component losses and diagnostics
    """
    prefix_len = int(cfg.prefix_len)
    horizon = int(cfg.horizon)
    B = int(batch["features"].size(0))

    with torch.no_grad():
        idx_c_full, idx_f_full = _encode_features(tokenizer, batch["features"])

    actual_returns_h = batch["actual_returns"][:, :horizon]

    # ── Phase A: Noisy Exploration + Oracle Filter ──
    golden_c, golden_f, has_golden, _, explore_stats = _star_cast_exploration_v5(
        model=model, tokenizer=tokenizer, batch=batch, cfg=cfg,
        device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
    )

    # ── Phase B: Dual-Engine Update ──
    model.train()

    train_len = int(golden_c.size(1))
    train_time = {key: value[:, :train_len] for key, value in batch["time"].items()}

    with _autocast_context(device, amp_enabled, amp_dtype):
        logits_c, logits_f, latent_states, hidden = model(
            golden_c[:, :-1],
            golden_f[:, :-1],
            train_time["minute"][:, :train_len - 1],
            train_time["day"][:, :train_len - 1],
            train_time["month"][:, :train_len - 1],
            train_time["year"][:, :train_len - 1],
            return_hidden=True,
            neftune_alpha=0.0,
        )

    start = prefix_len - 1
    end = start + horizon
    rollout_c = logits_c[:, start:end, :]
    rollout_f = logits_f[:, start:end, :]

    # ═══════════════════════════════════════════════════════════════
    # Engine 1: Magnitude-Anchored Asymmetric Loss (Phase 9 NEW)
    # ═══════════════════════════════════════════════════════════════

    expected_traj = get_differentiable_expected_returns(
        tokenizer, rollout_c, rollout_f,
        batch["means"], batch["stds"],
        top_k=int(cfg.top_k_expected_return),
        sharpening_temp=float(cfg.prob_sharpening_temp),
    )

    # Diagnostic: expected vs actual magnitude ratio
    expected_absmean = float(expected_traj.abs().mean().item())
    actual_absmean = float(actual_returns_h.abs().mean().item())
    expected_actual_ratio = expected_absmean / max(actual_absmean, 1e-8)

    # Step-level loss with magnitude anchoring
    step_loss_matrix = compute_asymmetric_direction_loss_v2(
        expected_traj, actual_returns_h,
        alpha=float(cfg.asymmetric_alpha),
        beta=float(cfg.asymmetric_beta),
        gamma=float(cfg.magnitude_anchor_gamma),
        dynamic_timidity_alpha=float(cfg.dynamic_timidity_alpha),
        dynamic_timidity_gamma=float(cfg.dynamic_timidity_gamma),
        magnitude_floor=float(cfg.magnitude_floor),
        magnitude_floor_weight=float(cfg.magnitude_floor_weight),
    )
    step_asym_loss = step_loss_matrix.mean()

    # Path-level loss with magnitude anchoring
    expected_path = torch.cumsum(expected_traj, dim=1)
    actual_path = torch.cumsum(actual_returns_h, dim=1)
    path_loss_matrix = compute_asymmetric_direction_loss_v2(
        expected_path, actual_path,
        alpha=float(cfg.path_asymmetric_alpha),
        beta=float(cfg.path_asymmetric_beta),
        gamma=float(cfg.magnitude_anchor_gamma),
        dynamic_timidity_alpha=float(cfg.dynamic_timidity_alpha),
        dynamic_timidity_gamma=float(cfg.dynamic_timidity_gamma),
        magnitude_floor=float(cfg.magnitude_floor),
        magnitude_floor_weight=float(cfg.magnitude_floor_weight),
    )
    path_asym_loss = path_loss_matrix.mean()

    # ═══════════════════════════════════════════════════════════════
    # Engine 2: STaR CE Reinforcement
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
    # Engine 3: Direction-Explicit Classification
    # ═══════════════════════════════════════════════════════════════
    if float(getattr(cfg, "direction_weight", 0.0)) > 0.0:
        dir_labels = compute_direction_labels(
            actual_returns_h,
            epsilon_scale=float(cfg.direction_epsilon_scale),
        )
        dir_logits = model.compute_direction_logits_at_positions(
            hidden, latent_states, start=start, end=end,
        )
        if bool(cfg.direction_use_class_weights):
            flat_w = float(cfg.direction_ce_flat_weight)
            class_weights = torch.tensor(
                [1.0, flat_w, 1.0],
                device=device, dtype=dir_logits.dtype,
            )
            direction_loss = F.cross_entropy(
                dir_logits.reshape(-1, 3).float(),
                dir_labels.reshape(-1),
                weight=class_weights,
            )
        else:
            direction_loss = F.cross_entropy(
                dir_logits.reshape(-1, 3).float(),
                dir_labels.reshape(-1),
            )
        direction_acc = (dir_logits.argmax(dim=-1) == dir_labels).float().mean()
    else:
        direction_loss = torch.tensor(0.0, device=device)
        direction_acc = torch.tensor(0.0, device=device)

    # ═══════════════════════════════════════════════════════════════
    # Engine 4: KL Anchor to Base Model (Phase 9 NEW)
    # ═══════════════════════════════════════════════════════════════
    kl_step_weights = torch.ones(horizon, device=device, dtype=torch.float32)
    kl_loss = torch.tensor(0.0, device=device)
    if reference_model is not None and float(cfg.kl_weight) > 0.0:
        with torch.no_grad():
            with _autocast_context(device, amp_enabled, amp_dtype):
                ref_logits_c, ref_logits_f, _ = reference_model(
                    golden_c[:, :-1],
                    golden_f[:, :-1],
                    train_time["minute"][:, :train_len - 1],
                    train_time["day"][:, :train_len - 1],
                    train_time["month"][:, :train_len - 1],
                    train_time["year"][:, :train_len - 1],
                )
            ref_sel_c = ref_logits_c[:, start:end, :]
            ref_sel_f = ref_logits_f[:, start:end, :]

        # KL divergence on rollout region logits
        kl_loss = _kl_to_reference(rollout_c, rollout_f, ref_sel_c, ref_sel_f, kl_step_weights)

    # ═══════════════════════════════════════════════════════════════
    # Total Loss
    # ═══════════════════════════════════════════════════════════════
    total_loss = (
        float(cfg.step_asym_weight) * step_asym_loss +
        float(cfg.path_asym_weight) * path_asym_loss +
        float(cfg.star_ce_weight) * star_ce_loss +
        float(getattr(cfg, "direction_weight", 0.0)) * direction_loss +
        float(cfg.kl_weight) * kl_loss
    )

    golden_rate = has_golden.float().mean()

    # Direction product stats for monitoring zero-collapse
    direction_product = expected_traj * actual_returns_h
    direction_correct_ratio = (direction_product > 0).float().mean()

    return total_loss, {
        "total_loss": float(total_loss.detach().item()),
        "step_asym": float(step_asym_loss.detach().item()),
        "path_asym": float(path_asym_loss.detach().item()),
        "star_ce": float(star_ce_loss.detach().item()) if has_golden.any() else 0.0,
        "direction_loss": float(direction_loss.detach().item()),
        "direction_acc": float(direction_acc.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
        "golden_rate": float(golden_rate.item()),
        "expected_absmean": expected_absmean,
        "actual_absmean": actual_absmean,
        "expected_actual_ratio": expected_actual_ratio,
        "direction_correct_ratio": float(direction_correct_ratio.item()),
    }

# ═══════════════════════════════════════════════════════════════════════
# Module 7: Argument parsing & namespace
# ═══════════════════════════════════════════════════════════════════════

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 9: STAR-CAST v5 self-training")
    cfg = PostTrainStarCastV5Config

    parser.add_argument("--checkpoint-path", default=cfg.checkpoint_path)
    parser.add_argument("--output-dir", default=cfg.output_dir)
    parser.add_argument("--save-name", default=cfg.save_name)
    parser.add_argument("--save-epoch-checkpoints", type=_as_bool, default=cfg.save_epoch_checkpoints)
    parser.add_argument("--prefix-len", type=int, default=cfg.prefix_len)
    parser.add_argument("--horizon", type=int, default=cfg.horizon)
    parser.add_argument("--stride-ratio", type=float, default=cfg.stride_ratio)
    parser.add_argument("--cache-dir", default=cfg.cache_dir)
    parser.add_argument("--cache-rebuild", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=cfg.max_stocks)
    parser.add_argument("--max-train-samples", type=int, default=cfg.max_train_samples)
    parser.add_argument("--max-val-samples", type=int, default=cfg.max_val_samples)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=cfg.eval_batch_size)
    parser.add_argument("--accumulation-steps", type=int, default=cfg.accumulation_steps)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    parser.add_argument("--max-train-updates", type=int, default=cfg.max_train_updates)
    parser.add_argument("--progress-interval", type=int, default=cfg.progress_interval)
    parser.add_argument("--checkpoint-interval", type=int, default=cfg.checkpoint_interval)

    # STAR-CAST hyperparameters
    parser.add_argument("--neftune-alpha", type=float, default=cfg.neftune_alpha)
    parser.add_argument("--num-trajectories", type=int, default=cfg.num_trajectories)
    parser.add_argument("--exploration-temperature", type=float, default=cfg.exploration_temperature)
    parser.add_argument("--top-k-expected-return", type=int, default=cfg.top_k_expected_return)

    # Asymmetric loss
    parser.add_argument("--asymmetric-alpha", type=float, default=cfg.asymmetric_alpha)
    parser.add_argument("--asymmetric-beta", type=float, default=cfg.asymmetric_beta)
    parser.add_argument("--path-asymmetric-alpha", type=float, default=cfg.path_asymmetric_alpha)
    parser.add_argument("--path-asymmetric-beta", type=float, default=cfg.path_asymmetric_beta)
    parser.add_argument("--step-asym-weight", type=float, default=cfg.step_asym_weight)
    parser.add_argument("--path-asym-weight", type=float, default=cfg.path_asym_weight)
    parser.add_argument("--star-ce-weight", type=float, default=cfg.star_ce_weight)

    # Phase 9 NEW: Magnitude anchoring
    parser.add_argument("--magnitude-anchor-gamma", type=float, default=cfg.magnitude_anchor_gamma)
    parser.add_argument("--magnitude-floor", type=float, default=cfg.magnitude_floor)
    parser.add_argument("--magnitude-floor-weight", type=float, default=cfg.magnitude_floor_weight)

    # Phase 9 NEW: Dynamic timidity
    parser.add_argument("--dynamic-timidity-alpha", type=float, default=cfg.dynamic_timidity_alpha)
    parser.add_argument("--dynamic-timidity-gamma", type=float, default=cfg.dynamic_timidity_gamma)

    # Oracle
    parser.add_argument("--timidity-penalty-weight", type=float, default=cfg.timidity_penalty_weight)
    parser.add_argument("--timidity-ratio-threshold", type=float, default=cfg.timidity_ratio_threshold)
    parser.add_argument("--oracle-magnitude-penalty", type=float, default=cfg.oracle_magnitude_penalty)
    parser.add_argument("--prob-sharpening-temp", type=float, default=cfg.prob_sharpening_temp)
    parser.add_argument("--actionable-da-threshold", type=float, default=cfg.actionable_da_threshold)
    parser.add_argument("--oracle-score-magnitude-weight", type=float, default=cfg.oracle_score_magnitude_weight)

    # Direction classification
    parser.add_argument("--direction-weight", type=float, default=cfg.direction_weight)
    parser.add_argument("--direction-epsilon-scale", type=float, default=cfg.direction_epsilon_scale)
    parser.add_argument("--direction-ce-flat-weight", type=float, default=cfg.direction_ce_flat_weight)
    parser.add_argument("--direction-use-class-weights", type=_as_bool, default=cfg.direction_use_class_weights)

    # Phase 9 NEW: KL anchor
    parser.add_argument("--kl-weight", type=float, default=cfg.kl_weight)

    # Optimisation
    parser.add_argument("--freeze-backbone", type=_as_bool, default=cfg.freeze_backbone)
    parser.add_argument("--trainable-scope", choices=["all", "heads"], default=cfg.trainable_scope)
    parser.add_argument("--use-gradient-checkpointing", type=_as_bool, default=cfg.use_gradient_checkpointing)
    parser.add_argument("--use-amp", type=_as_bool, default=cfg.use_amp)
    parser.add_argument("--amp-dtype", default=cfg.amp_dtype)
    parser.add_argument("--use-tf32", type=_as_bool, default=cfg.use_tf32)
    parser.add_argument("--mape-eps", type=float, default=cfg.mape_eps)
    parser.add_argument("--deterministic", type=_as_bool, default=cfg.deterministic)
    parser.add_argument("--seed", type=int, default=cfg.random_seed)

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
        magnitude_anchor_gamma=float(args.magnitude_anchor_gamma),
        magnitude_floor=float(args.magnitude_floor),
        magnitude_floor_weight=float(args.magnitude_floor_weight),
        dynamic_timidity_alpha=float(args.dynamic_timidity_alpha),
        dynamic_timidity_gamma=float(args.dynamic_timidity_gamma),
        timidity_penalty_weight=float(args.timidity_penalty_weight),
        timidity_ratio_threshold=float(args.timidity_ratio_threshold),
        oracle_magnitude_penalty=float(args.oracle_magnitude_penalty),
        prob_sharpening_temp=float(args.prob_sharpening_temp),
        actionable_da_threshold=float(args.actionable_da_threshold),
        oracle_score_magnitude_weight=float(args.oracle_score_magnitude_weight),
        direction_weight=float(args.direction_weight),
        direction_epsilon_scale=float(args.direction_epsilon_scale),
        direction_ce_flat_weight=float(args.direction_ce_flat_weight),
        direction_use_class_weights=bool(args.direction_use_class_weights),
        kl_weight=float(args.kl_weight),
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
# Module 8: Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train(cfg):
    """Full Phase 9 STAR-CAST v5 training loop.

    Phase 9 engineering fixes:
      - KL anchor to frozen base model
      - Complete logging of all loss components
      - Unified config (no HPO/training mismatch)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(int(cfg.random_seed), deterministic=bool(cfg.deterministic))

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.use_tf32)

    # ── Load model + tokenizer (matches Phase 8 pattern) ──
    model, tokenizer = load_model(
        device=device, checkpoint_path=cfg.checkpoint_path,
        strict_checkpoint_compat=False,
    )
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    model.to(device)
    if bool(cfg.use_gradient_checkpointing):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    param_groups = _configure_trainable(model, cfg)
    # ── Phase 9 NEW: Create frozen reference model for KL anchor ──
    reference_model = None
    if float(cfg.kl_weight) > 0.0:
        reference_model = copy.deepcopy(model).to(device)
        reference_model.eval()
        reference_model.requires_grad_(False)
        print(f"KL anchor enabled: kl_weight={cfg.kl_weight}")

    # ── Datasets ──
    train_ds = RolloutWindowDataset(
        "train", cfg,
        max_samples=int(cfg.max_train_samples),
        seed=int(cfg.random_seed),
    )
    val_ds = RolloutWindowDataset(
        "val", cfg,
        max_samples=int(cfg.max_val_samples),
        seed=int(cfg.random_seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.batch_size), shuffle=True,
        num_workers=int(cfg.num_workers), collate_fn=rollout_collate,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.eval_batch_size), shuffle=False,
        num_workers=int(cfg.num_workers), collate_fn=rollout_collate,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Phase 9 STAR-CAST v5: train={len(train_ds)}, val={len(val_ds)}, "
          f"N={cfg.num_trajectories}, top_k={cfg.top_k_expected_return}")
    print(f"Magnitude anchoring: gamma={cfg.magnitude_anchor_gamma}, "
          f"floor={cfg.magnitude_floor}, floor_weight={cfg.magnitude_floor_weight}")
    print(f"Dynamic timidity: alpha={cfg.dynamic_timidity_alpha}, "
          f"gamma={cfg.dynamic_timidity_gamma}")
    print(f"KL anchor: weight={cfg.kl_weight}")

    optimizer, optimizer_kwargs = _build_optimizer(param_groups, cfg, device)
    total_steps = max(1, ((len(train_ds) // max(1, int(cfg.batch_size) * int(cfg.accumulation_steps))) * int(cfg.epochs)))
    warmup = max(1, total_steps // 10)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup),
        eta_min=float(cfg.learning_rate) * 0.05,
    )

    amp_enabled = bool(cfg.use_amp) and device.type == "cuda"
    amp_dtype = _amp_dtype(cfg.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

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
            "star_ce": 0.0, "direction_loss": 0.0, "direction_acc": 0.0,
            "kl_loss": 0.0, "golden_rate": 0.0,
            "expected_absmean": 0.0, "actual_absmean": 0.0,
            "expected_actual_ratio": 0.0, "direction_correct_ratio": 0.0,
        }
        batches = 0
        pbar = tqdm(train_loader, desc=f"STAR-CAST-v5 epoch {epoch + 1}/{cfg.epochs}")

        for batch_idx, raw_batch in enumerate(pbar, start=1):
            batch = _move_batch(raw_batch, device)

            loss, stats = train_star_cast_v5_step(
                model=model, reference_model=reference_model,
                tokenizer=tokenizer, batch=batch, cfg=cfg,
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

                ci = int(getattr(cfg, "checkpoint_interval", 0))
                if ci > 0 and updates % ci == 0:
                    step_path = os.path.join(
                        cfg.output_dir,
                        f"{os.path.splitext(cfg.save_name)[0]}-step{updates}.pt",
                    )
                    _save_v5_checkpoint(step_path, model, tokenizer, cfg, {"updates": updates}, history)

            for key in epoch_totals:
                epoch_totals[key] += float(stats.get(key, 0.0))
            batches += 1

            if batch_idx % int(cfg.progress_interval) == 0:
                pbar.set_postfix({
                    "loss": f"{stats['total_loss']:.4f}",
                    "golden": f"{stats['golden_rate']:.2f}",
                    "dir_acc": f"{stats.get('direction_acc', 0):.2f}",
                    "exp/act": f"{stats.get('expected_actual_ratio', 0):.2f}",
                    "kl": f"{stats.get('kl_loss', 0):.4f}",
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
            _save_v5_checkpoint(epoch_path, model, tokenizer, cfg, val_metrics, history)

        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            _save_v5_checkpoint(best_path, model, tokenizer, cfg, val_metrics, history)

        if int(cfg.max_train_updates) > 0 and updates >= int(cfg.max_train_updates):
            break

    history_path = os.path.join(cfg.output_dir, "star_cast_v5_history.json")
    _write_history(history_path, cfg, history, best_metrics)
    print(f"Best Phase 9 checkpoint: {best_path}")
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