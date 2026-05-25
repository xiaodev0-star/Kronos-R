# -*- coding: utf-8 -*-
"""Phase 5 DA — Loss functions for all post-training methods.

Old methods (token-space preference optimization):
  T1. CE   — standard next-token cross-entropy (baseline)
  T2. ExPO — ExPO regression on token log-probabilities
  T3. DPO  — DPO sigmoid loss on token log-probabilities
  T4. RSFT — rejection-sampled best token → CE

New method (financial GRPO):
  T5. GRPO — Group Relative Policy Optimization for financial alignment
             - On-policy group sampling (K=16 from current policy)
             - Direction reward + magnitude penalty
             - Within-group advantage normalization
             - PPO-style clipped policy gradient + KL constraint
"""

import torch
import torch.nn.functional as F

from hpo.phase5.core import (
    IGNORE_INDEX, LABEL_UP, LABEL_DOWN,
    _ac, prepare_inputs,
    sample_tokens, token_returns, token_direction,
    candidate_logp, build_winner_loser,
)


# ═══════════════════════════════════════════════
# T1: CE — standard next-token cross-entropy
# ═══════════════════════════════════════════════

def loss_ce(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = prepare_inputs(batch)
    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    target_c = batch["idx_c_full"][:, -1]
    target_f = batch["idx_f_full"][:, -1]
    return (F.cross_entropy(logits_c[:, -1, :].float(), target_c)
            + F.cross_entropy(logits_f[:, -1, :].float(), target_f))


# ═══════════════════════════════════════════════
# T2: ExPO — regression on token log-probabilities
# ═══════════════════════════════════════════════

def loss_expo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = prepare_inputs(batch)

    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_last_c = logits_c[:, -1, :]
    pi_last_f = logits_f[:, -1, :]

    with torch.no_grad():
        with _ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_last_c = ref_lc[:, -1, :]
        ref_last_f = ref_lf[:, -1, :]

    w_c, w_f, l_c, l_f, valid_pair, w_dir, l_dir = build_winner_loser(
        tokenizer, ref_last_c, ref_last_f, batch, cfg)

    theta_win = candidate_logp(pi_last_c, pi_last_f, w_c, w_f)
    theta_lose = candidate_logp(pi_last_c, pi_last_f, l_c, l_f)
    with torch.no_grad():
        ref_win = candidate_logp(ref_last_c, ref_last_f, w_c, w_f)
        ref_lose = candidate_logp(ref_last_c, ref_last_f, l_c, l_f)
        ref_pref = torch.sigmoid(ref_win - ref_lose)

    theta_pref = torch.sigmoid(theta_win - theta_lose)
    lam = max(0.0, min(1.0, cfg.get("expo_reference_weight", 0.6)))
    target_pref = (lam * ref_pref + (1.0 - lam)).clamp(0.0, 1.0)

    per_row = (theta_pref - target_pref).pow(2)
    vw = valid_pair.to(dtype=per_row.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (per_row * vw).sum() / vw.sum().clamp_min(1.0)


# ═══════════════════════════════════════════════
# T3: DPO — sigmoid loss on token log-probabilities
# ═══════════════════════════════════════════════

def loss_dpo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = prepare_inputs(batch)

    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_last_c = logits_c[:, -1, :]
    pi_last_f = logits_f[:, -1, :]

    with torch.no_grad():
        with _ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_last_c = ref_lc[:, -1, :]
        ref_last_f = ref_lf[:, -1, :]

    w_c, w_f, l_c, l_f, valid_pair, _, _ = build_winner_loser(
        tokenizer, ref_last_c, ref_last_f, batch, cfg)

    pi_win = candidate_logp(pi_last_c, pi_last_f, w_c, w_f)
    pi_lose = candidate_logp(pi_last_c, pi_last_f, l_c, l_f)
    with torch.no_grad():
        ref_win = candidate_logp(ref_last_c, ref_last_f, w_c, w_f)
        ref_lose = candidate_logp(ref_last_c, ref_last_f, l_c, l_f)

    log_ratio = (pi_win - pi_lose) - (ref_win - ref_lose)
    per_row = -F.logsigmoid(cfg.get("dpo_beta", 0.5) * log_ratio)
    vw = valid_pair.to(dtype=per_row.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (per_row * vw).sum() / vw.sum().clamp_min(1.0)


# ═══════════════════════════════════════════════
# T4: RSFT — rejection-sampled best token → CE
# ═══════════════════════════════════════════════

def loss_rsft(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = prepare_inputs(batch)

    with torch.no_grad():
        with _ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_last_c = ref_lc[:, -1, :]
        ref_last_f = ref_lf[:, -1, :]

    B = ref_last_c.size(0)
    sampled_c = sample_tokens(ref_last_c, cfg["temperature"], cfg["num_candidates"])
    sampled_f = sample_tokens(ref_last_f, cfg["temperature"], cfg["num_candidates"])

    if cfg["include_gold"]:
        gold_c = batch["idx_c_full"][:, -1]
        gold_f = batch["idx_f_full"][:, -1]
        sampled_c = torch.cat([sampled_c, gold_c.long().unsqueeze(1)], dim=1)
        sampled_f = torch.cat([sampled_f, gold_f.long().unsqueeze(1)], dim=1)

    cand_dir = token_direction(tokenizer, sampled_c, sampled_f,
                                batch["prompt_means"], batch["prompt_stds"])
    cand_ret = token_returns(tokenizer, sampled_c, sampled_f,
                              batch["prompt_means"], batch["prompt_stds"])

    valid_label = batch["loss_labels"] != IGNORE_INDEX
    real = batch["real_returns"].to(device=cand_ret.device, dtype=cand_ret.dtype)
    err_scale = torch.maximum(real.abs(), torch.tensor(1e-6, device=cand_ret.device))
    while real.ndim < cand_ret.ndim:
        real = real.unsqueeze(-1); err_scale = err_scale.unsqueeze(-1)
    norm_err = torch.nan_to_num((cand_ret - real).abs() / err_scale, nan=1e6, posinf=1e6)

    labels_2d = batch["labels"].unsqueeze(1)
    dir_correct = cand_dir.eq(labels_2d) & valid_label.unsqueeze(1)
    scores = (dir_correct.to(dtype=cand_ret.dtype) * cfg["direction_bonus"]
              - norm_err * cfg["error_weight"])
    scores = scores.masked_fill(~valid_label.unsqueeze(1), float("-inf"))

    _, best_idx = scores.max(dim=1)
    rows = torch.arange(B, device=sampled_c.device)
    best_c = sampled_c[rows, best_idx]
    best_f = sampled_f[rows, best_idx]
    best_valid = valid_label & scores.max(dim=1).values.isfinite()

    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)

    if best_valid.any():
        return (F.cross_entropy(logits_c[best_valid, -1, :].float(), best_c[best_valid].long())
                + F.cross_entropy(logits_f[best_valid, -1, :].float(), best_f[best_valid].long()))
    return torch.tensor(0.0, device=device, requires_grad=True)


# ═══════════════════════════════════════════════
# T5: GRPO — Group Relative Policy Optimization
# ═══════════════════════════════════════════════

def loss_grpo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    """Financial GRPO: group sampling from current policy + advantage-normalized PG.

    Key differences from ExPO/DPO:
      - On-policy: samples from CURRENT policy (not reference), enabling true RL
      - Group advantage: normalizes rewards WITHIN each sample's K candidates
      - Direction + magnitude reward: handles flat samples naturally (no ignore needed)
      - PPO clipping + KL to reference: stable training, prevents distribution collapse
    """
    K = cfg.get("grpo_group_size", 16)
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = prepare_inputs(batch)

    # ── Policy forward ──
    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_c = logits_c[:, -1, :]  # [B, V]
    pi_f = logits_f[:, -1, :]  # [B, V]
    B = pi_c.size(0)

    # ── Group sampling from CURRENT policy (on-policy) ──
    grpo_temp = cfg.get("grpo_temperature", 1.2)
    sampled_c = sample_tokens(pi_c, grpo_temp, K)  # [B, K]
    sampled_f = sample_tokens(pi_f, grpo_temp, K)  # [B, K]

    # ── Decode all candidates ──
    cand_ret = token_returns(tokenizer, sampled_c, sampled_f,
                              batch["prompt_means"], batch["prompt_stds"])  # [B, K]

    # ── Compute multi-objective rewards (no reference model needed for rewards) ──
    real = batch["real_returns"].to(device=device, dtype=cand_ret.dtype)
    eps_ret = torch.tensor(1e-6, device=device, dtype=cand_ret.dtype)

    # Direction reward: +1 for correct sign, -1 for wrong
    true_dir = torch.zeros_like(real)
    true_dir[real > eps_ret] = 1.0
    true_dir[real < -eps_ret] = -1.0

    pred_dir = torch.zeros_like(cand_ret)
    pred_dir[cand_ret > eps_ret] = 1.0
    pred_dir[cand_ret < -eps_ret] = -1.0

    # Broadcast to [B, K]
    td = true_dir.unsqueeze(1).expand(B, K)
    r_dir = torch.where(pred_dir == td, torch.ones_like(pred_dir),
                        -torch.ones_like(pred_dir))

    # Flat compensation: for real-flat samples, reward proximity to zero
    flat_mask = (true_dir == 0.0).unsqueeze(1).expand(B, K)
    r_dir[flat_mask] = 1.0 - torch.abs(cand_ret[flat_mask]).clamp(max=2.0)

    # Magnitude penalty: normalized absolute error
    abs_err = torch.abs(cand_ret - real.unsqueeze(1))
    r_mag = -torch.clamp(abs_err / (torch.abs(real.unsqueeze(1)) + 1e-6), max=5.0)

    # Composite reward
    rewards = 1.0 * r_dir + 0.3 * r_mag  # [B, K]

    # Mask invalid samples (flat samples kept! GRPO handles them naturally)
    valid_label = batch["loss_labels"] != IGNORE_INDEX
    # But still compute advantage even for flat — the reward structure handles it
    # Only fully mask where we have NaN/Inf returns
    rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Group advantage normalization ──
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True).clamp_min(1e-6)
    advantages = (rewards - mean_r) / std_r  # [B, K]

    # ── Policy log-probabilities for sampled tokens ──
    logp_c = F.log_softmax(pi_c.float(), dim=-1)  # [B, V]
    logp_f = F.log_softmax(pi_f.float(), dim=-1)  # [B, V]
    pi_logp = logp_c.gather(1, sampled_c) + logp_f.gather(1, sampled_f)  # [B, K]

    # ── Reference log-probabilities (for KL and "old" policy in clipping) ──
    with torch.no_grad():
        with _ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_c = ref_lc[:, -1, :]
        ref_f = ref_lf[:, -1, :]
        ref_logp_c = F.log_softmax(ref_c.float(), dim=-1)
        ref_logp_f = F.log_softmax(ref_f.float(), dim=-1)
        ref_logp = ref_logp_c.gather(1, sampled_c) + ref_logp_f.gather(1, sampled_f)  # [B, K]
        old_logp = ref_logp.detach()  # use ref as proxy for old policy

    # ── PPO-style clipped policy loss ──
    ratio = torch.exp(pi_logp - old_logp)
    clip_eps = cfg.get("grpo_clip_eps", 0.2)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # ── KL penalty ──
    kl = (pi_logp - ref_logp).mean()

    kl_weight = cfg.get("grpo_kl_weight", 0.02)

    # Mask for valid samples (flat are still valid for GRPO)
    valid_mask = valid_label.unsqueeze(1).float()
    total_valid = valid_mask.sum().clamp_min(1.0)

    grpo_loss = (policy_loss * valid_mask).sum() / total_valid + kl_weight * kl

    return grpo_loss


# ═══════════════════════════════════════════════
# Method registry
# ═══════════════════════════════════════════════

METHOD_REGISTRY = {
    "ce":   {"name": "CE (Baseline)",         "loss_fn": loss_ce,   "needs_ref": False},
    "expo": {"name": "ExPO",                   "loss_fn": loss_expo, "needs_ref": True},
    "dpo":  {"name": "DPO",                    "loss_fn": loss_dpo,  "needs_ref": True},
    "rsft": {"name": "RSFT",                   "loss_fn": loss_rsft, "needs_ref": True},
    "grpo": {"name": "GRPO (Group Relative PO)","loss_fn": loss_grpo, "needs_ref": True},
}
