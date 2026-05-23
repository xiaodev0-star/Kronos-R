# -*- coding: utf-8 -*-
"""Phase 5 Sup — Full HPO with walk-forward validation and deep method adaptations.

Replaces the smoke-test runner (phase5_sup_train.py) with:
  - Optuna TPE + MedianPruner HPO (15 trials × 600 updates per method)
  - Walk-forward Val split (4 segments, score = mean − 0.5×std of slice ΔBalAcc)
  - Per-method hyperparameter search spaces tailored to each algorithm
  - Final best-config evaluation (1200 updates) + Demo inference

Methods: Conservative ExPO, Robust DPO, KTO, GRPO, SimPO, DAPO, Verifier Reranker.

Usage::

    python -m hpo.phase5_sup_hpo                     # full HPO
    python -m hpo.phase5_sup_hpo --method grpo        # single method
    python -m hpo.phase5_sup_hpo --eval-only          # final eval only
"""

from __future__ import annotations

import copy, json, os, sys, time, warnings
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

import hpo.phase5_da as p5d
import hpo.phase5_hpo as p5hpo
import hpo.phase5_sup_eval as p5e
from reproducibility import set_global_seed

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR = os.path.join(_PROJECT_ROOT, "trials", "phase5_sup")
HPO_DIR = os.path.join(OUT_DIR, "hpo")
os.makedirs(HPO_DIR, exist_ok=True)

# ═══════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════

HPO_TRIALS = 15
HPO_UPDATES = 600      # updates per trial
FINAL_UPDATES = 1200   # updates for best-config final run
BATCH_SIZE = 8

DEFAULT_CFG = {
    "batch_size": BATCH_SIZE, "weight_decay": 1e-4, "grad_clip": 1.0,
    "num_candidates": 128, "temperature": 1.0,
    "direction_bonus": 1.0, "error_weight": 0.25,
    "score_margin": 0.05, "include_gold": True,
    "label_mode": "global_median", "epsilon_scale": 0.5, "flat_policy": "ignore",
}


# ═══════════════════════════════════════════════
# Walk-forward val
# ═══════════════════════════════════════════════

def _build_wf_loaders(n_splits=4):
    """Build train loader + list of (val_loader, val_ds) for walk-forward segments."""
    train_payload = torch.load(os.path.join(_PROJECT_ROOT, "dataset_train.pt"),
                               map_location="cpu", weights_only=False)
    val_payload = torch.load(os.path.join(_PROJECT_ROOT, "dataset_val.pt"),
                             map_location="cpu", weights_only=False)
    train_returns = p5d._denorm_last_returns(train_payload)
    val_returns = p5d._denorm_last_returns(val_payload)
    train_abs = np.abs(train_returns); train_abs = train_abs[np.isfinite(train_abs)]
    eps = max(1e-5, float(np.median(train_abs)) * DEFAULT_CFG["epsilon_scale"])

    train_ds = p5d.DirectionDataset(train_payload, np.arange(len(train_returns), dtype=np.int64),
                                     train_returns, eps, DEFAULT_CFG["flat_policy"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=p5d.collate_fn, drop_last=True)

    # Walk-forward val splits
    wf_splits = p5e.build_walkforward_val_splits(
        os.path.join(_PROJECT_ROOT, "dataset_val.pt"), n_splits=n_splits)

    val_loaders = []
    for indices, date_range in wf_splits:
        val_ds = p5d.DirectionDataset(val_payload, indices, val_returns, eps, "class")
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=p5d.collate_fn)
        val_loaders.append((val_loader, val_ds, date_range))

    return train_loader, val_loaders, eps


def _wf_score(model, tokenizer, val_loaders, device, base_metrics=None):
    """Walk-forward score: mean(slice ΔBalAcc) − 0.5×std(slice ΔBalAcc)."""
    slice_scores = []
    for loader, ds, dr in val_loaders:
        metrics = p5d.evaluate(model, tokenizer, loader, device, False, None)
        ba = metrics.get("balanced_accuracy", 0.0)
        if base_metrics is not None:
            ba_base = base_metrics.get("balanced_accuracy", 0.5)
            slice_scores.append(ba - ba_base)
        else:
            slice_scores.append(ba)

    arr = np.array(slice_scores)
    return float(np.mean(arr) - 0.5 * np.std(arr))


# ═══════════════════════════════════════════════
# Model builder
# ═══════════════════════════════════════════════

def _build_model(device, use_lora=False, lora_rank=8, lora_alpha=16):
    if use_lora:
        return p5hpo.build_trainable_model_hpo(device, lora_rank, lora_alpha, use_lora=True)
    else:
        model = p5hpo.build_trainable_model_hpo(device, lora_rank, lora_alpha, use_lora=False)
        model.enable_gradient_checkpointing(True)
        return model


# ═══════════════════════════════════════════════
# Loss functions (improved, vectorized, well-adapted)
# ═══════════════════════════════════════════════

def _shared_forward(model, ref_model, batch, device, amp_enabled, amp_dtype):
    """Shared forward pass: returns policy + reference logits at last step."""
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = p5d._prepare_inputs(batch)
    with p5d._ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_c, pi_f = logits_c[:, -1, :], logits_f[:, -1, :]

    ref_c = ref_f = None
    if ref_model is not None:
        with torch.no_grad():
            with p5d._ac(device, amp_enabled, amp_dtype):
                ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
            ref_c, ref_f = ref_lc[:, -1, :], ref_lf[:, -1, :]

    return pi_c, pi_f, ref_c, ref_f, idx_c, idx_f, t_min, t_day, t_mon, t_yr


# ── Conservative ExPO ──

def _loss_conservative_expo_v2(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    pi_c, pi_f, ref_c, ref_f, *_ = _shared_forward(model, ref_model, batch, device, amp_enabled, amp_dtype)

    w_c, w_f, l_c, l_f, valid_pair, _, _ = p5d._build_winner_loser(tokenizer, ref_c, ref_f, batch, cfg)

    theta_win = p5d._candidate_logp(pi_c, pi_f, w_c, w_f)
    theta_lose = p5d._candidate_logp(pi_c, pi_f, l_c, l_f)
    with torch.no_grad():
        ref_win = p5d._candidate_logp(ref_c, ref_f, w_c, w_f)
        ref_lose = p5d._candidate_logp(ref_c, ref_f, l_c, l_f)
        ref_pref = torch.sigmoid(ref_win - ref_lose)

    theta_pref = torch.sigmoid(theta_win - theta_lose)
    lam = cfg.get("expo_reference_weight", 0.7)
    target_pref = (lam * ref_pref + (1.0 - lam)).clamp(0.0, 1.0)
    expo_loss = (theta_pref - target_pref).pow(2)
    vw = valid_pair.to(dtype=expo_loss.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    expo_loss = (expo_loss * vw).sum() / vw.sum().clamp_min(1.0)

    # Token CE anchor
    target_c = batch["idx_c_full"][:, -1]
    target_f = batch["idx_f_full"][:, -1]
    ce_weight = cfg.get("token_ce_weight", 0.05)
    ce_loss = F.cross_entropy(pi_c.float(), target_c) + F.cross_entropy(pi_f.float(), target_f)
    return expo_loss + ce_weight * ce_loss


# ── Robust DPO ──

def _loss_robust_dpo_v2(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    pi_c, pi_f, ref_c, ref_f, *_ = _shared_forward(model, ref_model, batch, device, amp_enabled, amp_dtype)

    w_c, w_f, l_c, l_f, valid_pair, _, _ = p5d._build_winner_loser(tokenizer, ref_c, ref_f, batch, cfg)

    pi_win = p5d._candidate_logp(pi_c, pi_f, w_c, w_f)
    pi_lose = p5d._candidate_logp(pi_c, pi_f, l_c, l_f)
    with torch.no_grad():
        ref_win = p5d._candidate_logp(ref_c, ref_f, w_c, w_f)
        ref_lose = p5d._candidate_logp(ref_c, ref_f, l_c, l_f)

    beta = cfg.get("dpo_beta", 0.5)
    log_ratio = (pi_win - pi_lose) - (ref_win - ref_lose)
    dpo_loss = -F.logsigmoid(beta * log_ratio)
    vw = valid_pair.to(dtype=dpo_loss.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    dpo_loss = (dpo_loss * vw).sum() / vw.sum().clamp_min(1.0)

    # KL regularization
    kl_weight = cfg.get("kl_weight", 0.02)
    kl_c = F.kl_div(F.log_softmax(pi_c.float(), dim=-1),
                    F.softmax(ref_c.float(), dim=-1), reduction="batchmean")
    kl_f = F.kl_div(F.log_softmax(pi_f.float(), dim=-1),
                    F.softmax(ref_f.float(), dim=-1), reduction="batchmean")
    return dpo_loss + kl_weight * (kl_c + kl_f)


# ── KTO (vectorized) ──

def _loss_kto_v2(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    pi_c, pi_f, ref_c, ref_f, *_ = _shared_forward(model, ref_model, batch, device, amp_enabled, amp_dtype)
    B = pi_c.size(0)
    K = min(cfg.get("num_candidates", 64), 64)

    # Sample candidates from reference
    sampled_c = p5d._sample_tokens(ref_c, cfg["temperature"], K)  # [B, K]
    sampled_f = p5d._sample_tokens(ref_f, cfg["temperature"], K)
    if cfg["include_gold"]:
        gold_c = batch["idx_c_full"][:, -1]; gold_f = batch["idx_f_full"][:, -1]
        sampled_c = torch.cat([sampled_c, gold_c.long().unsqueeze(1)], dim=1)
        sampled_f = torch.cat([sampled_f, gold_f.long().unsqueeze(1)], dim=1)
        K += 1

    # Decode
    cand_ret = p5d._token_returns(tokenizer, sampled_c, sampled_f,
                                   batch["prompt_means"], batch["prompt_stds"])
    cand_dir = torch.where(cand_ret > 0,
                           torch.tensor(p5d.LABEL_UP, device=device, dtype=torch.long),
                           torch.tensor(p5d.LABEL_DOWN, device=device, dtype=torch.long))

    valid_label = batch["loss_labels"] != p5d.IGNORE_INDEX
    dir_correct = cand_dir.eq(batch["labels"].unsqueeze(1)).float()

    real = batch["real_returns"].to(device=device, dtype=cand_ret.dtype)
    err_scale = torch.maximum(real.abs(), torch.tensor(1e-6, device=device))
    while real.ndim < cand_ret.ndim:
        real = real.unsqueeze(-1); err_scale = err_scale.unsqueeze(-1)
    norm_err = torch.nan_to_num((cand_ret - real).abs() / err_scale, nan=1e6, posinf=1e6)

    # Binary utility
    median_err = norm_err.median()
    desirable = dir_correct.bool() & (norm_err <= median_err)
    undesirable = (~dir_correct.bool()) | (norm_err > 2 * median_err)

    # Token logp for all candidates (vectorized via gather)
    logp_c = F.log_softmax(pi_c.float(), dim=-1)  # [B, V]
    logp_f = F.log_softmax(pi_f.float(), dim=-1)
    pi_logp = (logp_c.gather(1, sampled_c) + logp_f.gather(1, sampled_f))  # [B, K]

    ref_logp_c = F.log_softmax(ref_c.float(), dim=-1)
    ref_logp_f = F.log_softmax(ref_f.float(), dim=-1)
    ref_logp = (ref_logp_c.gather(1, sampled_c) + ref_logp_f.gather(1, sampled_f))

    kto_beta = cfg.get("kto_beta", 0.1)
    lambda_d = cfg.get("kto_lambda_d", 1.0)
    lambda_u = cfg.get("kto_lambda_u", 1.0)

    des_loss = -F.logsigmoid(kto_beta * (pi_logp - ref_logp))  # [B, K]
    undes_loss = -F.logsigmoid(kto_beta * (ref_logp - pi_logp))  # [B, K]

    des_mask = desirable.float() * valid_label.unsqueeze(1).float()
    undes_mask = undesirable.float() * valid_label.unsqueeze(1).float()

    n_des = des_mask.sum().clamp_min(1)
    n_undes = undes_mask.sum().clamp_min(1)

    total_loss = (lambda_d * (des_loss * des_mask).sum() / n_des
                  + lambda_u * (undes_loss * undes_mask).sum() / n_undes)
    return total_loss


# ── GRPO ──

def _loss_grpo_v2(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    pi_c, pi_f, ref_c, ref_f, *_ = _shared_forward(model, ref_model, batch, device, amp_enabled, amp_dtype)
    B = pi_c.size(0)
    K = cfg.get("grpo_K", 16)
    temp = cfg.get("grpo_temperature", 1.2)

    # Sample from current policy (on-policy)
    sampled_c = p5d._sample_tokens(pi_c, temp, K)
    sampled_f = p5d._sample_tokens(pi_f, temp, K)

    cand_ret = p5d._token_returns(tokenizer, sampled_c, sampled_f,
                                   batch["prompt_means"], batch["prompt_stds"])
    cand_dir = torch.where(cand_ret > 0,
                           torch.tensor(p5d.LABEL_UP, device=device, dtype=torch.long),
                           torch.tensor(p5d.LABEL_DOWN, device=device, dtype=torch.long))

    valid_label = batch["loss_labels"] != p5d.IGNORE_INDEX
    dir_correct = cand_dir.eq(batch["labels"].unsqueeze(1)).float()

    real = batch["real_returns"].to(device=device, dtype=cand_ret.dtype)
    err_scale = torch.maximum(real.abs(), torch.tensor(1e-6, device=device))
    while real.ndim < cand_ret.ndim:
        real = real.unsqueeze(-1); err_scale = err_scale.unsqueeze(-1)
    norm_err = torch.nan_to_num((cand_ret - real).abs() / err_scale, nan=1e6, posinf=1e6)

    rewards = dir_correct - 0.15 * norm_err  # [B, K]
    rewards = rewards.masked_fill(~valid_label.unsqueeze(1), 0.0)

    # Group advantage
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True).clamp_min(1e-6)
    advantages = (rewards - mean_r) / std_r

    # Policy logp
    logp_c = F.log_softmax(pi_c.float(), dim=-1)
    logp_f = F.log_softmax(pi_f.float(), dim=-1)
    pi_logp = logp_c.gather(1, sampled_c) + logp_f.gather(1, sampled_f)

    # Reference logp (for "old" policy in PPO clip + KL)
    ref_logp_c = F.log_softmax(ref_c.float(), dim=-1)
    ref_logp_f = F.log_softmax(ref_f.float(), dim=-1)
    ref_logp = ref_logp_c.gather(1, sampled_c) + ref_logp_f.gather(1, sampled_f)

    ratio = torch.exp(pi_logp - ref_logp.detach())
    clip_eps = cfg.get("grpo_clip_eps", 0.2)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages)

    kl_weight = cfg.get("grpo_kl_weight", 0.02)
    kl = (pi_logp - ref_logp).mean()

    valid_mask = valid_label.unsqueeze(1).float()
    total_valid = valid_mask.sum().clamp_min(1)
    return (policy_loss * valid_mask).sum() / total_valid + kl_weight * kl


# ── SimPO (reference-free) ──

def _loss_simpo_v2(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    pi_c, pi_f, ref_c, ref_f, *_ = _shared_forward(model, ref_model, batch, device, amp_enabled, amp_dtype)
    B = pi_c.size(0)
    K = min(cfg.get("num_candidates", 32), 32)
    gamma = cfg.get("simpo_gamma", 0.5)

    sampled_c = p5d._sample_tokens(pi_c, 1.0, K)
    sampled_f = p5d._sample_tokens(pi_f, 1.0, K)

    cand_ret = p5d._token_returns(tokenizer, sampled_c, sampled_f,
                                   batch["prompt_means"], batch["prompt_stds"])
    cand_dir = torch.where(cand_ret > 0,
                           torch.tensor(p5d.LABEL_UP, device=device, dtype=torch.long),
                           torch.tensor(p5d.LABEL_DOWN, device=device, dtype=torch.long))

    valid_label = batch["loss_labels"] != p5d.IGNORE_INDEX
    dir_correct = cand_dir.eq(batch["labels"].unsqueeze(1)).float()

    real = batch["real_returns"].to(device=device, dtype=cand_ret.dtype)
    err_scale = torch.maximum(real.abs(), torch.tensor(1e-6, device=device))
    while real.ndim < cand_ret.ndim:
        real = real.unsqueeze(-1); err_scale = err_scale.unsqueeze(-1)
    norm_err = torch.nan_to_num((cand_ret - real).abs() / err_scale, nan=1e6, posinf=1e6)

    scores = dir_correct - 0.15 * norm_err
    scores = scores.masked_fill(~valid_label.unsqueeze(1), float("-inf"))
    _, winner_idx = scores.max(dim=1)
    _, loser_idx = scores.min(dim=1)

    rows = torch.arange(B, device=device)
    w_c, w_f = sampled_c[rows, winner_idx], sampled_f[rows, winner_idx]
    l_c, l_f = sampled_c[rows, loser_idx], sampled_f[rows, loser_idx]

    logp_c = F.log_softmax(pi_c.float(), dim=-1)
    logp_f = F.log_softmax(pi_f.float(), dim=-1)
    # Average log-prob (SimPO style: reward = avg logp / |tokens|, here |tokens|=2)
    r_win = (logp_c.gather(1, w_c.unsqueeze(1)).squeeze(1)
             + logp_f.gather(1, w_f.unsqueeze(1)).squeeze(1)) / 2.0
    r_lose = (logp_c.gather(1, l_c.unsqueeze(1)).squeeze(1)
              + logp_f.gather(1, l_f.unsqueeze(1)).squeeze(1)) / 2.0

    valid_pair = valid_label & (winner_idx != loser_idx) & torch.isfinite(scores.max(dim=1).values)
    loss_per = -F.logsigmoid(r_win - r_lose - gamma)
    vw = valid_pair.to(dtype=loss_per.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (loss_per * vw).sum() / vw.sum().clamp_min(1.0)


# ── DAPO ──

def _loss_dapo_v2(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    pi_c, pi_f, ref_c, ref_f, *_ = _shared_forward(model, ref_model, batch, device, amp_enabled, amp_dtype)
    B = pi_c.size(0)
    K = cfg.get("dapo_K", 16)
    temp = cfg.get("dapo_temperature", 1.2)

    sampled_c = p5d._sample_tokens(pi_c, temp, K)
    sampled_f = p5d._sample_tokens(pi_f, temp, K)

    cand_ret = p5d._token_returns(tokenizer, sampled_c, sampled_f,
                                   batch["prompt_means"], batch["prompt_stds"])
    cand_dir = torch.where(cand_ret > 0,
                           torch.tensor(p5d.LABEL_UP, device=device, dtype=torch.long),
                           torch.tensor(p5d.LABEL_DOWN, device=device, dtype=torch.long))

    valid_label = batch["loss_labels"] != p5d.IGNORE_INDEX
    dir_correct = cand_dir.eq(batch["labels"].unsqueeze(1)).float()
    real = batch["real_returns"].to(device=device, dtype=cand_ret.dtype)
    err_scale = torch.maximum(real.abs(), torch.tensor(1e-6, device=device))
    while real.ndim < cand_ret.ndim:
        real = real.unsqueeze(-1); err_scale = err_scale.unsqueeze(-1)
    norm_err = torch.nan_to_num((cand_ret - real).abs() / err_scale, nan=1e6, posinf=1e6)
    rewards = dir_correct - 0.15 * norm_err

    # Dynamic sampling: skip groups with zero variance
    reward_range = rewards.max(dim=1).values - rewards.min(dim=1).values
    active = valid_label & (reward_range > 0.01)
    if active.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True).clamp_min(1e-6)
    advantages = (rewards - mean_r) / std_r

    logp_c = F.log_softmax(pi_c.float(), dim=-1)
    logp_f = F.log_softmax(pi_f.float(), dim=-1)
    pi_logp = logp_c.gather(1, sampled_c) + logp_f.gather(1, sampled_f)

    ref_logp_c = F.log_softmax(ref_c.float(), dim=-1)
    ref_logp_f = F.log_softmax(ref_f.float(), dim=-1)
    ref_logp = ref_logp_c.gather(1, sampled_c) + ref_logp_f.gather(1, sampled_f)

    ratio = torch.exp(pi_logp - ref_logp.detach())
    # Decoupled clip
    clip_pos = cfg.get("dapo_clip_pos", 1.2)
    clip_neg = cfg.get("dapo_clip_neg", 0.8)
    clipped = torch.where(advantages > 0,
                          torch.clamp(ratio, 1.0, clip_pos),
                          torch.clamp(ratio, clip_neg, 1.0))
    policy_loss = -torch.min(ratio * advantages, clipped * advantages)

    kl_weight = cfg.get("dapo_kl_weight", 0.02)
    kl = (pi_logp - ref_logp).mean()

    active_mask = active.unsqueeze(1).float()
    total_active = active_mask.sum().clamp_min(1)
    return (policy_loss * active_mask).sum() / total_active + kl_weight * kl


# ── Verifier Reranker ──

def _train_verifier_full(model, tokenizer, train_loader, val_loaders, device, cfg):
    """Train verifier with cross-fitting: train on train, validate on val[0]."""
    from hpo.phase5_sup_train import VerifierMLP, _build_verifier_dataset, _evaluate_verifier

    print("  Building verifier dataset...")
    X, y = _build_verifier_dataset(model, tokenizer, train_loader, device, k=cfg.get("verifier_k", 32))
    if X is None:
        return None, 0.0

    n = len(X)
    n_train = int(n * 0.8)
    perm = torch.randperm(n)
    X_tr, y_tr = X[perm[:n_train]].to(device), y[perm[:n_train]].to(device)
    X_va, y_va = X[perm[n_train:]].to(device), y[perm[n_train:]].to(device)

    verifier = VerifierMLP().to(device)
    opt = torch.optim.AdamW(verifier.parameters(), lr=cfg.get("verifier_lr", 1e-4), weight_decay=1e-4)

    best_acc = 0.0; best_state = None; patience = 5; no_improve = 0
    bs = 256

    for epoch in range(cfg.get("verifier_epochs", 20)):
        verifier.train()
        for i in range(0, len(X_tr), bs):
            xb, yb = X_tr[i:i+bs], y_tr[i:i+bs]
            logits = verifier(xb[:, :384], xb[:, 384:768], xb[:, 768], xb[:, 769], xb[:, 770])
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        verifier.eval()
        with torch.no_grad():
            val_logits = verifier(X_va[:, :384], X_va[:, 384:768], X_va[:, 768], X_va[:, 769], X_va[:, 770])
            val_acc = ((torch.sigmoid(val_logits) > 0.5).float() == y_va).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc; best_state = {k: v.cpu().clone() for k, v in verifier.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break

    if best_state: verifier.load_state_dict(best_state)

    # Evaluate reranked DA on val[0]
    metrics = _evaluate_verifier(model, verifier, tokenizer, val_loaders[0][0], device,
                                 k=cfg.get("verifier_k", 32))
    return verifier, metrics.get("direction_accuracy", 0.0)


# ═══════════════════════════════════════════════
# Search spaces
# ═══════════════════════════════════════════════

def _suggest_shared(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
        "use_lora": trial.suggest_categorical("use_lora", [True, False]),
    }

def _suggest_candidates(trial):
    return {
        "num_candidates": trial.suggest_categorical("num_candidates", [64, 128, 192]),
        "temperature": trial.suggest_categorical("temperature", [0.5, 0.8, 1.0, 1.5, 2.0]),
        "direction_bonus": trial.suggest_categorical("direction_bonus", [0.5, 1.0, 2.0]),
        "error_weight": trial.suggest_categorical("error_weight", [0.1, 0.25, 0.5]),
        "score_margin": trial.suggest_categorical("score_margin", [0.0, 0.05, 0.1]),
        "include_gold": trial.suggest_categorical("include_gold", [True, False]),
    }

METHOD_SPACES = {
    "con_expo": {
        "params": ["lr", "use_lora", "num_candidates", "temperature", "direction_bonus",
                    "error_weight", "score_margin", "include_gold",
                    "expo_reference_weight", "token_ce_weight"],
        "specific": lambda t: {
            "expo_reference_weight": t.suggest_categorical("expo_reference_weight", [0.3, 0.5, 0.7, 0.9]),
            "token_ce_weight": t.suggest_categorical("token_ce_weight", [0.0, 0.02, 0.05, 0.1]),
        },
    },
    "robust_dpo": {
        "params": ["lr", "use_lora", "num_candidates", "temperature", "direction_bonus",
                    "error_weight", "score_margin", "include_gold",
                    "dpo_beta", "kl_weight"],
        "specific": lambda t: {
            "dpo_beta": t.suggest_categorical("dpo_beta", [0.1, 0.3, 0.5, 0.7, 1.0]),
            "kl_weight": t.suggest_categorical("kl_weight", [0.0, 0.01, 0.02, 0.05]),
        },
    },
    "kto": {
        "params": ["lr", "use_lora", "num_candidates", "temperature", "direction_bonus",
                    "error_weight", "score_margin", "include_gold",
                    "kto_beta", "kto_lambda_d", "kto_lambda_u"],
        "specific": lambda t: {
            "kto_beta": t.suggest_categorical("kto_beta", [0.05, 0.1, 0.3, 0.5]),
            "kto_lambda_d": t.suggest_categorical("kto_lambda_d", [0.5, 1.0, 1.5]),
            "kto_lambda_u": t.suggest_categorical("kto_lambda_u", [0.5, 1.0, 1.5]),
        },
    },
    "grpo": {
        "params": ["lr", "use_lora", "grpo_K", "grpo_temperature", "grpo_clip_eps", "grpo_kl_weight"],
        "specific": lambda t: {
            "grpo_K": t.suggest_categorical("grpo_K", [8, 16, 32]),
            "grpo_temperature": t.suggest_categorical("grpo_temperature", [0.8, 1.0, 1.2, 1.6, 2.0]),
            "grpo_clip_eps": t.suggest_categorical("grpo_clip_eps", [0.1, 0.2, 0.3]),
            "grpo_kl_weight": t.suggest_categorical("grpo_kl_weight", [0.01, 0.02, 0.05, 0.1]),
        },
    },
    "simpo": {
        "params": ["lr", "use_lora", "num_candidates", "simpo_gamma"],
        "specific": lambda t: {
            "num_candidates": t.suggest_categorical("num_candidates", [16, 32, 64]),
            "simpo_gamma": t.suggest_categorical("simpo_gamma", [0.1, 0.3, 0.5, 0.7, 1.0]),
        },
    },
    "dapo": {
        "params": ["lr", "use_lora", "dapo_K", "dapo_temperature", "dapo_clip_pos", "dapo_clip_neg", "dapo_kl_weight"],
        "specific": lambda t: {
            "dapo_K": t.suggest_categorical("dapo_K", [8, 16, 32]),
            "dapo_temperature": t.suggest_categorical("dapo_temperature", [0.8, 1.0, 1.2, 1.6]),
            "dapo_clip_pos": t.suggest_categorical("dapo_clip_pos", [1.1, 1.2, 1.5]),
            "dapo_clip_neg": t.suggest_categorical("dapo_clip_neg", [0.7, 0.8, 0.9]),
            "dapo_kl_weight": t.suggest_categorical("dapo_kl_weight", [0.01, 0.02, 0.05]),
        },
    },
    "verifier": {
        "params": ["verifier_k", "verifier_lr", "verifier_epochs"],
        "specific": lambda t: {
            "verifier_k": t.suggest_categorical("verifier_k", [16, 32, 64]),
            "verifier_lr": t.suggest_categorical("verifier_lr", [5e-5, 1e-4, 5e-4]),
            "verifier_epochs": t.suggest_categorical("verifier_epochs", [10, 20, 40]),
        },
    },
}

LOSS_FNS = {
    "con_expo": _loss_conservative_expo_v2,
    "robust_dpo": _loss_robust_dpo_v2,
    "kto": _loss_kto_v2,
    "grpo": _loss_grpo_v2,
    "simpo": _loss_simpo_v2,
    "dapo": _loss_dapo_v2,
}


# ═══════════════════════════════════════════════
# HPO objective
# ═══════════════════════════════════════════════

def _make_objective(method, train_loader, val_loaders, device):
    tokenizer = p5d._load_tokenizer(device)
    base_model = p5hpo.build_ref_model(device)

    # Pre-compute base metrics for walk-forward slices
    base_wf_metrics = {}
    for i, (loader, ds, dr) in enumerate(val_loaders):
        base_wf_metrics[i] = p5d.evaluate(base_model, tokenizer, loader, device, False, None)

    def objective(trial):
        params = _suggest_shared(trial)
        space = METHOD_SPACES[method]
        # Only use shared candidate sampling if method doesn't override it
        specific_params = space["specific"](trial)
        if method not in ("simpo", "dapo", "grpo"):
            params.update(_suggest_candidates(trial))
        params.update(specific_params)

        cfg = dict(DEFAULT_CFG)
        cfg.update(params)

        use_lora = params.get("use_lora", False)
        model = _build_model(device, use_lora=use_lora)
        ref_model = p5hpo.build_ref_model(device) if method != "simpo" else None

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            fused=True if device.type == "cuda" else False)

        amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
        amp_enabled = device.type == "cuda" and amp_dtype is not None
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

        best_score = -float("inf")
        updates = 0

        # Verifier is special: train once, evaluate
        if method == "verifier":
            verifier, reranked_da = _train_verifier_full(
                base_model, tokenizer, train_loader, val_loaders, device, cfg)
            return float(reranked_da) if verifier is not None else 0.0

        loss_fn = LOSS_FNS[method]

        while updates < HPO_UPDATES:
            model.train()
            for raw_batch in train_loader:
                if updates >= HPO_UPDATES: break
                batch = p5d._move_batch(raw_batch, device)
                batch["tokenizer"] = tokenizer
                loss = loss_fn(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg)

                if not torch.isfinite(loss): continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1

            # Eval every epoch (1206 batches ≈ 1 epoch)
            wf_score = _wf_score(model, tokenizer, val_loaders, device)
            if wf_score > best_score:
                best_score = wf_score

            trial.report(wf_score, updates // len(train_loader))
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("use_lora", use_lora)
        return best_score

    return objective


# ═══════════════════════════════════════════════
# Main HPO runner
# ═══════════════════════════════════════════════

def _run_hpo(method, device):
    study_name = f"phase5sup_{method}"
    storage = f"sqlite:///{HPO_DIR}/{study_name}.db"

    train_loader, val_loaders, eps = _build_wf_loaders()
    print(f"\n{'='*60}\nPhase 5 Sup HPO: {method.upper()}")
    print(f"  Trials: {HPO_TRIALS}  |  Updates/trial: {HPO_UPDATES}")
    print(f"  Walk-forward val: {len(val_loaders)} slices")
    print(f"  Search params: {METHOD_SPACES[method]['params']}")
    print(f"{'='*60}")

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        load_if_exists=True)

    existing = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if existing >= HPO_TRIALS:
        print(f"  Already {existing} completed (≥ {HPO_TRIALS}), skipping.")
        return study
    to_run = HPO_TRIALS - existing
    if existing > 0:
        print(f"  Resuming: {existing} completed, running {to_run} more (target: {HPO_TRIALS})")

    study.optimize(_make_objective(method, train_loader, val_loaders, device),
                   n_trials=to_run, show_progress_bar=True)

    summary = {
        "method": method, "n_trials": HPO_TRIALS,
        "best_trial": study.best_trial.number, "best_score": study.best_value,
        "best_params": study.best_params,
    }
    with open(os.path.join(HPO_DIR, f"{study_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{method} best: {study.best_value:.4f} (trial {study.best_trial.number})")
    print(f"  params: {json.dumps(study.best_params)}")
    return study


# ═══════════════════════════════════════════════
# Final best-config run
# ═══════════════════════════════════════════════

def _run_final(method, best_params, device):
    run_dir = os.path.join(OUT_DIR, f"{method}_best")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n--- {method} final ({FINAL_UPDATES} updates) ---")

    cfg = dict(DEFAULT_CFG); cfg.update(best_params)
    use_lora = best_params.get("use_lora", False)

    train_loader, val_loaders, eps = _build_wf_loaders()
    tokenizer = p5d._load_tokenizer(device)
    model = _build_model(device, use_lora=use_lora)
    ref_model = p5hpo.build_ref_model(device) if method != "simpo" else None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"], fused=True)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    amp_enabled = device.type == "cuda" and amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    loss_fn = LOSS_FNS.get(method)
    if loss_fn is None:
        print(f"  No loss function for {method}, skipping")
        return None

    history = []; best_score = -float("inf"); best_metrics = None
    updates = 0; epoch = 0

    while updates < FINAL_UPDATES:
        epoch += 1
        model.train(); epoch_loss = 0.0; epoch_batches = 0
        for raw_batch in train_loader:
            if updates >= FINAL_UPDATES: break
            batch = p5d._move_batch(raw_batch, device)
            batch["tokenizer"] = tokenizer
            loss = loss_fn(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg)
            if not torch.isfinite(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1; epoch_loss += loss.item(); epoch_batches += 1

        val_metrics = p5d.evaluate(model, tokenizer, val_loaders[-1][0], device, False, None)
        score = float(val_metrics.get("balanced_accuracy", 0.0))
        record = {"epoch": epoch, "updates": updates, "train_loss": epoch_loss/max(1,epoch_batches),
                  "val_da": val_metrics.get("direction_accuracy", 0),
                  "val_balacc": score, "val_mape": val_metrics.get("mape", 0)}
        history.append(record)
        print(f"  ep{epoch} DA={record['val_da']:.4f} BalAcc={score:.4f} MAPE={record['val_mape']:.4f}")

        if score > best_score:
            best_score = score; best_metrics = val_metrics
            torch.save({"method": method, "epoch": epoch, "updates": updates,
                        "model_state_dict": model.state_dict(),
                        "hpo_params": best_params, "metrics": val_metrics, "history": history},
                       os.path.join(run_dir, f"{method}_best.pt"))

        # Save resume checkpoint
        torch.save({"epoch": epoch, "updates": updates, "best_score": best_score,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history},
                   os.path.join(run_dir, "ckpt.pt"))

    result = {"method": method, "best_score": best_score, "best_metrics": best_metrics,
              "best_params": best_params, "history": history}
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    return result


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Phase 5 Sup HPO")
    p.add_argument("--method", choices=list(METHOD_SPACES.keys()) + ["all"], default="all")
    p.add_argument("--trials", type=int, default=HPO_TRIALS)
    p.add_argument("--updates", type=int, default=HPO_UPDATES)
    p.add_argument("--final-updates", type=int, default=FINAL_UPDATES)
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    import hpo.phase5_sup_hpo as _mod
    _mod.HPO_TRIALS = args.trials
    _mod.HPO_UPDATES = args.updates
    _mod.FINAL_UPDATES = args.final_updates

    set_global_seed(42, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    methods = list(METHOD_SPACES.keys()) if args.method == "all" else [args.method]

    if not args.eval_only:
        for method in methods:
            _run_hpo(method, device)

    # Final evaluation
    all_results = {}
    for method in methods:
        summary_path = os.path.join(HPO_DIR, f"phase5sup_{method}_summary.json")
        if not os.path.exists(summary_path):
            print(f"No HPO summary for {method}, skipping final")
            continue
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        all_results[method] = _run_final(method, s["best_params"], device)

    print("\n" + "="*60)
    print("Phase 5 Sup HPO — Final Results")
    print("="*60)
    for method, r in all_results.items():
        if r is None: continue
        bm = r.get("best_metrics", {})
        print(f"  {method:15s}  DA={bm.get('direction_accuracy',0):.4f}  "
              f"BalAcc={bm.get('balanced_accuracy',0):.4f}  "
              f"MAPE={bm.get('mape',0):.4f}")

    with open(os.path.join(OUT_DIR, "cross_method_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
