# -*- coding: utf-8 -*-
"""Phase 5 Sup – Stage 2-4: smoke-test training methods.

Methods (each ~300 updates, 120 stocks subset):
  S2 – Conservative ExPO   (higher ref_weight, CE anchor)
  S3 – Robust DPO          (higher beta, KL reg)
  S4 – KTO                 (binary utility, no pairs needed)
  S5 – One-step GRPO       (group sampling + advantage normalization)

Usage::

    python -m hpo.phase5_sup_train --method expo --smoke
    python -m hpo.phase5_sup_train --method grpo --smoke --updates 300
"""

from __future__ import annotations

import argparse, copy, json, os, sys, time, warnings
from contextlib import nullcontext
from datetime import datetime

import numpy as np
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

OUT_DIR = os.path.join(_PROJECT_ROOT, "trials", "phase5_sup")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════
# Smoke test config
# ═══════════════════════════════════════════════

SMOKE_CFG = {
    "max_updates": 300,
    "batch_size": 8,
    "lr": 5e-6,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "num_candidates": 64,   # fewer for smoke speed
    "temperature": 1.0,
    "direction_bonus": 1.0,
    "error_weight": 0.25,
    "score_margin": 0.05,
    "include_gold": True,
    "label_mode": "global_median",
    "epsilon_scale": 0.5,
    "flat_policy": "ignore",
    "num_val_samples": 500,
}


# ═══════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════

def _ac(device, enabled, dtype):
    if device.type != "cuda" or not enabled or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_full_data(max_train_samples=0):
    """Load train + val data, optionally subset."""
    train_payload = torch.load(os.path.join(_PROJECT_ROOT, "dataset_train.pt"),
                               map_location="cpu", weights_only=False)
    val_payload = torch.load(os.path.join(_PROJECT_ROOT, "dataset_val.pt"),
                             map_location="cpu", weights_only=False)
    train_returns = p5d._denorm_last_returns(train_payload)
    val_returns = p5d._denorm_last_returns(val_payload)
    train_abs = np.abs(train_returns); train_abs = train_abs[np.isfinite(train_abs)]
    eps = max(1e-5, float(np.median(train_abs)) * SMOKE_CFG["epsilon_scale"])

    train_indices = np.arange(len(train_returns), dtype=np.int64)
    val_indices = np.arange(len(val_returns), dtype=np.int64)

    if max_train_samples > 0 and len(train_indices) > max_train_samples:
        rng = np.random.default_rng(42)
        train_indices = np.sort(rng.choice(train_indices, size=max_train_samples, replace=False))

    train_ds = p5d.DirectionDataset(train_payload, train_indices, train_returns, eps, SMOKE_CFG["flat_policy"])
    val_ds = p5d.DirectionDataset(val_payload, val_indices, val_returns, eps, "class")

    train_loader = DataLoader(train_ds, batch_size=SMOKE_CFG["batch_size"], shuffle=True,
                              collate_fn=p5d.collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=SMOKE_CFG["batch_size"], shuffle=False,
                            collate_fn=p5d.collate_fn)
    return train_loader, val_loader, eps, train_ds, val_ds


def _build_optimizer(model, lr):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=SMOKE_CFG["weight_decay"],
        fused=True if torch.cuda.is_available() else False)


def _get_val_metrics(model, tokenizer, val_loader, device):
    return p5d.evaluate(model, tokenizer, val_loader, device, False, None)


# ═══════════════════════════════════════════════
# S2: Conservative ExPO
# ═══════════════════════════════════════════════

def _loss_conservative_expo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype):
    """ExPO with strong reference anchor + token CE regularization."""
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = p5d._prepare_inputs(batch)

    with p5d._ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_c, pi_f = logits_c[:, -1, :], logits_f[:, -1, :]

    with torch.no_grad():
        with p5d._ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_c, ref_f = ref_lc[:, -1, :], ref_lf[:, -1, :]

    w_c, w_f, l_c, l_f, valid_pair, _, _ = p5d._build_winner_loser(
        tokenizer, ref_c, ref_f, batch, SMOKE_CFG)

    theta_win = p5d._candidate_logp(pi_c, pi_f, w_c, w_f)
    theta_lose = p5d._candidate_logp(pi_c, pi_f, l_c, l_f)
    with torch.no_grad():
        ref_win = p5d._candidate_logp(ref_c, ref_f, w_c, w_f)
        ref_lose = p5d._candidate_logp(ref_c, ref_f, l_c, l_f)
        ref_pref = torch.sigmoid(ref_win - ref_lose)

    theta_pref = torch.sigmoid(theta_win - theta_lose)
    # Conservative: higher reference weight (0.7 instead of 0.3)
    lam = 0.7
    target_pref = (lam * ref_pref + (1.0 - lam)).clamp(0.0, 1.0)
    expo_loss = (theta_pref - target_pref).pow(2)
    vw = valid_pair.to(dtype=expo_loss.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    expo_loss = (expo_loss * vw).sum() / vw.sum().clamp_min(1.0)

    # Token CE anchor (mild, 0.05 weight)
    target_c = batch["idx_c_full"][:, -1]
    target_f = batch["idx_f_full"][:, -1]
    ce_loss = (F.cross_entropy(pi_c.float(), target_c)
               + F.cross_entropy(pi_f.float(), target_f))

    return expo_loss + 0.05 * ce_loss


def run_conservative_expo(device, max_updates=300):
    return _run_smoke("conservative_expo", device, _loss_conservative_expo,
                      needs_ref=True, use_lora=False, lr=3e-6, max_updates=max_updates)


# ═══════════════════════════════════════════════
# S3: Robust DPO
# ═══════════════════════════════════════════════

def _loss_robust_dpo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype):
    """DPO with higher beta + KL regularization."""
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = p5d._prepare_inputs(batch)

    with p5d._ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_c, pi_f = logits_c[:, -1, :], logits_f[:, -1, :]

    with torch.no_grad():
        with p5d._ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_c, ref_f = ref_lc[:, -1, :], ref_lf[:, -1, :]

    w_c, w_f, l_c, l_f, valid_pair, _, _ = p5d._build_winner_loser(
        tokenizer, ref_c, ref_f, batch, SMOKE_CFG)

    pi_win = p5d._candidate_logp(pi_c, pi_f, w_c, w_f)
    pi_lose = p5d._candidate_logp(pi_c, pi_f, l_c, l_f)
    with torch.no_grad():
        ref_win = p5d._candidate_logp(ref_c, ref_f, w_c, w_f)
        ref_lose = p5d._candidate_logp(ref_c, ref_f, l_c, l_f)

    # Robust DPO: beta=0.5 (not 0.1)
    beta = 0.5
    log_ratio = (pi_win - pi_lose) - (ref_win - ref_lose)
    dpo_loss = -F.logsigmoid(beta * log_ratio)
    vw = valid_pair.to(dtype=dpo_loss.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    dpo_loss = (dpo_loss * vw).sum() / vw.sum().clamp_min(1.0)

    # KL regularization
    kl_c = F.kl_div(F.log_softmax(pi_c.float(), dim=-1),
                    F.softmax(ref_c.float(), dim=-1), reduction="batchmean")
    kl_f = F.kl_div(F.log_softmax(pi_f.float(), dim=-1),
                    F.softmax(ref_f.float(), dim=-1), reduction="batchmean")
    kl = (kl_c + kl_f) * 0.02

    return dpo_loss + kl


def run_robust_dpo(device, max_updates=300):
    return _run_smoke("robust_dpo", device, _loss_robust_dpo,
                      needs_ref=True, use_lora=False, lr=5e-6, max_updates=max_updates)


# ═══════════════════════════════════════════════
# S4: KTO — Kahneman-Tversky Optimization
# ═══════════════════════════════════════════════

def _loss_kto(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype):
    """KTO: binary desirable/undesirable utility without pairwise preference.

    desirable  = direction correct AND abs_error <= median bucket error
    undesirable = direction wrong OR abs_error too large
    """
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = p5d._prepare_inputs(batch)

    with p5d._ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_c, pi_f = logits_c[:, -1, :], logits_f[:, -1, :]

    with torch.no_grad():
        with p5d._ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_c, ref_f = ref_lc[:, -1, :], ref_lf[:, -1, :]

    B = pi_c.size(0)
    # Sample candidates
    sampled_c = p5d._sample_tokens(ref_c, SMOKE_CFG["temperature"], SMOKE_CFG["num_candidates"])
    sampled_f = p5d._sample_tokens(ref_f, SMOKE_CFG["temperature"], SMOKE_CFG["num_candidates"])
    if SMOKE_CFG["include_gold"]:
        gold_c = batch["idx_c_full"][:, -1]; gold_f = batch["idx_f_full"][:, -1]
        sampled_c = torch.cat([sampled_c, gold_c.long().unsqueeze(1)], dim=1)
        sampled_f = torch.cat([sampled_f, gold_f.long().unsqueeze(1)], dim=1)

    # Decode and score all candidates
    cand_ret = p5d._token_returns(tokenizer, sampled_c, sampled_f,
                                   batch["prompt_means"], batch["prompt_stds"])
    cand_dir = torch.where(cand_ret > 0,
                           torch.tensor(p5d.LABEL_UP, device=device, dtype=torch.long),
                           torch.tensor(p5d.LABEL_DOWN, device=device, dtype=torch.long))

    valid_label = batch["loss_labels"] != p5d.IGNORE_INDEX
    labels_2d = batch["labels"].unsqueeze(1)
    dir_correct = cand_dir.eq(labels_2d) & valid_label.unsqueeze(1)

    real = batch["real_returns"].to(device=cand_ret.device, dtype=cand_ret.dtype)
    err_scale = torch.maximum(real.abs(), torch.tensor(1e-6, device=device))
    while real.ndim < cand_ret.ndim:
        real = real.unsqueeze(-1); err_scale = err_scale.unsqueeze(-1)
    abs_err = (cand_ret - real).abs()
    norm_err = torch.nan_to_num(abs_err / err_scale, nan=1e6, posinf=1e6, neginf=1e6)

    # Binary utility: desirable = direction correct, undesirable = wrong or bad error
    median_err = norm_err.median()
    desirable = dir_correct & (norm_err <= median_err)
    undesirable = ~dir_correct | (norm_err > 2 * median_err)

    # For each sample, pick one desirable and one undesirable candidate (if available)
    # KTO loss: -λ_d * σ(β*(logp - ref_logp)) for desirables, -λ_u * σ(β*(ref_logp - logp)) for undesirables
    lambda_d = 1.0; lambda_u = 1.0; kto_beta = 0.1

    total_loss = torch.tensor(0.0, device=device)
    n_des, n_undes = 0, 0

    for i in range(B):
        if not valid_label[i]:
            continue
        des_idx = torch.where(desirable[i])[0]
        undes_idx = torch.where(undesirable[i])[0]
        if len(des_idx) == 0 or len(undes_idx) == 0:
            continue

        # Take the most extreme example of each
        best_des = des_idx[norm_err[i, des_idx].argmin()]
        best_undes = undes_idx[norm_err[i, undes_idx].argmax()]

        for idx, is_des in [(best_des, True), (best_undes, False)]:
            logp_pi = p5d._candidate_logp(pi_c[i:i+1], pi_f[i:i+1],
                                           sampled_c[i:i+1, idx], sampled_f[i:i+1, idx]).squeeze(0)
            logp_ref = p5d._candidate_logp(ref_c[i:i+1], ref_f[i:i+1],
                                            sampled_c[i:i+1, idx], sampled_f[i:i+1, idx]).squeeze(0)

            if is_des:
                loss_i = -F.logsigmoid(kto_beta * (logp_pi - logp_ref)) * lambda_d
                n_des += 1
            else:
                loss_i = -F.logsigmoid(kto_beta * (logp_ref - logp_pi)) * lambda_u
                n_undes += 1
            total_loss = total_loss + loss_i

    n_total = n_des + n_undes
    if n_total == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / n_total


def run_kto(device, max_updates=300):
    return _run_smoke("kto", device, _loss_kto,
                      needs_ref=True, use_lora=False, lr=5e-6, max_updates=max_updates)


# ═══════════════════════════════════════════════
# S5: One-Step GRPO
# ═══════════════════════════════════════════════

def _loss_grpo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype):
    """One-step GRPO: group sampling + advantage-normalized policy gradient.

    For each prefix, sample K candidates from the CURRENT policy (not reference),
    compute reward = direction_correct - 0.15 * norm_error, normalize within group,
    train with clipped policy gradient + KL constraint.
    """
    K = 16  # group size
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = p5d._prepare_inputs(batch)

    with p5d._ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_c, pi_f = logits_c[:, -1, :], logits_f[:, -1, :]

    # Sample K candidates from current policy
    B = pi_c.size(0)
    sampled_c = p5d._sample_tokens(pi_c, 1.2, K)  # [B, K]
    sampled_f = p5d._sample_tokens(pi_f, 1.2, K)  # [B, K]

    # Decode and compute rewards
    cand_ret = p5d._token_returns(tokenizer, sampled_c, sampled_f,
                                   batch["prompt_means"], batch["prompt_stds"])
    cand_dir = torch.where(cand_ret > 0,
                           torch.tensor(p5d.LABEL_UP, device=device, dtype=torch.long),
                           torch.tensor(p5d.LABEL_DOWN, device=device, dtype=torch.long))

    valid_label = batch["loss_labels"] != p5d.IGNORE_INDEX
    labels_2d = batch["labels"].unsqueeze(1)
    dir_correct = cand_dir.eq(labels_2d).float()

    real = batch["real_returns"].to(device=cand_ret.device, dtype=cand_ret.dtype)
    err_scale = torch.maximum(real.abs(), torch.tensor(1e-6, device=device))
    while real.ndim < cand_ret.ndim:
        real = real.unsqueeze(-1); err_scale = err_scale.unsqueeze(-1)
    norm_err = torch.nan_to_num((cand_ret - real).abs() / err_scale, nan=1e6, posinf=1e6)

    rewards = dir_correct - 0.15 * norm_err  # [B, K]
    rewards = rewards.masked_fill(~valid_label.unsqueeze(1), 0.0)

    # Group advantage normalization
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True).clamp_min(1e-6)
    advantages = (rewards - mean_r) / std_r  # [B, K]

    # Policy log-prob for each sampled token
    logp_c_full = F.log_softmax(pi_c.float(), dim=-1)  # [B, V]
    logp_f_full = F.log_softmax(pi_f.float(), dim=-1)  # [B, V]
    logp_sampled = (logp_c_full.gather(1, sampled_c)
                    + logp_f_full.gather(1, sampled_f))  # [B, K]

    # Reference log-prob (for KL)
    with torch.no_grad():
        with p5d._ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_c, ref_f = ref_lc[:, -1, :], ref_lf[:, -1, :]
        ref_logp_c = F.log_softmax(ref_c.float(), dim=-1)
        ref_logp_f = F.log_softmax(ref_f.float(), dim=-1)
        ref_logp = (ref_logp_c.gather(1, sampled_c) + ref_logp_f.gather(1, sampled_f))
        # Old log-prob (for clipping) — we don't have old_logp stored, use ref as proxy
        old_logp = ref_logp.detach()

    # PPO-style clipped loss
    ratio = torch.exp(logp_sampled - old_logp)
    clip_eps = 0.2
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # KL penalty
    kl = (logp_sampled - ref_logp).mean()

    valid_mask = valid_label.unsqueeze(1).float()
    total_valid = valid_mask.sum().clamp_min(1)
    grpo_loss = (policy_loss * valid_mask).sum() / total_valid + 0.02 * kl

    return grpo_loss


def run_grpo(device, max_updates=300):
    return _run_smoke("grpo", device, _loss_grpo,
                      needs_ref=True, use_lora=False, lr=3e-6, max_updates=max_updates)


# ═══════════════════════════════════════════════
# Smoke test runner
# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════
# S6: Verifier Reranker
# ═══════════════════════════════════════════════

class VerifierMLP(torch.nn.Module):
    """Small MLP that scores token candidates for direction correctness."""
    def __init__(self, hidden_dim=384, token_dim=384):
        super().__init__()
        # Input: last_hidden [D] + token_emb [D] + decoded_return [1] + logp [1] + rank [1]
        in_dim = hidden_dim + token_dim + 3
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1),  # score logit
        )

    def forward(self, last_hidden, token_emb, decoded_ret, logp, rank):
        x = torch.cat([last_hidden, token_emb,
                       decoded_ret.unsqueeze(-1),
                       logp.unsqueeze(-1),
                       rank.unsqueeze(-1).float()], dim=-1)
        return self.net(x).squeeze(-1)


@torch.no_grad()
def _build_verifier_dataset(model, tokenizer, loader, device, k=32):
    """Build a dataset of (features, label) for training the verifier.

    For each sample: get top-K token candidates from BaseModel, extract features,
    label = direction_correct.
    """
    model.eval()
    all_features, all_labels = [], []

    for raw_batch in tqdm(loader, desc="Build verifier data", leave=False):
        batch = p5d._move_batch(raw_batch, device)
        batch["tokenizer"] = tokenizer
        idx_c, idx_f, t_min, t_day, t_mon, t_yr = p5d._prepare_inputs(batch)

        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr,
                                       last_only=True)
        last_c = logits_c[:, -1, :].float()
        last_f = logits_f[:, -1, :].float()

        # Get last hidden state (need a separate forward without last_only)
        _, _, latent_states = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=False)
        # latent_states: [n_layers, B, n_latent, D] — use last position hidden via direction_head path
        # For simplicity, use direction logits as a proxy for last hidden
        # Actually, re-run forward_direction to get the hidden state
        # We'll pass None for now and use a simpler feature set

        # Extract top-K info
        info = p5e._extract_topk(tokenizer, last_c, last_f,
                                  batch["prompt_means"], batch["prompt_stds"],
                                  batch["real_returns"], k)

        B = last_c.size(0)
        for i in range(B):
            lbl = int(batch["labels"][i].item())
            if lbl == p5d.LABEL_FLAT:
                continue

            tok_c = info["tokens_c"][i]   # [K]
            tok_f = info["tokens_f"][i]   # [K]
            rets = info["decoded_returns"][i]  # [K]
            dir_corr = info["directions_correct"][i]  # [K]

            # Token embeddings
            c_emb = model.token_emb_coarse(torch.tensor(tok_c, device=device))  # [K, D]
            f_emb = model.token_emb_fine(torch.tensor(tok_f, device=device))    # [K, D]
            tok_emb = c_emb + f_emb  # [K, D]

            # Get last hidden state via forward_direction
            with torch.no_grad():
                ic = idx_c[i:i+1]; iff = idx_f[i:i+1]
                tm = t_min[i:i+1]; td = t_day[i:i+1]; tmo = t_mon[i:i+1]; tyr = t_yr[i:i+1]
                dir_logits, lstates = model.forward_direction(ic, iff, tm, td, tmo, tyr)
                # hidden at last position
                x_emb = model._compute_embedding(ic, iff, tm, td, tmo, tyr)
                for blk in model.blocks:
                    x_emb = blk(x_emb)
                last_hidden = x_emb[0, -1, :]  # [D]

            # Logp from logits
            logp_c = torch.nn.functional.log_softmax(last_c[i:i+1], dim=-1)
            logp_f = torch.nn.functional.log_softmax(last_f[i:i+1], dim=-1)
            tok_c_t = torch.tensor(tok_c, device=device)
            tok_f_t = torch.tensor(tok_f, device=device)
            logp = logp_c[0, tok_c_t] + logp_f[0, tok_f_t]  # [K]

            rank_t = torch.arange(len(tok_c), device=device).float()

            # For each candidate: feature = [last_hidden, token_emb, decoded_ret, logp, rank]
            for j in range(len(tok_c)):
                feat = torch.cat([last_hidden.cpu(), tok_emb[j].cpu(),
                                  torch.tensor([rets[j]]), torch.tensor([logp[j].item()]),
                                  torch.tensor([float(j)])])
                all_features.append(feat)
                all_labels.append(float(dir_corr[j]))

    if not all_features:
        return None, None
    X = torch.stack(all_features)  # [N, D*2+3]
    y = torch.tensor(all_labels, dtype=torch.float32)
    return X, y


def _train_verifier(X_train, y_train, X_val, y_val, device, max_epochs=20):
    """Train verifier MLP with early stopping."""
    model = VerifierMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Simple train/val split from training data if no separate val set
    n = len(X_train)
    n_train = int(n * 0.8)
    perm = torch.randperm(n)
    X_tr = X_train[perm[:n_train]].to(device)
    y_tr = y_train[perm[:n_train]].to(device)
    X_va = X_train[perm[n_train:]].to(device)
    y_va = y_train[perm[n_train:]].to(device)

    best_acc = 0.0
    best_state = None
    patience = 5
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        # Mini-batch training
        bs = 256
        total_loss = 0.0
        for i in range(0, len(X_tr), bs):
            xb = X_tr[i:i+bs]; yb = y_tr[i:i+bs]
            logits = model(xb[:, :384], xb[:, 384:768], xb[:, 768], xb[:, 769], xb[:, 770])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va[:, :384], X_va[:, 384:768], X_va[:, 768], X_va[:, 769], X_va[:, 770])
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_pred == y_va).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


@torch.no_grad()
def _evaluate_verifier(model, verifier, tokenizer, loader, device, k=32):
    """Evaluate: get top-K, rerank by verifier, compute DA after reranking."""
    model.eval()
    verifier.eval()
    all_preds, all_labels = [], []

    for raw_batch in tqdm(loader, desc="Verifier eval", leave=False):
        batch = p5d._move_batch(raw_batch, device)
        batch["tokenizer"] = tokenizer
        idx_c, idx_f, t_min, t_day, t_mon, t_yr = p5d._prepare_inputs(batch)

        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        last_c = logits_c[:, -1, :].float()
        last_f = logits_f[:, -1, :].float()

        info = p5e._extract_topk(tokenizer, last_c, last_f,
                                  batch["prompt_means"], batch["prompt_stds"],
                                  batch["real_returns"], k)

        B = last_c.size(0)
        for i in range(B):
            lbl = int(batch["labels"][i].item())
            if lbl == p5d.LABEL_FLAT:
                continue
            all_labels.append(lbl)

            tok_c = info["tokens_c"][i]
            tok_f = info["tokens_f"][i]
            rets = info["decoded_returns"][i]
            K_i = len(tok_c)

            if K_i == 0:
                all_preds.append(p5d.LABEL_DOWN)
                continue

            # Get features for all K candidates
            c_emb = model.token_emb_coarse(torch.tensor(tok_c, device=device))
            f_emb = model.token_emb_fine(torch.tensor(tok_f, device=device))
            tok_emb = c_emb + f_emb

            # Last hidden
            ic = idx_c[i:i+1]; iff = idx_f[i:i+1]
            tm = t_min[i:i+1]; td = t_day[i:i+1]; tmo = t_mon[i:i+1]; tyr = t_yr[i:i+1]
            x_emb = model._compute_embedding(ic, iff, tm, td, tmo, tyr)
            for blk in model.blocks:
                x_emb = blk(x_emb)
            last_hidden = x_emb[0, -1, :]

            logp_c = torch.nn.functional.log_softmax(last_c[i:i+1], dim=-1)
            logp_f = torch.nn.functional.log_softmax(last_f[i:i+1], dim=-1)
            tok_c_t = torch.tensor(tok_c, device=device)
            tok_f_t = torch.tensor(tok_f, device=device)
            logp = logp_c[0, tok_c_t] + logp_f[0, tok_f_t]
            rank_t = torch.arange(K_i, device=device).float()
            rets_t = torch.tensor(rets, device=device)

            # Verifier score
            last_hid_exp = last_hidden.unsqueeze(0).expand(K_i, -1)
            scores = verifier(last_hid_exp, tok_emb, rets_t, logp, rank_t)
            best_idx = scores.argmax().item()

            pred_ret = rets[best_idx]
            pred_dir = p5d.LABEL_UP if pred_ret > 0 else p5d.LABEL_DOWN
            all_preds.append(pred_dir)

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    true_dir = labels != p5d.LABEL_FLAT
    da = float(np.mean(preds[true_dir] == labels[true_dir])) if true_dir.any() else 0
    return {"direction_accuracy": da, "num_samples": len(labels)}


def run_verifier(device, max_updates=300):
    """Train a verifier reranker: extract top-K features, train MLP, evaluate."""
    name = "verifier"
    run_dir = os.path.join(OUT_DIR, f"smoke_{name}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'='*60}\nSmoke: Verifier Reranker\n{'='*60}")

    tokenizer = p5d._load_tokenizer(device)
    base_model = p5hpo.build_ref_model(device)  # frozen P3

    # Use small subset for speed
    train_loader, val_loader, eps, train_ds, val_ds = _load_full_data(max_train_samples=500)
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    # Build verifier training data from val set (cross-fitting: train on val, eval on...)
    # Actually: use train_loader to build data, split internally for verifier training
    print("Building verifier dataset...")
    t0 = time.time()
    X, y = _build_verifier_dataset(base_model, tokenizer, train_loader, device, k=32)
    if X is None:
        print("No verifier data built!")
        return {"method": name, "error": "no data"}
    print(f"Verifier dataset: {len(X)} candidates, pos={y.sum().item():.0f} ({y.mean().item():.2%})")

    # Train verifier
    verifier, best_acc = _train_verifier(X, y, None, None, device, max_epochs=20)
    print(f"Verifier trained: best val acc = {best_acc:.4f}")

    # Evaluate with verifier reranking
    val_metrics = _evaluate_verifier(base_model, verifier, tokenizer, val_loader, device, k=32)
    print(f"Verifier-reranked val DA = {val_metrics['direction_accuracy']:.4f}")

    elapsed = time.time() - t0
    result = {"method": name, "verifier_val_acc": best_acc,
              "reranked_da": val_metrics["direction_accuracy"],
              "n_candidates": len(X), "elapsed": elapsed}
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if best_acc > 0.5:
        torch.save({"verifier_state_dict": verifier.state_dict(),
                    "metrics": result},
                   os.path.join(run_dir, f"smoke_{name}_best.pt"))
    return result


# ═══════════════════════════════════════════════
# S7: SimPO — reference-free preference optimization
# ═══════════════════════════════════════════════

def _loss_simpo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype):
    """SimPO: reference-free with average log-prob reward and target margin.

    reward(c) = avg_logp(c) / |c|   (average log-prob per token)
    L = -log σ(reward_w - reward_l - γ)
    """
    gamma = 0.5  # target margin
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = p5d._prepare_inputs(batch)

    with p5d._ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_c, pi_f = logits_c[:, -1, :], logits_f[:, -1, :]

    # Sample candidates from POLICY (SimPO is on-policy)
    B = pi_c.size(0)
    K = 32
    sampled_c = p5d._sample_tokens(pi_c, 1.0, K)
    sampled_f = p5d._sample_tokens(pi_f, 1.0, K)

    # Decode and score
    cand_ret = p5d._token_returns(tokenizer, sampled_c, sampled_f,
                                   batch["prompt_means"], batch["prompt_stds"])
    cand_dir = torch.where(cand_ret > 0,
                           torch.tensor(p5d.LABEL_UP, device=device, dtype=torch.long),
                           torch.tensor(p5d.LABEL_DOWN, device=device, dtype=torch.long))

    valid_label = batch["loss_labels"] != p5d.IGNORE_INDEX
    labels_2d = batch["labels"].unsqueeze(1)
    dir_correct = cand_dir.eq(labels_2d).float()

    real = batch["real_returns"].to(device=cand_ret.device, dtype=cand_ret.dtype)
    err_scale = torch.maximum(real.abs(), torch.tensor(1e-6, device=device))
    while real.ndim < cand_ret.ndim:
        real = real.unsqueeze(-1); err_scale = err_scale.unsqueeze(-1)
    norm_err = torch.nan_to_num((cand_ret - real).abs() / err_scale, nan=1e6, posinf=1e6)

    scores = dir_correct - 0.15 * norm_err
    scores = scores.masked_fill(~valid_label.unsqueeze(1), float("-inf"))

    winner_score, winner_idx = scores.max(dim=1)
    loser_score, loser_idx = scores.min(dim=1)
    valid_pair = valid_label & (winner_idx != loser_idx) & torch.isfinite(winner_score)

    rows = torch.arange(B, device=device)
    w_c, w_f = sampled_c[rows, winner_idx], sampled_f[rows, winner_idx]
    l_c, l_f = sampled_c[rows, loser_idx], sampled_f[rows, loser_idx]

    # Reward = average log-prob
    logp_c = F.log_softmax(pi_c.float(), dim=-1)
    logp_f = F.log_softmax(pi_f.float(), dim=-1)
    r_win = (logp_c.gather(1, w_c.unsqueeze(1)).squeeze(1)
             + logp_f.gather(1, w_f.unsqueeze(1)).squeeze(1)) / 2.0
    r_lose = (logp_c.gather(1, l_c.unsqueeze(1)).squeeze(1)
              + logp_f.gather(1, l_f.unsqueeze(1)).squeeze(1)) / 2.0

    loss_per = -F.logsigmoid(r_win - r_lose - gamma)
    vw = valid_pair.to(dtype=loss_per.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (loss_per * vw).sum() / vw.sum().clamp_min(1.0)


def run_simpo(device, max_updates=300):
    return _run_smoke("simpo", device, _loss_simpo,
                      needs_ref=False, use_lora=False, lr=3e-6, max_updates=max_updates)


# ═══════════════════════════════════════════════
# S8: DAPO — dynamic sampling + decoupled clip
# ═══════════════════════════════════════════════

def _loss_dapo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype):
    """DAPO-style GRPO: dynamic sampling + decoupled clip + token-level reward."""
    K = 16
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = p5d._prepare_inputs(batch)

    with p5d._ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_c, pi_f = logits_c[:, -1, :], logits_f[:, -1, :]

    B = pi_c.size(0)
    sampled_c = p5d._sample_tokens(pi_c, 1.2, K)
    sampled_f = p5d._sample_tokens(pi_f, 1.2, K)

    cand_ret = p5d._token_returns(tokenizer, sampled_c, sampled_f,
                                   batch["prompt_means"], batch["prompt_stds"])
    cand_dir = torch.where(cand_ret > 0,
                           torch.tensor(p5d.LABEL_UP, device=device, dtype=torch.long),
                           torch.tensor(p5d.LABEL_DOWN, device=device, dtype=torch.long))

    valid_label = batch["loss_labels"] != p5d.IGNORE_INDEX
    dir_correct = cand_dir.eq(batch["labels"].unsqueeze(1)).float()

    real = batch["real_returns"].to(device=cand_ret.device, dtype=cand_ret.dtype)
    err_scale = torch.maximum(real.abs(), torch.tensor(1e-6, device=device))
    while real.ndim < cand_ret.ndim:
        real = real.unsqueeze(-1); err_scale = err_scale.unsqueeze(-1)
    norm_err = torch.nan_to_num((cand_ret - real).abs() / err_scale, nan=1e6, posinf=1e6)
    rewards = dir_correct - 0.15 * norm_err

    # Dynamic sampling: skip groups where all rewards identical (no learning signal)
    reward_range = rewards.max(dim=1).values - rewards.min(dim=1).values
    active = valid_label & (reward_range > 0.01)

    if active.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Group advantage (only for active samples)
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True).clamp_min(1e-6)
    advantages = (rewards - mean_r) / std_r

    # Policy log-prob
    logp_c = F.log_softmax(pi_c.float(), dim=-1)
    logp_f = F.log_softmax(pi_f.float(), dim=-1)
    logp = logp_c.gather(1, sampled_c) + logp_f.gather(1, sampled_f)

    # Reference log-prob (from base model, for KL)
    with torch.no_grad():
        with p5d._ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_c, ref_f = ref_lc[:, -1, :], ref_lf[:, -1, :]
        ref_logp_c = F.log_softmax(ref_c.float(), dim=-1)
        ref_logp_f = F.log_softmax(ref_f.float(), dim=-1)
        ref_logp = ref_logp_c.gather(1, sampled_c) + ref_logp_f.gather(1, sampled_f)

    ratio = torch.exp(logp - ref_logp.detach())

    # Decoupled clip: tighter for negative advantages
    clip_pos = 1.2
    clip_neg = 0.8
    clipped_ratio = torch.where(advantages > 0,
                                torch.clamp(ratio, 1.0, clip_pos),
                                torch.clamp(ratio, clip_neg, 1.0))
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # KL penalty
    kl = (logp - ref_logp).mean()

    active_mask = active.unsqueeze(1).float()
    total_active = active_mask.sum().clamp_min(1)
    dapo_loss = (policy_loss * active_mask).sum() / total_active + 0.02 * kl

    return dapo_loss


def run_dapo(device, max_updates=300):
    return _run_smoke("dapo", device, _loss_dapo,
                      needs_ref=True, use_lora=False, lr=3e-6, max_updates=max_updates)


METHOD_REGISTRY = {
    "conservative_expo": run_conservative_expo,
    "robust_dpo": run_robust_dpo,
    "kto": run_kto,
    "grpo": run_grpo,
    "verifier": run_verifier,
    "simpo": run_simpo,
    "dapo": run_dapo,
}


def _run_smoke(name, device, loss_fn, needs_ref, use_lora, lr, max_updates, resume=True):
    """Generic training loop with checkpoint/resume support.

    Saves full training state (model, optimizer, scheduler, scaler, history) to
    ``{run_dir}/ckpt.pt`` after each epoch.  On restart, auto-detects and resumes
    from the latest checkpoint.
    """
    run_dir = os.path.join(OUT_DIR, f"smoke_{name}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    best_path = os.path.join(run_dir, f"smoke_{name}_best.pt")

    # ── Resume detection ──
    history = []
    best_score = -float("inf")
    best_metrics = None
    start_epoch = 0
    updates = 0

    if resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        history = state.get("history", [])
        best_score = state.get("best_score", -float("inf"))
        start_epoch = state.get("epoch", 0)
        updates = state.get("updates", 0)
        print(f"  Resumed: epoch={start_epoch}, updates={updates}, best_score={best_score:.4f}")
        if updates >= max_updates:
            print(f"  Already completed ({updates} >= {max_updates}), skipping.")
            return {"method": name, "best_score": best_score,
                    "best_metrics": state.get("best_metrics", {}), "history": history}
    else:
        print(f"\n{'='*60}\nSmoke: {name}  (max {max_updates} updates, lr={lr})\n{'='*60}")

    # ── Data ──
    max_train = max(1, int(max_updates * SMOKE_CFG["batch_size"] * 1.5))
    train_loader, val_loader, eps, train_ds, val_ds = _load_full_data(max_train_samples=max_train)
    if not resume or not os.path.exists(ckpt_path):
        print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}  eps={eps:.6f}")

    # ── Model ──
    tokenizer = p5d._load_tokenizer(device)
    model = p5hpo.build_trainable_model_hpo(device, lora_rank=8, lora_alpha=16, use_lora=use_lora)
    if not use_lora:
        model.enable_gradient_checkpointing(True)
    ref_model = p5hpo.build_ref_model(device) if needs_ref else None

    if resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not resume or not os.path.exists(ckpt_path):
        print(f"Trainable: {trainable:,}")

    # ── Optimizer + scheduler ──
    optimizer = _build_optimizer(model, lr)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    amp_enabled = device.type == "cuda" and amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    if resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in state:
            try:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception as e:
                print(f"  Warning: could not load optimizer state ({e}), restarting optimizer")

    pbar = tqdm(total=max_updates, desc=f"  {name}", initial=updates)
    epoch = start_epoch

    while updates < max_updates:
        epoch += 1
        model.train()
        epoch_loss = 0.0; epoch_batches = 0

        for raw_batch in train_loader:
            if updates >= max_updates:
                break
            batch = p5d._move_batch(raw_batch, device)
            batch["tokenizer"] = tokenizer
            loss = loss_fn(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], SMOKE_CFG["grad_clip"])
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            epoch_loss += loss.detach().cpu().item()
            epoch_batches += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{epoch_loss/max(1,epoch_batches):.4f}", "upd": updates})

            if updates >= max_updates:
                break

        # ── Eval ──
        val_metrics = _get_val_metrics(model, tokenizer, val_loader, device)
        score = float(val_metrics.get("balanced_accuracy", 0.0))
        record = {"epoch": epoch, "updates": updates,
                  "train_loss": epoch_loss / max(1, epoch_batches),
                  "val_da": val_metrics.get("direction_accuracy", 0),
                  "val_balacc": val_metrics.get("balanced_accuracy", 0),
                  "val_mape": val_metrics.get("mape", 0)}
        history.append(record)
        print(f"\n  ep{epoch} upd={updates} DA={record['val_da']:.4f} "
              f"BalAcc={record['val_balacc']:.4f} MAPE={record['val_mape']:.4f}")

        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            torch.save({"method": name, "epoch": epoch, "updates": updates,
                        "model_state_dict": model.state_dict(),
                        "tokenizer_state_dict": tokenizer.state_dict(),
                        "metrics": val_metrics, "history": history},
                       best_path)

        # ── Save full resume checkpoint ──
        torch.save({
            "epoch": epoch, "updates": updates,
            "best_score": best_score, "best_metrics": best_metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer_state_dict": tokenizer.state_dict(),
            "history": history,
        }, ckpt_path)

    pbar.close()
    print(f"\n{name} best: DA={best_metrics['direction_accuracy']:.4f} BalAcc={best_metrics['balanced_accuracy']:.4f}")
    result = {"method": name, "best_score": best_score, "best_metrics": best_metrics, "history": history}
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    return result


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Phase 5 Sup – smoke test training")
    p.add_argument("--method", required=True,
                   choices=list(METHOD_REGISTRY.keys()) + ["all"],
                   help="Method to smoke-test")
    p.add_argument("--updates", type=int, default=300,
                   help="Max training updates (default: 300)")
    p.add_argument("--smoke", action="store_true", default=True,
                   help="Run in smoke-test mode (subset data, limited updates)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    set_global_seed(42, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    methods = list(METHOD_REGISTRY.keys()) if args.method == "all" else [args.method]
    results = {}
    for method in methods:
        t0 = time.time()
        results[method] = METHOD_REGISTRY[method](device, max_updates=args.updates)
        elapsed = time.time() - t0
        results[method]["elapsed"] = elapsed
        print(f"  {method}: {elapsed:.0f}s")

    print("\n=== Smoke Summary ===")
    for method, r in results.items():
        bm = r.get("best_metrics", {})
        print(f"  {method:20s}  DA={bm.get('direction_accuracy',0):.4f}  "
              f"BalAcc={bm.get('balanced_accuracy',0):.4f}  "
              f"MAPE={bm.get('mape',0):.4f}  time={r.get('elapsed',0):.0f}s")
