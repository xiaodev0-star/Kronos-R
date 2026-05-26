"""Phase 8 V2: STAR-CAST Post-Training HPO (refined).

V2 improvements over V1:
  - 1200 stocks (2x V1) for better signal
  - Narrowed search ranges around V1 best (temp~0.5, CE_w~0.17, lr~2e-5)
  - Longer training: 240-480 updates (vs 80-160)
  - 50 trials (vs 40)

V1 best: path_mape=4.86%, temp=0.517, neftune=5.78, ce_w=0.174, lr=1.98e-5

Usage:
    python -m hpo.phase8_star_cast_v2
"""

from __future__ import annotations

import copy, json, os, time, hashlib, math
from argparse import Namespace
from contextlib import nullcontext
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import DataConfig
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT
from posttrain.rollout.data import (
    RolloutWindowDataset,
    rollout_collate,
    resolve_project_path,
)
from reproducibility import set_global_seed

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
N_TRIALS = 50
STUDY_NAME = "phase8_star_cast_v2"
CLEAN_START = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE8_DIR = os.path.join(PROJECT_ROOT, "trials", "phase8_star_cast_v2")
STUDY_DB = os.path.join(PHASE8_DIR, "study.db")
SUMMARY_CSV = os.path.join(PHASE8_DIR, "summary.csv")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.pt")
TOKENIZER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer_config.json")
BASEMODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "base_model.pt")

TOKENIZER_BITS = 10
TOKENIZER_VOCAB = 1 << TOKENIZER_BITS
PREFIX_LEN = 1023
ROLLOUT_HORIZON = 10

# ── Fixed backbone (Phase 3 trial 047) ──
BACKBONE = {
    "dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
    "dsa_windows": [None, 512, 512],
    "position_encoding": "rope", "rope_base": 10000.0,
    "dropout": 0.1323, "use_revin": False, "num_factor_tokens": 0,
}

# ── Server resource config (override via env vars) ──
SERVER_BATCH_SIZE     = int(os.environ.get("STARCAST_BS",  "16"))
SERVER_NUM_WORKERS    = int(os.environ.get("STARCAST_NW",  "4"))
SERVER_EVAL_BATCHES   = int(os.environ.get("STARCAST_EVAL", "500"))
SERVER_GRAD_ACCUM     = int(os.environ.get("STARCAST_GA",  "2"))
SERVER_USE_COMPILE    = os.environ.get("STARCAST_COMPILE", "0") == "1"
# Max sequences per forward during exploration (controls attention matrix size)
# 4090 24G: 128 is safe (~1.2 GB attention), 192 borderline, 256+ OOM-risk
SERVER_EXPLORE_CHUNK  = int(os.environ.get("STARCAST_CHUNK", "96"))

# ═══════════════════════════════════════════════════════════════════════
# Search space
# ═══════════════════════════════════════════════════════════════════════
SEARCH_SPACE = {
    "exploration_temp":     (0.35, 1.0),       # narrowed around V1 best=0.517
    "neftune_alpha":        (2.0, 10.0),       # low-end clipped, V1 best=5.78
    "star_ce_weight":       (0.05, 0.5),       # narrowed around V1 best=0.174
    "lr":                   (5e-6, 5e-5),      # widened upper, V1 best=1.98e-5
    "max_updates":          [320, 480],        # increased from [80,160]
}

# Fixed STAR-CAST parameters (not searched)
FIXED_PARAMS = {
    "num_trajectories": 4,
    "step_asym_weight": 1.0,
    "path_asym_weight": 1.5,
    "asymmetric_alpha": 3.0,
    "asymmetric_beta": 10.0,
    "path_asymmetric_beta": 15.0,
    "top_k_expected_return": 16,
    "epochs": 1,
}


def sample_params(trial: optuna.Trial) -> dict:
    """Sample STAR-CAST hyperparameters from the search space."""
    p = {}
    p["exploration_temp"]  = round(trial.suggest_float("exploration_temp", *SEARCH_SPACE["exploration_temp"], log=True), 4)
    p["neftune_alpha"]     = round(trial.suggest_float("neftune_alpha", *SEARCH_SPACE["neftune_alpha"], log=True), 2)
    p["star_ce_weight"]    = round(trial.suggest_float("star_ce_weight", *SEARCH_SPACE["star_ce_weight"], log=True), 4)
    p["lr"]                = round(trial.suggest_float("lr", *SEARCH_SPACE["lr"], log=True), 10)
    p["max_updates"]       = trial.suggest_categorical("max_updates", SEARCH_SPACE["max_updates"])
    p.update(FIXED_PARAMS)
    return p


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_rollout_cfg():
    """Minimal config namespace for RolloutWindowDataset.

    Uses a fixed random subset of HPO_STOCKS stocks (default 600) for
    fast HPO iteration. The random seed ensures reproducibility across trials.
    Set STARCAST_STOCKS=0 env var to use all stocks.
    """
    n_stocks = int(os.environ.get("STARCAST_STOCKS", "1200"))
    return Namespace(
        prefix_len=PREFIX_LEN, horizon=ROLLOUT_HORIZON,
        stride_ratio=DataConfig.stride_ratio,
        cache_dir=os.path.join(PROJECT_ROOT, "posttrain", "rollout", "cache"),
        max_stocks=n_stocks, cache_rebuild=False,
    )


def _choose_amp_dtype(device):
    if device.type != "cuda": return None
    if torch.cuda.is_bf16_supported(): return torch.bfloat16
    return torch.float16


def _autocast_ctx(amp, dt):
    if not amp: return nullcontext()
    try: return torch.amp.autocast(device_type="cuda", dtype=dt)
    except Exception: return torch.cuda.amp.autocast(dtype=dt)


def _config_hash(params: dict) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════

def _load_tokenizer(device):
    ckpt = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg and os.path.exists(TOKENIZER_CONFIG_PATH):
        with open(TOKENIZER_CONFIG_PATH) as f: cfg = json.load(f)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tok.load_state_dict(ckpt["model_state_dict"], strict=False)
    tok.eval(); tok.requires_grad_(False)
    return tok


def _load_basemodel(device):
    bp = BACKBONE
    model = KronosReasoningGPT(
        dim=bp["dim"], depth=bp["depth"], heads=bp["heads"],
        num_kv_heads=bp["num_kv_heads"], dsa_windows=bp["dsa_windows"],
        dropout=bp["dropout"], vocab_size_coarse=TOKENIZER_VOCAB,
        vocab_size_fine=TOKENIZER_VOCAB,
        position_encoding=bp["position_encoding"], rope_base=bp["rope_base"],
        use_revin=bp["use_revin"], num_factor_tokens=bp["num_factor_tokens"],
    ).to(device)
    if os.path.exists(BASEMODEL_PATH):
        ckpt = torch.load(BASEMODEL_PATH, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded BaseModel from {BASEMODEL_PATH}")
    else:
        print(f"  WARNING: {BASEMODEL_PATH} not found. Using random init.")
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════

def _build_rollout_data(device, batch_size=None):
    """Build train/val DataLoaders with server-configured parallelism."""
    if batch_size is None:
        batch_size = SERVER_BATCH_SIZE
    cfg = _make_rollout_cfg()
    train_ds = RolloutWindowDataset("train", cfg=cfg, max_samples=0, seed=42)
    val_ds   = RolloutWindowDataset("val",   cfg=cfg, max_samples=0, seed=59)
    print(f"  Rollout train windows: {len(train_ds)}, val windows: {len(val_ds)}")
    print(f"  batch_size={batch_size}, num_workers={SERVER_NUM_WORKERS}, "
          f"grad_accum={SERVER_GRAD_ACCUM}")

    loader_kwargs = dict(
        num_workers=SERVER_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(SERVER_NUM_WORKERS > 0),
        prefetch_factor=2,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=rollout_collate, **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        collate_fn=rollout_collate, **loader_kwargs,
    )
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════
# STAR-CAST Core: Differentiable Expected Returns & Asymmetric Loss
# ═══════════════════════════════════════════════════════════════════════

def _expected_return_from_topk(tokenizer, logits_c, logits_f, means, stds, top_k):
    """Bridge discrete tokens to continuous expected returns via top-K soft expectation."""
    B, H, V_c = logits_c.shape
    K = min(int(top_k), V_c)

    top_logits_c, top_idx_c = torch.topk(logits_c.float(), k=K, dim=-1)
    top_logits_f, top_idx_f = torch.topk(logits_f.float(), k=K, dim=-1)
    prob_c = F.softmax(top_logits_c, dim=-1)
    prob_f = F.softmax(top_logits_f, dim=-1)
    prob_c = prob_c / prob_c.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prob_f = prob_f / prob_f.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    joint_prob = prob_c.unsqueeze(-1) * prob_f.unsqueeze(-2)  # [B, H, K, K]
    pair_c = top_idx_c.unsqueeze(-1).expand(B, H, K, K).reshape(B * H, K * K)
    pair_f = top_idx_f.unsqueeze(-2).expand(B, H, K, K).reshape(B * H, K * K)

    with torch.no_grad():
        decoded = tokenizer.decode(pair_c, pair_f)[..., 0].float()
        decoded = decoded.view(B, H, K, K)
        returns_grid = decoded * stds[:, 0].view(B, 1, 1, 1) + means[:, 0].view(B, 1, 1, 1)

    return (joint_prob * returns_grid).sum(dim=(-1, -2))  # [B, H]


def _asymmetric_direction_loss(expected, actual, alpha, beta, eps=1e-4):
    """Asymmetric penalty: wrong-direction errors are amplified."""
    abs_err = torch.abs(expected - actual)
    is_wrong = ((expected * actual) < 0) & (torch.abs(actual) > eps)
    penalty = torch.ones_like(abs_err)
    penalty = torch.where(is_wrong, alpha + beta * torch.abs(expected), penalty)
    return abs_err * penalty


# ═══════════════════════════════════════════════════════════════════════
# STAR-CAST: Noisy Exploration + Oracle Filter (Phase A, no grad)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _star_cast_exploration(
    model, tokenizer, idx_c_full, idx_f_full,
    time_feats, means, stds, actual_returns,
    prefix_len, horizon, num_traj, temp, neftune_alpha,
    device, amp_enabled, amp_dtype, chunk_size=96,
):
    """Noisy exploration with CHUNKED forward passes.

    Instead of feeding all B*N trajectories through the model at once
    (which creates enormous attention matrices), we split into chunks.
    Each chunk is processed independently; results are reassembled.

    Oracle Filter is fully vectorized across the batch dimension.
    """
    B = idx_c_full.size(0)
    N = num_traj
    H = horizon
    BN = B * N
    CS = min(chunk_size, BN)

    # ── Per-sample trajectory context ──
    # Store as [B, N, prefix_len+H] and reshape to [B*N, ...] chunk-by-chunk
    traj_c = torch.empty(B, N, prefix_len + H, dtype=torch.long, device=device)
    traj_f = torch.empty(B, N, prefix_len + H, dtype=torch.long, device=device)
    traj_ret = torch.empty(B, N, H, device=device, dtype=torch.float32)

    # Initial context: [B, N, prefix_len]
    traj_c[:, :, :prefix_len] = idx_c_full[:, :prefix_len].unsqueeze(1).expand(B, N, prefix_len)
    traj_f[:, :, :prefix_len] = idx_f_full[:, :prefix_len].unsqueeze(1).expand(B, N, prefix_len)

    means_3d = means.unsqueeze(1).expand(B, N, means.size(1))  # [B, N, 6]
    stds_3d  = stds.unsqueeze(1).expand(B, N, stds.size(1))

    # ── Autoregressive rollout, chunked per step ──
    for step in range(H):
        cur_len = prefix_len + step
        cur_c = traj_c[:, :, :cur_len].reshape(BN, cur_len)
        cur_f = traj_f[:, :, :cur_len].reshape(BN, cur_len)

        # Build time features for current length
        cur_time_all = {k: time_feats[k][:, :cur_len].unsqueeze(1).expand(B, N, cur_len).reshape(BN, cur_len)
                        for k in ("minute", "day", "month", "year")}

        # ── Chunked forward ──
        logits_c_chunks, logits_f_chunks = [], []
        for chunk_start in range(0, BN, CS):
            chunk_end = min(chunk_start + CS, BN)
            cc = cur_c[chunk_start:chunk_end]
            cf = cur_f[chunk_start:chunk_end]
            ct = {k: v[chunk_start:chunk_end] for k, v in cur_time_all.items()}
            with _autocast_ctx(amp_enabled, amp_dtype):
                lc, lf, _ = model(cc, cf, ct["minute"], ct["day"], ct["month"], ct["year"],
                                  last_only=True, neftune_alpha=neftune_alpha)
            logits_c_chunks.append(lc[:, -1, :].float())
            logits_f_chunks.append(lf[:, -1, :].float())

        logits_c_all = torch.cat(logits_c_chunks, dim=0)  # [BN, V]
        logits_f_all = torch.cat(logits_f_chunks, dim=0)

        # Temperature sampling (all trajectories at once, cheap)
        probs_c = F.softmax(logits_c_all / max(1e-4, temp), dim=-1)
        probs_f = F.softmax(logits_f_all / max(1e-4, temp), dim=-1)
        pred_c = torch.multinomial(probs_c, num_samples=1)  # [BN, 1]
        pred_f = torch.multinomial(probs_f, num_samples=1)

        # Decode + compute returns
        decoded = tokenizer.decode(pred_c, pred_f)[..., 0].float()  # [BN, 1]
        step_ret = decoded * stds.unsqueeze(1).expand(B, N, 6).reshape(BN, 6)[:, 0:1] \
                   + means.unsqueeze(1).expand(B, N, 6).reshape(BN, 6)[:, 0:1]
        traj_ret[:, :, step] = step_ret.view(B, N)

        # Append to trajectories
        traj_c[:, :, cur_len] = pred_c.view(B, N)
        traj_f[:, :, cur_len] = pred_f.view(B, N)

    # ── Reshape for Oracle Filter ──
    path_returns = traj_ret.sum(dim=2)                          # [B, N]
    actual_path = actual_returns[:, :H].sum(dim=1)              # [B]

    # ── Vectorized Oracle Filter ──
    correct_dir = (path_returns * actual_path.unsqueeze(1)) > 0
    is_valid = correct_dir.any(dim=1) & (torch.abs(actual_path) > 1e-6)

    errors = torch.abs(path_returns - actual_path.unsqueeze(1))
    errors[~correct_dir] = float('inf')
    best_idx = errors.argmin(dim=1)

    # Gather best trajectory per sample
    gather_idx = best_idx.view(B, 1, 1).expand(B, 1, prefix_len + H)
    golden_c = traj_c.gather(1, gather_idx).squeeze(1)
    golden_f = traj_f.gather(1, gather_idx).squeeze(1)

    # Fallback to ground truth where no valid golden trajectory
    gt_c = idx_c_full[:, :prefix_len + H]
    gt_f = idx_f_full[:, :prefix_len + H]
    mask = is_valid.view(B, 1).float()
    golden_c = (golden_c.float() * mask + gt_c.float() * (1 - mask)).long()
    golden_f = (golden_f.float() * mask + gt_f.float() * (1 - mask)).long()

    return golden_c, golden_f, is_valid


# ═══════════════════════════════════════════════════════════════════════
# STAR-CAST Training
# ═══════════════════════════════════════════════════════════════════════

def _train_star_cast(model, tokenizer, params: dict, tdir: str, device) -> dict:
    """STAR-CAST training loop with gradient accumulation. Returns best val path_mape."""
    p = params
    result_path = os.path.join(tdir, "result.json")
    resume_path = os.path.join(tdir, "star_cast_resume.pt")
    os.makedirs(tdir, exist_ok=True)

    if os.path.exists(result_path):
        with open(result_path) as f: return json.load(f)

    train_loader, val_loader = _build_rollout_data(device)

    lr = p["lr"]
    max_updates = p["max_updates"]
    ga_steps = max(1, SERVER_GRAD_ACCUM)

    opt = optim.AdamW(model.parameters(), lr=lr, fused=True if device.type == "cuda" else False)
    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # torch.compile for server GPU (2x speedup on forward pass)
    _compiled = False
    if SERVER_USE_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            _compiled = True
            print("  torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"  torch.compile failed ({e}), using eager mode")

    # Resume
    update_count = 0
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        update_count = ckpt["update_count"]
        print(f"  Resume from update {update_count}")

    prefix_len = PREFIX_LEN
    horizon = ROLLOUT_HORIZON
    total_updates_done = update_count
    total_loss_sum = 0.0

    model.train()
    pbar = tqdm(total=max_updates - update_count, desc="  STAR-CAST train")
    opt.zero_grad(set_to_none=True)
    microbatch_count = 0

    while total_updates_done < max_updates:
        for batch in train_loader:
            if total_updates_done >= max_updates: break

            feats  = batch["features"].to(device=device, dtype=torch.float32, non_blocking=True)
            means  = batch["means"].to(device=device, dtype=torch.float32, non_blocking=True)
            stds   = batch["stds"].to(device=device, dtype=torch.float32, non_blocking=True)
            actual = batch["actual_returns"].to(device=device, dtype=torch.float32, non_blocking=True)
            times  = {k: v.to(device=device, dtype=torch.long, non_blocking=True)
                      for k, v in batch["time"].items()}

            B = feats.size(0)
            if B == 0: continue

            idx_c_full, idx_f_full = tokenizer.encode(feats)
            actual_h = actual[:, :horizon]

            # ── Phase A: Noisy Exploration + Oracle Filter ──
            golden_c, golden_f, has_golden = _star_cast_exploration(
                model, tokenizer, idx_c_full, idx_f_full,
                times, means, stds, actual,
                prefix_len, horizon,
                p["num_trajectories"], p["exploration_temp"], p["neftune_alpha"],
                device, use_amp, amp_dtype, chunk_size=SERVER_EXPLORE_CHUNK,
            )

            # ── Phase B: Dual-Engine Update ──
            model.train()
            train_len = golden_c.size(1)
            train_time = {k: v[:, :train_len] for k, v in times.items()}

            with _autocast_ctx(use_amp, amp_dtype):
                logits_c, logits_f, _ = model(
                    golden_c[:, :-1], golden_f[:, :-1],
                    train_time["minute"][:, :train_len - 1],
                    train_time["day"][:, :train_len - 1],
                    train_time["month"][:, :train_len - 1],
                    train_time["year"][:, :train_len - 1],
                    neftune_alpha=0.0,
                )

                start = prefix_len - 1
                rollout_c = logits_c[:, start:start + horizon, :]
                rollout_f = logits_f[:, start:start + horizon, :]

                # Engine 1: Continuous — Asymmetric Direction Loss
                expected = _expected_return_from_topk(
                    tokenizer, rollout_c, rollout_f, means, stds,
                    p["top_k_expected_return"],
                )

                step_loss = _asymmetric_direction_loss(
                    expected, actual_h, p["asymmetric_alpha"], p["asymmetric_beta"],
                ).mean()

                expected_path = torch.cumsum(expected, dim=1)
                actual_path = torch.cumsum(actual_h, dim=1)
                path_loss = _asymmetric_direction_loss(
                    expected_path, actual_path,
                    p["asymmetric_alpha"] * 1.3, p["path_asymmetric_beta"],
                ).mean()

                # Engine 2: Discrete — STaR CE Reinforcement
                if has_golden.any():
                    target_c = golden_c[has_golden, prefix_len:prefix_len + horizon]
                    target_f = golden_f[has_golden, prefix_len:prefix_len + horizon]
                    ce_c = F.cross_entropy(
                        rollout_c[has_golden].reshape(-1, rollout_c.size(-1)).float(),
                        target_c.reshape(-1),
                    )
                    ce_f = F.cross_entropy(
                        rollout_f[has_golden].reshape(-1, rollout_f.size(-1)).float(),
                        target_f.reshape(-1),
                    )
                    star_ce = ce_c + ce_f
                else:
                    star_ce = torch.tensor(0.0, device=device)

                loss = (p["step_asym_weight"] * step_loss +
                        p["path_asym_weight"] * path_loss +
                        p["star_ce_weight"] * star_ce)

                # Scale loss by gradient accumulation steps
                loss = loss / ga_steps

            if not torch.isfinite(loss): continue

            scaler.scale(loss).backward()
            microbatch_count += 1

            # Step optimizer only after accumulating enough gradients
            if microbatch_count % ga_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

                total_loss_sum += loss.item() * ga_steps
                total_updates_done += 1
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item() * ga_steps:.3f}",
                    "golden": f"{has_golden.float().mean().item():.2f}",
                })

            if total_updates_done % 80 == 0 and total_updates_done > 0:
                torch.save({
                    "update_count": total_updates_done,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                }, resume_path)

            if total_updates_done >= max_updates: break

    pbar.close()

    # Save final model
    torch.save({"model_state_dict": model.state_dict(), "update_count": total_updates_done},
               os.path.join(tdir, "star_cast_model.pt"))

    # Evaluate
    eval_result = _eval_rollout(model, tokenizer, val_loader, device)
    if os.path.exists(resume_path): os.remove(resume_path)

    result = {
        **eval_result,
        "train_loss": round(total_loss_sum / max(total_updates_done, 1), 6),
        "total_updates": total_updates_done,
        "params": p,
    }
    with open(result_path, "w") as f: json.dump(result, f, indent=2)
    return result


# ═══════════════════════════════════════════════════════════════════════
# True 10-step AR evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval_rollout(model, tokenizer, val_loader, device, max_batches=None) -> dict:
    """Strict 10-step autoregressive rollout -> path_mape, daily_mape, da.

    Args:
        max_batches: override eval batch limit (None = use SERVER_EVAL_BATCHES).
                     Set lower (e.g. 80) for quick intermediate pruning checks.
    """
    model.eval()
    all_path_mape = []
    all_daily_mape = []
    all_da = []
    prefix_len = PREFIX_LEN
    horizon = ROLLOUT_HORIZON
    limit = max_batches if max_batches is not None else SERVER_EVAL_BATCHES

    n_batches = 0
    for batch in tqdm(val_loader, desc="  Eval AR10", leave=False):
        feats  = batch["features"].to(device=device, dtype=torch.float32)
        means  = batch["means"].to(device=device, dtype=torch.float32)
        stds   = batch["stds"].to(device=device, dtype=torch.float32)
        actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
        times  = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}

        B = feats.size(0)
        if B == 0: continue
        n_batches += 1
        if n_batches > limit: break

        idx_c, idx_f = tokenizer.encode(feats)
        cur_c = idx_c[:, :prefix_len].clone()
        cur_f = idx_f[:, :prefix_len].clone()
        actual_rets = actual.cpu()

        pred_rets = []
        for step in range(horizon):
            sl = int(cur_c.size(1))
            cur_time = {
                "minute": times["minute"][:, :sl],
                "day":    times["day"][:, :sl],
                "month":  times["month"][:, :sl],
                "year":   times["year"][:, :sl],
            }
            logits_c, logits_f, _ = model(
                cur_c, cur_f,
                cur_time["minute"], cur_time["day"],
                cur_time["month"], cur_time["year"],
                last_only=True,
            )
            if not torch.isfinite(logits_c).all(): break
            pc = logits_c[:, -1, :].argmax(dim=-1)
            pf = logits_f[:, -1, :].argmax(dim=-1)
            dec = tokenizer.decode(pc.unsqueeze(1), pf.unsqueeze(1))
            pred_norm = dec[:, 0, 0].cpu().float()
            pred_ret = pred_norm * stds[:, 0].cpu() + means[:, 0].cpu()
            pred_rets.append(pred_ret)

            if step < horizon - 1:
                cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
                cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)

        if len(pred_rets) < horizon: continue

        pred_rets = torch.stack(pred_rets, dim=1)  # [B, 10]

        # Path MAPE
        cum_pred   = torch.cumsum(pred_rets.float(), dim=1)
        cum_actual = torch.cumsum(actual_rets.float(), dim=1)
        for step in range(horizon):
            pr = torch.exp(torch.clamp(cum_pred[:, step], -20, 20))
            ar = torch.exp(torch.clamp(cum_actual[:, step], -20, 20))
            denom = torch.clamp(torch.abs(ar), min=1e-6)
            valid = torch.isfinite(pr) & torch.isfinite(ar) & (denom > 0)
            if valid.sum() > 0:
                all_path_mape.append(
                    (torch.abs(pr[valid] - ar[valid]) / denom[valid]).mean().item() * 100
                )

            dr = torch.exp(torch.clamp(pred_rets[:, step].float(), -20, 20))
            da_val = torch.exp(torch.clamp(actual_rets[:, step].float(), -20, 20))
            denom_d = torch.clamp(torch.abs(da_val), min=1e-6)
            valid_d = torch.isfinite(dr) & torch.isfinite(da_val) & (denom_d > 0)
            if valid_d.sum() > 0:
                all_daily_mape.append(
                    (torch.abs(dr[valid_d] - da_val[valid_d]) / denom_d[valid_d]).mean().item() * 100
                )

        # Directional Accuracy
        pred_sign = (pred_rets >= 0).float() * 2 - 1
        actual_sign = (actual_rets >= 0).float() * 2 - 1
        all_da.append((pred_sign == actual_sign).float().mean().item() * 100)

    if not all_path_mape:
        return {"path_mape": 999.0, "daily_mape": 999.0, "da": 0.0,
                "path_mape_std": 0.0, "num_eval_steps": 0}

    return {
        "path_mape":      round(float(np.mean(all_path_mape)), 6),
        "daily_mape":     round(float(np.mean(all_daily_mape)), 6),
        "da":             round(float(np.mean(all_da)), 4),
        "path_mape_std":  round(float(np.std(all_path_mape)), 6),
        "num_eval_steps": len(all_path_mape),
    }


# ═══════════════════════════════════════════════════════════════════════
# Trial directory management
# ═══════════════════════════════════════════════════════════════════════

def _assign_trial_dir():
    os.makedirs(PHASE8_DIR, exist_ok=True)
    counter_path = os.path.join(PHASE8_DIR, ".next_trial")

    existing = sorted([d for d in os.listdir(PHASE8_DIR)
                       if d.startswith("trial_") and os.path.isdir(os.path.join(PHASE8_DIR, d))],
                      key=lambda x: int(x.split("_")[1]))
    for d in existing:
        full = os.path.join(PHASE8_DIR, d)
        resume = os.path.join(full, "star_cast_resume.pt")
        result = os.path.join(full, "result.json")
        if os.path.exists(resume) and not os.path.exists(result):
            if os.path.exists(os.path.join(full, "config.json")):
                print(f"  Found incomplete trial: {d} — will resume")
                return full

    next_num = 0
    if os.path.exists(counter_path):
        with open(counter_path) as f: next_num = int(f.read().strip())
    new_dir = os.path.join(PHASE8_DIR, f"trial_{next_num:03d}")
    with open(counter_path, "w") as f: f.write(str(next_num + 1))
    return new_dir


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    if CLEAN_START and os.path.exists(PHASE8_DIR):
        import shutil; shutil.rmtree(PHASE8_DIR)
    os.makedirs(PHASE8_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 8 V2 — STAR-CAST Post-Training HPO (refined)")
    print(f"  Output: {PHASE8_DIR}")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"  Prefix: {PREFIX_LEN}, Horizon: {ROLLOUT_HORIZON}")
    print(f"  Server config: batch_size={SERVER_BATCH_SIZE}, workers={SERVER_NUM_WORKERS}, "
          f"grad_accum={SERVER_GRAD_ACCUM}, eval_batches={SERVER_EVAL_BATCHES}, "
          f"compile={SERVER_USE_COMPILE}, explore_chunk={SERVER_EXPLORE_CHUNK}, "
          f"stocks={os.environ.get('STARCAST_STOCKS', '1200')}")
    print(f"  Search space: {list(SEARCH_SPACE.keys())}")
    print(f"  Fixed params: {list(FIXED_PARAMS.keys())}")
    print()

    # Load tokenizer + BaseModel once
    print("Loading tokenizer + BaseModel...")
    tokenizer = _load_tokenizer(device)
    base_model = _load_basemodel(device)
    print(f"  BaseModel: {sum(p.numel() for p in base_model.parameters()):,} params")
    print()

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=f"sqlite:///{STUDY_DB}",
        direction="minimize", load_if_exists=True,
    )

    seen_hashes = set()
    completed = 0

    while completed < N_TRIALS:
        trial = study.ask()
        tdir = _assign_trial_dir()
        os.makedirs(tdir, exist_ok=True)

        params = sample_params(trial)
        config_path = os.path.join(tdir, "config.json")

        if os.path.exists(config_path):
            with open(config_path) as f: params = json.load(f)
        else:
            with open(config_path, "w") as f: json.dump(params, f, indent=2)

        ch = _config_hash(params)
        if ch in seen_hashes:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue

        print(f"\n{'='*60}")
        print(f"Trial {trial.number:03d} (dir={os.path.basename(tdir)})")
        print(f"Params: temp={params['exploration_temp']} neftune={params['neftune_alpha']} "
              f"ce_w={params['star_ce_weight']} lr={params['lr']:.1e} "
              f"updates={params['max_updates']}")
        print(f"{'='*60}")

        # Clone model for this trial
        model = copy.deepcopy(base_model)

        t0 = time.time()
        try:
            result = _train_star_cast(model, tokenizer, params, tdir, device)
            path_mape = result["path_mape"]
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue

        elapsed = time.time() - t0

        trial.set_user_attr("trial_dir", tdir)
        trial.set_user_attr("dir_name", os.path.basename(tdir))
        trial.set_user_attr("elapsed_min", round(elapsed / 60, 1))
        trial.set_user_attr("daily_mape", result["daily_mape"])
        trial.set_user_attr("da", result.get("da", 0))
        trial.set_user_attr("train_loss", result.get("train_loss", 0))

        study.tell(trial, path_mape)

        seen_hashes.add(ch)
        completed += 1

        print(f"  path_mape={path_mape:.4f}%  daily_mape={result['daily_mape']:.4f}%  "
              f"da={result.get('da', 0):.2f}%  time={elapsed/60:.1f}min  "
              f"[{completed}/{N_TRIALS}]")
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    # Summary
    export_summary(study)
    if len(study.trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        best = study.best_trial
        print(f"\nPhase 8 complete. Best path_mape: {best.value:.4f}% (trial {best.number})")
        print(f"Best params: {best.params}")
    else:
        print(f"\nPhase 8 complete. No successful trials.")


def export_summary(study: optuna.Study):
    import csv
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE: continue
        row = {"trial": t.number, "value": t.value, **t.params}
        for k, v in t.user_attrs.items():
            if isinstance(v, (int, float, str, bool)): row[k] = v
        rows.append(row)
    if not rows: return
    all_keys = set(); [all_keys.update(r.keys()) for r in rows]
    ordered = ["trial", "value"] + sorted(k for k in all_keys if k not in ("trial", "value"))
    os.makedirs(PHASE8_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    ranked = sorted(rows, key=lambda r: r["value"])
    print(f"\nTop-10 by path_mape:")
    for r in ranked[:10]:
        print(f"  Trial {r['trial']:03d}  path_mape={r['value']:.4f}  "
              f"temp={r.get('exploration_temp','?')}  "
              f"neftune={r.get('neftune_alpha','?')}  "
              f"lr={r.get('lr','?'):.1e}  ce_w={r.get('star_ce_weight','?')}")
    print(f"Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
