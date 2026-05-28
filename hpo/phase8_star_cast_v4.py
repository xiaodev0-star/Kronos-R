"""Phase 8-2 V4: Progressive-Sample Direction-Explicit STAR-CAST HPO.

Gradually increases stock count over 3 phases to efficiently explore
the direction classification (Engine 3) parameter space within 11 hours.

Phase 1: 300 stocks, 160 updates (~18min/trial) — coarse exploration
Phase 2: 600 stocks, 200 updates (~32min/trial) — refinement
Phase 3: 1200 stocks, 240 updates (~42min/trial) — precision

Usage:
    python -m hpo.phase8_star_cast_v4
"""

from __future__ import annotations

import copy, json, os, time, hashlib, math, sys
from argparse import Namespace
from contextlib import nullcontext
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np
import optuna
import torch
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
STUDY_NAME = "phase8_star_cast_v4"
CLEAN_START = True  # fresh start

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE8_DIR = os.path.join(PROJECT_ROOT, "trials", "phase8_star_cast_v4")
STUDY_DB = os.path.join(PHASE8_DIR, "study.db")
SUMMARY_CSV = os.path.join(PHASE8_DIR, "summary.csv")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.pt")
TOKENIZER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer_config.json")
BASEMODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "base_model.pt")

TOKENIZER_BITS = 10
TOKENIZER_VOCAB = 1 << TOKENIZER_BITS
PREFIX_LEN = 1023
ROLLOUT_HORIZON = 10
TOTAL_BUDGET_HOURS = 11.0

# ── Progressive phases ──
PHASES = [
    {"stocks": 300,  "updates": 160, "max_trials": 8,  "label": "Phase1-coarse"},
    {"stocks": 600,  "updates": 200, "max_trials": 10, "label": "Phase2-refine"},
    {"stocks": 1200, "updates": 240, "max_trials": 12, "label": "Phase3-precise"},
]

# ── Fixed backbone ──
BACKBONE = {
    "dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
    "dsa_windows": [None, 512, 512],
    "position_encoding": "rope", "rope_base": 10000.0,
    "dropout": 0.1323, "use_revin": False, "num_factor_tokens": 0,
}

# ── Resource config ──
SERVER_BATCH_SIZE     = int(os.environ.get("STARCAST_BS",  "2"))
SERVER_NUM_WORKERS    = int(os.environ.get("STARCAST_NW",  "0"))
SERVER_EVAL_BATCHES   = int(os.environ.get("STARCAST_EVAL", "200"))
SERVER_GRAD_ACCUM     = int(os.environ.get("STARCAST_GA",  "16"))
SERVER_USE_COMPILE    = os.environ.get("STARCAST_COMPILE", "0") == "1"
SERVER_EXPLORE_CHUNK  = int(os.environ.get("STARCAST_CHUNK", "96"))

# ═══════════════════════════════════════════════════════════════════════
# Search space — Phase 8-2 direction classification params
# ═══════════════════════════════════════════════════════════════════════
SEARCH_SPACE = {
    "direction_weight":         (0.1, 0.8),     # Engine 3 loss weight
    "direction_epsilon_scale":  (0.2, 1.0),     # up/down/flat threshold factor
    "direction_ce_flat_weight": (0.1, 0.8),     # FLAT class CE weight
}

# ── Fixed from previous HPO best ──
FIXED_PARAMS = {
    "num_trajectories": 4,
    "exploration_temp": 0.414,          # V2 best
    "neftune_alpha": 2.5,               # V2 best
    "star_ce_weight": 0.334,            # V2 best
    "lr": 9.59e-6,                      # V2 best
    "step_asym_weight": 1.0,
    "path_asym_weight": 1.5,
    "asymmetric_alpha": 3.0,
    "asymmetric_beta": 10.0,
    "path_asymmetric_beta": 15.0,
    "top_k_expected_return": 16,
    # Phase 8-sup improvements (V3 best)
    "timidity_penalty_weight": 1.03,    # V3 best
    "timidity_ratio_threshold": 0.5,
    "oracle_magnitude_penalty": 3.99,   # V3 best
    "prob_sharpening_temp": 0.933,      # V3 best
    "actionable_da_threshold": 0.005,
    "direction_use_class_weights": True,
    "epochs": 1,
}


def sample_params(trial: optuna.Trial) -> dict:
    p = {}
    p["direction_weight"]         = round(trial.suggest_float("direction_weight",         *SEARCH_SPACE["direction_weight"],         log=True), 3)
    p["direction_epsilon_scale"]  = round(trial.suggest_float("direction_epsilon_scale",  *SEARCH_SPACE["direction_epsilon_scale"],  log=True), 3)
    p["direction_ce_flat_weight"] = round(trial.suggest_float("direction_ce_flat_weight", *SEARCH_SPACE["direction_ce_flat_weight"], log=True), 3)
    p.update(FIXED_PARAMS)
    return p


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_rollout_cfg(n_stocks):
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
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════

_BASE_CACHE_DIR = os.path.join(PROJECT_ROOT, "posttrain", "rollout", "cache")

def _build_rollout_data(device, n_stocks):
    cfg = _make_rollout_cfg(n_stocks)
    # Use phase-specific cache files to avoid rebuilds when switching stocks
    phase_cache_dir = os.path.join(_BASE_CACHE_DIR, f"v4_n{n_stocks}")
    os.makedirs(phase_cache_dir, exist_ok=True)
    train_cache = os.path.join(phase_cache_dir, "rollout_train.pt")
    val_cache = os.path.join(phase_cache_dir, "rollout_val.pt")

    train_ds = RolloutWindowDataset("train", cfg=cfg, max_samples=0, seed=42)
    val_ds   = RolloutWindowDataset("val",   cfg=cfg, max_samples=0, seed=59)

    loader_kwargs = dict(
        num_workers=SERVER_NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=SERVER_BATCH_SIZE, shuffle=True,
        collate_fn=rollout_collate, **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=SERVER_BATCH_SIZE * 2, shuffle=False,
        collate_fn=rollout_collate, **loader_kwargs,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


# ═══════════════════════════════════════════════════════════════════════
# Improved STAR-CAST Core (with direction labels)
# ═══════════════════════════════════════════════════════════════════════

def _expected_return_from_topk(tokenizer, logits_c, logits_f, means, stds, top_k,
                                sharpening_temp=1.0):
    B, H, V_c = logits_c.shape
    K = min(int(top_k), V_c)
    top_logits_c, top_idx_c = torch.topk(logits_c.float(), k=K, dim=-1)
    top_logits_f, top_idx_f = torch.topk(logits_f.float(), k=K, dim=-1)
    prob_c = F.softmax(top_logits_c / sharpening_temp, dim=-1)
    prob_f = F.softmax(top_logits_f / sharpening_temp, dim=-1)
    prob_c = prob_c / prob_c.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prob_f = prob_f / prob_f.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    joint_prob = prob_c.unsqueeze(-1) * prob_f.unsqueeze(-2)
    pair_c = top_idx_c.unsqueeze(-1).expand(B, H, K, K).reshape(B * H, K * K)
    pair_f = top_idx_f.unsqueeze(-2).expand(B, H, K, K).reshape(B * H, K * K)
    with torch.no_grad():
        decoded = tokenizer.decode(pair_c, pair_f)[..., 0].float()
        decoded = decoded.view(B, H, K, K)
        returns_grid = decoded * stds[:, 0].view(B, 1, 1, 1) + means[:, 0].view(B, 1, 1, 1)
    return (joint_prob * returns_grid).sum(dim=(-1, -2))


def _asymmetric_direction_loss(expected, actual, alpha, beta, eps=1e-4,
                                timidity_weight=2.0, timidity_ratio=0.5):
    abs_err = torch.abs(expected - actual)
    is_wrong = ((expected * actual) < 0) & (torch.abs(actual) > eps)
    is_correct_but_timid = (
        ((expected * actual) > 0)
        & (torch.abs(expected) < torch.abs(actual) * timidity_ratio)
        & (torch.abs(actual) > eps)
    )
    penalty = torch.ones_like(abs_err)
    penalty = torch.where(is_wrong, alpha + beta * torch.abs(expected), penalty)
    penalty = torch.where(is_correct_but_timid, timidity_weight, penalty)
    return abs_err * penalty


_DIR_LABEL_DOWN = 0
_DIR_LABEL_FLAT = 1
_DIR_LABEL_UP   = 2


def _direction_labels(actual_returns, epsilon_scale=0.5):
    per_sample_abs_mean = torch.mean(torch.abs(actual_returns), dim=1, keepdim=True)
    epsilons = per_sample_abs_mean * epsilon_scale
    labels = torch.full_like(actual_returns, _DIR_LABEL_FLAT, dtype=torch.long)
    labels = torch.where(actual_returns > epsilons,
                         torch.full_like(labels, _DIR_LABEL_UP), labels)
    labels = torch.where(actual_returns < -epsilons,
                         torch.full_like(labels, _DIR_LABEL_DOWN), labels)
    return labels


# ═══════════════════════════════════════════════════════════════════════
# STAR-CAST: Noisy Exploration + Oracle Filter
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _star_cast_exploration(
    model, tokenizer, idx_c_full, idx_f_full,
    time_feats, means, stds, actual_returns,
    prefix_len, horizon, num_traj, temp, neftune_alpha,
    device, amp_enabled, amp_dtype, chunk_size=96,
    oracle_magnitude_penalty=2.0,
):
    B = idx_c_full.size(0); N = num_traj; H = horizon; BN = B * N; CS = min(chunk_size, BN)
    traj_c = torch.empty(B, N, prefix_len + H, dtype=torch.long, device=device)
    traj_f = torch.empty(B, N, prefix_len + H, dtype=torch.long, device=device)
    traj_ret = torch.empty(B, N, H, device=device, dtype=torch.float32)
    traj_c[:, :, :prefix_len] = idx_c_full[:, :prefix_len].unsqueeze(1).expand(B, N, prefix_len)
    traj_f[:, :, :prefix_len] = idx_f_full[:, :prefix_len].unsqueeze(1).expand(B, N, prefix_len)

    for step in range(H):
        cur_len = prefix_len + step
        cur_c = traj_c[:, :, :cur_len].reshape(BN, cur_len)
        cur_f = traj_f[:, :, :cur_len].reshape(BN, cur_len)
        cur_time_all = {k: time_feats[k][:, :cur_len].unsqueeze(1).expand(B, N, cur_len).reshape(BN, cur_len)
                        for k in ("minute", "day", "month", "year")}
        logits_c_chunks, logits_f_chunks = [], []
        for chunk_start in range(0, BN, CS):
            chunk_end = min(chunk_start + CS, BN)
            cc = cur_c[chunk_start:chunk_end]; cf = cur_f[chunk_start:chunk_end]
            ct = {k: v[chunk_start:chunk_end] for k, v in cur_time_all.items()}
            with _autocast_ctx(amp_enabled, amp_dtype):
                lc, lf, _ = model(cc, cf, ct["minute"], ct["day"], ct["month"], ct["year"],
                                  last_only=True, neftune_alpha=neftune_alpha)
            logits_c_chunks.append(lc[:, -1, :].float()); logits_f_chunks.append(lf[:, -1, :].float())
        logits_c_all = torch.cat(logits_c_chunks, dim=0); logits_f_all = torch.cat(logits_f_chunks, dim=0)
        probs_c = F.softmax(logits_c_all / max(1e-4, temp), dim=-1)
        probs_f = F.softmax(logits_f_all / max(1e-4, temp), dim=-1)
        pred_c = torch.multinomial(probs_c, num_samples=1); pred_f = torch.multinomial(probs_f, num_samples=1)
        decoded = tokenizer.decode(pred_c, pred_f)[..., 0].float()
        step_ret = (decoded * stds.unsqueeze(1).expand(B, N, 6).reshape(BN, 6)[:, 0:1]
                    + means.unsqueeze(1).expand(B, N, 6).reshape(BN, 6)[:, 0:1])
        traj_ret[:, :, step] = step_ret.view(B, N)
        traj_c[:, :, cur_len] = pred_c.view(B, N); traj_f[:, :, cur_len] = pred_f.view(B, N)

    path_returns = traj_ret.sum(dim=2)
    actual_path = actual_returns[:, :H].sum(dim=1)
    correct_dir = (path_returns * actual_path.unsqueeze(1)) > 0
    is_valid = correct_dir.any(dim=1) & (torch.abs(actual_path) > 1e-6)
    errors = torch.abs(path_returns - actual_path.unsqueeze(1))
    mag_penalty = torch.clamp(torch.abs(actual_path.unsqueeze(1)) - torch.abs(path_returns), min=0)
    errors = errors + oracle_magnitude_penalty * mag_penalty
    errors[~correct_dir] = float('inf')
    best_idx = errors.argmin(dim=1)
    gather_idx = best_idx.view(B, 1, 1).expand(B, 1, prefix_len + H)
    golden_c = traj_c.gather(1, gather_idx).squeeze(1)
    golden_f = traj_f.gather(1, gather_idx).squeeze(1)
    gt_c = idx_c_full[:, :prefix_len + H]; gt_f = idx_f_full[:, :prefix_len + H]
    mask = is_valid.view(B, 1).float()
    golden_c = (golden_c.float() * mask + gt_c.float() * (1 - mask)).long()
    golden_f = (golden_f.float() * mask + gt_f.float() * (1 - mask)).long()
    return golden_c, golden_f, is_valid


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def _train_star_cast(model, tokenizer, params: dict, tdir: str, device,
                     n_stocks: int) -> dict:
    p = params
    result_path = os.path.join(tdir, "result.json")
    resume_path = os.path.join(tdir, "star_cast_resume.pt")
    os.makedirs(tdir, exist_ok=True)

    if os.path.exists(result_path):
        with open(result_path) as f: return json.load(f)

    train_loader, val_loader, n_train, n_val = _build_rollout_data(device, n_stocks)
    print(f"  Data: {n_stocks} stocks, train={n_train}, val={n_val}")

    lr = p["lr"]
    max_updates = p["max_updates"]
    ga_steps = max(1, SERVER_GRAD_ACCUM)

    opt = optim.AdamW(model.parameters(), lr=lr, fused=True if device.type == "cuda" else False)
    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    if SERVER_USE_COMPILE and hasattr(torch, "compile"):
        try: model = torch.compile(model, mode="reduce-overhead"); print("  torch.compile ON")
        except Exception as e: print(f"  compile failed ({e})")

    # Cosine annealing with warmup
    warmup_steps = max(2, max_updates // 10)
    cosine_t_max = max_updates - warmup_steps
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, cosine_t_max), eta_min=lr * 0.05)

    update_count = 0
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        update_count = ckpt["update_count"]
        print(f"  RESUME from update {update_count}")

    prefix_len = PREFIX_LEN; horizon = ROLLOUT_HORIZON
    total_updates_done = update_count; total_loss_sum = 0.0

    model.train()
    pbar = tqdm(total=max_updates - update_count, desc="  STAR-CAST")
    opt.zero_grad(set_to_none=True)
    microbatch_count = 0

    while total_updates_done < max_updates:
        for batch in train_loader:
            if total_updates_done >= max_updates: break
            feats  = batch["features"].to(device=device, dtype=torch.float32, non_blocking=True)
            means  = batch["means"].to(device=device, dtype=torch.float32, non_blocking=True)
            stds   = batch["stds"].to(device=device, dtype=torch.float32, non_blocking=True)
            actual = batch["actual_returns"].to(device=device, dtype=torch.float32, non_blocking=True)
            times  = {k: v.to(device=device, dtype=torch.long, non_blocking=True) for k, v in batch["time"].items()}
            B = feats.size(0)
            if B == 0: continue

            idx_c_full, idx_f_full = tokenizer.encode(feats)
            actual_h = actual[:, :horizon]

            golden_c, golden_f, has_golden = _star_cast_exploration(
                model, tokenizer, idx_c_full, idx_f_full,
                times, means, stds, actual,
                prefix_len, horizon,
                p["num_trajectories"], p["exploration_temp"], p["neftune_alpha"],
                device, use_amp, amp_dtype, chunk_size=SERVER_EXPLORE_CHUNK,
                oracle_magnitude_penalty=p["oracle_magnitude_penalty"],
            )

            model.train()
            train_len = golden_c.size(1)
            train_time = {k: v[:, :train_len] for k, v in times.items()}

            with _autocast_ctx(use_amp, amp_dtype):
                logits_c, logits_f, latent_states, hidden = model(
                    golden_c[:, :-1], golden_f[:, :-1],
                    train_time["minute"][:, :train_len - 1],
                    train_time["day"][:, :train_len - 1],
                    train_time["month"][:, :train_len - 1],
                    train_time["year"][:, :train_len - 1],
                    return_hidden=True, neftune_alpha=0.0,
                )
                start = prefix_len - 1
                rollout_c = logits_c[:, start:start + horizon, :]
                rollout_f = logits_f[:, start:start + horizon, :]

                expected = _expected_return_from_topk(
                    tokenizer, rollout_c, rollout_f, means, stds,
                    p["top_k_expected_return"], sharpening_temp=p["prob_sharpening_temp"])
                step_loss = _asymmetric_direction_loss(
                    expected, actual_h, p["asymmetric_alpha"], p["asymmetric_beta"],
                    timidity_weight=p["timidity_penalty_weight"],
                    timidity_ratio=p["timidity_ratio_threshold"]).mean()
                expected_path = torch.cumsum(expected, dim=1)
                actual_path = torch.cumsum(actual_h, dim=1)
                path_loss = _asymmetric_direction_loss(
                    expected_path, actual_path,
                    p["asymmetric_alpha"] * 1.3, p["path_asymmetric_beta"],
                    timidity_weight=p["timidity_penalty_weight"],
                    timidity_ratio=p["timidity_ratio_threshold"]).mean()

                if has_golden.any():
                    target_c = golden_c[has_golden, prefix_len:prefix_len + horizon]
                    target_f = golden_f[has_golden, prefix_len:prefix_len + horizon]
                    ce_c = F.cross_entropy(rollout_c[has_golden].reshape(-1, rollout_c.size(-1)).float(), target_c.reshape(-1))
                    ce_f = F.cross_entropy(rollout_f[has_golden].reshape(-1, rollout_f.size(-1)).float(), target_f.reshape(-1))
                    star_ce = ce_c + ce_f
                else:
                    star_ce = torch.tensor(0.0, device=device)

                # Engine 3: Direction-Explicit Classification
                if p.get("direction_weight", 0.0) > 0.0:
                    dir_labels = _direction_labels(actual_h, p["direction_epsilon_scale"])
                    dir_logits = model.compute_direction_logits_at_positions(
                        hidden, latent_states, start=start, end=start + horizon)
                    flat_w = p["direction_ce_flat_weight"]
                    cw = torch.tensor([1.0, flat_w, 1.0], device=device, dtype=dir_logits.dtype)
                    dir_loss = F.cross_entropy(dir_logits.reshape(-1, 3).float(), dir_labels.reshape(-1), weight=cw)
                else:
                    dir_loss = torch.tensor(0.0, device=device)

                loss = (p["step_asym_weight"] * step_loss +
                        p["path_asym_weight"] * path_loss +
                        p["star_ce_weight"] * star_ce +
                        p.get("direction_weight", 0.0) * dir_loss)
                loss = loss / ga_steps

            if not torch.isfinite(loss): continue

            scaler.scale(loss).backward()
            microbatch_count += 1

            if microbatch_count % ga_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                if total_updates_done < warmup_steps:
                    warmup_sched.step()
                else:
                    cosine_sched.step()
                total_loss_sum += loss.item() * ga_steps
                total_updates_done += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item() * ga_steps:.3f}",
                                 golden=f"{has_golden.float().mean().item():.2f}",
                                 dir_loss=f"{dir_loss.item():.3f}",
                                 lr=f"{opt.param_groups[0]['lr']:.2e}")

            if total_updates_done % 60 == 0 and total_updates_done > 0:
                torch.save({"update_count": total_updates_done,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict()}, resume_path)
            if total_updates_done >= max_updates: break

    pbar.close()
    torch.save({"model_state_dict": model.state_dict(), "update_count": total_updates_done},
               os.path.join(tdir, "star_cast_model.pt"))

    eval_result = _eval_rollout(model, tokenizer, val_loader, device,
                                actionable_da_threshold=p["actionable_da_threshold"])
    if os.path.exists(resume_path): os.remove(resume_path)

    result = {**eval_result,
              "train_loss": round(total_loss_sum / max(total_updates_done, 1), 6),
              "total_updates": total_updates_done, "params": p, "n_stocks": n_stocks}
    with open(result_path, "w") as f: json.dump(result, f, indent=2)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval_rollout(model, tokenizer, val_loader, device, max_batches=None,
                  actionable_da_threshold=0.005) -> dict:
    model.eval()
    all_path_mape, all_daily_mape, all_da = [], [], []
    all_actionable_da, all_actionable_ratio = [], []
    prefix_len = PREFIX_LEN; horizon = ROLLOUT_HORIZON
    limit = max_batches if max_batches is not None else SERVER_EVAL_BATCHES

    n_batches = 0
    for batch in tqdm(val_loader, desc="  Eval", leave=False):
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
        cur_c = idx_c[:, :prefix_len].clone(); cur_f = idx_f[:, :prefix_len].clone()
        actual_rets = actual.cpu()
        pred_rets = []
        for step in range(horizon):
            sl = int(cur_c.size(1))
            cur_time = {"minute": times["minute"][:, :sl], "day": times["day"][:, :sl],
                        "month": times["month"][:, :sl], "year": times["year"][:, :sl]}
            logits_c, logits_f, _ = model(cur_c, cur_f, cur_time["minute"], cur_time["day"],
                                          cur_time["month"], cur_time["year"], last_only=True)
            if not torch.isfinite(logits_c).all(): break
            pc = logits_c[:, -1, :].argmax(dim=-1); pf = logits_f[:, -1, :].argmax(dim=-1)
            dec = tokenizer.decode(pc.unsqueeze(1), pf.unsqueeze(1))
            pred_ret = dec[:, 0, 0].cpu().float() * stds[:, 0].cpu() + means[:, 0].cpu()
            pred_rets.append(pred_ret)
            if step < horizon - 1:
                cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
                cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)
        if len(pred_rets) < horizon: continue
        pred_rets = torch.stack(pred_rets, dim=1)

        cum_pred = torch.cumsum(pred_rets.float(), dim=1)
        cum_actual = torch.cumsum(actual_rets.float(), dim=1)
        for step in range(horizon):
            pr = torch.exp(torch.clamp(cum_pred[:, step], -20, 20))
            ar = torch.exp(torch.clamp(cum_actual[:, step], -20, 20))
            denom = torch.clamp(torch.abs(ar), min=1e-6)
            valid = torch.isfinite(pr) & torch.isfinite(ar) & (denom > 0)
            if valid.sum() > 0:
                all_path_mape.append((torch.abs(pr[valid] - ar[valid]) / denom[valid]).mean().item() * 100)
            dr = torch.exp(torch.clamp(pred_rets[:, step].float(), -20, 20))
            da_val = torch.exp(torch.clamp(actual_rets[:, step].float(), -20, 20))
            denom_d = torch.clamp(torch.abs(da_val), min=1e-6)
            valid_d = torch.isfinite(dr) & torch.isfinite(da_val) & (denom_d > 0)
            if valid_d.sum() > 0:
                all_daily_mape.append((torch.abs(dr[valid_d] - da_val[valid_d]) / denom_d[valid_d]).mean().item() * 100)

        pred_sign = (pred_rets >= 0).float() * 2 - 1
        actual_sign = (actual_rets >= 0).float() * 2 - 1
        all_da.append((pred_sign == actual_sign).float().mean().item() * 100)
        confident_mask = torch.abs(pred_rets) > actionable_da_threshold
        if confident_mask.sum() > 0:
            all_actionable_da.append((pred_sign[confident_mask] == actual_sign[confident_mask]).float().mean().item() * 100)
            all_actionable_ratio.append(confident_mask.float().mean().item() * 100)

    if not all_path_mape:
        return {"path_mape": 999.0, "daily_mape": 999.0, "da": 0.0,
                "actionable_da": 0.0, "actionable_ratio": 0.0, "path_mape_std": 0.0, "num_eval_steps": 0}

    result = {"path_mape": round(float(np.mean(all_path_mape)), 6),
              "daily_mape": round(float(np.mean(all_daily_mape)), 6),
              "da": round(float(np.mean(all_da)), 4),
              "path_mape_std": round(float(np.std(all_path_mape)), 6),
              "num_eval_steps": len(all_path_mape)}
    if all_actionable_da:
        result["actionable_da"] = round(float(np.mean(all_actionable_da)), 4)
        result["actionable_ratio"] = round(float(np.mean(all_actionable_ratio)), 4)
    else:
        result["actionable_da"] = 0.0; result["actionable_ratio"] = 0.0
    return result


# ═══════════════════════════════════════════════════════════════════════
# Trial management
# ═══════════════════════════════════════════════════════════════════════

def _assign_trial_dir(phase_label):
    phase_dir = os.path.join(PHASE8_DIR, phase_label)
    os.makedirs(phase_dir, exist_ok=True)
    counter_path = os.path.join(phase_dir, ".next_trial")

    existing = sorted([d for d in os.listdir(phase_dir)
                       if d.startswith("trial_") and os.path.isdir(os.path.join(phase_dir, d))],
                      key=lambda x: int(x.split("_")[1]))
    for d in existing:
        full = os.path.join(phase_dir, d)
        resume = os.path.join(full, "star_cast_resume.pt")
        result = os.path.join(full, "result.json")
        if os.path.exists(resume) and not os.path.exists(result):
            if os.path.exists(os.path.join(full, "config.json")):
                print(f"  Found incomplete trial: {phase_label}/{d} — RESUME")
                return full

    next_num = 0
    if os.path.exists(counter_path):
        with open(counter_path) as f: next_num = int(f.read().strip())
    new_dir = os.path.join(phase_dir, f"trial_{next_num:03d}")
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
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if device.type == "cuda" else 0
    print(f"Phase 8-2 V4 — Progressive-Sample Direction-Explicit HPO")
    print(f"  Device: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    print(f"  Budget: {TOTAL_BUDGET_HOURS}h")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    for ph in PHASES:
        print(f"  {ph['label']}: {ph['stocks']} stocks, {ph['updates']} updates, max {ph['max_trials']} trials")
    print(f"  Search: {list(SEARCH_SPACE.keys())}")
    print(f"  Fixed base: oracle_mag={FIXED_PARAMS['oracle_magnitude_penalty']}, "
          f"timidity_w={FIXED_PARAMS['timidity_penalty_weight']}, "
          f"sharpening={FIXED_PARAMS['prob_sharpening_temp']}")
    print()

    print("Loading tokenizer + BaseModel...")
    tokenizer = _load_tokenizer(device)
    base_model = _load_basemodel(device)
    print(f"  Params: {sum(p.numel() for p in base_model.parameters()):,}")
    print()

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=f"sqlite:///{STUDY_DB}",
        direction="minimize", load_if_exists=True,
    )

    start_time = time.time()
    deadline = start_time + TOTAL_BUDGET_HOURS * 3600
    seen_hashes = set()
    global_completed = 0
    all_results = []  # (path_mape, params, phase_label, n_stocks)

    for phase_idx, phase in enumerate(PHASES):
        n_stocks = phase["stocks"]
        phase_label = phase["label"]
        max_updates = phase["updates"]
        max_trials = phase["max_trials"]

        # Update fixed params for this phase
        phase_params = dict(FIXED_PARAMS)
        phase_params["max_updates"] = max_updates

        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 1200:  # < 20 min
            print(f"\n[{phase_label}] Time budget exhausted ({elapsed/3600:.1f}h) — stopping")
            break

        print(f"\n{'#'*70}")
        print(f"# {phase_label}: {n_stocks} stocks, {max_updates} updates, max {max_trials} trials")
        print(f"# Elapsed: {elapsed/3600:.1f}h, Remaining: {remaining/3600:.1f}h")
        print(f"{'#'*70}")

        phase_completed = 0
        while phase_completed < max_trials:
            elapsed = time.time() - start_time
            remaining = deadline - time.time()
            # Estimate if we have time for another trial
            if phase_idx == 0: est_min = 20
            elif phase_idx == 1: est_min = 35
            else: est_min = 45
            if remaining < est_min * 60 + 300:  # need trial time + 5min margin
                print(f"  [{phase_label}] Not enough time for another trial "
                      f"(remaining={remaining/60:.0f}min, need ~{est_min}min)")
                break

            trial = study.ask()
            tdir = _assign_trial_dir(phase_label)
            os.makedirs(tdir, exist_ok=True)

            params = sample_params(trial)
            params.update(phase_params)
            config_path = os.path.join(tdir, "config.json")

            if os.path.exists(config_path):
                with open(config_path) as f: params = json.load(f)
            else:
                with open(config_path, "w") as f: json.dump(params, f, indent=2)

            ch = _config_hash(params)
            if ch in seen_hashes:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue

            print(f"\n  [{phase_label}] Trial {trial.number:03d} (dir={os.path.basename(tdir)})")
            print(f"    dir_w={params['direction_weight']:.3f}  "
                  f"eps_scale={params['direction_epsilon_scale']:.3f}  "
                  f"flat_w={params['direction_ce_flat_weight']:.3f}")
            print(f"    stocks={n_stocks}  updates={max_updates}")

            model = copy.deepcopy(base_model)
            t0 = time.time()
            try:
                result = _train_star_cast(model, tokenizer, params, tdir, device, n_stocks)
                path_mape = result["path_mape"]
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback; traceback.print_exc()
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                continue

            elapsed_trial = time.time() - t0
            elapsed_total = time.time() - start_time

            trial.set_user_attr("phase", phase_label)
            trial.set_user_attr("n_stocks", n_stocks)
            trial.set_user_attr("elapsed_min", round(elapsed_trial / 60, 1))
            trial.set_user_attr("daily_mape", result["daily_mape"])
            trial.set_user_attr("da", result.get("da", 0))
            trial.set_user_attr("actionable_da", result.get("actionable_da", 0))
            trial.set_user_attr("actionable_ratio", result.get("actionable_ratio", 0))
            trial.set_user_attr("train_loss", result.get("train_loss", 0))

            study.tell(trial, path_mape)
            seen_hashes.add(ch)
            phase_completed += 1
            global_completed += 1
            all_results.append((path_mape, params, phase_label, n_stocks))

            print(f"    path_mape={path_mape:.4f}%  da={result.get('da',0):.2f}%  "
                  f"act_da={result.get('actionable_da',0):.2f}% "
                  f"(ratio={result.get('actionable_ratio',0):.1f}%)  "
                  f"time={elapsed_trial/60:.1f}min")
            print(f"    Phase: {phase_completed}/{max_trials}  "
                  f"Total: {global_completed}  "
                  f"Elapsed: {elapsed_total/3600:.1f}h  "
                  f"Remaining: {(deadline - time.time())/3600:.1f}h")

            del model
            if device.type == "cuda": torch.cuda.empty_cache()

            # Save intermediate summary
            export_summary(study, all_results)

        if remaining < 600:  # < 10 min
            break

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"HPO complete. Total: {total_time/3600:.1f}h, trials: {global_completed}")
    export_summary(study, all_results)

    if all_results:
        all_results.sort(key=lambda x: x[0])
        print(f"\n{'='*70}")
        print(f"Top-10 by path_mape:")
        for i, (mape, p, label, stocks) in enumerate(all_results[:10]):
            print(f"  #{i+1}: path_mape={mape:.4f}%  [{label}]  "
                  f"dir_w={p['direction_weight']:.3f}  "
                  f"eps_scale={p['direction_epsilon_scale']:.3f}  "
                  f"flat_w={p['direction_ce_flat_weight']:.3f}  "
                  f"stocks={stocks}")

        best = all_results[0]
        print(f"\nBest: path_mape={best[0]:.4f}%  [{best[2]}]  "
              f"dir_w={best[1]['direction_weight']:.3f}  "
              f"eps_scale={best[1]['direction_epsilon_scale']:.3f}  "
              f"flat_w={best[1]['direction_ce_flat_weight']:.3f}")


def export_summary(study: optuna.Study, all_results: list = None):
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

    if rows:
        ranked = sorted(rows, key=lambda r: r["value"])
        print(f"\n  Current top-5:")
        for r in ranked[:5]:
            print(f"    {r['trial']:03d}  path_mape={r['value']:.4f}  "
                  f"dir_w={r.get('direction_weight','?')}  "
                  f"eps_scale={r.get('direction_epsilon_scale','?')}  "
                  f"flat_w={r.get('direction_ce_flat_weight','?')}  "
                  f"da={r.get('da','?')}  act_da={r.get('actionable_da','?')}")


if __name__ == "__main__":
    main()
