"""Exploration: Diagnose and fix the zero-collapse root cause.

Hypothesis: soft expectation E[r] = sum(P_i * r_i) collapses to ~0 because
the top-K token distribution is nearly symmetric around zero for financial
time series. This is an architectural problem, not a training problem.

Experiments:
  1. Diagnose: analyze top-K token distribution symmetry
  2. Gumbel-Softmax: replace soft expectation with differentiable sampling
  3. Straight-Through: forward=argmax, backward=softmax gradient
  4. Compare prediction distributions
"""
import os, sys, math, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace

from evaluate_predictions import load_model
from posttrain.rollout.data import RolloutWindowDataset, rollout_collate, resolve_project_path
from posttrain.rollout.train_rollout import _amp_dtype, _autocast_context, _move_batch, _encode_features

device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high"); torch.cuda.empty_cache()
amp_dtype = _amp_dtype("bfloat16"); amp_enabled = True

# Load best model
V4_CKPT = resolve_project_path("trials/phase8_star_cast_v4/Phase2-refine/trial_009/star_cast_model.pt")
print("Loading model...")
model, tokenizer = load_model(device=device, checkpoint_path=V4_CKPT, strict_checkpoint_compat=False)
tokenizer.eval(); tokenizer.requires_grad_(False); model.eval()

# Data
cfg = Namespace(prefix_len=1023, horizon=10, stride_ratio=0.5,
    cache_dir=resolve_project_path("posttrain/rollout/cache"),
    max_stocks=100, cache_rebuild=False, mape_eps=1e-4)
dataset = RolloutWindowDataset("val", cfg=cfg, max_samples=0, seed=42)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=rollout_collate, num_workers=0)
print(f"Val windows: {len(dataset)}")

TOP_K = 16

# ═══════════════════════════════════════════════════════════════════
# Experiment 1: Analyze top-K token symmetry
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Experiment 1: Top-K token distribution symmetry analysis")
print("="*60)

all_soft_expectations = []
all_argmax_returns = []
all_gumbel_returns = []
all_token_symmetries = []  # measure of pos/neg balance in top-K

for batch_idx, batch in enumerate(loader):
    if batch_idx >= 20: break  # 20 batches = 80 samples
    batch = _move_batch(batch, device)
    idx_c, idx_f = _encode_features(tokenizer, batch["features"])
    B = idx_c.size(0)
    means = batch["means"]; stds = batch["stds"]

    with torch.no_grad():
        with _autocast_context(device, amp_enabled, amp_dtype):
            logits_c, logits_f, _ = model(
                idx_c[:, :1023], idx_f[:, :1023],
                batch["time"]["minute"][:, :1023],
                batch["time"]["day"][:, :1023],
                batch["time"]["month"][:, :1023],
                batch["time"]["year"][:, :1023],
                last_only=True,
            )
        lc = logits_c[:, -1, :].float()  # [B, Vc]
        lf = logits_f[:, -1, :].float()  # [B, Vf]
        K = min(TOP_K, lc.size(-1))

        # Current: soft expectation
        top_logits_c, top_idx_c = torch.topk(lc, k=K, dim=-1)
        top_logits_f, top_idx_f = torch.topk(lf, k=K, dim=-1)
        prob_c = F.softmax(top_logits_c, dim=-1)
        prob_f = F.softmax(top_logits_f, dim=-1)

        joint_prob = prob_c.unsqueeze(-1) * prob_f.unsqueeze(-2)  # [B, K, K]
        # Use same pattern as train_star_cast.py with H=1
        # top_idx_c: [B, K], top_idx_f: [B, K]
        H_dim = 1  # single position
        pair_c = top_idx_c.unsqueeze(1).unsqueeze(-1).expand(B, H_dim, K, K).reshape(B * H_dim, K * K)
        pair_f = top_idx_f.unsqueeze(1).unsqueeze(-2).expand(B, H_dim, K, K).reshape(B * H_dim, K * K)
        decoded = tokenizer.decode(pair_c, pair_f)[..., 0].float()  # [B*1, K*K]
        returns_grid = decoded.view(B, H_dim, K, K).reshape(B, -1) * stds[:, 0:1] + means[:, 0:1]

        soft_expectation = (joint_prob.reshape(B, -1) * returns_grid).sum(dim=-1)
        all_soft_expectations.extend(soft_expectation.cpu().tolist())

        # Argmax
        am_c = top_idx_c[:, 0]; am_f = top_idx_f[:, 0]
        am_dec = tokenizer.decode(am_c.unsqueeze(1), am_f.unsqueeze(1))[:, 0, 0].float()
        am_ret = am_dec * stds[:, 0] + means[:, 0]
        all_argmax_returns.extend(am_ret.cpu().tolist())

        # Token symmetry: ratio of positive returns in top-K decoded values
        for b in range(B):
            rets_b = returns_grid[b].cpu().numpy()
            pos_ratio = np.mean(rets_b > 0)
            # symmetry score: 0.5 = perfectly symmetric, 0 or 1 = all one sign
            symmetry = 1.0 - abs(pos_ratio - 0.5) * 2  # 1 = perfect symmetry, 0 = all one sign
            all_token_symmetries.append(symmetry)

        # Gumbel-softmax sample (hard, straight-through style)
        gumbel_c = -torch.log(-torch.log(torch.rand_like(top_logits_c).clamp_min(1e-8) + 1e-8) + 1e-8)
        gumbel_f = -torch.log(-torch.log(torch.rand_like(top_logits_f).clamp_min(1e-8) + 1e-8) + 1e-8)
        gumbel_logits_c = (top_logits_c + gumbel_c) / 0.5
        gumbel_logits_f = (top_logits_f + gumbel_f) / 0.5
        gs_idx_c = gumbel_logits_c.argmax(dim=-1)
        gs_idx_f = gumbel_logits_f.argmax(dim=-1)
        gs_c = top_idx_c[torch.arange(B), gs_idx_c]
        gs_f = top_idx_f[torch.arange(B), gs_idx_f]
        gs_dec = tokenizer.decode(gs_c.unsqueeze(1), gs_f.unsqueeze(1))[:, 0, 0].float()
        gs_ret = gs_dec * stds[:, 0] + means[:, 0]
        all_gumbel_returns.extend(gs_ret.cpu().tolist())

print(f"\n  Samples analyzed: {len(all_soft_expectations)}")

soft_arr = np.array(all_soft_expectations)
am_arr = np.array(all_argmax_returns)
gs_arr = np.array(all_gumbel_returns)
sym_arr = np.array(all_token_symmetries)

print(f"\n  Soft Expectation:  mean={np.mean(soft_arr):.6f}  median={np.median(soft_arr):.6f}  "
      f"std={np.std(soft_arr):.6f}  |mean|={np.mean(np.abs(soft_arr)):.6f}")
print(f"  Argmax:            mean={np.mean(am_arr):.6f}  median={np.median(am_arr):.6f}  "
      f"std={np.std(am_arr):.6f}  |mean|={np.mean(np.abs(am_arr)):.6f}")
print(f"  Gumbel-Softmax:    mean={np.mean(gs_arr):.6f}  median={np.median(gs_arr):.6f}  "
      f"std={np.std(gs_arr):.6f}  |mean|={np.mean(np.abs(gs_arr)):.6f}")

print(f"\n  Fraction near zero (|pred| < 0.001):")
print(f"    Soft:  {np.mean(np.abs(soft_arr)<0.001)*100:.1f}%")
print(f"    Argmax:{np.mean(np.abs(am_arr)<0.001)*100:.1f}%")
print(f"    Gumbel:{np.mean(np.abs(gs_arr)<0.001)*100:.1f}%")

print(f"\n  Top-K token symmetry (1.0=perfectly symmetric):")
print(f"    Mean: {np.mean(sym_arr):.4f}  Median: {np.median(sym_arr):.4f}")
print(f"    P25: {np.percentile(sym_arr, 25):.4f}  P75: {np.percentile(sym_arr, 75):.4f}")

# ═══════════════════════════════════════════════════════════════════
# Experiment 2: Replace soft expectation with Gumbel-Softmax in expected returns
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Experiment 2: Gumbel-Softmax expected returns vs Soft expectation")
print("="*60)

def gumbel_softmax_expected_returns(tokenizer, logits_c, logits_f, means, stds, top_k=16, temp=0.5, hard=True):
    """Gumbel-Softmax version: sample instead of average.

    When hard=True: forward uses argmax (non-zero prediction), backward uses softmax gradient.
    When hard=False: forward uses soft sample (still has some averaging but less than pure softmax).
    """
    B, H, V_c = logits_c.shape
    K = min(int(top_k), V_c)

    top_logits_c, top_idx_c = torch.topk(logits_c.float(), k=K, dim=-1)  # [B, H, K]
    top_logits_f, top_idx_f = torch.topk(logits_f.float(), k=K, dim=-1)

    # Gumbel noise
    gumbel_c = -torch.log(-torch.log(torch.rand_like(top_logits_c).clamp_min(1e-8) + 1e-8) + 1e-8)
    gumbel_f = -torch.log(-torch.log(torch.rand_like(top_logits_f).clamp_min(1e-8) + 1e-8) + 1e-8)

    # Soft sample
    soft_c = F.softmax((top_logits_c + gumbel_c) / temp, dim=-1)  # [B, H, K]
    soft_f = F.softmax((top_logits_f + gumbel_f) / temp, dim=-1)

    if hard:
        # Straight-through: forward = one-hot(argmax), backward = softmax grad
        hard_c = F.one_hot(soft_c.argmax(dim=-1), num_classes=K).float()
        hard_f = F.one_hot(soft_f.argmax(dim=-1), num_classes=K).float()
        prob_c = hard_c - soft_c.detach() + soft_c  # STE
        prob_f = hard_f - soft_f.detach() + soft_f
    else:
        prob_c = soft_c
        prob_f = soft_f

    prob_c = prob_c / prob_c.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prob_f = prob_f / prob_f.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # Joint probability
    joint_prob = prob_c.unsqueeze(-1) * prob_f.unsqueeze(-2)  # [B, H, K, K]

    # Decode all K*K pairs
    pair_c = top_idx_c.unsqueeze(-1).expand(B, H, K, K).reshape(B * H, K * K)
    pair_f = top_idx_f.unsqueeze(-2).expand(B, H, K, K).reshape(B * H, K * K)

    with torch.no_grad():
        decoded = tokenizer.decode(pair_c, pair_f)[..., 0].float()
        decoded = decoded.view(B, H, K, K)
        returns_grid = decoded * stds[:, 0].view(B, 1, 1, 1) + means[:, 0].view(B, 1, 1, 1)

    # Expected return (weighted by sharp/one-hot probs)
    expected_returns = (joint_prob * returns_grid).sum(dim=(-1, -2))  # [B, H]
    return expected_returns


def soft_expected_returns(tokenizer, logits_c, logits_f, means, stds, top_k=16, sharpening_temp=1.0):
    """Original soft expectation (baseline)."""
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


# Collect distributions over multiple samples
soft_exps = []
gs_soft_exps = []   # Gumbel-softmax (soft sample)
gs_hard_exps = []   # Gumbel-softmax (hard/STE)

for batch_idx, batch in enumerate(loader):
    if batch_idx >= 20: break
    batch = _move_batch(batch, device)
    idx_c, idx_f = _encode_features(tokenizer, batch["features"])
    B = idx_c.size(0); H = 10

    with torch.no_grad():
        # We need hidden states for multi-step — just do 1-step for this test
        with _autocast_context(device, amp_enabled, amp_dtype):
            logits_c, logits_f, _ = model(
                idx_c[:, :1023], idx_f[:, :1023],
                batch["time"]["minute"][:, :1023],
                batch["time"]["day"][:, :1023],
                batch["time"]["month"][:, :1023],
                batch["time"]["year"][:, :1023],
                last_only=True,
            )
        lc = logits_c[:, -1:, :].float()
        lf = logits_f[:, -1:, :].float()

        # Soft expectation (current)
        se = soft_expected_returns(tokenizer, lc, lf, batch["means"], batch["stds"], TOP_K, sharpening_temp=0.5)
        soft_exps.extend(se.reshape(-1).cpu().tolist())

        # Gumbel-softmax (soft sample, temp=0.5)
        gs_s = gumbel_softmax_expected_returns(tokenizer, lc, lf, batch["means"], batch["stds"], TOP_K, temp=0.5, hard=False)
        gs_soft_exps.extend(gs_s.reshape(-1).cpu().tolist())

        # Gumbel-softmax (hard/STE, temp=0.5)
        gs_h = gumbel_softmax_expected_returns(tokenizer, lc, lf, batch["means"], batch["stds"], TOP_K, temp=0.5, hard=True)
        gs_hard_exps.extend(gs_h.reshape(-1).cpu().tolist())

se_arr = np.array(soft_exps)
gs_s_arr = np.array(gs_soft_exps)
gs_h_arr = np.array(gs_hard_exps)

print(f"\n  Multi-step (H=1) expected return distributions:")
print(f"  {'Method':<25} {'|Mean|':>10} {'Std':>10} {'%|<0.001':>10} {'%|=0':>10}")
print(f"  {'-'*55}")
for name, arr in [("Soft Expectation", se_arr), ("Gumbel-Soft (soft)", gs_s_arr), ("Gumbel-Hard (STE)", gs_h_arr)]:
    near_zero = np.mean(np.abs(arr) < 0.001) * 100
    exactly_zero = np.mean(np.abs(arr) < 1e-8) * 100
    print(f"  {name:<25} {np.mean(np.abs(arr)):>10.6f} {np.std(arr):>10.6f} {near_zero:>10.1f} {exactly_zero:>10.1f}")

# ═══════════════════════════════════════════════════════════════════
# Experiment 3: Train 20 steps with Gumbel-Hard (STE) and compare
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Experiment 3: 20-step training comparison (Soft vs Gumbel-STE)")
print("="*60)

from posttrain.rollout.train_star_cast import compute_asymmetric_direction_loss, compute_direction_labels
from posttrain.rollout.train_star_cast import _star_cast_exploration
from posttrain.rollout.train_rollout import _configure_trainable, _build_optimizer
import copy

# Quick config
train_cfg = Namespace(
    prefix_len=1023, horizon=10, stride_ratio=0.5,
    cache_dir=resolve_project_path("posttrain/rollout/cache"),
    max_stocks=50, cache_rebuild=False,
    batch_size=2, epochs=1,
    num_trajectories=4, exploration_temperature=0.414, neftune_alpha=2.5,
    top_k_expected_return=16,
    asymmetric_alpha=3.0, asymmetric_beta=10.0,
    path_asymmetric_alpha=4.0, path_asymmetric_beta=15.0,
    step_asym_weight=1.0, path_asym_weight=1.5, star_ce_weight=0.334,
    timidity_penalty_weight=1.03, timidity_ratio_threshold=0.5,
    oracle_magnitude_penalty=3.99, prob_sharpening_temp=0.933,
    direction_weight=0.336, direction_epsilon_scale=0.361,
    direction_ce_flat_weight=0.540, direction_use_class_weights=True,
    mape_eps=1e-4, accumulation_steps=1,
    learning_rate=9.59e-6, weight_decay=1e-4, grad_clip=0.3,
    use_gradient_checkpointing=False,
    freeze_backbone=False, trainable_scope="all",
    use_amp=True, amp_dtype="bfloat16",
)

# Small train dataset
train_cfg2 = Namespace(**{**vars(train_cfg), "max_stocks": 30})
train_dataset = RolloutWindowDataset("train", cfg=train_cfg2, max_samples=0, seed=42)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=rollout_collate, num_workers=0)
print(f"Train windows: {len(train_dataset)}")

def run_training_steps(model_base, tokenizer, n_steps, use_gumbel=False):
    """Run N training steps and track expected return magnitudes."""
    model = copy.deepcopy(model_base)
    model.train()
    param_groups = _configure_trainable(model, train_cfg)
    optimizer, _ = _build_optimizer(param_groups, train_cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    exp_magnitudes = []
    step_losses = []

    updates = 0
    for batch in train_loader:
        if updates >= n_steps: break
        batch = _move_batch(batch, device)
        # Phase A: Exploration (train_star_cast version handles encoding internally)
        golden_c, golden_f, has_golden, _ = _star_cast_exploration(
            model, tokenizer, batch, train_cfg,
            device, amp_enabled, amp_dtype,
        )

        # Phase B: Forward
        train_len = golden_c.size(1)
        train_time = {k: v[:, :train_len] for k, v in batch["time"].items()}

        with _autocast_context(device, amp_enabled, amp_dtype):
            logits_c, logits_f, latent_states, hidden = model(
                golden_c[:, :-1], golden_f[:, :-1],
                train_time["minute"][:, :train_len - 1],
                train_time["day"][:, :train_len - 1],
                train_time["month"][:, :train_len - 1],
                train_time["year"][:, :train_len - 1],
                return_hidden=True, neftune_alpha=0.0,
            )
            start = train_cfg.prefix_len - 1
            horizon = train_cfg.horizon
            rollout_c = logits_c[:, start:start+horizon, :]
            rollout_f = logits_f[:, start:start+horizon, :]

            # Expected returns (Gumbel vs Soft)
            if use_gumbel:
                expected = gumbel_softmax_expected_returns(
                    tokenizer, rollout_c, rollout_f, batch["means"], batch["stds"],
                    train_cfg.top_k_expected_return, temp=0.3, hard=True)
            else:
                expected = soft_expected_returns(
                    tokenizer, rollout_c, rollout_f, batch["means"], batch["stds"],
                    train_cfg.top_k_expected_return, sharpening_temp=train_cfg.prob_sharpening_temp)

            actual_h = batch["actual_returns"][:, :horizon]

            step_loss = compute_asymmetric_direction_loss(
                expected, actual_h,
                alpha=train_cfg.asymmetric_alpha, beta=train_cfg.asymmetric_beta,
                timidity_weight=train_cfg.timidity_penalty_weight,
                timidity_ratio=train_cfg.timidity_ratio_threshold).mean()

            exp_path = torch.cumsum(expected, dim=1)
            actual_path = torch.cumsum(actual_h, dim=1)
            path_loss = compute_asymmetric_direction_loss(
                exp_path, actual_path,
                alpha=train_cfg.path_asymmetric_alpha, beta=train_cfg.path_asymmetric_beta,
                timidity_weight=train_cfg.timidity_penalty_weight,
                timidity_ratio=train_cfg.timidity_ratio_threshold).mean()

            if has_golden.any():
                target_c = golden_c[has_golden, train_cfg.prefix_len:train_cfg.prefix_len+horizon]
                target_f = golden_f[has_golden, train_cfg.prefix_len:train_cfg.prefix_len+horizon]
                ce_c = F.cross_entropy(rollout_c[has_golden].reshape(-1, rollout_c.size(-1)).float(), target_c.reshape(-1))
                ce_f = F.cross_entropy(rollout_f[has_golden].reshape(-1, rollout_f.size(-1)).float(), target_f.reshape(-1))
                star_ce = ce_c + ce_f
            else:
                star_ce = torch.tensor(0.0, device=device)

            # Direction loss
            if train_cfg.direction_weight > 0:
                dir_labels = compute_direction_labels(actual_h, train_cfg.direction_epsilon_scale)
                dir_logits = model.compute_direction_logits_at_positions(
                    hidden, latent_states, start=start, end=start+horizon)
                flat_w = train_cfg.direction_ce_flat_weight
                cw = torch.tensor([1.0, flat_w, 1.0], device=device, dtype=dir_logits.dtype)
                dir_loss = F.cross_entropy(dir_logits.reshape(-1, 3).float(), dir_labels.reshape(-1), weight=cw)
            else:
                dir_loss = torch.tensor(0.0, device=device)

            loss = (train_cfg.step_asym_weight * step_loss +
                    train_cfg.path_asym_weight * path_loss +
                    train_cfg.star_ce_weight * star_ce +
                    train_cfg.direction_weight * dir_loss)

        if not torch.isfinite(loss): continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], 0.3)
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)

        updates += 1
        exp_magnitudes.append(torch.mean(torch.abs(expected)).item())
        step_losses.append(loss.item())

    return exp_magnitudes, step_losses

print("\nTraining with SOFT expectation...")
soft_mags, soft_losses = run_training_steps(model, tokenizer, 20, use_gumbel=False)

print("Training with GUMBEL-STE expectation...")
gumbel_mags, gumbel_losses = run_training_steps(model, tokenizer, 20, use_gumbel=True)

print(f"\n  {'Method':<20} {'|E[r]| Start':>15} {'|E[r]| End':>15} {' |E[r]| Trend ':>15}")
print(f"  {'-'*65}")
print(f"  {'Soft Expectation':<20} {soft_mags[0]:>15.6f} {np.mean(soft_mags[-5:]):>15.6f} {'STABLE (near 0)':>15}")
print(f"  {'Gumbel-STE':<20} {gumbel_mags[0]:>15.6f} {np.mean(gumbel_mags[-5:]):>15.6f} {'?':>15}")

print(f"\n  Full training trajectories:")
for i in range(0, 20, 2):
    print(f"    Step {i+1:2d}: Soft |E[r]|={soft_mags[i]:.6f}  Gumbel |E[r]|={gumbel_mags[i]:.6f}  "
          f"Soft loss={soft_losses[i]:.4f}  Gumbel loss={gumbel_losses[i]:.4f}")

print("\nDone!")
