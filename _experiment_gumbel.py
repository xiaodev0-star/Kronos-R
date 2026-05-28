"""Phase 8-2 experiment: Gumbel-Softmax vs Soft Expectation — 200-step A/B test.

Budget: 3 hours. Strategy:
  Run A: Soft Expectation (baseline) — 200 steps, ~30 min
  Run B: Gumbel-Soft (tau=0.5) — 200 steps, ~30 min
  Run C: Gumbel-Soft (tau=0.3) — 200 steps, ~30 min
  Eval all 3 on Demo — ~30 min
  Analysis — remaining time
"""
import os, sys, math, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace
import copy

from evaluate_predictions import load_model
from posttrain.rollout.data import RolloutWindowDataset, rollout_collate, resolve_project_path
from posttrain.rollout.train_rollout import (
    _amp_dtype, _autocast_context, _move_batch, _encode_features,
    _configure_trainable, _build_optimizer, compute_rollout_metrics,
)
from posttrain.rollout.train_star_cast import (
    _star_cast_exploration, compute_asymmetric_direction_loss,
    compute_direction_labels, _save_star_cast_checkpoint,
)
from reproducibility import set_global_seed

set_global_seed(42, deterministic=False)
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high"); torch.cuda.empty_cache()
amp_dtype = _amp_dtype("bfloat16"); amp_enabled = True

# ═══════════════ Config ═══════════════
N_STOCKS = 100
N_UPDATES = 200
GA = 16  # gradient accumulation
TOP_K = 16

OUTPUT_DIR = resolve_project_path("outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

base_cfg = Namespace(
    prefix_len=1023, horizon=10, stride_ratio=0.5,
    cache_dir=resolve_project_path("posttrain/rollout/cache"),
    max_stocks=N_STOCKS, cache_rebuild=False,
    batch_size=2, epochs=1,
    num_trajectories=4, exploration_temperature=0.414, neftune_alpha=2.5,
    top_k_expected_return=TOP_K,
    asymmetric_alpha=3.0, asymmetric_beta=10.0,
    path_asymmetric_alpha=4.0, path_asymmetric_beta=15.0,
    step_asym_weight=1.0, path_asym_weight=1.5, star_ce_weight=0.334,
    timidity_penalty_weight=1.03, timidity_ratio_threshold=0.5,
    oracle_magnitude_penalty=3.99, prob_sharpening_temp=0.933,
    direction_weight=0.336, direction_epsilon_scale=0.361,
    direction_ce_flat_weight=0.540, direction_use_class_weights=True,
    mape_eps=1e-4, accumulation_steps=GA,
    learning_rate=9.59e-6, weight_decay=1e-4, grad_clip=0.3,
    use_gradient_checkpointing=False,
    freeze_backbone=False, trainable_scope="all",
    use_amp=True, amp_dtype="bfloat16",
    progress_interval=1000, checkpoint_interval=1000,
    save_epoch_checkpoints=False,
)

# ═══════════════ Expected Return Variants ═══════════════
def soft_expected_returns(tokenizer, logits_c, logits_f, means, stds, top_k=16, sharpening=1.0):
    """Original soft expectation (baseline)."""
    B, H, V_c = logits_c.shape
    K = min(int(top_k), V_c)
    top_logits_c, top_idx_c = torch.topk(logits_c.float(), k=K, dim=-1)
    top_logits_f, top_idx_f = torch.topk(logits_f.float(), k=K, dim=-1)
    prob_c = F.softmax(top_logits_c / sharpening, dim=-1)
    prob_f = F.softmax(top_logits_f / sharpening, dim=-1)
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


def gumbel_soft_expected_returns(tokenizer, logits_c, logits_f, means, stds, top_k=16, temp=0.5):
    """Gumbel-Softmax: soft sample instead of pure softmax expectation.

    Adding Gumbel noise + lowered temperature breaks the symmetry-averaging
    effect that causes soft expectations to collapse toward zero.
    """
    B, H, V_c = logits_c.shape
    K = min(int(top_k), V_c)
    top_logits_c, top_idx_c = torch.topk(logits_c.float(), k=K, dim=-1)
    top_logits_f, top_idx_f = torch.topk(logits_f.float(), k=K, dim=-1)

    # Gumbel noise
    g_c = -torch.log(-torch.log(torch.rand_like(top_logits_c).clamp_min(1e-8) + 1e-8) + 1e-8)
    g_f = -torch.log(-torch.log(torch.rand_like(top_logits_f).clamp_min(1e-8) + 1e-8) + 1e-8)

    prob_c = F.softmax((top_logits_c + g_c) / temp, dim=-1)
    prob_f = F.softmax((top_logits_f + g_f) / temp, dim=-1)
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


# ═══════════════ Training Loop ═══════════════
def train_model(model_base, tokenizer, cfg, device, expected_return_fn, label, n_updates=200):
    """Train for n_updates using the given expected_return_fn."""
    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"{'='*60}")

    model = copy.deepcopy(model_base)
    model.train()
    param_groups = _configure_trainable(model, cfg)
    optimizer, _ = _build_optimizer(param_groups, cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # bf16 doesn't need scaling

    # Data
    train_dataset = RolloutWindowDataset("train", cfg=cfg, max_samples=0, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                              collate_fn=rollout_collate, num_workers=0)
    print(f"  Train windows: {len(train_dataset)}")

    # LR schedule
    warmup = max(2, n_updates // 10)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_updates - warmup),
        eta_min=float(cfg.learning_rate) * 0.05)

    updates = 0; microbatch = 0
    history = {"abs_er": [], "loss": [], "step_asym": [], "star_ce": [], "dir_loss": []}
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=n_updates, desc=f"  {label}")

    while updates < n_updates:
        for batch in train_loader:
            if updates >= n_updates: break
            batch = _move_batch(batch, device)
            try:
                golden_c, golden_f, has_golden, _ = _star_cast_exploration(
                    model, tokenizer, batch, cfg, device, amp_enabled, amp_dtype)
            except Exception as e:
                print(f"  Exploration error: {e}"); continue

            model.train()
            train_len = golden_c.size(1)
            train_time = {k: v[:, :train_len] for k, v in batch["time"].items()}

            with _autocast_context(device, amp_enabled, amp_dtype):
                logits_c, logits_f, latent_states, hidden = model(
                    golden_c[:, :-1], golden_f[:, :-1],
                    train_time["minute"][:, :train_len - 1],
                    train_time["day"][:, :train_len - 1],
                    train_time["month"][:, :train_len - 1],
                    train_time["year"][:, :train_len - 1],
                    return_hidden=True, neftune_alpha=0.0)

                start = cfg.prefix_len - 1; horizon = cfg.horizon
                rollout_c = logits_c[:, start:start+horizon, :]
                rollout_f = logits_f[:, start:start+horizon, :]
                actual_h = batch["actual_returns"][:, :horizon]

                # Use the specified expected return function
                expected = expected_return_fn(
                    tokenizer, rollout_c, rollout_f,
                    batch["means"], batch["stds"], cfg.top_k_expected_return)

                step_loss = compute_asymmetric_direction_loss(
                    expected, actual_h,
                    alpha=cfg.asymmetric_alpha, beta=cfg.asymmetric_beta,
                    timidity_weight=cfg.timidity_penalty_weight,
                    timidity_ratio=cfg.timidity_ratio_threshold).mean()

                exp_path = torch.cumsum(expected, dim=1)
                actual_path = torch.cumsum(actual_h, dim=1)
                path_loss = compute_asymmetric_direction_loss(
                    exp_path, actual_path,
                    alpha=cfg.path_asymmetric_alpha, beta=cfg.path_asymmetric_beta,
                    timidity_weight=cfg.timidity_penalty_weight,
                    timidity_ratio=cfg.timidity_ratio_threshold).mean()

                if has_golden.any():
                    t_c = golden_c[has_golden, cfg.prefix_len:cfg.prefix_len+horizon]
                    t_f = golden_f[has_golden, cfg.prefix_len:cfg.prefix_len+horizon]
                    ce_c = F.cross_entropy(rollout_c[has_golden].reshape(-1, rollout_c.size(-1)).float(), t_c.reshape(-1))
                    ce_f = F.cross_entropy(rollout_f[has_golden].reshape(-1, rollout_f.size(-1)).float(), t_f.reshape(-1))
                    star_ce = ce_c + ce_f
                else:
                    star_ce = torch.tensor(0.0, device=device)

                if cfg.direction_weight > 0:
                    dir_labels = compute_direction_labels(actual_h, cfg.direction_epsilon_scale)
                    dir_logits = model.compute_direction_logits_at_positions(
                        hidden, latent_states, start=start, end=start+horizon)
                    cw = torch.tensor([1.0, cfg.direction_ce_flat_weight, 1.0],
                                      device=device, dtype=dir_logits.dtype)
                    dir_loss = F.cross_entropy(dir_logits.reshape(-1, 3).float(), dir_labels.reshape(-1), weight=cw)
                else:
                    dir_loss = torch.tensor(0.0, device=device)

                loss = (cfg.step_asym_weight * step_loss +
                        cfg.path_asym_weight * path_loss +
                        cfg.star_ce_weight * star_ce +
                        cfg.direction_weight * dir_loss)
                loss = loss / GA

            if not torch.isfinite(loss): continue
            scaler.scale(loss).backward()
            microbatch += 1

            if microbatch % GA == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], cfg.grad_clip)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if updates < warmup: warmup_sched.step()
                else: cosine_sched.step()
                updates += 1; pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item()*GA:.3f}",
                                 abs_er=f"{torch.mean(torch.abs(expected)).item():.4f}")
                history["abs_er"].append(torch.mean(torch.abs(expected)).item())
                history["loss"].append(loss.item() * GA)
                history["step_asym"].append(step_loss.item())
                history["star_ce"].append(star_ce.item())
                history["dir_loss"].append(dir_loss.item())

    pbar.close()

    # Save model
    save_path = os.path.join(OUTPUT_DIR, f"experiment_{label.replace(' ','_').replace('=','')}.pt")
    _save_star_cast_checkpoint(save_path, model, tokenizer, cfg, {"label": label}, history)

    return model, history, save_path


# ═══════════════ Evaluation ═══════════════
@torch.inference_mode()
def eval_model(model, tokenizer, loader, label, max_batches=200):
    """Autoregressive 10-step evaluation."""
    torch.cuda.empty_cache()
    model.eval()
    all_pred, all_actual = [], []
    n = 0
    for batch in tqdm(loader, desc=f"Eval {label}", total=max_batches):
        batch = _move_batch(batch, device)
        idx_c, idx_f = _encode_features(tokenizer, batch["features"])
        cur_c = idx_c[:, :1023].clone(); cur_f = idx_f[:, :1023].clone()
        preds = []
        for step in range(10):
            sl = cur_c.size(1)
            ct = {k: v[:, :sl] for k, v in batch["time"].items()}
            with _autocast_context(device, amp_enabled, amp_dtype):
                lc, lf, _ = model(cur_c, cur_f, ct["minute"], ct["day"], ct["month"], ct["year"], last_only=True)
            pc = lc[:, -1, :].argmax(dim=-1); pf = lf[:, -1, :].argmax(dim=-1)
            dec = tokenizer.decode(pc.unsqueeze(1), pf.unsqueeze(1))
            ret = dec[:, 0, 0].cpu().float() * batch["stds"][:, 0].cpu() + batch["means"][:, 0].cpu()
            preds.append(ret)
            if step < 9:
                cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
                cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)
        all_pred.append(torch.stack(preds, dim=1))
        all_actual.append(batch["actual_returns"].cpu())
        n += 1
        if n >= max_batches:
            break
    pred = torch.cat(all_pred, dim=0).numpy()
    actual = torch.cat(all_actual, dim=0).numpy()
    return compute_rollout_metrics(pred, actual, mape_eps=1e-4), pred, actual


# ═══════════════ Main ═══════════════
def main():
    t0_total = time.time()
    print("Loading base model...")
    model_base, tokenizer = load_model(
        device=device,
        checkpoint_path=resolve_project_path("checkpoints/base_model.pt"),
        strict_checkpoint_compat=False)
    tokenizer.eval(); tokenizer.requires_grad_(False)
    print(f"  Params: {sum(p.numel() for p in model_base.parameters()):,}")

    # Eval dataset (Demo)
    eval_cfg = Namespace(**{**vars(base_cfg), "max_stocks": 0})
    eval_dataset = RolloutWindowDataset("val", cfg=eval_cfg, max_samples=0, seed=999)
    eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False,
                             collate_fn=rollout_collate, num_workers=0)
    print(f"  Eval windows: {len(eval_dataset)}")

    # ── Run A: Soft Expectation Baseline ──
    cfg_a = copy.deepcopy(base_cfg)
    def soft_fn(tok, lc, lf, m, s, k): return soft_expected_returns(tok, lc, lf, m, s, k, sharpening=cfg_a.prob_sharpening_temp)
    model_a, hist_a, path_a = train_model(model_base, tokenizer, cfg_a, device, soft_fn, "Soft-Baseline", N_UPDATES)
    metrics_a, pred_a, actual_a = eval_model(model_a, tokenizer, eval_loader, "Soft-Baseline")
    del model_a; torch.cuda.empty_cache()

    # ── Run B: Gumbel-Soft tau=0.5 ──
    cfg_b = copy.deepcopy(base_cfg)
    def gumbel05_fn(tok, lc, lf, m, s, k): return gumbel_soft_expected_returns(tok, lc, lf, m, s, k, temp=0.5)
    model_b, hist_b, path_b = train_model(model_base, tokenizer, cfg_b, device, gumbel05_fn, "Gumbel-tau=0.5", N_UPDATES)
    metrics_b, pred_b, actual_b = eval_model(model_b, tokenizer, eval_loader, "Gumbel-tau=0.5")
    del model_b; torch.cuda.empty_cache()

    # ── Check remaining time for Run C ──
    elapsed = time.time() - t0_total
    remaining = 3 * 3600 - elapsed
    run_c = remaining > 2400  # only run if >40 min left

    if run_c:
        cfg_c = copy.deepcopy(base_cfg)
        def gumbel03_fn(tok, lc, lf, m, s, k): return gumbel_soft_expected_returns(tok, lc, lf, m, s, k, temp=0.3)
        model_c, hist_c, path_c = train_model(model_base, tokenizer, cfg_c, device, gumbel03_fn, "Gumbel-tau=0.3", N_UPDATES)
        metrics_c, pred_c, actual_c = eval_model(model_c, tokenizer, eval_loader, "Gumbel-tau=0.3")
        del model_c; torch.cuda.empty_cache()
    else:
        print(f"\n  Skipping Run C (Gumbel tau=0.3) — only {remaining/60:.0f}min remaining")
        metrics_c = None

    # ── Results ──
    total_time = time.time() - t0_total
    print(f"\n{'='*80}")
    print(f"EXPERIMENT RESULTS ({total_time/60:.1f} min)")
    print(f"{'='*80}")

    # Training stats
    for name, hist in [("Soft-Baseline", hist_a), ("Gumbel-tau=0.5", hist_b)] + \
                      ([("Gumbel-tau=0.3", hist_c)] if run_c else []):
        em = np.array(hist["abs_er"])
        print(f"\n  {name} Training:")
        print(f"    abs_er: start={em[0]:.5f}  end={np.mean(em[-10:]):.5f}  "
              f"mean={np.mean(em):.5f}  max={np.max(em):.5f}  "
              f"%<0.001={np.mean(em<0.001)*100:.1f}%")
        print(f"    loss: mean={np.mean(hist['loss']):.4f}  final={np.mean(hist['loss'][-10:]):.4f}")
        print(f"    step_asym: mean={np.mean(hist['step_asym']):.4f}")
        print(f"    star_ce: mean={np.mean(hist['star_ce']):.4f}")
        print(f"    dir_loss: mean={np.mean(hist['dir_loss']):.4f}")

    # Eval metrics
    print(f"\n  Demo Evaluation (200 batches):")
    print(f"  {'Method':<22} {'PathMAPE':>10} {'DA':>8} {'PathDA':>8} {'ActDA':>8} {'ActRatio':>8} {'PathMAE':>10}")
    print(f"  {'-'*72}")
    for name, m in [("Soft-Baseline", metrics_a), ("Gumbel-tau=0.5", metrics_b)] + \
                   ([("Gumbel-tau=0.3", metrics_c)] if run_c else []):
        # Compute actionable DA
        pt = torch.from_numpy(pred_a if name == "Soft-Baseline" else (pred_b if "0.5" in name else pred_c))
        at_ = torch.from_numpy(actual_a)
        ps = (pt >= 0).float() * 2 - 1; _as = (at_ >= 0).float() * 2 - 1
        cm = torch.abs(pt) > 0.005
        act_da = (ps[cm] == _as[cm]).float().mean().item() * 100 if cm.sum() > 0 else 0
        act_ratio = cm.float().mean().item() * 100
        print(f"  {name:<22} {m['path_mape']:>10.4f} {m['da']:>8.2f} {m.get('path_da',0):>8.2f} "
              f"{act_da:>8.2f} {act_ratio:>8.2f} {m['path_mae']:>10.6f}")

    # Prediction distribution comparison
    print(f"\n  Prediction Distribution (10-step AR rollout):")
    print(f"  {'Method':<22} {'|Pred|Mean':>12} {'|Pred|Med':>12} {'%<0.001':>10} {'%<0.005':>10} {'%<0.01':>10}")
    print(f"  {'-'*76}")
    for name, pred_arr in [("Soft-Baseline", pred_a), ("Gumbel-tau=0.5", pred_b)] + \
                          ([("Gumbel-tau=0.3", pred_c)] if run_c else []):
        ap = np.abs(pred_arr.ravel())
        print(f"  {name:<22} {np.mean(ap):>12.6f} {np.median(ap):>12.6f} "
              f"{np.mean(ap<0.001)*100:>10.1f} {np.mean(ap<0.005)*100:>10.1f} {np.mean(ap<0.01)*100:>10.1f}")

    # Save results JSON
    import json
    result = {
        "soft": {"path_mape": metrics_a["path_mape"], "da": metrics_a["da"],
                 "abs_er_train": float(np.mean(hist_a["abs_er"])),
                 "abs_pred_eval": float(np.mean(np.abs(pred_a)))},
        "gumbel_05": {"path_mape": metrics_b["path_mape"], "da": metrics_b["da"],
                      "abs_er_train": float(np.mean(hist_b["abs_er"])),
                      "abs_pred_eval": float(np.mean(np.abs(pred_b)))},
    }
    if run_c:
        result["gumbel_03"] = {"path_mape": metrics_c["path_mape"], "da": metrics_c["da"],
                               "abs_er_train": float(np.mean(hist_c["abs_er"])),
                               "abs_pred_eval": float(np.mean(np.abs(pred_c)))}
    with open(os.path.join(OUTPUT_DIR, "experiment_gumbel_vs_soft.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Results saved to: {OUTPUT_DIR}/experiment_gumbel_vs_soft.json")
    print(f"  Total time: {total_time/60:.1f} min")
    print("Done!")


if __name__ == "__main__":
    main()
