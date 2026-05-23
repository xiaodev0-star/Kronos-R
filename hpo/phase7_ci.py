"""Phase 7a: CI Post-Training HPO (Idea 2 — CI-aware training).

Optimises the concentration-loss and interval-score hyperparameters for
confidence-interval-aware post-training on top of Kronos-R BaseModel.

Evaluation metric: average interval score (lower = better) at 80% confidence
on the val set via distribution-quantile-based CI construction.

Usage:
    python -m hpo.phase7_ci
"""

from __future__ import annotations

import copy, csv, hashlib, json, os, time
from argparse import Namespace
from contextlib import nullcontext
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DataConfig
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT
from posttrain.rollout.data import RolloutWindowDataset, rollout_collate
from posttrain.ci.eval_ci import compute_ci_metrics

# ── Config ──
N_TRIALS = 100
STUDY_NAME = "phase7_ci_training"
CLEAN_START = False  # already cleaned manually

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE7_DIR = os.path.join(PROJECT_ROOT, "trials", "phase7_ci")
STUDY_DB = os.path.join(PHASE7_DIR, "study_training.db")
SUMMARY_CSV = os.path.join(PHASE7_DIR, "summary_training.csv")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.pt")
TOKENIZER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer_config.json")
BASEMODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "base_model.pt")
ROLLOUT_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "post_train_rollout", "rollout_scheduled.pt")

TOKENIZER_VOCAB = 1 << 10
PREFIX_LEN = 1023
HORIZON = 10

BACKBONE = {
    "dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
    "dsa_windows": [None, 512, 512],
    "position_encoding": "rope", "rope_base": 10000.0,
    "dropout": 0.1323, "use_revin": False, "num_factor_tokens": 0,
}

# ── CI Training search space (expanded for 10-hr budget) ──
SEARCH_SPACE = {
    "concentration_weight": [0.1, 0.3, 1.0, 3.0, 10.0],
    "interval_score_weight": [0.0, 0.05, 0.1, 0.3, 1.0],
    "ci_confidence_level": [0.68, 0.80, 0.90],
    "ci_top_k": [8, 16, 32, 64],
    "lr": (5e-6, 5e-5),
    "kl_weight": (0.005, 0.2),
    "max_updates": [240, 480, 960, 1440],
    "step_weight_gamma": [0.3, 0.5, 0.7, 1.0],
    "start_model": ["basemodel", "rollout"],
}


def sample_params(trial: optuna.Trial) -> dict:
    conc_w = trial.suggest_categorical("concentration_weight", SEARCH_SPACE["concentration_weight"])
    is_w = trial.suggest_categorical("interval_score_weight", SEARCH_SPACE["interval_score_weight"])
    conf = trial.suggest_categorical("ci_confidence_level", SEARCH_SPACE["ci_confidence_level"])
    topk = trial.suggest_categorical("ci_top_k", SEARCH_SPACE["ci_top_k"])
    lr = trial.suggest_float("lr", *SEARCH_SPACE["lr"], log=True)
    kl_w = trial.suggest_float("kl_weight", *SEARCH_SPACE["kl_weight"], log=True)
    max_up = trial.suggest_categorical("max_updates", SEARCH_SPACE["max_updates"])
    gamma = trial.suggest_categorical("step_weight_gamma", SEARCH_SPACE["step_weight_gamma"])
    start_m = trial.suggest_categorical("start_model", SEARCH_SPACE["start_model"])
    return {
        "concentration_weight": conc_w,
        "interval_score_weight": is_w,
        "ci_confidence_level": conf,
        "ci_top_k": topk,
        "lr": round(lr, 10),
        "kl_weight": round(kl_w, 6),
        "max_updates": max_up,
        "step_weight_gamma": gamma,
        "start_model": start_m,
        "batch_size": 2,
    }


def _make_ci_cfg():
    return Namespace(
        prefix_len=PREFIX_LEN, horizon=HORIZON,
        stride_ratio=DataConfig.stride_ratio,
        cache_dir=os.path.join(PROJECT_ROOT, "posttrain", "rollout", "cache"),
        max_stocks=0, cache_rebuild=False,
    )


def _config_hash(params: dict) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def _load_tokenizer(device):
    ckpt = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg and os.path.exists(TOKENIZER_CONFIG_PATH):
        with open(TOKENIZER_CONFIG_PATH) as f:
            cfg = json.load(f)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tok.load_state_dict(ckpt["model_state_dict"], strict=False)
    tok.eval()
    tok.requires_grad_(False)
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
    ckpt = torch.load(BASEMODEL_PATH, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _build_data(device):
    cfg = _make_ci_cfg()
    train_ds = RolloutWindowDataset("train", cfg=cfg, max_samples=0, seed=42)
    val_ds = RolloutWindowDataset("val", cfg=cfg, max_samples=500, seed=59)
    print(f"  Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True,
        collate_fn=rollout_collate, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=rollout_collate, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════
# CI Training
# ═══════════════════════════════════════════════════════════════

def _train_ci(model, tokenizer, params: dict, tdir: str, device) -> dict:
    """CI-aware post-training. Returns best val interval_score."""
    bp = params
    conc_w = bp["concentration_weight"]
    is_w = bp["interval_score_weight"]
    conf_level = bp["ci_confidence_level"]
    top_k = bp["ci_top_k"]
    lr = bp["lr"]
    kl_w = bp["kl_weight"]
    max_updates = bp["max_updates"]
    batch_size = bp["batch_size"]
    gamma = float(bp.get("step_weight_gamma", 0.5))

    result_path = os.path.join(tdir, "result.json")
    resume_path = os.path.join(tdir, "ci_resume.pt")
    os.makedirs(tdir, exist_ok=True)

    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    train_loader, val_loader = _build_data(device)

    import torch.optim as optim
    opt = optim.AdamW(model.parameters(), lr=lr)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    use_amp = device.type == "cuda" and amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    update_count = 0
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        update_count = ckpt["update_count"]
        print(f"  Resume from update {update_count}")

    ref_model = copy.deepcopy(model)
    ref_model.eval()
    ref_model.requires_grad_(False)

    total_updates = update_count
    prefix_len = PREFIX_LEN
    horizon = HORIZON
    last_ce = 0.0  # track for result logging

    model.train()
    pbar = tqdm(total=max_updates - update_count, desc="  CI train")

    while total_updates < max_updates:
        for batch in train_loader:
            if total_updates >= max_updates:
                break

            feats = batch["features"].to(device=device, dtype=torch.float32)
            means = batch["means"].to(device=device, dtype=torch.float32)
            stds = batch["stds"].to(device=device, dtype=torch.float32)
            actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
            times_f = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}

            B = feats.shape[0]
            if B == 0:
                continue

            idx_c_full, idx_f_full = tokenizer.encode(feats)

            # Ground-truth context for stable HPO comparison
            ctx_c = idx_c_full[:, :prefix_len + horizon - 1]
            ctx_f = idx_f_full[:, :prefix_len + horizon - 1]
            ctx_time = {k: times_f[k][:, :ctx_c.size(1)] for k in ("minute", "day", "month", "year")}
            target_c = idx_c_full[:, prefix_len:prefix_len + horizon]
            target_f = idx_f_full[:, prefix_len:prefix_len + horizon]

            opt.zero_grad(set_to_none=True)

            with (torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()):
                logits_c, logits_f, _ = model(
                    ctx_c, ctx_f,
                    ctx_time["minute"], ctx_time["day"],
                    ctx_time["month"], ctx_time["year"],
                )

                r_c = logits_c[:, prefix_len - 1 : prefix_len - 1 + horizon, :].float()
                r_f = logits_f[:, prefix_len - 1 : prefix_len - 1 + horizon, :].float()

                # ── Step weights (later steps weighted more) ──
                H_steps = int(r_c.size(1))
                steps_t = torch.arange(H_steps, device=r_c.device, dtype=torch.float32)
                step_w = 1.0 + gamma * steps_t / max(1, H_steps - 1)  # [H]
                step_w = step_w / step_w.mean()  # normalise so mean=1

                # ── Anchor CE (step-weighted) ──
                loss_c = F.cross_entropy(
                    r_c.reshape(-1, r_c.size(-1)), target_c.reshape(-1), reduction="none"
                ).view(-1, H_steps)
                loss_f = F.cross_entropy(
                    r_f.reshape(-1, r_f.size(-1)), target_f.reshape(-1), reduction="none"
                ).view(-1, H_steps)
                ce_per_step = loss_c + loss_f  # [B, H]
                ce = (ce_per_step * step_w.view(1, -1)).sum() / step_w.sum().clamp_min(1.0) / ce_per_step.size(0)

                # ── Shared: build top-K return distribution ──
                K_dist = min(int(top_k), r_c.size(-1), r_f.size(-1))
                need_dist = (conc_w > 0) or (is_w > 0)
                pair_probs = None
                ret_denorm = None
                B_s = H_s = Kc = Kf = 0

                if need_dist:
                    probs_c = F.softmax(r_c, dim=-1)
                    probs_f = F.softmax(r_f, dim=-1)
                    top_pc, top_ic = torch.topk(probs_c, k=K_dist, dim=-1)
                    top_pf, top_if = torch.topk(probs_f, k=K_dist, dim=-1)
                    top_pc = top_pc / top_pc.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    top_pf = top_pf / top_pf.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    pair_probs = top_pc.unsqueeze(-1) * top_pf.unsqueeze(-2)
                    B_s, H_s, Kc, Kf = pair_probs.shape

                    pc_flat = top_ic.unsqueeze(-1).expand(B_s, H_s, Kc, Kf).reshape(B_s * H_s, Kc * Kf)
                    pf_flat = top_if.unsqueeze(-2).expand(B_s, H_s, Kc, Kf).reshape(B_s * H_s, Kc * Kf)
                    with torch.no_grad():
                        decoded = tokenizer.decode(pc_flat, pf_flat)[..., 0].float()
                        returns = decoded.view(B_s, H_s, Kc, Kf)
                        ret_denorm = returns * stds[:, 0].view(B_s, 1, 1, 1) + means[:, 0].view(B_s, 1, 1, 1)

                # ── Concentration loss (step-weighted) ──
                conc_loss_val = ce.new_zeros(())
                if conc_w > 0 and pair_probs is not None:
                    y_conc = actual[:, :horizon].view(B_s, H_s, 1, 1)
                    abs_err = (ret_denorm - y_conc).abs()
                    per_step_conc = (pair_probs * abs_err).sum(dim=(-1, -2))  # [B, H]
                    sw = step_w.to(device=per_step_conc.device, dtype=per_step_conc.dtype)
                    conc_loss_val = (per_step_conc * sw.view(1, -1)).sum() / sw.sum().clamp_min(1.0) / per_step_conc.size(0)

                # ── KL to reference ──
                kl_val = ce.new_zeros(())
                if kl_w > 0:
                    with torch.no_grad():
                        ref_lc, ref_lf, _ = ref_model(
                            ctx_c, ctx_f,
                            ctx_time["minute"], ctx_time["day"],
                            ctx_time["month"], ctx_time["year"],
                        )
                    ref_rc = ref_lc[:, prefix_len - 1 : prefix_len - 1 + horizon, :].float()
                    ref_rf = ref_lf[:, prefix_len - 1 : prefix_len - 1 + horizon, :].float()
                    kl_val = F.kl_div(F.log_softmax(r_c, dim=-1), F.softmax(ref_rc, dim=-1), reduction="batchmean")
                    kl_val = kl_val + F.kl_div(F.log_softmax(r_f, dim=-1), F.softmax(ref_rf, dim=-1), reduction="batchmean")

                loss = ce + conc_w * conc_loss_val + kl_w * kl_val

                # ── Interval-score surrogate: scale CE by IS ──
                if is_w > 0 and pair_probs is not None and K_dist >= 8:
                    alpha = 1.0 - float(conf_level)
                    low_q = alpha / 2.0
                    high_q = 1.0 - alpha / 2.0

                    pair_probs_f = pair_probs.view(B_s, H_s, -1)
                    ret_flat = ret_denorm.view(B_s, H_s, -1)

                    sort_idx = ret_flat.argsort(dim=-1)
                    sorted_ret = ret_flat.gather(-1, sort_idx)
                    sorted_prob = pair_probs_f.gather(-1, sort_idx)
                    cum_prob = sorted_prob.cumsum(dim=-1)
                    cum_prob = cum_prob / cum_prob[..., -1:].clamp_min(1e-8)

                    Np = cum_prob.shape[-1]
                    mask_low = cum_prob >= low_q
                    idx_low = mask_low.float().argmax(dim=-1).clamp(0, Np - 1)
                    mask_high = cum_prob >= high_q
                    idx_high = mask_high.float().argmax(dim=-1).clamp(0, Np - 1)

                    rows_i = torch.arange(B_s, device=ret_flat.device).view(B_s, 1).expand(B_s, H_s)
                    cols_i = torch.arange(H_s, device=ret_flat.device).view(1, H_s).expand(B_s, H_s)
                    L = sorted_ret[rows_i, cols_i, idx_low]
                    U = sorted_ret[rows_i, cols_i, idx_high]

                    y_is = actual[:, :horizon].view(B_s, H_s)
                    width = U - L
                    miss_low = torch.clamp(L - y_is, min=0)
                    miss_high = torch.clamp(y_is - U, min=0)
                    iscore_per_step = width + (2.0 / max(alpha, 1e-8)) * (miss_low + miss_high)  # [B, H]
                    sw_is = step_w.to(device=iscore_per_step.device, dtype=iscore_per_step.dtype)
                    iscore_mean = (iscore_per_step * sw_is.view(1, -1)).sum() / sw_is.sum().clamp_min(1.0) / iscore_per_step.size(0)
                    iscore_mean = iscore_mean.detach()

                    loss = loss + is_w * iscore_mean * ce.detach()

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()

            total_updates += 1
            last_ce = float(ce.item())
            pbar.update(1)
            pbar.set_postfix({"ce": f"{last_ce:.3f}", "conc": f"{conc_loss_val.item():.4f}"})

            if total_updates % 120 == 0:
                torch.save({
                    "update_count": total_updates,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                }, resume_path)

            if total_updates >= max_updates:
                break

    pbar.close()

    torch.save({"model_state_dict": model.state_dict(), "update_count": total_updates},
               os.path.join(tdir, "ci_model.pt"))

    # ── Evaluate CI metrics ──
    eval_result = _eval_ci(model, tokenizer, val_loader, device, conf_level, top_k)
    if os.path.exists(resume_path):
        os.remove(resume_path)

    result = {**eval_result, "train_ce": round(last_ce, 6),
              "total_updates": total_updates, "params": bp}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


@torch.no_grad()
def _eval_ci(model, tokenizer, val_loader, device, conf_level, top_k):
    """CI evaluation via distribution quantiles."""
    model.eval()
    prefix_len = PREFIX_LEN
    horizon = HORIZON

    all_lower, all_upper, all_actual = [], [], []

    for batch in val_loader:
        feats = batch["features"].to(device=device, dtype=torch.float32)
        means = batch["means"].to(device=device, dtype=torch.float32)
        stds = batch["stds"].to(device=device, dtype=torch.float32)
        actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
        times_f = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}

        B = feats.shape[0]
        if B == 0:
            continue

        idx_c_full, idx_f_full = tokenizer.encode(feats)
        context_c = idx_c_full[:, :prefix_len].clone()
        context_f = idx_f_full[:, :prefix_len].clone()

        step_lower, step_upper = [], []
        alpha = 1.0 - float(conf_level)
        low_q = alpha / 2.0
        high_q = 1.0 - alpha / 2.0
        K = min(int(top_k), TOKENIZER_VOCAB)

        for step in range(horizon):
            cur_len = int(context_c.size(1))
            cur_time = {k: times_f[k][:, :cur_len] for k in ("minute", "day", "month", "year")}

            logits_c, logits_f, _ = model(
                context_c, context_f,
                cur_time["minute"], cur_time["day"],
                cur_time["month"], cur_time["year"],
                last_only=True,
            )

            last_c = logits_c[:, -1, :].float()
            last_f = logits_f[:, -1, :].float()

            probs_c = F.softmax(last_c, dim=-1)
            probs_f = F.softmax(last_f, dim=-1)
            top_pc, top_ic = torch.topk(probs_c, k=K, dim=-1)
            top_pf, top_if = torch.topk(probs_f, k=K, dim=-1)
            top_pc = top_pc / top_pc.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            top_pf = top_pf / top_pf.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pair_probs = top_pc.unsqueeze(-1) * top_pf.unsqueeze(-2)
            B_s, Kc, Kf = pair_probs.shape

            pc_flat = top_ic.unsqueeze(-1).expand(B_s, Kc, Kf).reshape(B_s, Kc * Kf)
            pf_flat = top_if.unsqueeze(-2).expand(B_s, Kc, Kf).reshape(B_s, Kc * Kf)
            with torch.no_grad():
                decoded = tokenizer.decode(pc_flat, pf_flat)[..., 0].float()
                returns = decoded.view(B_s, Kc, Kf)
                ret_denorm = returns * stds[:, 0].view(B_s, 1, 1) + means[:, 0].view(B_s, 1, 1)

            ret_flat = ret_denorm.view(B_s, -1)
            prob_flat = pair_probs.view(B_s, -1)

            sort_idx = ret_flat.argsort(dim=-1)
            sorted_ret = ret_flat.gather(-1, sort_idx)
            sorted_prob = prob_flat.gather(-1, sort_idx)
            cum_prob = sorted_prob.cumsum(dim=-1)
            cum_prob = cum_prob / cum_prob[..., -1:].clamp_min(1e-8)

            Np = cum_prob.shape[-1]
            mask_low = cum_prob >= low_q
            idx_low = mask_low.float().argmax(dim=-1).clamp(0, Np - 1)
            mask_high = cum_prob >= high_q
            idx_high = mask_high.float().argmax(dim=-1).clamp(0, Np - 1)
            rows = torch.arange(B_s, device=ret_flat.device)

            L = sorted_ret[rows, idx_low].cpu()
            U = sorted_ret[rows, idx_high].cpu()

            step_lower.append(L)
            step_upper.append(U)

            if step < horizon - 1:
                next_c = last_c.argmax(dim=-1)
                next_f = last_f.argmax(dim=-1)
                context_c = torch.cat([context_c, next_c.unsqueeze(1)], dim=1)
                context_f = torch.cat([context_f, next_f.unsqueeze(1)], dim=1)

        all_lower.append(torch.stack(step_lower, dim=1))
        all_upper.append(torch.stack(step_upper, dim=1))
        all_actual.append(actual.cpu())

    if not all_lower:
        return {"avg_interval_score": 999.0, "coverage": 0.0, "avg_width": 0.0}

    pl = torch.cat(all_lower, dim=0).numpy()
    pu = torch.cat(all_upper, dim=0).numpy()
    aa = torch.cat(all_actual, dim=0).numpy()

    m = compute_ci_metrics(pl, pu, aa, confidence_level=float(conf_level))
    return {
        "avg_interval_score": round(m["avg_interval_score"], 6),
        "coverage": round(m["coverage"], 6),
        "avg_width": round(m["avg_width"], 6),
        "path_coverage": round(m["path_coverage"], 6),
        "path_avg_width": round(m["path_avg_width"], 6),
        "path_avg_interval_score": round(m["path_avg_interval_score"], 6),
        "mape_midpoint": round(m["mape_midpoint"], 4),
        "da_midpoint": round(m["da_midpoint"], 4),
        "per_step_coverage": [round(s["coverage"], 4) for s in m["per_step"]],
        "per_step_width": [round(s["avg_width"], 6) for s in m["per_step"]],
    }


# ═══════════════════════════════════════════════════════════════
# HPO Loop
# ═══════════════════════════════════════════════════════════════

def _assign_trial_dir():
    os.makedirs(PHASE7_DIR, exist_ok=True)
    counter_path = os.path.join(PHASE7_DIR, ".next_trial_training")
    existing = sorted([
        d for d in os.listdir(PHASE7_DIR)
        if d.startswith("trial_train_") and os.path.isdir(os.path.join(PHASE7_DIR, d))
    ], key=lambda x: int(x.split("_")[2]) if len(x.split("_")) >= 3 else 0)

    for d in existing:
        full = os.path.join(PHASE7_DIR, d)
        resume = os.path.join(full, "ci_resume.pt")
        result = os.path.join(full, "result.json")
        if os.path.exists(resume) and not os.path.exists(result):
            if os.path.exists(os.path.join(full, "config.json")):
                print(f"  Found incomplete trial: {d} — will resume")
                return full

    next_num = 0
    if os.path.exists(counter_path):
        with open(counter_path) as f:
            next_num = int(f.read().strip())
    new_dir = os.path.join(PHASE7_DIR, f"trial_train_{next_num:03d}")
    with open(counter_path, "w") as f:
        f.write(str(next_num + 1))
    return new_dir


def export_summary(study: optuna.Study):
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "value": t.value, **t.params}
        for k, v in t.user_attrs.items():
            if isinstance(v, (int, float, str, bool)):
                row[k] = v
        rows.append(row)
    if not rows:
        return
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    ordered = ["trial", "value"] + sorted(k for k in all_keys if k not in ("trial", "value"))
    os.makedirs(PHASE7_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ranked = sorted(rows, key=lambda r: r["value"])
    print(f"\nTop-10 by interval_score:")
    for r in ranked[:10]:
        print(f"  Trial {r['trial']:03d}  IS={r['value']:.6f}  "
              f"conc={r.get('concentration_weight','?')}  "
              f"is_w={r.get('interval_score_weight','?')}  "
              f"topk={r.get('ci_top_k','?')}")
    print(f"Summary: {SUMMARY_CSV}")


def main():
    if CLEAN_START and os.path.exists(PHASE7_DIR):
        import shutil
        for f in os.listdir(PHASE7_DIR):
            if f.startswith("trial_train_"):
                shutil.rmtree(os.path.join(PHASE7_DIR, f))
        for f in ["study_training.db", "summary_training.csv", ".next_trial_training"]:
            p = os.path.join(PHASE7_DIR, f)
            if os.path.exists(p):
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)

    os.makedirs(PHASE7_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 7a — CI Post-Training HPO")
    print(f"  Output: {PHASE7_DIR}")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    tokenizer = _load_tokenizer(device)
    base_model_ref = _load_basemodel(device)  # always loaded as template
    rollout_model_ref = None
    if os.path.exists(ROLLOUT_MODEL_PATH):
        rollout_model_ref = _load_basemodel(device)
        ckpt = torch.load(ROLLOUT_MODEL_PATH, map_location=device, weights_only=False)
        rollout_model_ref.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        rollout_model_ref.eval()
        print(f"  RolloutModel loaded from Phase 6 trial_006")
    print(f"  BaseModel: {sum(p.numel() for p in base_model_ref.parameters()):,} params")

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
            with open(config_path) as f:
                params = json.load(f)
        else:
            with open(config_path, "w") as f:
                json.dump(params, f, indent=2)

        ch = _config_hash(params)
        if ch in seen_hashes:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue

        start_m = params.get("start_model", "basemodel")
        if start_m == "rollout" and rollout_model_ref is not None:
            model = copy.deepcopy(rollout_model_ref)
        else:
            model = copy.deepcopy(base_model_ref)

        print(f"\n{'=' * 60}")
        print(f"Trial {trial.number:03d} (dir={os.path.basename(tdir)})")
        print(f"Start: {start_m} | conc_w={params['concentration_weight']} "
              f"is_w={params['interval_score_weight']} "
              f"conf={params['ci_confidence_level']} topk={params['ci_top_k']}")
        print(f"        lr={params['lr']:.1e} kl={params['kl_weight']} "
              f"updates={params['max_updates']} gamma={params['step_weight_gamma']}")
        print(f"{'=' * 60}")
        t0 = time.time()
        try:
            result = _train_ci(model, tokenizer, params, tdir, device)
            score = result["avg_interval_score"]
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue

        elapsed = time.time() - t0
        trial.set_user_attr("trial_dir", tdir)
        trial.set_user_attr("dir_name", os.path.basename(tdir))
        trial.set_user_attr("elapsed_min", round(elapsed / 60, 1))
        trial.set_user_attr("coverage", result.get("coverage", 0))
        trial.set_user_attr("avg_width", result.get("avg_width", 0))
        trial.set_user_attr("path_avg_interval_score", result.get("path_avg_interval_score", 0))

        study.tell(trial, score)
        seen_hashes.add(ch)
        completed += 1

        print(f"  IS={score:.6f}  cov={result.get('coverage',0):.4f}  "
              f"width={result.get('avg_width',0):.6f}  "
              f"time={elapsed/60:.1f}min  [{completed}/{N_TRIALS}]")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    export_summary(study)
    if len(study.trials) > 0:
        best = study.best_trial
        print(f"\nPhase 7a complete. Best IS: {best.value:.6f} (trial {best.number})")


if __name__ == "__main__":
    main()
