"""Phase 6: Rollout Post-Training HPO (Fixed).

Fixes vs original:
  1. Time features: real day/month/year instead of zeros.
  2. True 10-step AR eval on 1033-token windows (1023 prefix + 10 horizon).
  3. Multi-step Oracle-Guided rollout training (not single-step).
  4. Joint coarse+fine candidate search via temperature-scaled sampling.
  5. Independent data pipeline via RolloutWindowDataset.

Usage:
    python -m hpo.phase6_rollout
"""

from __future__ import annotations

import copy, json, os, time, hashlib
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

# ── Config ──
N_TRIALS = 110
STUDY_NAME = "phase6_rollout"
CLEAN_START = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE6_DIR = os.path.join(PROJECT_ROOT, "trials", "phase6_rollout")
STUDY_DB = os.path.join(PHASE6_DIR, "study.db")
SUMMARY_CSV = os.path.join(PHASE6_DIR, "summary.csv")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.pt")
TOKENIZER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer_config.json")
BASEMODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "base_model.pt")

TOKENIZER_BITS = 10
TOKENIZER_VOCAB = 1 << TOKENIZER_BITS
PREFIX_LEN = 1023          # observed prefix
ROLLOUT_HORIZON = 10       # 10-step autoregressive future

# ── Fixed backbone (Phase 3 trial 047) ──
BACKBONE = {
    "dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
    "dsa_windows": [None, 512, 512],
    "position_encoding": "rope", "rope_base": 10000.0,
    "dropout": 0.1323, "use_revin": False, "num_factor_tokens": 0,
}

# ── Rollout search space ──
SEARCH_SPACE = {
    "oracle_top_k":    [4, 8, 16, 32],
    "oracle_temp":     (0.5, 3.0),
    "kl_weight":       (0.005, 0.2),
    "lr":              (1e-6, 5e-5),
    "max_updates":     [240, 480, 960],
}


def sample_params(trial: optuna.Trial) -> dict:
    top_k = trial.suggest_categorical("oracle_top_k", SEARCH_SPACE["oracle_top_k"])
    temp  = trial.suggest_float("oracle_temp", *SEARCH_SPACE["oracle_temp"], log=True)
    kl_w  = trial.suggest_float("kl_weight", *SEARCH_SPACE["kl_weight"], log=True)
    lr    = trial.suggest_float("lr", *SEARCH_SPACE["lr"], log=True)
    max_up = trial.suggest_categorical("max_updates", SEARCH_SPACE["max_updates"])

    return {
        "oracle_top_k": top_k, "oracle_temp": round(temp, 4),
        "kl_weight": round(kl_w, 6), "lr": round(lr, 10),
        "max_updates": max_up,
        "batch_size": 2,
    }


# ── Helpers ──
def _make_rollout_cfg():
    """Minimal config namespace for RolloutWindowDataset."""
    return Namespace(
        prefix_len=PREFIX_LEN,
        horizon=ROLLOUT_HORIZON,
        stride_ratio=DataConfig.stride_ratio,
        cache_dir=os.path.join(PROJECT_ROOT, "posttrain", "rollout", "cache"),
        max_stocks=0,
        cache_rebuild=False,
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


# ── Load tokenizer + BaseModel ──
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


# ── Data loaders via RolloutWindowDataset ──
def _build_rollout_data(device):
    """Build train/val DataLoaders from RolloutWindowDataset (1033-token windows)."""
    cfg = _make_rollout_cfg()
    train_ds = RolloutWindowDataset("train", cfg=cfg, max_samples=0, seed=42)
    val_ds   = RolloutWindowDataset("val",   cfg=cfg, max_samples=0, seed=59)
    print(f"  Rollout train windows: {len(train_ds)}, val windows: {len(val_ds)}")

    loader_kwargs = dict(num_workers=2, pin_memory=(device.type == "cuda"),
                         persistent_workers=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True,
        collate_fn=rollout_collate, **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=rollout_collate, **loader_kwargs,
    )
    return train_loader, val_loader


# ── Oracle-guided multi-step rollout ──
@torch.no_grad()
def _oracle_guided_rollout(
    model, tokenizer, idx_c_full, idx_f_full,
    time_feats, means, stds, actual_returns,
    prefix_len, horizon, top_k, temp, device, amp_enabled, amp_dtype,
):
    """Multi-step oracle-guided rollout with joint (coarse, fine) sampling.

    At each future step, sample K (coarse, fine) token pairs from the
    temperature-scaled distribution.  Decode each pair, compare with the
    actual return (Oracle), and keep the best pair as context for the next
    step.  Returns the oracle-built context of length
    prefix_len + horizon - 1.
    """
    if horizon <= 1:
        return idx_c_full[:, :prefix_len], idx_f_full[:, :prefix_len]

    was_training = model.training
    model.eval()

    context_c = idx_c_full[:, :prefix_len].clone()  # [B, 1023]
    context_f = idx_f_full[:, :prefix_len].clone()
    K = max(2, int(top_k))
    temp = max(1e-4, float(temp))
    B = int(idx_c_full.size(0))

    for step in range(horizon - 1):  # 9 oracle-guided steps
        cur_len = int(context_c.size(1))
        cur_time = {
            "minute": time_feats["minute"][:, :cur_len],
            "day":    time_feats["day"][:, :cur_len],
            "month":  time_feats["month"][:, :cur_len],
            "year":   time_feats["year"][:, :cur_len],
        }

        # ONE forward pass (model is eval, same input → same output)
        with _autocast_ctx(amp_enabled, amp_dtype):
            logits_c, logits_f, _ = model(
                context_c, context_f,
                cur_time["minute"], cur_time["day"],
                cur_time["month"], cur_time["year"],
                last_only=True,
            )

        # Sample K (coarse, fine) pairs at once from the same logits
        probs_c = torch.softmax(logits_c[:, -1, :].float() / temp, dim=-1)
        probs_f = torch.softmax(logits_f[:, -1, :].float() / temp, dim=-1)
        samples_c = torch.multinomial(probs_c, num_samples=K, replacement=True)  # [B, K]
        samples_f = torch.multinomial(probs_f, num_samples=K, replacement=True)  # [B, K]

        # Decode ALL K candidates in one call
        decoded = tokenizer.decode(samples_c, samples_f)  # [B, K, 6]
        pred_norms = decoded[:, :, 0].float()              # [B, K]
        pred_returns = pred_norms * stds[:, 0:1] + means[:, 0:1]  # [B, K]
        errs = (pred_returns - actual_returns[:, step:step+1]).abs()  # [B, K]

        # Pick best pair for each sample in batch
        best_k = errs.argmin(dim=1)  # [B]
        rows = torch.arange(B, device=device)
        best_c = samples_c[rows, best_k]
        best_f = samples_f[rows, best_k]

        # Append oracle-best token to context for next step
        context_c = torch.cat([context_c, best_c.unsqueeze(1)], dim=1)
        context_f = torch.cat([context_f, best_f.unsqueeze(1)], dim=1)

    if was_training:
        model.train()

    return context_c, context_f


# ── Training ──
def _train_rollout(model, tokenizer, params: dict, tdir: str, device) -> dict:
    """Oracle-guided multi-step rollout training. Returns best val path_mape."""
    bp = params
    top_k   = bp["oracle_top_k"]
    temp    = bp["oracle_temp"]
    kl_weight = bp["kl_weight"]
    lr      = bp["lr"]
    max_updates = bp["max_updates"]
    batch_size  = bp["batch_size"]

    result_path = os.path.join(tdir, "result.json")
    resume_path = os.path.join(tdir, "rollout_resume.pt")
    os.makedirs(tdir, exist_ok=True)

    if os.path.exists(result_path):
        with open(result_path) as f: return json.load(f)

    train_loader, val_loader = _build_rollout_data(device)

    opt = optim.AdamW(model.parameters(), lr=lr)
    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # Resume
    update_count = 0
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        update_count = ckpt["update_count"]
        print(f"  Resume from update {update_count}")

    # Reference model for KL
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    ref_model.requires_grad_(False)

    total_updates_done = update_count
    total_ce_loss = 0.0
    prefix_len = PREFIX_LEN
    horizon = ROLLOUT_HORIZON

    model.train()
    pbar = tqdm(total=max_updates - update_count, desc="  Rollout train")

    while total_updates_done < max_updates:
        for batch in train_loader:
            if total_updates_done >= max_updates: break

            feats  = batch["features"].to(device=device, dtype=torch.float32)
            means  = batch["means"].to(device=device, dtype=torch.float32)
            stds   = batch["stds"].to(device=device, dtype=torch.float32)
            actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
            times  = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}

            B = feats.shape[0]
            if B == 0: continue

            # Encode full 1033-token window
            idx_c_full, idx_f_full = tokenizer.encode(feats)

            # ── Multi-step Oracle-Guided Rollout ──
            oracle_c, oracle_f = _oracle_guided_rollout(
                model, tokenizer, idx_c_full, idx_f_full,
                times, means, stds, actual,
                prefix_len, horizon, top_k, temp, device,
                use_amp, amp_dtype,
            )
            # oracle_c/oracle_f length = prefix_len + horizon - 1 = 1032

            train_len = int(oracle_c.size(1))  # 1032
            train_time = {k: v[:, :train_len] for k, v in times.items()}
            target_c = idx_c_full[:, prefix_len:prefix_len + horizon]  # [B, 10]
            target_f = idx_f_full[:, prefix_len:prefix_len + horizon]

            opt.zero_grad(set_to_none=True)
            with _autocast_ctx(use_amp, amp_dtype):
                # Full forward on oracle-built context
                logits_c, logits_f, _ = model(
                    oracle_c, oracle_f,
                    train_time["minute"], train_time["day"],
                    train_time["month"], train_time["year"],
                )
                # Select rollout positions: prefix_len-1 : prefix_len-1+horizon
                r_c = logits_c[:, prefix_len - 1 : prefix_len - 1 + horizon, :]
                r_f = logits_f[:, prefix_len - 1 : prefix_len - 1 + horizon, :]

                ce = F.cross_entropy(r_c.reshape(-1, r_c.size(-1)).float(),
                                     target_c.reshape(-1))
                ce = ce + F.cross_entropy(r_f.reshape(-1, r_f.size(-1)).float(),
                                          target_f.reshape(-1))

                # KL with reference model (same oracle context)
                with torch.no_grad():
                    ref_lc, ref_lf, _ = ref_model(
                        oracle_c, oracle_f,
                        train_time["minute"], train_time["day"],
                        train_time["month"], train_time["year"],
                    )
                ref_rc = ref_lc[:, prefix_len - 1 : prefix_len - 1 + horizon, :]
                ref_rf = ref_lf[:, prefix_len - 1 : prefix_len - 1 + horizon, :]

                kl = F.kl_div(
                    F.log_softmax(r_c.float(), dim=-1),
                    F.softmax(ref_rc.float(), dim=-1),
                    reduction="batchmean",
                ) + F.kl_div(
                    F.log_softmax(r_f.float(), dim=-1),
                    F.softmax(ref_rf.float(), dim=-1),
                    reduction="batchmean",
                )
                loss = ce + kl_weight * kl

            if not torch.isfinite(loss): continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            scaler.step(opt); scaler.update()

            total_ce_loss += ce.item()
            total_updates_done += 1
            pbar.update(1)
            pbar.set_postfix({"loss": ce.item(), "top_k": top_k})

            if total_updates_done % 120 == 0:
                torch.save({
                    "update_count": total_updates_done,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                }, resume_path)

            if total_updates_done >= max_updates: break

    pbar.close()

    # Save final model
    torch.save({"model_state_dict": model.state_dict(), "update_count": total_updates_done},
               os.path.join(tdir, "rollout_model.pt"))

    # Evaluate: true 10-step AR
    eval_result = _eval_rollout(model, tokenizer, val_loader, device)
    if os.path.exists(resume_path): os.remove(resume_path)

    result = {**eval_result, "train_ce": round(total_ce_loss / max(total_updates_done, 1), 6),
              "total_updates": total_updates_done, "params": bp}
    with open(result_path, "w") as f: json.dump(result, f, indent=2)
    return result


# ── True 10-step AR evaluation ──
@torch.no_grad()
def _eval_rollout(model, tokenizer, val_loader, device) -> dict:
    """Strict 10-step autoregressive rollout -> path_mape.

    Uses RolloutWindowDataset which provides 1033-token windows
    (1023 prefix + 10 future steps with ground truth).
    """
    model.eval()
    all_path_mape = []
    all_daily_mape = []
    prefix_len = PREFIX_LEN
    horizon = ROLLOUT_HORIZON

    n_batches = 0
    for batch in tqdm(val_loader, desc="  Eval AR10", leave=False):
        feats  = batch["features"].to(device=device, dtype=torch.float32)
        means  = batch["means"].to(device=device, dtype=torch.float32)
        stds   = batch["stds"].to(device=device, dtype=torch.float32)
        actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
        times  = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}

        B = feats.shape[0]
        if B == 0: continue
        n_batches += 1
        if n_batches > 100: break

        idx_c, idx_f = tokenizer.encode(feats)  # [B, 1033]
        cur_c = idx_c[:, :prefix_len].clone()   # [B, 1023]
        cur_f = idx_f[:, :prefix_len].clone()

        # Actual returns for all 10 steps (raw log_ret)
        actual_rets = actual.cpu()  # [B, 10]

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
            if not torch.isfinite(logits_c).all():
                break
            pc = logits_c[:, -1, :].argmax(dim=-1)
            pf = logits_f[:, -1, :].argmax(dim=-1)
            dec = tokenizer.decode(pc.unsqueeze(1), pf.unsqueeze(1))
            pred_norm = dec[:, 0, 0].cpu().float()
            pred_ret = pred_norm * stds[:, 0].cpu() + means[:, 0].cpu()
            pred_rets.append(pred_ret)

            # Append prediction to context for next step
            if step < horizon - 1:
                cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
                cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)

        if len(pred_rets) < horizon:
            continue

        pred_rets = torch.stack(pred_rets, dim=1)  # [B, 10]

        # Cumulative path returns
        cum_pred   = torch.cumsum(pred_rets.float(), dim=1)
        cum_actual = torch.cumsum(actual_rets.float(), dim=1)

        # Path MAPE per step
        for step in range(horizon):
            pred_ratio   = torch.exp(torch.clamp(cum_pred[:, step], -20, 20))
            actual_ratio = torch.exp(torch.clamp(cum_actual[:, step], -20, 20))
            denom = torch.clamp(torch.abs(actual_ratio), min=1e-6)
            valid = torch.isfinite(pred_ratio) & torch.isfinite(actual_ratio) & (denom > 0)
            if valid.sum() > 0:
                mape_step = (torch.abs(pred_ratio[valid] - actual_ratio[valid])
                             / denom[valid]).mean().item() * 100
                all_path_mape.append(mape_step)

            # Daily MAPE
            dr = torch.exp(torch.clamp(pred_rets[:, step].float(), -20, 20))
            da = torch.exp(torch.clamp(actual_rets[:, step].float(), -20, 20))
            denom_d = torch.clamp(torch.abs(da), min=1e-6)
            valid_d = torch.isfinite(dr) & torch.isfinite(da) & (denom_d > 0)
            if valid_d.sum() > 0:
                mape_d = (torch.abs(dr[valid_d] - da[valid_d])
                          / denom_d[valid_d]).mean().item() * 100
                all_daily_mape.append(mape_d)

    if not all_path_mape:
        print(f"  WARNING: all path_mape values filtered as NaN/Inf — returning 999")
        return {"path_mape": 999.0, "daily_mape": 999.0,
                "path_mape_std": 0.0, "num_eval_steps": 0}

    return {
        "path_mape":      round(float(np.mean(all_path_mape)), 6),
        "daily_mape":     round(float(np.mean(all_daily_mape)), 6),
        "path_mape_std":  round(float(np.std(all_path_mape)), 6),
        "num_eval_steps": len(all_path_mape),
    }


# ── Resume-aware directory assignment ──
def _assign_trial_dir():
    os.makedirs(PHASE6_DIR, exist_ok=True)
    counter_path = os.path.join(PHASE6_DIR, ".next_trial")

    existing = sorted([d for d in os.listdir(PHASE6_DIR)
                       if d.startswith("trial_") and os.path.isdir(os.path.join(PHASE6_DIR, d))],
                      key=lambda x: int(x.split("_")[1]))
    for d in existing:
        full = os.path.join(PHASE6_DIR, d)
        resume = os.path.join(full, "rollout_resume.pt")
        result = os.path.join(full, "result.json")
        if os.path.exists(resume) and not os.path.exists(result):
            if os.path.exists(os.path.join(full, "config.json")):
                print(f"  Found incomplete trial: {d} — will resume")
                return full

    next_num = 0
    if os.path.exists(counter_path):
        with open(counter_path) as f: next_num = int(f.read().strip())
    new_dir = os.path.join(PHASE6_DIR, f"trial_{next_num:03d}")
    with open(counter_path, "w") as f: f.write(str(next_num + 1))
    return new_dir


# ── Main ──
def main():
    if CLEAN_START and os.path.exists(PHASE6_DIR):
        import shutil; shutil.rmtree(PHASE6_DIR)
    os.makedirs(PHASE6_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 6 — Rollout Post-Training HPO (FIXED)")
    print(f"  Output: {PHASE6_DIR}")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"  Prefix: {PREFIX_LEN}, Horizon: {ROLLOUT_HORIZON}")
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

    # Manual loop (study.ask/tell) — fixes resume double-count
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
        print(f"Params: top_k={params['oracle_top_k']} temp={params['oracle_temp']} "
              f"kl={params['kl_weight']} lr={params['lr']:.1e}")
        print(f"        updates={params['max_updates']}")
        print(f"{'='*60}")

        # Clone model for this trial
        model = copy.deepcopy(base_model)

        t0 = time.time()
        try:
            result = _train_rollout(model, tokenizer, params, tdir, device)
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
        trial.set_user_attr("train_ce", result.get("train_ce", 0))

        study.tell(trial, path_mape)

        seen_hashes.add(ch)
        completed += 1

        print(f"  path_mape={path_mape:.4f}%  daily_mape={result['daily_mape']:.4f}%  "
              f"time={elapsed/60:.1f}min  [{completed}/{N_TRIALS}]")
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    # Summary
    export_summary(study)
    if len(study.trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        best = study.best_trial
        print(f"\nPhase 6 complete. Best path_mape: {best.value:.4f}% (trial {best.number})")
    else:
        print(f"\nPhase 6 complete. No successful trials.")


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
    os.makedirs(PHASE6_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    ranked = sorted(rows, key=lambda r: r["value"])
    print(f"\nTop-10 by path_mape:")
    for r in ranked[:10]:
        print(f"  Trial {r['trial']:03d}  path_mape={r['value']:.4f}  "
              f"top_k={r.get('oracle_top_k','?')}  temp={r.get('oracle_temp','?')}")
    print(f"Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
