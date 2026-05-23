"""Phase 7 Sup V2 — Quantile-head CI post-training with conformal, dual, etc.

Stages:
  --stage baselines       Lock reference baselines + split manifest.
  --stage quantile_smoke  A1 head-only smoke test.
  --stage quantile_hpo    A/B/C HPO (head-only, head-LoRA, dual, conformal).
  --stage final_eval      Evaluate top checkpoints on val_eval / demo.

Usage:
  python -m hpo.phase7_sup_v2 --stage baselines
  python -m hpo.phase7_sup_v2 --stage quantile_smoke
  python -m hpo.phase7_sup_v2 --stage quantile_hpo --trials 52
"""

import argparse, copy, csv, hashlib, json, math, os, sys, time
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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DataConfig
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT
from posttrain.rollout.data import RolloutWindowDataset, rollout_collate
from posttrain.ci.eval_ci import compute_ci_metrics
from posttrain.ci.quantile_head import CIQuantileHead, CONFIDENCE_LEVELS, K_CONSTANTS
from posttrain.ci.quantile_train import (
    compute_quantile_training_loss,
    CoverageDualController,
    dual_pinball_loss,
    interval_score_loss,
)
from posttrain.ci.conformal import (
    calibrate_and_evaluate,
    split_indices_time_ordered,
    split_indices_random,
)
from reproducibility import set_global_seed

# ═══════════════════════════════════════════════════════════════
# Paths & Constants
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUP_DIR = os.path.join(PROJECT_ROOT, "trials", "phase7_sup")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.pt")
TOKENIZER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer_config.json")
BASEMODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "base_model.pt")
PHASE7_DIR = os.path.join(PROJECT_ROOT, "trials", "phase7_ci")

TOKENIZER_VOCAB = 1 << 10
PREFIX_LEN = 1023
HORIZON = 10

BACKBONE = {
    "dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
    "dsa_windows": [None, 512, 512],
    "position_encoding": "rope", "rope_base": 10000.0,
    "dropout": 0.1323, "use_revin": False, "num_factor_tokens": 0,
}

DEFAULT_CONFIDENCE_LEVELS = (0.68, 0.80, 0.90)


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

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


def _load_model(device):
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
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    return model


def _make_cfg():
    return Namespace(
        prefix_len=PREFIX_LEN, horizon=HORIZON,
        stride_ratio=DataConfig.stride_ratio,
        cache_dir=os.path.join(PROJECT_ROOT, "posttrain", "rollout", "cache"),
        max_stocks=0, cache_rebuild=False,
    )


def _build_data(device, max_train=2048, max_val=500):
    cfg = _make_cfg()
    train_ds = RolloutWindowDataset("train", cfg=cfg, max_samples=max_train, seed=42)
    val_ds = RolloutWindowDataset("val", cfg=cfg, max_samples=max_val, seed=59)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=rollout_collate, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=rollout_collate, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


def _get_hidden_states(model, tokenizer, batch, device):
    """Full forward on ground-truth context → hidden states at horizon positions."""
    feats = batch["features"].to(device=device, dtype=torch.float32)
    times_f = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}
    idx_c, idx_f = tokenizer.encode(feats)
    B = feats.size(0)

    ctx_c = idx_c[:, :PREFIX_LEN + HORIZON - 1]
    ctx_f = idx_f[:, :PREFIX_LEN + HORIZON - 1]
    ctx_time = {k: times_f[k][:, :ctx_c.size(1)] for k in ("minute", "day", "month", "year")}

    with torch.no_grad():
        _, _, _, hidden = model(
            ctx_c, ctx_f,
            ctx_time["minute"], ctx_time["day"],
            ctx_time["month"], ctx_time["year"],
            return_hidden=True,
        )
    # hidden: [B, T, dim]; extract horizon positions
    h = hidden[:, PREFIX_LEN - 1 : PREFIX_LEN - 1 + HORIZON, :]  # [B, H, dim]
    actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
    means = batch["means"].to(device=device, dtype=torch.float32)
    stds = batch["stds"].to(device=device, dtype=torch.float32)
    actual_denorm = actual[:, :HORIZON] * stds[:, 0:1] + means[:, 0:1]
    return h, actual_denorm


# ═══════════════════════════════════════════════════════════════
# Teacher quantile caching
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_teacher_quantiles(model, tokenizer, loader, device, confidence_levels,
                               N=512, temperature=1.5):
    """Pre-compute high-N sampling quantiles for teacher distillation."""
    model.eval()
    teacher_data = {}  # sample_id → {c: (low, high)}

    for batch_idx, batch in enumerate(tqdm(loader, desc="Teacher quantiles", leave=False)):
        feats = batch["features"].to(device=device, dtype=torch.float32)
        means = batch["means"].to(device=device, dtype=torch.float32)
        stds = batch["stds"].to(device=device, dtype=torch.float32)
        times_f = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}
        idx_c, idx_f = tokenizer.encode(feats)
        B = feats.size(0)

        context_c = idx_c[:, :PREFIX_LEN].clone()
        context_f = idx_f[:, :PREFIX_LEN].clone()

        all_quantiles = {c: ([], []) for c in confidence_levels}

        for step in range(HORIZON):
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
            temp = max(temperature, 1e-5)
            probs_c = torch.softmax(last_c / temp, dim=-1)
            probs_f = torch.softmax(last_f / temp, dim=-1)
            sc = torch.multinomial(probs_c, num_samples=N, replacement=True)
            sf = torch.multinomial(probs_f, num_samples=N, replacement=True)
            decoded = tokenizer.decode(sc, sf)
            pred_rets = decoded[:, :, 0].float() * stds[:, 0:1] + means[:, 0:1]
            sorted_r = pred_rets.sort(dim=1).values

            for c in confidence_levels:
                alpha = 1.0 - c
                idx_low = max(0, min(N-1, int(alpha/2 * N)))
                idx_high = max(0, min(N-1, int((1-alpha/2) * N)))
                all_quantiles[c][0].append(sorted_r[:, idx_low].cpu())
                all_quantiles[c][1].append(sorted_r[:, idx_high].cpu())

            if step < HORIZON - 1:
                next_c = last_c.argmax(dim=-1)
                next_f = last_f.argmax(dim=-1)
                context_c = torch.cat([context_c, next_c.unsqueeze(1)], dim=1)
                context_f = torch.cat([context_f, next_f.unsqueeze(1)], dim=1)

        for c in confidence_levels:
            t_low = torch.stack(all_quantiles[c][0], dim=1)  # [B, H]
            t_high = torch.stack(all_quantiles[c][1], dim=1)
            for b in range(B):
                sid = f"{batch_idx}_{b}"
                if sid not in teacher_data:
                    teacher_data[sid] = {}
                teacher_data[sid][c] = (t_low[b], t_high[b])

    return teacher_data


# ═══════════════════════════════════════════════════════════════
# Quantile-head evaluation
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_quantile_head(model, tokenizer, quantile_head, loader, device,
                            confidence_level=0.80):
    """Evaluate CI metrics via quantile head on val set."""
    model.eval()
    quantile_head.eval()

    all_lower, all_upper, all_actual = [], [], []
    for batch in loader:
        h, actual_denorm = _get_hidden_states(model, tokenizer, batch, device)
        lower, upper = quantile_head.get_interval(h, confidence_level)
        all_lower.append(lower.cpu())
        all_upper.append(upper.cpu())
        all_actual.append(actual_denorm.cpu())

    if not all_lower:
        return {"avg_interval_score": 999.0}
    pl = torch.cat(all_lower, dim=0).detach().cpu().numpy()
    pu = torch.cat(all_upper, dim=0).detach().cpu().numpy()
    aa = torch.cat(all_actual, dim=0).detach().cpu().numpy()
    return compute_ci_metrics(pl, pu, aa, confidence_level=float(confidence_level))


# ═══════════════════════════════════════════════════════════════
# Stage 0: Lock baselines
# ═══════════════════════════════════════════════════════════════

def stage_baselines():
    """Lock reference baselines and create split manifest."""
    os.makedirs(SUP_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tokenizer = _load_tokenizer(device)
    model = _load_model(device)
    _, val_loader, _, n_val = _build_data(device, max_train=0, max_val=500)

    baselines = {}

    # Baseline 1: sampling best quality (T=1.5, N=128, C=0.68)
    print("Computing baseline: sampling_best_quality (T=1.5, N=128, C=0.68)...")
    from posttrain.ci.ci_sampling import predict_ci_sampling
    cfg_s = Namespace(prefix_len=PREFIX_LEN, horizon=HORIZON, batch_size=8)
    pl, pu, _, aa = predict_ci_sampling(
        model=model, tokenizer=tokenizer, loader=val_loader, cfg=cfg_s,
        device=device, amp_enabled=True, amp_dtype=torch.bfloat16,
        num_samples=128, temperature=1.5, confidence_level=0.68, feed_mode="argmax",
    )
    m = compute_ci_metrics(pl, pu, aa, confidence_level=0.68)
    baselines["sampling_best_quality"] = {
        "config": "T=1.5, N=128, C=0.68, argmax",
        "avg_interval_score": m["avg_interval_score"],
        "coverage": m["coverage"], "avg_width": m["avg_width"],
        "path_avg_interval_score": m["path_avg_interval_score"],
        "path_coverage": m["path_coverage"],
    }
    print(f"  IS={m['avg_interval_score']:.6f} cov={m['coverage']:.4f}")

    # Baseline 2: sampling prod (T=1.5, N=64, C=0.80)
    print("Computing baseline: sampling_prod (T=1.5, N=64, C=0.80)...")
    pl, pu, _, aa = predict_ci_sampling(
        model=model, tokenizer=tokenizer, loader=val_loader, cfg=cfg_s,
        device=device, amp_enabled=True, amp_dtype=torch.bfloat16,
        num_samples=64, temperature=1.5, confidence_level=0.80, feed_mode="argmax",
    )
    m = compute_ci_metrics(pl, pu, aa, confidence_level=0.80)
    baselines["sampling_prod"] = {
        "config": "T=1.5, N=64, C=0.80, argmax",
        "avg_interval_score": m["avg_interval_score"],
        "coverage": m["coverage"], "avg_width": m["avg_width"],
        "path_avg_interval_score": m["path_avg_interval_score"],
        "path_coverage": m["path_coverage"],
    }
    print(f"  IS={m['avg_interval_score']:.6f} cov={m['coverage']:.4f}")

    # Baseline 3: CI training best (from Phase 7)
    training_csv = os.path.join(PHASE7_DIR, "summary_training.csv")
    if os.path.exists(training_csv):
        with open(training_csv, newline="") as f:
            tr = list(csv.DictReader(f))
        for r in tr:
            for k in list(r.keys()):
                try: r[k] = float(r[k])
                except (ValueError, TypeError): pass
            if "value" in r: r["avg_interval_score"] = r["value"]
        if tr:
            best_t = min(tr, key=lambda r: r.get("avg_interval_score", 999))
            baselines["ci_training_best"] = {
                "config": f"conc_w={best_t.get('concentration_weight','?')} is_w={best_t.get('interval_score_weight','?')}",
                "avg_interval_score": best_t.get("avg_interval_score", 0),
                "coverage": best_t.get("coverage", 0),
                "avg_width": best_t.get("avg_width", 0),
            }
            print(f"  CI training best: IS={best_t.get('avg_interval_score',0):.6f}")

    # Create split manifest
    n_total = n_val
    calib_idx, eval_idx = split_indices_time_ordered(n_total, calib_ratio=0.4)
    random_calib, random_eval = split_indices_random(n_total, calib_ratio=0.4)
    split_manifest = {
        "n_total": n_total,
        "calib_ratio": 0.4,
        "time_split": {
            "calib_n": int(len(calib_idx)),
            "eval_n": int(len(eval_idx)),
            "calib_indices": calib_idx.tolist(),
            "eval_indices": eval_idx.tolist(),
        },
        "random_split": {
            "calib_n": int(len(random_calib)),
            "eval_n": int(len(random_eval)),
        },
    }

    # Save
    with open(os.path.join(SUP_DIR, "locked_baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)
    with open(os.path.join(SUP_DIR, "split_manifest.json"), "w") as f:
        json.dump(split_manifest, f, indent=2)

    # Write baseline CSV
    csv_path = os.path.join(SUP_DIR, "baseline_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "avg_interval_score", "coverage", "avg_width"])
        w.writeheader()
        for name, b in baselines.items():
            w.writerow({"name": name, **{k: b.get(k, "") for k in ["avg_interval_score", "coverage", "avg_width"]}})

    print(f"\nStage 0 complete. Baselines saved to {SUP_DIR}")
    print(json.dumps(baselines, indent=2))


# ═══════════════════════════════════════════════════════════════
# Stage 1: Quantile smoke test
# ═══════════════════════════════════════════════════════════════

def stage_quantile_smoke():
    """A1: Head-only quantile student smoke test."""
    os.makedirs(SUP_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    set_global_seed(42)

    tokenizer = _load_tokenizer(device)
    model = _load_model(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    train_loader, val_loader, n_train, n_val = _build_data(device, max_train=2048, max_val=500)

    # Compute teacher quantiles
    print("Computing teacher quantiles (T=1.5, N=512)...")
    teacher_data = compute_teacher_quantiles(
        model, tokenizer, train_loader, device,
        confidence_levels=DEFAULT_CONFIDENCE_LEVELS, N=512, temperature=1.5,
    )
    print(f"  Teacher quantiles computed for {len(teacher_data)} samples")

    # Build CIQuantileHead
    quantile_head = CIQuantileHead(
        hidden_dim=BACKBONE["dim"],
        num_steps=HORIZON,
        step_embedding_dim=16,
        head_hidden_dim=128,
        share_aC=True,
    ).to(device)

    opt = torch.optim.AdamW(quantile_head.parameters(), lr=1e-3)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    use_amp = device.type == "cuda" and amp_dtype is not None

    # Smoke params
    max_updates = 300
    lambda_pinball = 1.0
    lambda_teacher = 0.3
    lambda_is_score = 0.3
    lambda_mono_step = 0.05

    total_updates = 0
    history = []
    best_is = float("inf")
    best_path = os.path.join(SUP_DIR, "quantile_smoke_best.pt")

    quantile_head.train()
    pbar = tqdm(total=max_updates, desc="Quantile smoke")

    while total_updates < max_updates:
        for batch_idx, batch in enumerate(train_loader):
            if total_updates >= max_updates:
                break

            h, actual_denorm = _get_hidden_states(model, tokenizer, batch, device)
            B = h.size(0)

            # Gather teacher quantiles for this batch
            teacher_batch = {}
            for c in DEFAULT_CONFIDENCE_LEVELS:
                t_low_list, t_high_list = [], []
                for b in range(B):
                    sid = f"{batch_idx}_{b}"
                    if sid in teacher_data and c in teacher_data[sid]:
                        t_low_list.append(teacher_data[sid][c][0])
                        t_high_list.append(teacher_data[sid][c][1])
                if t_low_list:
                    teacher_batch[c] = (
                        torch.stack(t_low_list).to(device),
                        torch.stack(t_high_list).to(device),
                    )

            opt.zero_grad()
            loss, stats = compute_quantile_training_loss(
                quantile_head=quantile_head,
                hidden_states=h,
                actual_returns=actual_denorm,
                teacher_quantiles=teacher_batch if teacher_batch else None,
                confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
                lambda_pinball=lambda_pinball,
                lambda_teacher=lambda_teacher,
                lambda_is_score=lambda_is_score,
                lambda_mono_step=lambda_mono_step,
            )

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(quantile_head.parameters(), 1.0)
            opt.step()

            total_updates += 1
            pbar.update(1)
            pbar.set_postfix({k: f"{v:.4f}" for k, v in stats.items() if isinstance(v, float)})

            if total_updates % 50 == 0 or total_updates >= max_updates:
                # Quick eval
                metrics = evaluate_quantile_head(model, tokenizer, quantile_head, val_loader, device, confidence_level=0.80)
                is_val = metrics.get("avg_interval_score", 999)
                history.append({"update": total_updates, "is": is_val, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
                if is_val < best_is:
                    best_is = is_val
                    torch.save({"quantile_head": quantile_head.state_dict(), "update": total_updates, "metrics": metrics}, best_path)
                print(f"  [update {total_updates}] IS={is_val:.6f} cov={metrics.get('coverage',0):.4f} w={metrics.get('avg_width',0):.6f}")

            if total_updates >= max_updates:
                break

    pbar.close()

    # Final eval at all confidence levels
    final_metrics = {}
    for c in CONFIDENCE_LEVELS:
        m = evaluate_quantile_head(model, tokenizer, quantile_head, val_loader, device, confidence_level=c)
        final_metrics[f"C{c}"] = {k: v for k, v in m.items() if isinstance(v, (int, float, str, bool))}
        print(f"  C={c:.0%}: IS={m.get('avg_interval_score',0):.6f} cov={m.get('coverage',0):.4f}")

    smoke_result = {
        "stage": "quantile_smoke",
        "best_is": best_is,
        "final_metrics": final_metrics,
        "history": history,
        "params": {
            "max_updates": max_updates, "lambda_pinball": lambda_pinball,
            "lambda_teacher": lambda_teacher, "lambda_is_score": lambda_is_score,
            "lambda_mono_step": lambda_mono_step,
        },
    }
    with open(os.path.join(SUP_DIR, "quantile_smoke_result.json"), "w") as f:
        json.dump(smoke_result, f, indent=2)
    print(f"\nStage 1 complete. Best IS={best_is:.6f} saved to {best_path}")


# ═══════════════════════════════════════════════════════════════
# Stage 2: Quantile HPO
# ═══════════════════════════════════════════════════════════════

SEARCH_SPACE_V2 = {
    "lambda_pinball": [0.5, 1.0, 2.0],
    "lambda_teacher": [0.1, 0.3, 1.0],
    "lambda_is_score": [0.1, 0.3, 1.0],
    "lambda_mono_step": [0.01, 0.05, 0.1],
    "lr_head": [1e-4, 3e-4, 1e-3],
    "max_updates": [200, 400],
    "use_dual": [False, True],
    "use_conformal": [False, True],
    "trainable_scope": ["head", "head_lora"],
}

SEARCH_SPACE_BC = {
    "lambda_pinball": [0.5, 1.0, 2.0],
    "lambda_teacher": [0.1, 0.3, 1.0],
    "lambda_is_score": [0.1, 0.3, 1.0],
    "lambda_mono_step": [0.01, 0.05],
    "lr_head": [1e-4, 3e-4, 1e-3],
    "max_updates": [200, 400],
    "use_dual": [True],
    "use_conformal": [True],
    "trainable_scope": ["head"],
}

_global_teacher_cache = None  # set once before HPO loop

STUDY_NAME_V2 = "phase7_sup_quantile_hpo"
STUDY_DB_V2 = os.path.join(SUP_DIR, "study_quantile_hpo.db")


def _train_quantile_trial(model, tokenizer, quantile_head, train_loader, val_loader,
                           device, params, tdir):
    """Single quantile HPO trial."""
    bp = params
    max_updates = bp["max_updates"]
    lr = bp["lr_head"]
    lambda_pinball = bp["lambda_pinball"]
    lambda_teacher = bp["lambda_teacher"]
    lambda_is_score = bp["lambda_is_score"]
    lambda_mono_step = bp["lambda_mono_step"]
    use_dual = bp.get("use_dual", False)
    use_conformal = bp.get("use_conformal", False)
    trainable_scope = bp.get("trainable_scope", "head")

    result_path = os.path.join(tdir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    # ── Configure trainable params ──
    if trainable_scope == "head_lora":
        from model.lora import inject_lora
        inject_lora(model, rank=8, alpha=16, dropout=0.05,
                     target_keywords=("to_qkv", "to_out", "head_coarse", "head_fine"),
                     freeze_base=True)
        # Train LoRA params + quantile head
        trainable_params = list(quantile_head.parameters())
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable_params.append(p)
        print(f"  LoRA: {sum(p.numel() for p in trainable_params)} trainable params")
    else:
        for p in model.parameters():
            p.requires_grad = False
        trainable_params = list(quantile_head.parameters())

    # Reset head
    head_init = CIQuantileHead(
        hidden_dim=BACKBONE["dim"], num_steps=HORIZON,
        step_embedding_dim=16, head_hidden_dim=128, share_aC=True,
    )
    quantile_head.load_state_dict(head_init.state_dict())

    opt = torch.optim.AdamW(trainable_params, lr=lr)
    dual_ctrl = CoverageDualController(confidence_levels=DEFAULT_CONFIDENCE_LEVELS) if use_dual else None

    # Load teacher quantiles from GLOBAL cache (set by stage_quantile_hpo)
    teacher_data = _global_teacher_cache
    if teacher_data is None:
        teacher_data = {}  # no teacher, just pinball loss

    total_updates = 0
    quantile_head.train()
    if trainable_scope == "head_lora":
        model.train()
    last_stats = {}

    while total_updates < max_updates:
        for batch_idx, batch in enumerate(train_loader):
            if total_updates >= max_updates:
                break

            if trainable_scope == "head_lora":
                # LoRA: forward through backbone needs grad
                feats = batch["features"].to(device=device, dtype=torch.float32)
                times_f = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}
                idx_c, idx_f = tokenizer.encode(feats)
                ctx_c = idx_c[:, :PREFIX_LEN + HORIZON - 1]
                ctx_f = idx_f[:, :PREFIX_LEN + HORIZON - 1]
                ctx_time = {k: times_f[k][:, :ctx_c.size(1)] for k in ("minute", "day", "month", "year")}
                _, _, _, hidden = model(ctx_c, ctx_f, ctx_time["minute"], ctx_time["day"],
                                         ctx_time["month"], ctx_time["year"], return_hidden=True)
                h = hidden[:, PREFIX_LEN - 1 : PREFIX_LEN - 1 + HORIZON, :]
                actual_denorm = (batch["actual_returns"].to(device=device, dtype=torch.float32)[:, :HORIZON]
                                 * batch["stds"].to(device=device, dtype=torch.float32)[:, 0:1]
                                 + batch["means"].to(device=device, dtype=torch.float32)[:, 0:1])
            else:
                h, actual_denorm = _get_hidden_states(model, tokenizer, batch, device)
            B = h.size(0)

            teacher_batch = {}
            for c in DEFAULT_CONFIDENCE_LEVELS:
                t_low_list, t_high_list = [], []
                for b in range(B):
                    sid = f"{batch_idx}_{b}"
                    if sid in teacher_data and c in teacher_data[sid]:
                        t_low_list.append(teacher_data[sid][c][0])
                        t_high_list.append(teacher_data[sid][c][1])
                if t_low_list:
                    teacher_batch[c] = (
                        torch.stack(t_low_list).to(device),
                        torch.stack(t_high_list).to(device),
                    )

            opt.zero_grad()
            loss, stats = compute_quantile_training_loss(
                quantile_head=quantile_head,
                hidden_states=h,
                actual_returns=actual_denorm,
                teacher_quantiles=teacher_batch if teacher_batch else None,
                confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
                lambda_pinball=lambda_pinball,
                lambda_teacher=lambda_teacher,
                lambda_is_score=lambda_is_score,
                lambda_mono_step=lambda_mono_step,
                dual_controller=dual_ctrl,
            )

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(quantile_head.parameters(), 1.0)
            opt.step()
            total_updates += 1
            last_stats = stats

            # Early prune: if coverage < target - 10pp after 100 updates
            if total_updates == 100 and stats.get("coverage", 0) < 0.70:
                return {"avg_interval_score": 999.0, "coverage": stats["coverage"],
                       "pruned": True, "prune_reason": "low_coverage"}

            if total_updates >= max_updates:
                break

    # Evaluate
    primary_c = 0.80
    metrics = evaluate_quantile_head(model, tokenizer, quantile_head, val_loader, device, confidence_level=primary_c)

    # Conformal correction
    if use_conformal:
        # Split val into calib/eval
        n_val = len(val_loader.dataset)
        calib_idx, eval_idx = split_indices_time_ordered(n_val, calib_ratio=0.4)

        all_lower, all_upper, all_actual = [], [], []
        quantile_head.eval()
        for batch_idx, batch in enumerate(val_loader):
            h, actual_denorm = _get_hidden_states(model, tokenizer, batch, device)
            lower, upper = quantile_head.get_interval(h, primary_c)
            all_lower.append(lower.cpu())
            all_upper.append(upper.cpu())
            all_actual.append(actual_denorm.cpu())

        pl = torch.cat(all_lower, dim=0).detach().cpu().numpy()
        pu = torch.cat(all_upper, dim=0).detach().cpu().numpy()
        aa = torch.cat(all_actual, dim=0).detach().cpu().numpy()

        calib_result = calibrate_and_evaluate(
            pred_lower_calib=pl[calib_idx], pred_upper_calib=pu[calib_idx],
            actual_calib=aa[calib_idx],
            pred_lower_eval=pl[eval_idx], pred_upper_eval=pu[eval_idx],
            actual_eval=aa[eval_idx],
            confidence_level=primary_c,
        )
        metrics_conformal = compute_ci_metrics(
            calib_result["pred_lower_corrected"],
            calib_result["pred_upper_corrected"],
            aa[eval_idx],
            confidence_level=primary_c,
        )
        metrics["conformal_is"] = metrics_conformal["avg_interval_score"]
        metrics["conformal_coverage"] = metrics_conformal["coverage"]
        metrics["conformal_offset"] = calib_result.get("offset", 0)

    score = metrics.get("conformal_is", metrics["avg_interval_score"]) if use_conformal else metrics["avg_interval_score"]

    result = {
        "avg_interval_score": round(score, 6),
        "coverage": round(metrics.get("coverage", 0), 4),
        "avg_width": round(metrics.get("avg_width", 0), 6),
        "train_coverage": round(last_stats.get("coverage", 0), 4),
        "total_updates": total_updates,
        "params": bp,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    torch.save({"quantile_head": quantile_head.state_dict(), "metrics": result},
               os.path.join(tdir, "quantile_head.pt"))
    return result


def stage_quantile_hpo(n_trials=40, search_space=None):
    """A/B/C HPO for quantile head."""
    global _global_teacher_cache
    if search_space is None:
        search_space = SEARCH_SPACE_V2
    os.makedirs(SUP_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    set_global_seed(42)
    tokenizer = _load_tokenizer(device)
    model = _load_model(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    train_loader, val_loader, n_train, n_val = _build_data(device, max_train=2048, max_val=500)
    print(f"Train samples: {n_train}, Val samples: {n_val}")

    # ── Pre-compute teacher quantiles ONCE ──
    teacher_cache_path = os.path.join(SUP_DIR, "teacher_cache.pt")
    if os.path.exists(teacher_cache_path):
        print("Loading cached teacher quantiles...")
        _global_teacher_cache = torch.load(teacher_cache_path, map_location="cpu", weights_only=False)
    else:
        print("Computing teacher quantiles (T=1.5, N=512) — ONE TIME only...")
        _global_teacher_cache = compute_teacher_quantiles(
            model, tokenizer, train_loader, device,
            confidence_levels=DEFAULT_CONFIDENCE_LEVELS, N=512, temperature=1.5,
        )
        torch.save(_global_teacher_cache, teacher_cache_path)
    print(f"  Teacher cache: {len(_global_teacher_cache)} samples")

    study = optuna.create_study(
        study_name=STUDY_NAME_V2, storage=f"sqlite:///{STUDY_DB_V2}",
        direction="minimize", load_if_exists=True,
    )

    # Base quantile head (re-initialized per trial)
    base_head = CIQuantileHead(
        hidden_dim=BACKBONE["dim"], num_steps=HORIZON,
        step_embedding_dim=16, head_hidden_dim=128, share_aC=True,
    ).to(device)

    seen_hashes = set()
    completed = 0
    trial_counter = 0
    tdir_counter_path = os.path.join(SUP_DIR, ".next_quantile_trial")

    while completed < n_trials:
        trial = study.ask()
        params = {}
        for key, values in search_space.items():
            if isinstance(values, list) and all(isinstance(v, bool) for v in values):
                params[key] = trial.suggest_categorical(key, values)
            elif isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                params[key] = trial.suggest_categorical(key, values)
        params["lr_head"] = trial.suggest_categorical("lr_head", search_space["lr_head"])

        ch = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        if ch in seen_hashes:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue

        # Assign directory
        existing = sorted([d for d in os.listdir(SUP_DIR) if d.startswith("quantile_trial_")],
                         key=lambda x: int(x.split("_")[2]) if len(x.split("_")) >= 3 else 0)
        tdir = os.path.join(SUP_DIR, f"quantile_trial_{len(existing):03d}")
        os.makedirs(tdir, exist_ok=True)

        # Save config
        with open(os.path.join(tdir, "config.json"), "w") as f:
            json.dump(params, f, indent=2)

        head = copy.deepcopy(base_head).to(device)

        print(f"\nTrial {trial.number:03d}: {params}")
        t0 = time.time()
        try:
            result = _train_quantile_trial(model, tokenizer, head, train_loader, val_loader,
                                            device, params, tdir)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue

        elapsed = time.time() - t0
        trial.set_user_attr("elapsed_min", round(elapsed / 60, 1))
        trial.set_user_attr("coverage", result.get("coverage", 0))
        trial.set_user_attr("conformal_coverage", result.get("conformal_coverage", result.get("coverage", 0)))

        study.tell(trial, result["avg_interval_score"])
        seen_hashes.add(ch)
        completed += 1

        print(f"  IS={result['avg_interval_score']:.6f} cov={result['coverage']:.4f} "
              f"time={elapsed/60:.1f}min [{completed}/{n_trials}]")
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    summary_path = os.path.join(SUP_DIR, "summary_quantile_hpo.csv")
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = {"trial": t.number, "value": t.value, **t.params}
            for k, v in t.user_attrs.items():
                if isinstance(v, (int, float, str, bool)):
                    row[k] = v
            rows.append(row)
    if rows:
        keys = ["trial", "value"] + sorted(k for k in rows[0].keys() if k not in ("trial", "value"))
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        ranked = sorted(rows, key=lambda r: r["value"])
        print(f"\nTop-5 by IS:")
        for r in ranked[:5]:
            print(f"  Trial {r['trial']:03d} IS={r['value']:.6f} "
                  f"pinball={r.get('lambda_pinball','?')} teacher={r.get('lambda_teacher','?')} "
                  f"is={r.get('lambda_is_score','?')} lr={r.get('lr_head','?')}")

    print(f"\nStage 2 complete. Best IS: {study.best_value:.6f}")


# ═══════════════════════════════════════════════════════════════
# Method E: Logit Calibration Adapter
# ═══════════════════════════════════════════════════════════════

class LogitCalibrationHead(nn.Module):
    """Tiny head that predicts per-step temperature for logit calibration.

    calibrated_logits = logits / softplus(raw_T_t + T_base)
    Then CI is constructed from the calibrated token distribution.
    """

    def __init__(self, hidden_dim=384, num_steps=10, step_emb_dim=8, T_base=1.5):
        super().__init__()
        self.T_base = T_base
        self.step_embed = nn.Embedding(num_steps, step_emb_dim)
        self.fc = nn.Linear(hidden_dim + step_emb_dim, 32)
        self.head_T = nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=0.5)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.head_T.weight)
        nn.init.zeros_(self.head_T.bias)

    def forward(self, hidden_states):
        """Returns per-step calibrated temperature [B, H]."""
        B, H, _ = hidden_states.shape
        steps = torch.arange(H, device=hidden_states.device)
        step_emb = self.step_embed(steps).unsqueeze(0).expand(B, H, -1)
        x = torch.cat([hidden_states, step_emb], dim=-1)
        x = F.silu(self.fc(x))
        raw_T = self.head_T(x).squeeze(-1)  # [B, H]
        T = F.softplus(raw_T + 1.0) + 0.1  # > 0.1, starts near 1.0
        return T


@torch.no_grad()
def _eval_calibrated_dist_ci(model, tokenizer, calib_head, loader, device,
                               confidence_level=0.80, top_k=32):
    """Evaluate CI via calibrated token distribution."""
    model.eval()
    calib_head.eval()
    all_lower, all_upper, all_actual = [], [], []
    alpha = 1.0 - float(confidence_level)
    low_q, high_q = alpha / 2.0, 1.0 - alpha / 2.0
    K = min(int(top_k), TOKENIZER_VOCAB)

    for batch in loader:
        feats = batch["features"].to(device=device, dtype=torch.float32)
        means = batch["means"].to(device=device, dtype=torch.float32)
        stds = batch["stds"].to(device=device, dtype=torch.float32)
        actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
        times_f = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}
        B = feats.size(0)
        if B == 0: continue

        idx_c, idx_f = tokenizer.encode(feats)
        ctx_c = idx_c[:, :PREFIX_LEN + HORIZON - 1]
        ctx_f = idx_f[:, :PREFIX_LEN + HORIZON - 1]
        ctx_time = {k: times_f[k][:, :ctx_c.size(1)] for k in ("minute", "day", "month", "year")}
        logits_c, logits_f, _, hidden = model(ctx_c, ctx_f,
                                               ctx_time["minute"], ctx_time["day"],
                                               ctx_time["month"], ctx_time["year"],
                                               return_hidden=True)
        h = hidden[:, PREFIX_LEN - 1 : PREFIX_LEN - 1 + HORIZON, :]
        T_t = calib_head(h)  # [B, H]

        step_lower, step_upper = [], []
        for step in range(HORIZON):
            lc = logits_c[:, PREFIX_LEN - 1 + step, :].float()
            lf = logits_f[:, PREFIX_LEN - 1 + step, :].float()
            T_step = T_t[:, step].view(B, 1)
            # Calibrate
            lc_cal = lc / T_step.clamp_min(0.1)
            lf_cal = lf / T_step.clamp_min(0.1)

            probs_c = F.softmax(lc_cal, dim=-1)
            probs_f = F.softmax(lf_cal, dim=-1)
            top_pc, top_ic = torch.topk(probs_c, k=K, dim=-1)
            top_pf, top_if = torch.topk(probs_f, k=K, dim=-1)
            top_pc = top_pc / top_pc.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            top_pf = top_pf / top_pf.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pair_probs = top_pc.unsqueeze(-1) * top_pf.unsqueeze(-2)

            pc_flat = top_ic.unsqueeze(-1).expand(B, K, K).reshape(B, K * K)
            pf_flat = top_if.unsqueeze(-2).expand(B, K, K).reshape(B, K * K)
            decoded = tokenizer.decode(pc_flat, pf_flat)[..., 0].float()
            returns = decoded.view(B, K, K)
            ret_denorm = returns * stds[:, 0].view(B, 1, 1) + means[:, 0].view(B, 1, 1)

            ret_flat = ret_denorm.view(B, -1)
            prob_flat = pair_probs.view(B, -1)
            sort_idx = ret_flat.argsort(dim=-1)
            sorted_ret = ret_flat.gather(-1, sort_idx)
            sorted_prob = prob_flat.gather(-1, sort_idx)
            cum_prob = sorted_prob.cumsum(dim=-1)
            cum_prob = cum_prob / cum_prob[..., -1:].clamp_min(1e-8)
            Np = K * K
            idx_low = (cum_prob >= low_q).float().argmax(dim=-1).clamp(0, Np - 1)
            idx_high = (cum_prob >= high_q).float().argmax(dim=-1).clamp(0, Np - 1)
            rows = torch.arange(B, device=ret_flat.device)
            step_lower.append(sorted_ret[rows, idx_low].detach().cpu())
            step_upper.append(sorted_ret[rows, idx_high].detach().cpu())

        all_lower.append(torch.stack(step_lower, dim=1))
        all_upper.append(torch.stack(step_upper, dim=1))
        all_actual.append(actual[:, :HORIZON].cpu())

    if not all_lower: return {"avg_interval_score": 999.0}
    pl = torch.cat(all_lower, dim=0).numpy()
    pu = torch.cat(all_upper, dim=0).numpy()
    aa = torch.cat(all_actual, dim=0).numpy()
    return compute_ci_metrics(pl, pu, aa, confidence_level=float(confidence_level))


def stage_logit_calibration():
    """Method E: Logit Calibration Adapter."""
    os.makedirs(SUP_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    set_global_seed(42)

    tokenizer = _load_tokenizer(device)
    model = _load_model(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    train_loader, val_loader, n_train, n_val = _build_data(device, max_train=2048, max_val=500)

    calib_head = LogitCalibrationHead(hidden_dim=BACKBONE["dim"], num_steps=HORIZON).to(device)

    # Hyperparam grid for E
    lr_values = [3e-4, 1e-3, 3e-3]
    lambda_cov_values = [0.5, 1.0, 2.0]
    lambda_width_values = [0.1, 0.3, 1.0]
    max_updates_values = [200, 400]
    T_base_values = [1.0, 1.5]

    results = []
    run_idx = 0
    total = len(lr_values) * len(lambda_cov_values) * len(lambda_width_values) * len(max_updates_values) * len(T_base_values)

    for lr in lr_values:
        for lam_cov in lambda_cov_values:
            for lam_width in lambda_width_values:
                for max_up in max_updates_values:
                    for T_base in T_base_values:
                        run_idx += 1
                        tag = f"E_lr{lr}_cov{lam_cov}_w{lam_width}_up{max_up}_T{T_base}"
                        tdir = os.path.join(SUP_DIR, tag)
                        rpath = os.path.join(tdir, "result.json")
                        if os.path.exists(rpath):
                            with open(rpath) as f:
                                results.append(json.load(f))
                            continue

                        os.makedirs(tdir, exist_ok=True)
                        # Reset head
                        head_init = LogitCalibrationHead(hidden_dim=BACKBONE["dim"], num_steps=HORIZON, T_base=T_base)
                        calib_head.load_state_dict(head_init.state_dict())
                        calib_head.train()

                        opt = torch.optim.AdamW(calib_head.parameters(), lr=lr)
                        total_updates = 0

                        while total_updates < max_up:
                            for batch in train_loader:
                                if total_updates >= max_up: break
                                feats = batch["features"].to(device=device, dtype=torch.float32)
                                times_f = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}
                                means = batch["means"].to(device=device, dtype=torch.float32)
                                stds = batch["stds"].to(device=device, dtype=torch.float32)
                                actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
                                B = feats.size(0)
                                if B == 0: continue

                                idx_c, idx_f = tokenizer.encode(feats)
                                ctx_c = idx_c[:, :PREFIX_LEN + HORIZON - 1]
                                ctx_f = idx_f[:, :PREFIX_LEN + HORIZON - 1]
                                ctx_time = {k: times_f[k][:, :ctx_c.size(1)] for k in ("minute", "day", "month", "year")}

                                with torch.no_grad():
                                    logits_c, logits_f, _, hidden = model(ctx_c, ctx_f,
                                        ctx_time["minute"], ctx_time["day"],
                                        ctx_time["month"], ctx_time["year"], return_hidden=True)

                                h = hidden[:, PREFIX_LEN - 1 : PREFIX_LEN - 1 + HORIZON, :]
                                T_t = calib_head(h)  # [B, H]

                                # Compute calibrated distribution CI at each step
                                K = 16
                                y = actual[:, :HORIZON] * stds[:, 0:1] + means[:, 0:1]
                                total_loss = T_t.new_zeros(())

                                for step in range(HORIZON):
                                    lc = logits_c[:, PREFIX_LEN - 1 + step, :].float()
                                    lf = logits_f[:, PREFIX_LEN - 1 + step, :].float()
                                    T_step = T_t[:, step].view(B, 1)
                                    lc_cal = lc / T_step.clamp_min(0.1)
                                    lf_cal = lf / T_step.clamp_min(0.1)

                                    probs_c = F.softmax(lc_cal, dim=-1)
                                    probs_f = F.softmax(lf_cal, dim=-1)
                                    top_pc, top_ic = torch.topk(probs_c, k=K, dim=-1)
                                    top_pf, top_if = torch.topk(probs_f, k=K, dim=-1)
                                    top_pc = top_pc / top_pc.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                                    top_pf = top_pf / top_pf.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                                    pair_probs = top_pc.unsqueeze(-1) * top_pf.unsqueeze(-2)

                                    pc_flat = top_ic.unsqueeze(-1).expand(B, K, K).reshape(B, K * K)
                                    pf_flat = top_if.unsqueeze(-2).expand(B, K, K).reshape(B, K * K)
                                    decoded = tokenizer.decode(pc_flat, pf_flat)[..., 0].float()
                                    returns = decoded.view(B, K, K)

                                    ret_flat = returns.view(B, -1)
                                    prob_flat = pair_probs.view(B, -1)
                                    sort_idx = ret_flat.argsort(dim=-1)
                                    sorted_ret = ret_flat.gather(-1, sort_idx)
                                    sorted_prob = prob_flat.gather(-1, sort_idx)
                                    cum_prob = sorted_prob.cumsum(dim=-1)
                                    cum_prob = cum_prob / cum_prob[..., -1:].clamp_min(1e-8)

                                    alpha = 0.2  # C=0.80
                                    Np = K * K
                                    idx_low = (cum_prob >= alpha / 2).float().argmax(dim=-1).clamp(0, Np - 1)
                                    idx_high = (cum_prob >= 1 - alpha / 2).float().argmax(dim=-1).clamp(0, Np - 1)
                                    rows = torch.arange(B, device=ret_flat.device)
                                    L = sorted_ret[rows, idx_low]
                                    U = sorted_ret[rows, idx_high]

                                    y_step = actual[:, step] * stds[:, 0] + means[:, 0]
                                    width = (U - L).mean()
                                    soft_cov = (torch.sigmoid((y_step - L) / 0.01) * torch.sigmoid((U - y_step) / 0.01)).mean()
                                    loss_step = lam_width * width + lam_cov * F.relu(0.80 - soft_cov)
                                    total_loss = total_loss + loss_step

                                total_loss = total_loss / HORIZON
                                opt.zero_grad()
                                total_loss.backward()
                                torch.nn.utils.clip_grad_norm_(calib_head.parameters(), 1.0)
                                opt.step()
                                total_updates += 1

                        # Eval
                        m = _eval_calibrated_dist_ci(model, tokenizer, calib_head, val_loader, device, confidence_level=0.80)
                        row = {"method": "E_logit_cal", "lr": lr, "lambda_cov": lam_cov,
                               "lambda_width": lam_width, "max_updates": max_up, "T_base": T_base,
                               "avg_interval_score": m.get("avg_interval_score", 999),
                               "coverage": m.get("coverage", 0), "avg_width": m.get("avg_width", 0),
                               "path_avg_interval_score": m.get("path_avg_interval_score", 0)}
                        results.append(row)
                        with open(rpath, "w") as f:
                            json.dump(row, f, indent=2)
                        print(f"  [{run_idx}/{total}] {tag}: IS={row['avg_interval_score']:.6f} cov={row['coverage']:.4f} w={row['avg_width']:.6f}")

    # Summary
    results.sort(key=lambda r: r.get("avg_interval_score", 999))
    print(f"\n=== Method E Best Results ===")
    for r in results[:5]:
        print(f"  IS={r['avg_interval_score']:.6f} cov={r['coverage']:.4f} w={r['avg_width']:.6f} "
              f"lr={r['lr']} cov_w={r['lambda_cov']} w_w={r['lambda_width']} T_base={r['T_base']}")

    summary_path = os.path.join(SUP_DIR, "summary_logit_calibration.csv")
    if results:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
    print(f"Method E complete. Summary: {summary_path}")


# ═══════════════════════════════════════════════════════════════
# Stage 5: Final eval
# ═══════════════════════════════════════════════════════════════

def stage_final_eval():
    """Evaluate top-5 checkpoints across all confidence levels."""
    os.makedirs(SUP_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tokenizer = _load_tokenizer(device)
    model = _load_model(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    _, val_loader, _, n_val = _build_data(device, max_train=0, max_val=500)

    # Find top checkpoints
    summary_csv = os.path.join(SUP_DIR, "summary_quantile_hpo.csv")
    if not os.path.exists(summary_csv):
        print("No HPO summary found. Run --stage quantile_hpo first.")
        return

    with open(summary_csv, newline="") as f:
        all_rows = list(csv.DictReader(f))
    for r in all_rows:
        for k in list(r.keys()):
            try: r[k] = float(r[k])
            except (ValueError, TypeError): pass
    top5 = sorted(all_rows, key=lambda r: r.get("value", 999))[:5]

    results = []
    for rank, row in enumerate(top5):
        trial_num = int(row.get("trial", -1))
        tdir = os.path.join(SUP_DIR, f"quantile_trial_{trial_num:03d}")
        ckpt_path = os.path.join(tdir, "quantile_head.pt")
        if not os.path.exists(ckpt_path):
            continue

        quantile_head = CIQuantileHead(
            hidden_dim=BACKBONE["dim"], num_steps=HORIZON,
            step_embedding_dim=16, head_hidden_dim=128, share_aC=True,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        quantile_head.load_state_dict(ckpt["quantile_head"])
        quantile_head.eval()

        for c in CONFIDENCE_LEVELS:
            m = evaluate_quantile_head(model, tokenizer, quantile_head, val_loader, device, confidence_level=c)
            results.append({
                "rank": rank + 1, "trial": trial_num, "confidence_level": c,
                "avg_interval_score": m["avg_interval_score"],
                "coverage": m["coverage"], "avg_width": m["avg_width"],
                "path_avg_interval_score": m["path_avg_interval_score"],
                "path_coverage": m["path_coverage"],
                "mape_midpoint": m.get("mape_midpoint", 0),
                "da_midpoint": m.get("da_midpoint", 0),
            })
            print(f"  Rank {rank+1} Trial {trial_num} C={c:.0%}: "
                  f"IS={m['avg_interval_score']:.6f} cov={m['coverage']:.4f}")

    # Save
    final_path = os.path.join(SUP_DIR, "final_eval_results.json")
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)

    # Leaderboard
    print(f"\n=== QUALITY LEADERBOARD (C=0.80, coverage within ±0.03 of target) ===")
    eligible = [r for r in results if r["confidence_level"] == 0.80 and abs(r["coverage"] - 0.80) <= 0.03]
    eligible.sort(key=lambda r: r["avg_interval_score"])
    for i, r in enumerate(eligible[:5]):
        print(f"  {i+1}. Trial {r['trial']} IS={r['avg_interval_score']:.6f} cov={r['coverage']:.4f}")

    print(f"\nFinal eval saved to {final_path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 7 Sup V2")
    parser.add_argument("--stage", required=True,
                        choices=["baselines", "quantile_smoke", "quantile_hpo", "quantile_hpo_bc",
                                 "logit_calibration", "final_eval"])
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args()

    if args.stage == "quantile_hpo_bc":
        stage_quantile_hpo(args.trials, search_space=SEARCH_SPACE_BC)
    elif args.stage == "logit_calibration":
        stage_logit_calibration()
    else:
        stage_map = {
            "baselines": stage_baselines,
            "quantile_smoke": stage_quantile_smoke,
            "quantile_hpo": lambda: stage_quantile_hpo(args.trials),
            "final_eval": stage_final_eval,
        }
        stage_map[args.stage]()


if __name__ == "__main__":
    main()
