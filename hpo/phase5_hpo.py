# -*- coding: utf-8 -*-
"""Phase 5 HPO: Hyperparameter optimisation for token-space DA post-training.

Compares four methods (CE / ExPO / DPO / RSFT) with Optuna-driven
hyperparameter search.  Each method gets its own study; the best config
per method is then evaluated with a longer run.

Shared infrastructure (model, dataset, evaluation) is imported from
``hpo.phase5_da``.

Usage::

    python -m hpo.phase5_hpo          # full HPO (all methods)
    python -m hpo.phase5_hpo --method dpo   # single-method HPO
    python -m hpo.phase5_hpo --eval-only    # re-evaluate best configs only

Resume:  studies are stored in SQLite (trials/phase5_da/hpo/).  Re-running
the same command will load existing trials and continue from where it left
off — no work is duplicated.
"""

from __future__ import annotations

import copy, json, os, sys, warnings
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

# Re-use the solid foundation from the fixed Phase 5 script.
from hpo.phase5_da import (
    CFG,
    LABEL_DOWN, LABEL_FLAT, LABEL_UP, IGNORE_INDEX,
    P3, P3_CKPT, VOCAB, SEED, OUT_DIR,
    _load_tokenizer, _build_model, _load_base_weights,
    _get_dataloaders, _move_batch, _prepare_inputs,
    _ac, _sample_tokens, _token_returns, _token_direction,
    _candidate_logp, _build_winner_loser, evaluate,
    build_ref_model,
    DirectionDataset, collate_fn,
)
from model.lora import inject_lora, lora_state_dict
from reproducibility import set_global_seed

warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════
# HPO config
# ═══════════════════════════════════════════════

HPO_EPOCHS = 3          # epochs per trial (fast iteration)
HPO_TRIALS = 15         # trials per method
FINAL_EPOCHS = 15       # epochs for the final best-config run
HPO_BATCH_SIZE = 8      # batch size for HPO trials (same as CFG)

HPO_DIR = os.path.join(OUT_DIR, "hpo")
os.makedirs(HPO_DIR, exist_ok=True)


# ═══════════════════════════════════════════════
# Shared search space
# ═══════════════════════════════════════════════

def _suggest_shared(trial: optuna.Trial) -> dict:
    """Hyperparameters shared by all methods."""
    return {
        "lr": trial.suggest_float("lr", 1e-6, 2e-4, log=True),
        "use_lora": trial.suggest_categorical("use_lora", [True, False]),
        "lora_rank": trial.suggest_categorical("lora_rank", [4, 8, 16]),
        "lora_alpha": trial.suggest_categorical("lora_alpha", [8, 16, 32]),
    }


def _suggest_candidate_cfg(trial: optuna.Trial, prefix: str = "") -> dict:
    """Candidate-sampling hyperparameters (shared by ExPO/DPO/RSFT)."""
    p = prefix + "_" if prefix else ""
    return {
        f"{p}num_candidates": trial.suggest_categorical(f"{p}num_candidates", [64, 128, 192, 256]),
        f"{p}temperature": trial.suggest_categorical(f"{p}temperature", [0.5, 0.8, 1.0, 1.5, 2.0]),
        f"{p}direction_bonus": trial.suggest_categorical(f"{p}direction_bonus", [0.5, 1.0, 2.0]),
        f"{p}error_weight": trial.suggest_categorical(f"{p}error_weight", [0.1, 0.25, 0.5]),
        f"{p}score_margin": trial.suggest_categorical(f"{p}score_margin", [0.0, 0.05, 0.1]),
        f"{p}include_gold": trial.suggest_categorical(f"{p}include_gold", [True, False]),
    }


# ═══════════════════════════════════════════════
# Per-method search spaces
# ═══════════════════════════════════════════════

SEARCH_SPACES: Dict[str, dict] = {
    "ce": {
        "description": "CE baseline — LoRA vs full-FT, LR, rank.",
        "params": ["lr", "use_lora", "lora_rank", "lora_alpha"],
    },
    "expo": {
        "description": "ExPO regression on token log-probabilities.",
        "params": [
            "lr", "use_lora", "lora_rank", "lora_alpha",
            "num_candidates", "temperature",
            "direction_bonus", "error_weight", "score_margin", "include_gold",
            "expo_reference_weight",
        ],
    },
    "dpo": {
        "description": "DPO sigmoid loss on token log-probabilities.",
        "params": [
            "lr", "use_lora", "lora_rank", "lora_alpha",
            "num_candidates", "temperature",
            "direction_bonus", "error_weight", "score_margin", "include_gold",
            "dpo_beta",
        ],
    },
    "rsft": {
        "description": "Rejection-sampled best-token fine-tuning.",
        "params": [
            "lr", "use_lora", "lora_rank", "lora_alpha",
            "num_candidates", "temperature",
            "direction_bonus", "error_weight", "score_margin", "include_gold",
        ],
    },
}

METHOD_SPECIFIC_SUGGEST = {
    "expo": lambda t: {"expo_reference_weight": t.suggest_categorical("expo_reference_weight", [0.3, 0.5, 0.6, 0.7, 0.9])},
    "dpo":  lambda t: {"dpo_beta": t.suggest_categorical("dpo_beta", [0.05, 0.1, 0.3, 0.5, 0.7, 1.0])},
    "rsft": lambda t: {},
    "ce":   lambda t: {},
}


def suggest_params(method: str, trial: optuna.Trial) -> dict:
    """Build a complete hyperparameter dict for *method* from *trial*."""
    params = {}
    params.update(_suggest_shared(trial))

    if method != "ce":
        params.update(_suggest_candidate_cfg(trial))

    specific = METHOD_SPECIFIC_SUGGEST.get(method, lambda t: {})
    params.update(specific(trial))

    # Normalise key names to match CFG convention
    params.setdefault("num_candidates", 192)
    params.setdefault("temperature", 1.0)
    params.setdefault("direction_bonus", 1.0)
    params.setdefault("error_weight", 0.25)
    params.setdefault("score_margin", 0.05)
    params.setdefault("include_gold", True)
    params.setdefault("expo_reference_weight", 0.6)
    params.setdefault("dpo_beta", 0.5)

    return params


def params_to_cfg(params: dict, method: str) -> dict:
    """Merge *params* into a runtime config dict (does not mutate global CFG)."""
    cfg = dict(CFG)
    cfg["lr"] = params["lr"]
    cfg["epochs"] = HPO_EPOCHS
    cfg["batch_size"] = HPO_BATCH_SIZE

    if method == "ce":
        return cfg

    cfg["num_candidates"]   = params["num_candidates"]
    cfg["temperature"]      = params["temperature"]
    cfg["direction_bonus"]  = params["direction_bonus"]
    cfg["error_weight"]     = params["error_weight"]
    cfg["score_margin"]     = params["score_margin"]
    cfg["include_gold"]     = params["include_gold"]

    if method == "expo":
        cfg["expo_reference_weight"] = params["expo_reference_weight"]
    elif method == "dpo":
        cfg["dpo_beta"] = params["dpo_beta"]

    return cfg


# ═══════════════════════════════════════════════
# Model factory (HPO version — per-trial LoRA config)
# ═══════════════════════════════════════════════

def build_trainable_model_hpo(device, lora_rank: int, lora_alpha: float,
                               use_lora: bool = True):
    """Build model with P3 weights, optionally injecting LoRA.

    When *use_lora* is True: inject LoRA into Q/K/V projections, freeze base.
    When *use_lora* is False: full fine-tuning — all parameters unfrozen.
    """
    model = _build_model(device)
    _load_base_weights(model, P3_CKPT)
    model.eval()

    if use_lora:
        from model.lora import inject_lora
        lora_targets = ("q_proj", "k_proj", "v_proj")
        inject_lora(model, rank=lora_rank, alpha=lora_alpha,
                    dropout=0.05, target_keywords=lora_targets, freeze_base=True)
    else:
        # Full fine-tuning: unfreeze everything
        for p in model.parameters():
            p.requires_grad = True

    return model


# ═══════════════════════════════════════════════
# Per-method training + objective
# ═══════════════════════════════════════════════

def _train_one_epoch(model, ref_model, optimizer, scaler, scheduler, cosine,
                     train_loader, tokenizer, device, amp_enabled, amp_dtype,
                     loss_fn, cfg, warmup_steps, epoch, updates, technique_name):
    """Run one training epoch.  Returns (total_loss, batches, updates)."""
    model.train()
    total_loss = 0.0; batches = 0
    pbar = tqdm(train_loader, desc=f"  {technique_name} ep{epoch+1}", leave=False)

    for raw_batch in pbar:
        batch = _move_batch(raw_batch, device)
        batch["tokenizer"] = tokenizer
        loss = loss_fn(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        if updates <= warmup_steps:
            scheduler.step()
        else:
            cosine.step()

        batches += 1; total_loss += loss.detach().cpu().item()

    return total_loss, batches, updates


# ── Loss functions (with cfg override) ──

def _loss_ce_hpo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = _prepare_inputs(batch)
    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    target_c = batch["idx_c_full"][:, -1]
    target_f = batch["idx_f_full"][:, -1]
    return (F.cross_entropy(logits_c[:, -1, :].float(), target_c)
            + F.cross_entropy(logits_f[:, -1, :].float(), target_f))


def _loss_expo_hpo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = _prepare_inputs(batch)

    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_last_c, pi_last_f = logits_c[:, -1, :], logits_f[:, -1, :]

    with torch.no_grad():
        with _ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_last_c, ref_last_f = ref_lc[:, -1, :], ref_lf[:, -1, :]

    w_c, w_f, l_c, l_f, valid_pair, _, _ = _build_winner_loser(
        tokenizer, ref_last_c, ref_last_f, batch, cfg)

    theta_win  = _candidate_logp(pi_last_c, pi_last_f, w_c, w_f)
    theta_lose = _candidate_logp(pi_last_c, pi_last_f, l_c, l_f)
    with torch.no_grad():
        ref_win  = _candidate_logp(ref_last_c, ref_last_f, w_c, w_f)
        ref_lose = _candidate_logp(ref_last_c, ref_last_f, l_c, l_f)
        ref_pref = torch.sigmoid(ref_win - ref_lose)

    theta_pref = torch.sigmoid(theta_win - theta_lose)
    lam = max(0.0, min(1.0, cfg.get("expo_reference_weight", 0.6)))
    target_pref = (lam * ref_pref + (1.0 - lam)).clamp(0.0, 1.0)
    per_row = (theta_pref - target_pref).pow(2)
    vw = valid_pair.to(dtype=per_row.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (per_row * vw).sum() / vw.sum().clamp_min(1.0)


def _loss_dpo_hpo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = _prepare_inputs(batch)

    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_last_c, pi_last_f = logits_c[:, -1, :], logits_f[:, -1, :]

    with torch.no_grad():
        with _ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_last_c, ref_last_f = ref_lc[:, -1, :], ref_lf[:, -1, :]

    w_c, w_f, l_c, l_f, valid_pair, _, _ = _build_winner_loser(
        tokenizer, ref_last_c, ref_last_f, batch, cfg)

    pi_win  = _candidate_logp(pi_last_c, pi_last_f, w_c, w_f)
    pi_lose = _candidate_logp(pi_last_c, pi_last_f, l_c, l_f)
    with torch.no_grad():
        ref_win  = _candidate_logp(ref_last_c, ref_last_f, w_c, w_f)
        ref_lose = _candidate_logp(ref_last_c, ref_last_f, l_c, l_f)

    log_ratio = (pi_win - pi_lose) - (ref_win - ref_lose)
    per_row = -F.logsigmoid(cfg.get("dpo_beta", 0.5) * log_ratio)
    vw = valid_pair.to(dtype=per_row.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (per_row * vw).sum() / vw.sum().clamp_min(1.0)


def _loss_rsft_hpo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = _prepare_inputs(batch)

    with torch.no_grad():
        with _ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_last_c, ref_last_f = ref_lc[:, -1, :], ref_lf[:, -1, :]

    B = ref_last_c.size(0)
    sampled_c = _sample_tokens(ref_last_c, cfg["temperature"], cfg["num_candidates"])
    sampled_f = _sample_tokens(ref_last_f, cfg["temperature"], cfg["num_candidates"])

    if cfg["include_gold"]:
        gold_c = batch["idx_c_full"][:, -1]; gold_f = batch["idx_f_full"][:, -1]
        sampled_c = torch.cat([sampled_c, gold_c.long().unsqueeze(1)], dim=1)
        sampled_f = torch.cat([sampled_f, gold_f.long().unsqueeze(1)], dim=1)

    cand_dir = _token_direction(tokenizer, sampled_c, sampled_f,
                                batch["prompt_means"], batch["prompt_stds"])
    cand_ret = _token_returns(tokenizer, sampled_c, sampled_f,
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
    best_c = sampled_c[rows, best_idx]; best_f = sampled_f[rows, best_idx]
    best_valid = valid_label & scores.max(dim=1).values.isfinite()

    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    if best_valid.any():
        return (F.cross_entropy(logits_c[best_valid, -1, :].float(), best_c[best_valid].long())
                + F.cross_entropy(logits_f[best_valid, -1, :].float(), best_f[best_valid].long()))
    return torch.tensor(0.0, device=device, requires_grad=True)


LOSS_FNS = {"ce": _loss_ce_hpo, "expo": _loss_expo_hpo,
            "dpo": _loss_dpo_hpo, "rsft": _loss_rsft_hpo}


# ═══════════════════════════════════════════════
# Optuna objective
# ═══════════════════════════════════════════════

def make_objective(method: str):
    """Return an Optuna objective function for *method*.

    The objective handles CUDA OOM gracefully: full-FT trials that run out of
    memory are marked as pruned rather than crashing the study.
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(method, trial)
        cfg = params_to_cfg(params, method)

        # ── Build model ──
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            model = build_trainable_model_hpo(device, params["lora_rank"], params["lora_alpha"],
                                               use_lora=params.get("use_lora", True))
        except torch.cuda.OutOfMemoryError:
            raise optuna.TrialPruned(f"OOM during model build for {method}")

        # Enable gradient checkpointing for full-FT to save memory
        if not params.get("use_lora", True):
            model.enable_gradient_checkpointing(True)

        tokenizer = _load_tokenizer(device)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        needs_ref = method != "ce"
        ref_model = build_ref_model(device) if needs_ref else None

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            fused=True if device.type == "cuda" else False)

        amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
        amp_enabled = device.type == "cuda" and amp_dtype is not None
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

        train_loader, val_loader, _eps = _get_dataloaders()
        total_steps = HPO_EPOCHS * len(train_loader)
        warmup_steps = max(1, total_steps // 10)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=cfg["lr"] * 0.01)

        loss_fn = LOSS_FNS[method]
        updates = 0
        best_score = -float("inf")

        for epoch in range(HPO_EPOCHS):
            try:
                total_loss, batches, updates = _train_one_epoch(
                    model, ref_model, optimizer, scaler, scheduler, cosine,
                    train_loader, tokenizer, device, amp_enabled, amp_dtype,
                    loss_fn, cfg, warmup_steps, epoch, updates, method)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"OOM during training epoch {epoch+1} for {method}")

            val_metrics = evaluate(model, tokenizer, val_loader, device, amp_enabled, amp_dtype)
            score = float(val_metrics.get("balanced_accuracy", 0.0))
            if score > best_score:
                best_score = score

            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            trial.set_user_attr(f"epoch_{epoch+1}_da", float(val_metrics.get("direction_accuracy", 0)))
            trial.set_user_attr(f"epoch_{epoch+1}_mape", float(val_metrics.get("mape", 0)))
            trial.set_user_attr(f"epoch_{epoch+1}_loss", float(total_loss / max(1, batches)))

        trial.set_user_attr("trainable_params", trainable)
        trial.set_user_attr("total_params", total)
        trial.set_user_attr("use_lora", params.get("use_lora", True))
        trial.set_user_attr("best_epoch_da", float(
            max(trial.user_attrs.get(f"epoch_{e}_da", 0) for e in range(1, HPO_EPOCHS+1))))

        return best_score

    return objective


# ═══════════════════════════════════════════════
# Final long-run evaluation of best config
# ═══════════════════════════════════════════════

def run_best_config(method: str, best_params: dict, best_score: float):
    """Train *method* with *best_params* for FINAL_EPOCHS and save."""
    out_dir = os.path.join(OUT_DIR, f"{method}_best")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Phase 5 HPO — {method} BEST config (final {FINAL_EPOCHS}-epoch run)")
    print(f"  HPO score: {best_score:.4f}")
    print(f"  Params: {json.dumps(best_params, indent=2)}")
    print(f"{'='*60}")

    cfg = params_to_cfg(best_params, method)
    cfg["epochs"] = FINAL_EPOCHS
    cfg["batch_size"] = HPO_BATCH_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_trainable_model_hpo(device, best_params["lora_rank"], best_params["lora_alpha"],
                                       use_lora=best_params.get("use_lora", True))
    tokenizer = _load_tokenizer(device)

    needs_ref = method != "ce"
    ref_model = build_ref_model(device) if needs_ref else None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        fused=True if device.type == "cuda" else False)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    amp_enabled = device.type == "cuda" and amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    train_loader, val_loader, _eps = _get_dataloaders()
    total_steps = FINAL_EPOCHS * len(train_loader)
    warmup_steps = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=cfg["lr"] * 0.01)

    loss_fn = LOSS_FNS[method]
    updates = 0
    history = []
    best_score_final = -float("inf")
    best_path = os.path.join(out_dir, f"phase5_hpo_{method}_best.pt")

    for epoch in range(FINAL_EPOCHS):
        total_loss, batches, updates = _train_one_epoch(
            model, ref_model, optimizer, scaler, scheduler, cosine,
            train_loader, tokenizer, device, amp_enabled, amp_dtype,
            loss_fn, cfg, warmup_steps, epoch, updates, method)

        val_metrics = evaluate(model, tokenizer, val_loader, device, amp_enabled, amp_dtype)
        record = {"epoch": epoch+1, "updates": updates,
                  "train_loss": total_loss/max(1,batches), "val": val_metrics}
        score = float(val_metrics.get("balanced_accuracy", 0.0))
        record["selection_score"] = score
        history.append(record)

        print(f"  ep{epoch+1:2d}  DA={val_metrics.get('direction_accuracy',0):.4f}  "
              f"BalAcc={val_metrics.get('balanced_accuracy',0):.4f}  "
              f"MAPE={val_metrics.get('mape',0):.4f}  "
              f"Pred={val_metrics.get('pred_counts',{})}")

        if score > best_score_final:
            best_score_final = score
            torch.save({
                "method": method, "epoch": epoch+1,
                "hpo_params": best_params, "hpo_score": best_score,
                "model_state_dict": model.state_dict(),
                "lora_state_dict": lora_state_dict(model),
                "tokenizer_state_dict": tokenizer.state_dict(),
                "model_config": P3,
                "lora_config": {"rank": best_params["lora_rank"],
                                "alpha": best_params["lora_alpha"]},
                "metrics": val_metrics, "history": history,
            }, best_path)

    result = {"method": method, "hpo_best_score": best_score,
              "best_params": best_params,
              "final_best_score": best_score_final,
              "final_val": history[-1]["val"] if history else {}}
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"  {method}_best done. HPO={best_score:.4f} → Final={best_score_final:.4f}")
    return result


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Phase 5 HPO: DA post-training hyperparameter search")
    p.add_argument("--method", choices=["ce","expo","dpo","rsft","all"], default="all",
                   help="Which method to optimise (default: all)")
    p.add_argument("--trials", type=int, default=HPO_TRIALS,
                   help=f"Number of Optuna trials per method (default: {HPO_TRIALS})")
    p.add_argument("--epochs", type=int, default=HPO_EPOCHS,
                   help=f"Epochs per HPO trial (default: {HPO_EPOCHS})")
    p.add_argument("--final-epochs", type=int, default=FINAL_EPOCHS,
                   help=f"Epochs for final best-config run (default: {FINAL_EPOCHS})")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip HPO, only re-run best configs from saved studies")
    p.add_argument("--no-final", action="store_true",
                   help="Skip the final long-run evaluation")
    return p.parse_args()


def _run_hpo_for_method(method: str, n_trials: int, n_epochs: int):
    global HPO_EPOCHS
    saved_epochs = HPO_EPOCHS
    HPO_EPOCHS = n_epochs

    study_name = f"phase5_{method}"
    storage_path = os.path.join(HPO_DIR, f"{study_name}.db")
    storage_url = f"sqlite:///{storage_path}"

    print(f"\n{'='*60}")
    print(f"Phase 5 HPO: {method.upper()}")
    print(f"  Trials: {n_trials}  |  Epochs/trial: {n_epochs}")
    print(f"  Storage: {storage_path}")
    print(f"  Search space: {SEARCH_SPACES[method]['description']}")
    print(f"  Params: {SEARCH_SPACES[method]['params']}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    set_global_seed(SEED, deterministic=False)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        load_if_exists=True,
    )

    existing_complete = len([t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE])
    existing_pruned  = len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.PRUNED])
    if existing_complete > 0:
        print(f"  Resuming study: {existing_complete} completed + {existing_pruned} pruned trials")
        print(f"  Current best: {study.best_value:.4f} (trial {study.best_trial.number})")

    objective_fn = make_objective(method)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)

    print(f"\n{method} HPO complete.")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best score:  {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=2)}")

    # Save summary
    summary = {
        "method": method,
        "n_trials": n_trials,
        "best_trial": study.best_trial.number,
        "best_score": study.best_value,
        "best_params": study.best_params,
        "search_space": SEARCH_SPACES[method],
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params,
             "state": str(t.state), "user_attrs": dict(t.user_attrs)}
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ],
    }
    with open(os.path.join(HPO_DIR, f"{study_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    HPO_EPOCHS = saved_epochs
    return study


def main():
    args = _parse_args()
    methods = ["ce", "expo", "dpo", "rsft"] if args.method == "all" else [args.method]

    # ── Phase 1: HPO ──
    studies = {}
    if not args.eval_only:
        for method in methods:
            studies[method] = _run_hpo_for_method(method, args.trials, args.epochs)

    # ── Phase 2: Final best-config evaluation ──
    if args.no_final:
        return

    all_results = {}
    for method in methods:
        # Load best params from saved study
        summary_path = os.path.join(HPO_DIR, f"phase5_{method}_summary.json")
        if not os.path.exists(summary_path):
            print(f"No HPO summary for {method}, skipping final run.")
            continue
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        best_params = summary["best_params"]
        best_score = summary["best_score"]
        all_results[method] = run_best_config(method, best_params, best_score)

    # ── Cross-method summary ──
    print(f"\n{'='*60}")
    print("Phase 5 HPO — Cross-Method Summary")
    print(f"{'='*60}")
    print(f"{'Method':8s}  {'HPO Score':>10s}  {'Final DA':>10s}  {'Final BalAcc':>10s}  {'Final MAPE':>10s}  {'Best Params'}")
    print("-" * 90)
    for method in methods:
        r = all_results.get(method, {})
        fv = r.get("final_val", {})
        bp = r.get("best_params", {})
        bp_short = {k: bp[k] for k in sorted(bp) if k in SEARCH_SPACES.get(method,{}).get("params",[])}
        print(f"  {method:6s}  {r.get('hpo_best_score',0):10.4f}  "
              f"{fv.get('direction_accuracy',0):10.4f}  {fv.get('balanced_accuracy',0):10.4f}  "
              f"{fv.get('mape',0):10.4f}  {json.dumps(bp_short)}")

    cross = {
        "phase": "5_hpo",
        "timestamp": datetime.now().isoformat(),
        "n_trials_per_method": args.trials,
        "hpo_epochs": args.epochs,
        "final_epochs": args.final_epochs,
        "results": all_results,
    }
    with open(os.path.join(OUT_DIR, "cross_method_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cross, f, indent=2, ensure_ascii=False)
    print(f"\nFull summary: {os.path.join(OUT_DIR, 'cross_method_summary.json')}")


if __name__ == "__main__":
    main()
