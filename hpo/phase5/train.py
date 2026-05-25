# -*- coding: utf-8 -*-
"""Phase 5 DA — Unified training loop with checkpointing and evaluation."""

from __future__ import annotations

import json, os, sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hpo.phase5.core import (
    _ac, DEFAULT_CFG, SEED, OUT_DIR, P3, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
    load_tokenizer, build_trainable_model, build_ref_model,
    get_dataloaders, move_batch, prepare_inputs, evaluate, compute_metrics,
)
from hpo.phase5.methods import METHOD_REGISTRY
from model.lora import lora_state_dict
from reproducibility import set_global_seed


def run_method(
    method: str,
    device: torch.device,
    cfg_override: dict | None = None,
    use_lora: bool = True,
    lora_rank: int = LORA_RANK,
    lora_alpha: float = LORA_ALPHA,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    output_dir: str | None = None,
    resume: bool = True,
):
    """Run a post-training method and return results.

    Parameters
    ----------
    method : str
        One of: ce, expo, dpo, rsft, grpo.
    device : torch.device
    cfg_override : dict or None
        Override specific config keys.
    use_lora : bool
        Whether to use LoRA (default True).
    lora_rank, lora_alpha : int, float
        LoRA hyperparameters.
    epochs, batch_size, lr : int, int, float or None
        Override training hyperparameters.
    output_dir : str or None
        Output directory for checkpoints and results.
    resume : bool
        Whether to resume from existing checkpoint.

    Returns
    -------
    dict with keys: method, best_score, best_metrics, history, final_val.
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method}. Choose from {list(METHOD_REGISTRY)}")

    info = METHOD_REGISTRY[method]
    loss_fn = info["loss_fn"]
    needs_ref = info["needs_ref"]
    method_name = info["name"]

    # ── Config ──
    cfg = dict(DEFAULT_CFG)
    if epochs is not None:
        cfg["epochs"] = epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    if lr is not None:
        cfg["lr"] = lr
    if cfg_override:
        cfg.update(cfg_override)

    out_dir = os.path.join(output_dir, method) if output_dir else os.path.join(OUT_DIR, method)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase 5 DA: {method_name} ({method})")
    print(f"{'='*60}")
    print(f"  LoRA={use_lora}  rank={lora_rank}  alpha={lora_alpha}")
    print(f"  Epochs={cfg['epochs']}  batch_size={cfg['batch_size']}  lr={cfg['lr']}")

    # ── Data ──
    train_loader, val_loader, eps = get_dataloaders()

    # ── Models ──
    tokenizer = load_tokenizer(device)
    model = build_trainable_model(device, use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/max(1,total):.1f}%)")

    ref_model = build_ref_model(device) if needs_ref else None

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        fused=True if device.type == "cuda" else False)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    amp_enabled = device.type == "cuda" and amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    # ── Scheduler ──
    total_steps = cfg["epochs"] * len(train_loader)
    warmup_steps = max(1, total_steps // 10)
    lin_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=cfg["lr"] * 0.01)

    # ── Checkpoint paths ──
    best_path = os.path.join(out_dir, f"phase5_{method}_best.pt")
    history_path = os.path.join(out_dir, f"phase5_{method}_history.json")
    result_path = os.path.join(out_dir, "result.json")

    # ── Resume ──
    history = []
    best_score = -float("inf")
    best_metrics = None
    start_epoch = 0
    updates = 0
    best_epoch = 0

    if resume and os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("completed", False):
            print(f"  Already completed. Best: {prev.get('best_score', 0):.4f}")
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            return {
                "method": method, "method_name": method_name,
                "best_score": prev["best_score"],
                "best_metrics": prev.get("best_metrics", {}),
                "final_val": history[-1]["val"] if history else {},
                "history": history, "best_epoch": prev.get("best_epoch", 0),
            }

    # ── Training ──
    print(f"  Training {cfg['epochs']} epochs...")
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"  {method} ep {epoch+1}/{cfg['epochs']}", leave=False)
        for raw_batch in pbar:
            batch = move_batch(raw_batch, device)
            batch["tokenizer"] = tokenizer

            loss = loss_fn(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype, cfg)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            updates += 1
            if updates <= warmup_steps:
                lin_scheduler.step()
            else:
                cosine_scheduler.step()

            batches += 1
            total_loss += loss.detach().cpu().item()

            if batches % cfg["progress_interval"] == 0:
                pbar.set_postfix({"loss": f"{total_loss/max(1,batches):.4f}", "upd": updates})

        # ── Validation ──
        val_metrics = evaluate(model, tokenizer, val_loader, device, amp_enabled, amp_dtype)
        record = {
            "epoch": epoch + 1, "updates": updates,
            "train_loss": total_loss / max(1, batches),
            "val": val_metrics,
        }
        score = float(val_metrics.get("balanced_accuracy", val_metrics.get("direction_accuracy", 0.0)))
        record["selection_score"] = score
        history.append(record)

        print(f"  ep{epoch+1:2d}  DA={val_metrics.get('direction_accuracy',0):.4f}  "
              f"BalAcc={val_metrics.get('balanced_accuracy',0):.4f}  "
              f"MAPE={val_metrics.get('mape',0):.4f}  "
              f"Preds={val_metrics.get('pred_counts',{})}")

        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            best_epoch = epoch + 1
            save_dict = {
                "method": method, "method_name": method_name,
                "epoch": epoch + 1, "updates": updates,
                "model_state_dict": model.state_dict(),
                "lora_state_dict": lora_state_dict(model) if use_lora else {},
                "tokenizer_state_dict": tokenizer.state_dict(),
                "model_config": P3,
                "lora_config": {"rank": lora_rank, "alpha": lora_alpha,
                                "dropout": LORA_DROPOUT, "targets": LORA_TARGETS},
                "cfg": cfg, "metrics": val_metrics, "history": history,
            }
            torch.save(save_dict, best_path)
            print(f"  -> Saved best: {best_path}")

    # ── Save results ──
    final_val = history[-1]["val"] if history else {}
    result = {
        "method": method, "method_name": method_name,
        "best_score": best_score, "best_epoch": best_epoch,
        "best_metrics": best_metrics, "final_val": final_val,
        "completed": True,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"  {method} done. Best score: {best_score:.4f} at epoch {best_epoch}")
    result["history"] = history
    return result
