"""Phase 1: Bits-per-quantizer search.

Searches bits ∈ [6, 12] to find the optimal vocabulary size.
For each bits value:
  1. Train BSQ tokenizer with FIXED reasonable params
  2. Train BaseModel (DSA + GQA) with early stopping
  3. Evaluate: val_ce + token distribution health

Token distribution metrics reveal whether the BaseModel actually
uses the vocabulary — the key signal for picking the right bits.

Usage:
    python -m hpo.phase1_bits_search
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import DataConfig, TokenizerConfig, ModelConfig, TrainingConfig
from data_processor import get_dataloaders, get_datasets
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs, export_tokenizer_config
from model.kronos_reasoning import KronosReasoningGPT
from reproducibility import set_global_seed

# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1_DIR = os.path.join(PROJECT_ROOT, "trials", "phase1_bits_search")
SUMMARY_PATH = os.path.join(PHASE1_DIR, "bits_summary.json")


def bits_dir(bits: int) -> str:
    return os.path.join(PHASE1_DIR, f"bits_{bits:02d}")


# ──────────────────────────────────────────────────────────
# Batch size overrides — edit for your GPU
#   RTX 4060 (8GB):    TOKENIZER_BS=512,  BASEMODEL_BS=16
#   A100 (40/80GB):    TOKENIZER_BS=2048, BASEMODEL_BS=128
#   A10 (24GB):        TOKENIZER_BS=1024, BASEMODEL_BS=64
# ──────────────────────────────────────────────────────────
TOKENIZER_BATCH_SIZE = 512
BASEMODEL_BATCH_SIZE = 16

# ──────────────────────────────────────────────────────────
# Fixed hyperparams — reasonable defaults, not searched
# ──────────────────────────────────────────────────────────

TOKENIZER_PARAMS = {
    "hidden_dim": 128,
    "embedding_dim": 64,
    "num_quantizers": 2,
    "bsq_commitment_cost": 0.05,
    "bsq_entropy_weight": 0.05,
    "learning_rate": 1e-4,
    "epochs": 100,
    "batch_size": TOKENIZER_BATCH_SIZE,
    "grad_clip": 1.0,
}

BASEMODEL_PARAMS = {
    "dim": 256,
    "depth": 4,
    "heads": 4,
    "num_kv_heads": 2,
    "dsa_windows": [None, 512, 512, None],
    "position_encoding": "rope",
    "rope_base": 10000.0,
    "dropout": 0.08,
    "learning_rate": 6e-4,
    "weight_decay": 8e-5,
    "batch_size": BASEMODEL_BATCH_SIZE,
    "accumulation_steps": 2,
    "grad_clip": 0.3,
    "early_stop_patience": 5,
    "max_epochs": 30,
    "use_gradient_checkpointing": True,
    "diversity_weight": 0.6,
    "collapse_weight": 0.0005,
}

BITS_RANGE = list(range(6, 13))  # 6, 7, 8, 9, 10, 11, 12


# ──────────────────────────────────────────────────────────
# Tokenizer training
# ──────────────────────────────────────────────────────────

def _override_config(params: dict, target):
    for k, v in params.items():
        if hasattr(target, k):
            setattr(target, k, v)


def _choose_amp_dtype(device):
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_ctx(amp_enabled, amp_dtype):
    if not amp_enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    except Exception:
        return torch.cuda.amp.autocast(dtype=amp_dtype)


def train_tokenizer_fixed(bits: int, tdir: str, device: torch.device):
    """Train BSQ tokenizer with fixed params. Supports epoch-level resume."""
    params = {**TOKENIZER_PARAMS, "bits_per_quantizer": bits}
    _override_config(params, TokenizerConfig)

    set_global_seed(int(getattr(DataConfig, "random_seed", 42)), deterministic=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    bs = params["batch_size"]
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=bs, include_demo=False,
        loader_overrides={"num_workers": 0, "persistent_workers": False, "pin_memory": True},
    )

    tokenizer = HierarchicalQuantizer(**build_tokenizer_kwargs({
        **params,
        "bits_per_quantizer": bits,
    })).to(device)

    total_p = sum(p.numel() for p in tokenizer.parameters())
    lr = params["learning_rate"]
    opt = optim.Adam(tokenizer.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params["epochs"], eta_min=1e-5)

    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    use_scaler = bool(use_amp and amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # Resume / skip-if-done
    resume_path = os.path.join(tdir, "tokenizer_resume.pt")
    best_path = os.path.join(tdir, "tokenizer.pt")
    start_epoch = 0
    best_val = float("inf")
    os.makedirs(tdir, exist_ok=True)

    # If final model exists but resume checkpoint doesn't → already completed
    if os.path.exists(best_path) and not os.path.exists(resume_path):
        print(f"  Tokenizer already completed, loading from {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        tokenizer.load_state_dict(ckpt["model_state_dict"], strict=False)
        tokenizer.eval()
        tokenizer.requires_grad_(False)
        return tokenizer

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        tokenizer.load_state_dict(ckpt["model_state_dict"], strict=False)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"  Tokenizer resume from epoch {start_epoch}")

    print(f"  Tokenizer: {total_p:,} params, bits={bits}, vocab={1<<bits}")

    for epoch in range(start_epoch, params["epochs"]):
        tokenizer.train()
        train_loss = 0.0
        batches = 0
        for data in train_loader:
            data = data[0].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast_ctx(use_amp, amp_dtype):
                vq_loss, x_recon, _, _ = tokenizer(data, return_all=True)
                loss = F.mse_loss(x_recon, data) + vq_loss
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), params["grad_clip"])
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item()
            batches += 1
            del data, vq_loss, x_recon, loss

        tokenizer.eval()
        val_loss = 0.0
        vb = 0
        with torch.no_grad():
            for batch_data in val_loader:
                data = batch_data[0].to(device, non_blocking=True)
                with _autocast_ctx(use_amp, amp_dtype):
                    vq_loss, x_recon, _, _ = tokenizer(data, return_all=True)
                    vl = F.mse_loss(x_recon, data) + vq_loss
                if torch.isfinite(vl):
                    val_loss += vl.item()
                    vb += 1
                del data, vq_loss, x_recon, vl
        sched.step()

        avg_t = train_loss / max(batches, 1)
        avg_v = val_loss / max(vb, 1)

        if (epoch + 1) % 50 == 0 or epoch == start_epoch:
            print(f"  Tokenizer epoch {epoch+1:3d}/{params['epochs']}  train={avg_t:.4f}  val={avg_v:.4f}")

        if avg_v < best_val:
            best_val = avg_v
            torch.save({
                "epoch": epoch, "model_state_dict": tokenizer.state_dict(),
                "config": export_tokenizer_config(), "loss": avg_v,
            }, best_path)

        # Periodic resume checkpoint (every 50 epochs)
        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": tokenizer.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "best_val": best_val,
            }, resume_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Load best
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    tokenizer.load_state_dict(ckpt["model_state_dict"], strict=False)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    # Cleanup resume checkpoint
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(f"  Tokenizer done. best_val={best_val:.6f}")
    return tokenizer


# ──────────────────────────────────────────────────────────
# BaseModel training with early stopping
# ──────────────────────────────────────────────────────────

def _prepare_batch(features, time_features, tokenizer, device, non_blocking,
                   encoding_coarse=None, encoding_fine=None):
    """Build training tensors from a batch."""
    if encoding_coarse is not None and encoding_fine is not None:
        idx_coarse = encoding_coarse.to(device, non_blocking=non_blocking)
        idx_fine = encoding_fine.to(device, non_blocking=non_blocking)
    else:
        tokenizer_device = next(tokenizer.parameters()).device
        features_on_device = features.to(tokenizer_device, non_blocking=non_blocking)
        with torch.no_grad():
            idx_coarse, idx_fine = tokenizer.encode(features_on_device)
        del features_on_device
        if tokenizer_device != device:
            idx_coarse = idx_coarse.to(device, non_blocking=non_blocking)
            idx_fine = idx_fine.to(device, non_blocking=non_blocking)

    input_coarse = idx_coarse[:, :-1].long()
    input_fine = idx_fine[:, :-1].long()
    target_coarse = idx_coarse[:, 1:].long()
    target_fine = idx_fine[:, 1:].long()

    t_min = time_features["minute"][:, :-1].to(device, non_blocking=non_blocking).long()
    t_day = time_features["day"][:, :-1].to(device, non_blocking=non_blocking).long()
    t_month = time_features["month"][:, :-1].to(device, non_blocking=non_blocking).long()
    t_year = time_features["year"][:, :-1].to(device, non_blocking=non_blocking).long()

    return {
        "input_coarse": input_coarse, "input_fine": input_fine,
        "target_coarse": target_coarse, "target_fine": target_fine,
        "t_min": t_min, "t_day": t_day, "t_month": t_month, "t_year": t_year,
    }


def _unpack_batch(batch_data):
    # batch_data: (features, sector_ids_ignored, time_features, encodings)
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 3:
        return batch_data[0], batch_data[2], batch_data[3]
    return batch_data[0], None, None


def latent_regularization_loss(latent_states):
    if latent_states is None or latent_states.shape[0] < 2:
        return torch.tensor(0.0, device=latent_states.device if latent_states is not None else "cpu")
    k_steps, batch_size, num_tokens, channels = latent_states.shape
    diff = latent_states[1:] - latent_states[:-1]
    diversity_loss = torch.exp(-diff.pow(2).sum(-1).sqrt().mean())
    latent_flat = latent_states.reshape(k_steps, batch_size * num_tokens, channels)
    collapse_loss = torch.exp(-latent_flat.var(dim=1).mean())
    return (
        BASEMODEL_PARAMS["diversity_weight"] * diversity_loss
        + BASEMODEL_PARAMS["collapse_weight"] * collapse_loss
    )


def compute_token_distribution_from_model(model, tokenizer, dataloader, device, vocab_size):
    """Run model on validation data and compute token distribution metrics.

    This captures what the model ACTUALLY predicts — the key signal for
    whether the vocabulary size is appropriate.
    """
    model.eval()
    all_preds_coarse = []
    all_preds_fine = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="  Token dist eval", leave=False):
            features, time_features, encodings = _unpack_batch(batch_data)
            batch = _prepare_batch(
                features, time_features, tokenizer, device,
                non_blocking=True,
                encoding_coarse=encodings["idx_coarse"] if encodings else None,
                encoding_fine=encodings["idx_fine"] if encodings else None,
            )

            logits_c, logits_f, _ = model(
                batch["input_coarse"], batch["input_fine"],
                batch["t_min"], batch["t_day"], batch["t_month"], batch["t_year"],
            )
            preds_c = logits_c.argmax(dim=-1).reshape(-1).cpu()
            preds_f = logits_f.argmax(dim=-1).reshape(-1).cpu()
            all_preds_coarse.append(preds_c)
            all_preds_fine.append(preds_f)

            del batch, logits_c, logits_f

    preds_c = torch.cat(all_preds_coarse)
    preds_f = torch.cat(all_preds_fine)
    return _compute_distribution_metrics(preds_c, vocab_size, "coarse"), \
           _compute_distribution_metrics(preds_f, vocab_size, "fine")


def _compute_distribution_metrics(token_ids: torch.Tensor, vocab_size: int, label: str) -> dict:
    counts = torch.bincount(token_ids, minlength=vocab_size).float()
    total = counts.sum()

    if total == 0:
        return {"label": label, "dead_tokens": vocab_size, "utilization": 0.0,
                "norm_entropy": 0.0, "low_freq_share": 0.0, "top10_conc": 0.0}

    probs = counts / total
    raw_entropy = -(probs * (probs + 1e-12).log()).sum().item()
    max_entropy = math.log(vocab_size)
    norm_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

    used = int((counts > 0).sum().item())
    dead = vocab_size - used

    # Low-frequency token share: bottom 50% of tokens
    sorted_counts = counts.sort().values
    n = len(sorted_counts)
    low_half = sorted_counts[: n // 2]
    low_freq_share = (low_half.sum() / total).item()

    top10_conc = probs.topk(max(1, vocab_size // 10)).values.sum().item()

    return {
        "label": label,
        "vocab_size": vocab_size,
        "utilization": round(used / vocab_size, 6),
        "dead_tokens": dead,
        "norm_entropy": round(norm_entropy, 6),
        "low_freq_share": round(low_freq_share, 6),
        "top10_concentration": round(top10_conc, 6),
    }


def train_basemodel(tokenizer, bits: int, tdir: str, device: torch.device) -> dict:
    """Train BaseModel with DSA + GQA and early stopping. Supports epoch-level resume."""
    bp = BASEMODEL_PARAMS
    vocab_size = 1 << bits
    patience = bp["early_stop_patience"]
    max_epochs = bp["max_epochs"]

    set_global_seed(int(getattr(DataConfig, "random_seed", 42)), deterministic=True)

    # Data
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=bp["batch_size"], include_demo=False,
        loader_overrides={"num_workers": 0, "persistent_workers": False, "pin_memory": True},
    )

    # Model
    model = KronosReasoningGPT(
        dim=bp["dim"], depth=bp["depth"], heads=bp["heads"],
        num_kv_heads=bp["num_kv_heads"], dsa_windows=bp["dsa_windows"],
        dropout=bp["dropout"],
        vocab_size_coarse=vocab_size, vocab_size_fine=vocab_size,
        position_encoding=bp["position_encoding"], rope_base=bp["rope_base"],
    ).to(device)

    if bp["use_gradient_checkpointing"]:
        model.enable_gradient_checkpointing(True)

    total_params = sum(p.numel() for p in model.parameters())

    # Optimizer
    opt = optim.AdamW(model.parameters(), lr=bp["learning_rate"], weight_decay=bp["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-8)
    criterion = nn.CrossEntropyLoss()
    accumulation = bp["accumulation_steps"]

    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # Resume / skip-if-done
    result_path = os.path.join(tdir, "result.json")
    resume_path = os.path.join(tdir, "basemodel_resume.pt")
    ckpt_path = os.path.join(tdir, "basemodel.pt")
    hist_path = os.path.join(tdir, "basemodel_history.json")
    os.makedirs(tdir, exist_ok=True)

    # If result.json exists → BaseModel training was completed
    if os.path.exists(result_path):
        print(f"  BaseModel already completed, loading result from {result_path}")
        with open(result_path) as f:
            return json.load(f)

    start_epoch = 0
    best_val_ce = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_ce": [], "lr": []}

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_ce = ckpt.get("best_val_ce", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
        print(f"  BaseModel resume from epoch {start_epoch} (best_val_ce={best_val_ce:.4f}, patience={patience_counter})")

    print(f"  BaseModel: {total_params:,} params, vocab={vocab_size}, DSA windows={bp['dsa_windows']}")

    stopped_epoch = max_epochs
    for epoch in range(start_epoch, max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches_done = 0

        pbar = tqdm(train_loader, desc=f"  BM epoch {epoch+1}/{max_epochs}", leave=False)
        for batch_idx, batch_data in enumerate(pbar, start=1):
            features, time_features, encodings = _unpack_batch(batch_data)
            batch = _prepare_batch(
                features, time_features, tokenizer, device,
                non_blocking=True,
                encoding_coarse=encodings["idx_coarse"] if encodings else None,
                encoding_fine=encodings["idx_fine"] if encodings else None,
            )
            del features, time_features, encodings

            with _autocast_ctx(use_amp, amp_dtype):
                logits_c, logits_f, latent_states = model(
                    batch["input_coarse"], batch["input_fine"],
                    batch["t_min"], batch["t_day"], batch["t_month"], batch["t_year"],
                )
                ce_loss = criterion(logits_c.reshape(-1, vocab_size), batch["target_coarse"].reshape(-1))
                ce_loss = ce_loss + criterion(logits_f.reshape(-1, vocab_size), batch["target_fine"].reshape(-1))
                latent_loss = latent_regularization_loss(latent_states)
                step_loss = (ce_loss + latent_loss) / accumulation

            if not torch.isfinite(step_loss):
                opt.zero_grad(set_to_none=True)
                del batch, logits_c, logits_f, latent_states, ce_loss, latent_loss, step_loss
                continue

            scaler.scale(step_loss).backward()
            total_loss += (ce_loss + latent_loss).item()
            batches_done += 1

            if batch_idx % accumulation == 0 or batch_idx == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), bp["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": total_loss / max(batches_done, 1)})
            del batch, logits_c, logits_f, latent_states, ce_loss, latent_loss, step_loss

        avg_train = total_loss / max(batches_done, 1)
        history["train_loss"].append(avg_train)

        # Validation
        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                features, time_features, encodings = _unpack_batch(batch_data)
                batch = _prepare_batch(
                    features, time_features, tokenizer, device,
                    non_blocking=True,
                    encoding_coarse=encodings["idx_coarse"] if encodings else None,
                    encoding_fine=encodings["idx_fine"] if encodings else None,
                )
                del features, time_features, encodings
                with _autocast_ctx(use_amp, amp_dtype):
                    logits_c, logits_f, _ = model(
                        batch["input_coarse"], batch["input_fine"],
                        batch["t_min"], batch["t_day"], batch["t_month"], batch["t_year"],
                    )
                    ce = criterion(logits_c.reshape(-1, vocab_size), batch["target_coarse"].reshape(-1))
                    ce = ce + criterion(logits_f.reshape(-1, vocab_size), batch["target_fine"].reshape(-1))
                val_total += ce.item()
                val_batches += 1
                del batch, logits_c, logits_f, ce

        avg_val_ce = val_total / max(val_batches, 1)
        history["val_ce"].append(avg_val_ce)
        history["lr"].append(opt.param_groups[0]["lr"])
        sched.step()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Early stopping logic
        if avg_val_ce < best_val_ce:
            best_val_ce = avg_val_ce
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_ce": avg_val_ce, "bits": bits,
            }, ckpt_path)
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1:2d}/{max_epochs}  train={avg_train:.4f}  val_ce={avg_val_ce:.4f}  best={best_val_ce:.4f}  patience={patience_counter}/{patience}")

        # Save resume checkpoint (every epoch)
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "best_val_ce": best_val_ce, "patience_counter": patience_counter,
        }, resume_path)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        if patience_counter >= patience:
            stopped_epoch = epoch + 1
            print(f"  Early stop at epoch {stopped_epoch}")
            break

    # Load best model
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Token distribution
    print(f"\n  Evaluating token distribution...")
    metrics_c, metrics_f = compute_token_distribution_from_model(
        model, tokenizer, val_loader, device, vocab_size
    )

    result = {
        "bits": bits, "vocab_size": vocab_size,
        "best_val_ce": round(best_val_ce, 6),
        "epoch_stopped": stopped_epoch, "max_epochs": max_epochs,
        "params": bp,
        "token_metrics": {"coarse": metrics_c, "fine": metrics_f},
    }
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Cleanup resume checkpoint
    if os.path.exists(resume_path):
        os.remove(resume_path)

    print(f"  Result: val_ce={best_val_ce:.4f}, "
          f"c_util={metrics_c['utilization']:.3f}, c_low50={metrics_c['low_freq_share']:.4f}, "
          f"f_util={metrics_f['utilization']:.3f}, f_low50={metrics_f['low_freq_share']:.4f}")
    return result


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    os.makedirs(PHASE1_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 1 — Bits-per-quantizer search (DSA + GQA)")
    print(f"  Output dir: {PHASE1_DIR}")
    print(f"  Bits range: {BITS_RANGE}")
    print(f"  Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print()

    all_results = []

    for bits in BITS_RANGE:
        t0 = time.time()
        bdir = bits_dir(bits)
        os.makedirs(bdir, exist_ok=True)

        print(f"{'='*60}")
        print(f"Bits = {bits}  (vocab_size = {1<<bits})")
        print(f"{'='*60}")

        # 1. Train tokenizer
        print(f"\n[1/2] Training tokenizer (bits={bits})...")
        tokenizer = train_tokenizer_fixed(bits, bdir, device)

        # 2. Train BaseModel with early stopping
        print(f"\n[2/2] Training BaseModel (DSA, bits={bits})...")
        result = train_basemodel(tokenizer, bits, bdir, device)

        elapsed = time.time() - t0
        result["elapsed_minutes"] = round(elapsed / 60, 1)
        all_results.append(result)

        # Free memory
        del tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Intermediate save
        with open(SUMMARY_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n  bits={bits} done in {elapsed/60:.1f} min\n")

    # ── Final report ──
    print(f"\n{'='*60}")
    print(f"Phase 1 complete — Bits Comparison")
    print(f"{'='*60}")
    print(f"{'bits':>5} {'vocab':>6} {'val_ce':>10} {'epoch':>6} "
          f"{'c_util':>8} {'c_low50%':>10} {'f_util':>8} {'f_low50%':>10}")
    print("-" * 70)

    for r in all_results:
        c = r["token_metrics"]["coarse"]
        f = r["token_metrics"]["fine"]
        print(f"{r['bits']:5d} {r['vocab_size']:6d} {r['best_val_ce']:10.4f} {r['epoch_stopped']:6d} "
              f"{c['utilization']:8.3f} {c['low_freq_share']:10.4f} "
              f"{f['utilization']:8.3f} {f['low_freq_share']:10.4f}")

    print(f"\nFull results: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
