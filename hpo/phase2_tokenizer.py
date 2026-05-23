"""Phase 2: Tokenizer hyperparameter HPO.

With bits-per-quantizer fixed (from Phase 1), searches:
  - hidden_dim, embedding_dim
  - bsq_commitment_cost, bsq_entropy_weight
  - tokenizer learning_rate

Each trial trains a tokenizer, then a BaseModel with early
stopping.  The objective is val_ce (minimize).  Token distribution metrics
are saved as secondary signals for post-hoc filtering.

Usage:
    python -m hpo.phase2_tokenizer
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import DataConfig, TokenizerConfig
from data_processor import get_dataloaders
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs, export_tokenizer_config
from model.kronos_reasoning import KronosReasoningGPT
from reproducibility import set_global_seed

# ──────────────────────────────────────────────────────────
# Config — edit here, not via CLI
# ──────────────────────────────────────────────────────────

N_TRIALS = 50
STUDY_NAME = "phase2_tokenizer"
CLEAN_START = False

# Fixed from Phase 1 result — edit after Phase 1 completes
BITS = 10

# Batch size overrides
TOKENIZER_BATCH_SIZE = 512
BASEMODEL_BATCH_SIZE = 16

# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE2_DIR = os.path.join(PROJECT_ROOT, "trials", "phase2_tokenizer")
STUDY_DB = os.path.join(PHASE2_DIR, "study.db")
SUMMARY_CSV = os.path.join(PHASE2_DIR, "summary.csv")


def trial_dir(trial_number: int) -> str:
    return os.path.join(PHASE2_DIR, f"trial_{trial_number:03d}")


# ──────────────────────────────────────────────────────────
# Search space
# ──────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "hidden_dim":         [96, 128, 192, 256],
    "embedding_dim":      [48, 64, 96, 128, 192],
    "bsq_commitment_cost": (0.01, 0.20),
    "bsq_entropy_weight":  (0.01, 0.20),
    "learning_rate":       (1e-5, 5e-4),
}

# Fixed tokenizer params
TOKENIZER_FIXED = {
    "num_quantizers": 2,
    "epochs": 100,
    "grad_clip": 1.0,
}

# Fixed BaseModel params
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
    "accumulation_steps": 2,
    "grad_clip": 0.3,
    "early_stop_patience": 5,
    "max_epochs": 30,
    "use_gradient_checkpointing": True,
    "diversity_weight": 0.6,
    "collapse_weight": 0.0005,
}


# ──────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────

def sample_params(trial: optuna.Trial) -> dict:
    """Sample tokenizer hyperparams. Clamp embedding_dim <= hidden_dim."""
    hidden_dim = trial.suggest_categorical("hidden_dim", SEARCH_SPACE["hidden_dim"])
    embedding_dim_raw = trial.suggest_categorical("embedding_dim", SEARCH_SPACE["embedding_dim"])
    # Also clamp to BaseModel dim (256) to keep things reasonable
    embedding_dim = min(embedding_dim_raw, hidden_dim)

    commit = trial.suggest_float("bsq_commitment_cost", *SEARCH_SPACE["bsq_commitment_cost"], log=True)
    ent = trial.suggest_float("bsq_entropy_weight", *SEARCH_SPACE["bsq_entropy_weight"], log=True)
    lr = trial.suggest_float("lr_tokenizer", *SEARCH_SPACE["learning_rate"], log=True)

    return {
        **TOKENIZER_FIXED,
        "bits_per_quantizer": BITS,
        "hidden_dim": hidden_dim,
        "embedding_dim_raw": embedding_dim_raw,
        "embedding_dim": embedding_dim,
        "bsq_commitment_cost": round(commit, 6),
        "bsq_entropy_weight": round(ent, 6),
        "learning_rate": round(lr, 8),
        "batch_size": TOKENIZER_BATCH_SIZE,
    }


# ──────────────────────────────────────────────────────────
# Helpers
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


# ──────────────────────────────────────────────────────────
# Tokenizer training
# ──────────────────────────────────────────────────────────

def train_tokenizer(params: dict, tdir: str, device: torch.device):
    """Train BSQ tokenizer. Skip if already done."""
    _override_config(params, TokenizerConfig)

    best_path = os.path.join(tdir, "tokenizer.pt")
    resume_path = os.path.join(tdir, "tokenizer_resume.pt")
    os.makedirs(tdir, exist_ok=True)

    # Already completed?
    if os.path.exists(best_path) and not os.path.exists(resume_path):
        print(f"  Tokenizer already done, loading.")
        return

    set_global_seed(int(getattr(DataConfig, "random_seed", 42)), deterministic=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=params["batch_size"], include_demo=False,
        loader_overrides={"num_workers": 0, "persistent_workers": False, "pin_memory": True},
    )

    tokenizer = HierarchicalQuantizer(**build_tokenizer_kwargs(params)).to(device)
    total_p = sum(p.numel() for p in tokenizer.parameters())
    opt = optim.Adam(tokenizer.parameters(), lr=params["learning_rate"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params["epochs"], eta_min=1e-5)

    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    start_epoch = 0
    best_val = float("inf")

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        tokenizer.load_state_dict(ckpt["model_state_dict"], strict=False)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"  Tokenizer resume from epoch {start_epoch}")

    print(f"  Tokenizer: {total_p:,} params, bits={BITS}, vocab={1<<BITS}")

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
            print(f"  T tok epoch {epoch+1:3d}/{params['epochs']}  train={avg_t:.4f}  val={avg_v:.4f}")

        if avg_v < best_val:
            best_val = avg_v
            torch.save({
                "epoch": epoch, "model_state_dict": tokenizer.state_dict(),
                "config": export_tokenizer_config(), "loss": avg_v,
            }, best_path)

        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": tokenizer.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(), "best_val": best_val,
            }, resume_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Load best
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    tokenizer.load_state_dict(ckpt["model_state_dict"], strict=False)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(f"  Tokenizer done. best_val={best_val:.6f}")


# ──────────────────────────────────────────────────────────
# BaseModel training
# ──────────────────────────────────────────────────────────

def _prepare_batch(features, time_features, tokenizer, device, non_blocking,
                   encoding_coarse=None, encoding_fine=None):
    if encoding_coarse is not None and encoding_fine is not None:
        idx_coarse = encoding_coarse.to(device, non_blocking=non_blocking)
        idx_fine = encoding_fine.to(device, non_blocking=non_blocking)
    else:
        tk_device = next(tokenizer.parameters()).device
        f_on_dev = features.to(tk_device, non_blocking=non_blocking)
        with torch.no_grad():
            idx_coarse, idx_fine = tokenizer.encode(f_on_dev)
        del f_on_dev
        if tk_device != device:
            idx_coarse = idx_coarse.to(device, non_blocking=non_blocking)
            idx_fine = idx_fine.to(device, non_blocking=non_blocking)

    return {
        "input_coarse": idx_coarse[:, :-1].long(),
        "input_fine": idx_fine[:, :-1].long(),
        "target_coarse": idx_coarse[:, 1:].long(),
        "target_fine": idx_fine[:, 1:].long(),
        "t_min": time_features["minute"][:, :-1].to(device, non_blocking=non_blocking).long(),
        "t_day": time_features["day"][:, :-1].to(device, non_blocking=non_blocking).long(),
        "t_month": time_features["month"][:, :-1].to(device, non_blocking=non_blocking).long(),
        "t_year": time_features["year"][:, :-1].to(device, non_blocking=non_blocking).long(),
    }


def _unpack_batch(batch_data):
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 3:
        return batch_data[0], batch_data[2], batch_data[3]
    return batch_data[0], None, None


def _latent_reg(latent_states):
    if latent_states is None or latent_states.shape[0] < 2:
        return torch.tensor(0.0, device=latent_states.device if latent_states is not None else "cpu")
    k, B, N, C = latent_states.shape
    div = torch.exp(-(latent_states[1:] - latent_states[:-1]).pow(2).sum(-1).sqrt().mean())
    col = torch.exp(-latent_states.reshape(k, B * N, C).var(dim=1).mean())
    return BASEMODEL_PARAMS["diversity_weight"] * div + BASEMODEL_PARAMS["collapse_weight"] * col


def train_basemodel(tdir: str, device: torch.device) -> float:
    """Train BaseModel with DSA. Returns best_val_ce (Optuna objective).

    Loads tokenizer from tdir/tokenizer.pt, skips if result.json exists.
    """
    bp = BASEMODEL_PARAMS
    vocab_size = 1 << BITS

    result_path = os.path.join(tdir, "result.json")
    resume_path = os.path.join(tdir, "basemodel_resume.pt")
    ckpt_path = os.path.join(tdir, "basemodel.pt")
    hist_path = os.path.join(tdir, "basemodel_history.json")
    os.makedirs(tdir, exist_ok=True)

    # Already completed?
    if os.path.exists(result_path):
        with open(result_path) as f:
            r = json.load(f)
        print(f"  BaseModel already done. val_ce={r['best_val_ce']:.4f}")
        return r["best_val_ce"]

    # Load tokenizer
    tokenizer_ckpt = torch.load(os.path.join(tdir, "tokenizer.pt"),
                                map_location=device, weights_only=False)
    tokenizer = HierarchicalQuantizer(**build_tokenizer_kwargs(
        tokenizer_ckpt.get("config", {}))).to(device)
    tokenizer.load_state_dict(tokenizer_ckpt["model_state_dict"], strict=False)
    tokenizer.eval()
    tokenizer.requires_grad_(False)

    set_global_seed(int(getattr(DataConfig, "random_seed", 42)), deterministic=True)

    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=BASEMODEL_BATCH_SIZE, include_demo=False,
        loader_overrides={"num_workers": 0, "persistent_workers": False, "pin_memory": True},
    )

    model = KronosReasoningGPT(
        dim=bp["dim"], depth=bp["depth"], heads=bp["heads"],
        num_kv_heads=bp["num_kv_heads"], dsa_windows=bp["dsa_windows"],
        dropout=bp["dropout"],
        vocab_size_coarse=vocab_size, vocab_size_fine=vocab_size,
        position_encoding=bp["position_encoding"], rope_base=bp["rope_base"],
    ).to(device)

    if bp["use_gradient_checkpointing"]:
        model.enable_gradient_checkpointing(True)

    total_p = sum(p.numel() for p in model.parameters())
    opt = optim.AdamW(model.parameters(), lr=bp["learning_rate"], weight_decay=bp["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=bp["max_epochs"], eta_min=1e-8)
    criterion = nn.CrossEntropyLoss()
    acc = bp["accumulation_steps"]

    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

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
        print(f"  BM resume epoch {start_epoch} (best={best_val_ce:.4f}, patience={patience_counter})")

    print(f"  BaseModel: {total_p:,} params, vocab={vocab_size}")

    patience = bp["early_stop_patience"]
    max_epochs = bp["max_epochs"]
    stopped_epoch = max_epochs

    for epoch in range(start_epoch, max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches_done = 0

        for batch_idx, batch_data in enumerate(
            tqdm(train_loader, desc=f"  BM {epoch+1}/{max_epochs}", leave=False), start=1
        ):
            features, time_features, encodings = _unpack_batch(batch_data)
            batch = _prepare_batch(
                features, time_features, tokenizer, device, non_blocking=True,
                encoding_coarse=encodings["idx_coarse"] if encodings else None,
                encoding_fine=encodings["idx_fine"] if encodings else None,
            )
            del features, time_features, encodings

            with _autocast_ctx(use_amp, amp_dtype):
                logits_c, logits_f, latent_states = model(
                    batch["input_coarse"], batch["input_fine"],
                    batch["t_min"], batch["t_day"], batch["t_month"], batch["t_year"],
                )
                ce = criterion(logits_c.reshape(-1, vocab_size), batch["target_coarse"].reshape(-1))
                ce = ce + criterion(logits_f.reshape(-1, vocab_size), batch["target_fine"].reshape(-1))
                step_loss = (ce + _latent_reg(latent_states)) / acc

            if not torch.isfinite(step_loss):
                opt.zero_grad(set_to_none=True)
                del batch, logits_c, logits_f, latent_states, ce, step_loss
                continue

            scaler.scale(step_loss).backward()
            total_loss += (ce + _latent_reg(latent_states)).item()
            batches_done += 1

            if batch_idx % acc == 0 or batch_idx == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), bp["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            del batch, logits_c, logits_f, latent_states, ce, step_loss

        # Val
        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                features, time_features, encodings = _unpack_batch(batch_data)
                batch = _prepare_batch(
                    features, time_features, tokenizer, device, non_blocking=True,
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

        avg_val = val_total / max(val_batches, 1)
        history["train_loss"].append(total_loss / max(batches_done, 1))
        history["val_ce"].append(avg_val)
        history["lr"].append(opt.param_groups[0]["lr"])
        sched.step()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if avg_val < best_val_ce:
            best_val_ce = avg_val
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_ce": avg_val, "bits": BITS}, ckpt_path)
        else:
            patience_counter += 1

        print(f"  BM epoch {epoch+1:2d}/{max_epochs}  train={history['train_loss'][-1]:.4f}  "
              f"val_ce={avg_val:.4f}  best={best_val_ce:.4f}  patience={patience_counter}/{patience}")

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

    # Token distribution (quick eval — just coarse utilization as sanity)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Compute coarse token utilization
    all_preds = []
    with torch.no_grad():
        for batch_data in val_loader:
            features, time_features, encodings = _unpack_batch(batch_data)
            batch = _prepare_batch(
                features, time_features, tokenizer, device, non_blocking=True,
                encoding_coarse=encodings["idx_coarse"] if encodings else None,
                encoding_fine=encodings["idx_fine"] if encodings else None,
            )
            del features, time_features, encodings
            logits_c, _, _ = model(
                batch["input_coarse"], batch["input_fine"],
                batch["t_min"], batch["t_day"], batch["t_month"], batch["t_year"],
            )
            all_preds.append(logits_c.argmax(dim=-1).reshape(-1).cpu())
            del batch, logits_c

    preds = torch.cat(all_preds)
    counts = torch.bincount(preds, minlength=vocab_size).float()
    used = int((counts > 0).sum().item())
    dead = vocab_size - used
    probs = counts / counts.sum()
    ent = -(probs * (probs + 1e-12).log()).sum().item()
    norm_ent = ent / math.log(vocab_size) if vocab_size > 1 else 0.0

    # low_freq_share
    sorted_c = counts.sort().values
    low_half = sorted_c[: len(sorted_c) // 2]
    low_share = (low_half.sum() / counts.sum()).item()

    result = {
        "bits": BITS,
        "vocab_size": vocab_size,
        "best_val_ce": round(best_val_ce, 6),
        "epoch_stopped": stopped_epoch,
        "max_epochs": max_epochs,
        "coarse_utilization": round(used / vocab_size, 4),
        "coarse_dead_tokens": dead,
        "coarse_norm_entropy": round(norm_ent, 4),
        "coarse_low_freq_share": round(low_share, 4),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    if os.path.exists(resume_path):
        os.remove(resume_path)

    print(f"  val_ce={best_val_ce:.4f}  util={used}/{vocab_size}  low50={low_share:.4f}")
    return best_val_ce


# ──────────────────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    t_number = trial.number
    tdir = trial_dir(t_number)
    os.makedirs(tdir, exist_ok=True)

    params = sample_params(trial)
    config_path = os.path.join(tdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(params, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Trial {t_number:03d} -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Params: {json.dumps({k: v for k, v in params.items() if k != 'embedding_dim_raw'}, indent=2)}")
    print(f"Trial dir: {tdir}")
    print(f"{'='*60}")

    # 1. Train tokenizer
    train_tokenizer(params, tdir, device)

    # 2. Train BaseModel
    t0 = time.time()
    val_ce = train_basemodel(tdir, device)
    elapsed = time.time() - t0

    # Store attributes
    result_path = os.path.join(tdir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            r = json.load(f)
        for key in ["coarse_utilization", "coarse_dead_tokens",
                     "coarse_norm_entropy", "coarse_low_freq_share"]:
            trial.set_user_attr(key, r.get(key, -1))
    trial.set_user_attr("trial_dir", tdir)
    trial.set_user_attr("elapsed_min", round(elapsed / 60, 1))
    trial.set_user_attr("actual_embedding_dim", params["embedding_dim"])

    # Cleanup resume checkpoint
    ckpt_path = os.path.join(tdir, "basemodel_resume.pt")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"  Score (val_ce): {val_ce:.6f}  time: {elapsed/60:.1f} min")
    return val_ce


# ──────────────────────────────────────────────────────────
# Summary export
# ──────────────────────────────────────────────────────────

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

    import csv
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    ordered = ["trial", "value"] + sorted(k for k in all_keys if k not in ("trial", "value"))

    os.makedirs(PHASE2_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    sorted_rows = sorted(rows, key=lambda r: r["value"])
    print(f"\n{'='*60}")
    print(f"Top-10 trials (by val_ce, lower=better):")
    print(f"{'='*60}")
    for r in sorted_rows[:10]:
        print(f"  Trial {r['trial']:03d}  val_ce={r['value']:.6f}  "
              f"c_util={r.get('coarse_utilization','?'):.4f}  "
              f"c_low50={r.get('coarse_low_freq_share','?'):.4f}")
    print(f"\nFull summary: {SUMMARY_CSV}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    if CLEAN_START and os.path.exists(PHASE2_DIR):
        import shutil
        shutil.rmtree(PHASE2_DIR)
        print(f"Cleaned: {PHASE2_DIR}")

    os.makedirs(PHASE2_DIR, exist_ok=True)

    print(f"Phase 2 -- Tokenizer HPO (bits={BITS} fixed)")
    print(f"  Trials dir: {PHASE2_DIR}")
    print(f"  Study DB:   {STUDY_DB}")
    print(f"  N trials:   {N_TRIALS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print()

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\n{'='*60}")
    print(f"Phase 2 complete.")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best val_ce: {study.best_trial.value:.6f}")
    print(f"  Best params: {json.dumps(study.best_trial.params, indent=4)}")
    print(f"{'='*60}")

    export_summary(study)


if __name__ == "__main__":
    main()
