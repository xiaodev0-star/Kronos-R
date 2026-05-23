"""Phase 3: BaseModel Architecture HPO.

With tokenizer fixed (Phase 2 trial 015), searches:
  - dim, depth, heads, num_kv_heads (GQA ratio)
  - DSA window patterns
  - lr, weight_decay, dropout, diversity_weight, collapse_weight

Each trial trains a BaseModel with DSA + early stopping, then runs a
1-step downstream MAPE eval.  val_ce is the Optuna objective; MAPE is
recorded as a trial user attribute for final selection.

Usage:
    python -m hpo.phase3_basemodel
"""

from __future__ import annotations

import copy, json, math, os, time
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
from data_processor import get_dataloaders
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT
from reproducibility import set_global_seed

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────

N_TRIALS = 52 
STUDY_NAME = "phase3_basemodel"
CLEAN_START = False

BASEMODEL_BATCH_SIZE = 16
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "checkpoints", "tokenizer.pt")
TOKENIZER_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "checkpoints", "tokenizer_config.json")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE3_DIR = os.path.join(PROJECT_ROOT, "trials", "phase3_basemodel")
STUDY_DB = os.path.join(PHASE3_DIR, "study.db")
SUMMARY_CSV = os.path.join(PHASE3_DIR, "summary.csv")

# Tokenizer config (loaded once at startup)
TOKENIZER_BITS = 10
TOKENIZER_VOCAB = 1 << TOKENIZER_BITS

# ──────────────────────────────────────────────────────────
# Search space
# ──────────────────────────────────────────────────────────

DSA_PATTERNS = [
    [None, 512, 512, None],       # 0: current (2 full + 2 window)
    [None, 256, 256, None],       # 1: smaller windows
    [None, None, 512, 512],       # 2: more full attention
    [512, 512, 512, 512],         # 3: all sliding window
    [None, 256, 512, None],       # 4: graduated windows
]


def _expand_dsa_pattern(pattern: list, depth: int) -> list:
    """Extend/truncate DSA pattern to match model depth."""
    if depth <= len(pattern):
        return pattern[:depth]
    # Repeat: extend by cycling through the original pattern
    extended = list(pattern)
    while len(extended) < depth:
        extended.append(pattern[len(extended) % len(pattern)])
    return extended


def sample_params(trial: optuna.Trial) -> dict:
    """Sample BaseModel hyperparams with validity constraints."""
    dim = trial.suggest_categorical("dim", [128, 192, 256, 384])
    depth = trial.suggest_categorical("depth", [2, 3, 4, 6])

    # heads: must divide dim
    valid_heads = [h for h in [4, 8] if dim % h == 0]
    heads = trial.suggest_categorical("heads", valid_heads)

    # num_kv_heads: must divide heads and <= heads
    valid_kv = [k for k in [1, 2, 4] if heads % k == 0 and k <= heads]
    num_kv_heads = trial.suggest_categorical("num_kv_heads", valid_kv)

    # DSA pattern
    dsa_idx = trial.suggest_categorical("dsa_pattern", list(range(len(DSA_PATTERNS))))
    dsa_windows = _expand_dsa_pattern(DSA_PATTERNS[dsa_idx], depth)

    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    div_w = trial.suggest_float("diversity_weight", 0.1, 2.0, log=True)
    col_w = trial.suggest_float("collapse_weight", 1e-5, 5e-3, log=True)

    return {
        "dim": dim, "depth": depth, "heads": heads,
        "num_kv_heads": num_kv_heads, "dsa_windows": dsa_windows,
        "dsa_pattern_idx": dsa_idx,
        "lr": round(lr, 8), "weight_decay": round(wd, 8),
        "dropout": round(dropout, 4),
        "diversity_weight": round(div_w, 6),
        "collapse_weight": round(col_w, 8),
        "early_stop_patience": 5, "max_epochs": 30,
        "accumulation_steps": 2, "grad_clip": 0.3,
        "batch_size": BASEMODEL_BATCH_SIZE,
    }


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _choose_amp_dtype(device):
    if device.type != "cuda": return None
    if torch.cuda.is_bf16_supported(): return torch.bfloat16
    return torch.float16

def _autocast_ctx(amp, dt):
    if not amp: return nullcontext()
    try: return torch.amp.autocast(device_type="cuda", dtype=dt)
    except Exception: return torch.cuda.amp.autocast(dtype=dt)


def _load_tokenizer(device):
    """Load the fixed tokenizer from checkpoints/."""
    ckpt = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg and os.path.exists(TOKENIZER_CONFIG_PATH):
        with open(TOKENIZER_CONFIG_PATH) as f:
            cfg = json.load(f)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tok.load_state_dict(ckpt["model_state_dict"], strict=False)
    tok.eval(); tok.requires_grad_(False)
    print(f"  Loaded tokenizer: bits={cfg.get('bits_per_quantizer','?')}, "
          f"hd={cfg.get('hidden_dim','?')}, ed={cfg.get('embedding_dim','?')}")
    return tok


# ──────────────────────────────────────────────────────────
# Batch prep (no sector)
# ──────────────────────────────────────────────────────────

def _prepare_batch(features, time_features, tokenizer, device, non_blocking,
                   encoding_coarse=None, encoding_fine=None):
    if encoding_coarse is not None and encoding_fine is not None:
        idx_c = encoding_coarse.to(device, non_blocking=non_blocking)
        idx_f = encoding_fine.to(device, non_blocking=non_blocking)
    else:
        tk_dev = next(tokenizer.parameters()).device
        f_on_dev = features.to(tk_dev, non_blocking=non_blocking)
        with torch.no_grad():
            idx_c, idx_f = tokenizer.encode(f_on_dev)
        del f_on_dev
        if tk_dev != device:
            idx_c = idx_c.to(device, non_blocking=non_blocking)
            idx_f = idx_f.to(device, non_blocking=non_blocking)
    return {
        "input_coarse": idx_c[:, :-1].long(),
        "input_fine": idx_f[:, :-1].long(),
        "target_coarse": idx_c[:, 1:].long(),
        "target_fine": idx_f[:, 1:].long(),
        "t_min": time_features["minute"][:, :-1].to(device, non_blocking=non_blocking).long(),
        "t_day": time_features["day"][:, :-1].to(device, non_blocking=non_blocking).long(),
        "t_month": time_features["month"][:, :-1].to(device, non_blocking=non_blocking).long(),
        "t_year": time_features["year"][:, :-1].to(device, non_blocking=non_blocking).long(),
    }

def _unpack_batch(batch_data):
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 3:
        return batch_data[0], batch_data[2], batch_data[3]
    return batch_data[0], None, None


# ──────────────────────────────────────────────────────────
# Downstream 1-step eval (per-trial, lightweight)
# ──────────────────────────────────────────────────────────

_VAL_CACHE = os.path.join(PROJECT_ROOT, "dataset_val.pt")
_EVAL_BATCH_SIZE = 32


def _load_val_data():
    """Load cached val dataset for downstream eval."""
    payload = torch.load(_VAL_CACHE, map_location="cpu", weights_only=False)
    features = payload["features"]
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)

    time_features = {}
    for key in ("minute", "day", "month", "year"):
        t = payload["time_features"][key]
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.long)
        time_features[key] = t

    seq_stats = payload["seq_stats"]
    N = len(seq_stats)
    means = np.zeros((N, 6), dtype=np.float32)
    stds  = np.zeros((N, 6), dtype=np.float32)
    for i, s in enumerate(seq_stats):
        means[i] = np.asarray(s["mean"], dtype=np.float32)
        stds[i]  = np.asarray(s["std"],  dtype=np.float32)
    return features, time_features, torch.from_numpy(means), torch.from_numpy(stds)


_VAL_DATA = None  # lazy-loaded once

def _get_val_data():
    global _VAL_DATA
    if _VAL_DATA is None:
        print("  Loading val cache for downstream eval...")
        _VAL_DATA = _load_val_data()
    return _VAL_DATA


@torch.inference_mode()
def _run_downstream_eval(model, tokenizer, device) -> dict:
    """1-step downstream prediction on validation set. Single pass, no bootstrap."""
    features, time_features, means, stds = _get_val_data()

    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    N = features.shape[0]
    all_preds, all_actuals = [], []
    indices = np.arange(N)

    for start in tqdm(range(0, N, _EVAL_BATCH_SIZE), desc="  Eval 1-step", leave=False):
        end = min(start + _EVAL_BATCH_SIZE, N)
        idx = indices[start:end]

        batch_feats = features[idx].to(device, non_blocking=True)
        batch_means = means[idx].to(device, non_blocking=True)
        batch_stds  = stds[idx].to(device, non_blocking=True)

        input_feats = batch_feats[:, :1023, :]
        actual_norm = batch_feats[:, 1023, 0]

        idx_c, idx_f = tokenizer.encode(input_feats)

        t_min  = time_features["minute"][idx][:, :1023].to(device, non_blocking=True).long()
        t_day  = time_features["day"][idx][:, :1023].to(device, non_blocking=True).long()
        t_month= time_features["month"][idx][:, :1023].to(device, non_blocking=True).long()
        t_year = time_features["year"][idx][:, :1023].to(device, non_blocking=True).long()

        with _autocast_ctx(use_amp, amp_dtype):
            lc, lf, _ = model(idx_c, idx_f, t_min, t_day, t_month, t_year, last_only=True)

        pred_c = lc[:, -1, :].float().argmax(dim=-1)
        pred_f = lf[:, -1, :].float().argmax(dim=-1)
        decoded = tokenizer.decode(pred_c.unsqueeze(1), pred_f.unsqueeze(1))
        pred_norm = decoded[:, 0, 0]

        pred_log_ret   = pred_norm * batch_stds[:, 0] + batch_means[:, 0]
        actual_log_ret = actual_norm * batch_stds[:, 0] + batch_means[:, 0]

        all_preds.append(pred_log_ret.cpu())
        all_actuals.append(actual_log_ret.cpu())

        del batch_feats, batch_means, batch_stds, input_feats, idx_c, idx_f, lc, lf, decoded

    preds   = torch.cat(all_preds).numpy().astype(np.float64)
    actuals = torch.cat(all_actuals).numpy().astype(np.float64)

    # Filter non-finite
    finite = np.isfinite(preds) & np.isfinite(actuals)
    preds, actuals = preds[finite], actuals[finite]

    # Metrics
    pred_ratio   = np.exp(np.clip(preds, -50, 50))
    actual_ratio = np.exp(np.clip(actuals, -50, 50))
    mape = float(np.mean(np.abs((pred_ratio - actual_ratio) /
                                 np.maximum(np.abs(actual_ratio), 1e-4))) * 100)
    pred_sign   = np.where(preds >= 0, 1, -1)
    actual_sign = np.where(actuals >= 0, 1, -1)
    da = float(np.mean(pred_sign == actual_sign) * 100)
    err = preds - actuals

    return {
        "mape": round(mape, 6),
        "da": round(da, 4),
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(np.sqrt(np.mean(err * err))), 6),
        "num_samples": int(len(preds)),
    }


# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────

def train_basemodel(params: dict, tdir: str, tokenizer, device: torch.device) -> float:
    """Train BaseModel with DSA. Returns best_val_ce. Supports resume."""
    bp = params
    vocab = TOKENIZER_VOCAB
    patience = bp["early_stop_patience"]
    max_epochs = bp["max_epochs"]

    result_path = os.path.join(tdir, "result.json")
    resume_path = os.path.join(tdir, "basemodel_resume.pt")
    ckpt_path = os.path.join(tdir, "basemodel.pt")
    hist_path = os.path.join(tdir, "basemodel_history.json")
    os.makedirs(tdir, exist_ok=True)

    if os.path.exists(result_path):
        with open(result_path) as f:
            r = json.load(f)
        print(f"  Already done. val_ce={r['best_val_ce']:.4f}")
        return r["best_val_ce"]

    set_global_seed(int(getattr(DataConfig, "random_seed", 42)), deterministic=True)

    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=bp["batch_size"], include_demo=False,
        loader_overrides={"num_workers": 0, "persistent_workers": False, "pin_memory": True},
    )

    model = KronosReasoningGPT(
        dim=bp["dim"], depth=bp["depth"], heads=bp["heads"],
        num_kv_heads=bp["num_kv_heads"], dsa_windows=bp["dsa_windows"],
        dropout=bp["dropout"], vocab_size_coarse=vocab, vocab_size_fine=vocab,
        position_encoding="rope", rope_base=10000.0,
    ).to(device)
    model.enable_gradient_checkpointing(True)

    total_p = sum(p.numel() for p in model.parameters())
    opt = optim.AdamW(model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-8)
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
        print(f"  Resume epoch {start_epoch} (best={best_val_ce:.4f})")

    print(f"  Model: {total_p:,} params  dim={bp['dim']} depth={bp['depth']} "
          f"heads={bp['heads']}/{bp['num_kv_heads']}  windows={bp['dsa_windows']}")

    stopped_epoch = max_epochs
    for epoch in range(start_epoch, max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0; batches_done = 0

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
                lc, lf, ls = model(
                    batch["input_coarse"], batch["input_fine"],
                    batch["t_min"], batch["t_day"], batch["t_month"], batch["t_year"],
                )
                ce = criterion(lc.reshape(-1, vocab), batch["target_coarse"].reshape(-1))
                ce = ce + criterion(lf.reshape(-1, vocab), batch["target_fine"].reshape(-1))
                # Latent regularization
                if ls is not None and ls.shape[0] >= 2:
                    k, B, N, C = ls.shape
                    div_l = torch.exp(-(ls[1:]-ls[:-1]).pow(2).sum(-1).sqrt().mean())
                    col_l = torch.exp(-ls.reshape(k, B*N, C).var(dim=1).mean())
                    lat = bp["diversity_weight"] * div_l + bp["collapse_weight"] * col_l
                else:
                    lat = torch.tensor(0.0, device=device)
                step_loss = (ce + lat) / acc

            if not torch.isfinite(step_loss):
                opt.zero_grad(set_to_none=True)
                del batch, lc, lf, ls, ce, lat, step_loss
                continue

            scaler.scale(step_loss).backward()
            total_loss += (ce + lat).item()
            batches_done += 1

            if batch_idx % acc == 0 or batch_idx == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), bp["grad_clip"])
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

            del batch, lc, lf, ls, ce, lat, step_loss

        # Val
        model.eval()
        val_total = 0.0; val_batches = 0
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
                    lc, lf, _ = model(
                        batch["input_coarse"], batch["input_fine"],
                        batch["t_min"], batch["t_day"], batch["t_month"], batch["t_year"],
                    )
                    ce = criterion(lc.reshape(-1, vocab), batch["target_coarse"].reshape(-1))
                    ce = ce + criterion(lf.reshape(-1, vocab), batch["target_fine"].reshape(-1))
                val_total += ce.item(); val_batches += 1
                del batch, lc, lf, ce

        avg_val = val_total / max(val_batches, 1)
        avg_train = total_loss / max(batches_done, 1)
        history["train_loss"].append(avg_train)
        history["val_ce"].append(avg_val)
        history["lr"].append(opt.param_groups[0]["lr"])
        sched.step()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if avg_val < best_val_ce:
            best_val_ce = avg_val; patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_ce": avg_val}, ckpt_path)
        else:
            patience_counter += 1

        print(f"  BM epoch {epoch+1:2d}/{max_epochs}  train={avg_train:.4f}  "
              f"val_ce={avg_val:.4f}  best={best_val_ce:.4f}  patience={patience_counter}/{patience}")

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
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

    # Downstream eval
    print(f"  Running 1-step downstream eval...")
    eval_metrics = _run_downstream_eval(model, tokenizer, device)

    result = {"best_val_ce": round(best_val_ce, 6), "epoch_stopped": stopped_epoch,
              "max_epochs": max_epochs, "params": bp, "downstream": eval_metrics}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    if os.path.exists(resume_path):
        os.remove(resume_path)

    print(f"  MAPE={eval_metrics['mape']:.4f}%  DA={eval_metrics['da']:.2f}%  "
          f"MAE={eval_metrics['mae']:.6f}  RMSE={eval_metrics['rmse']:.6f}")
    return best_val_ce, eval_metrics


# ──────────────────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────────────────

# Global tokenizer loaded once — set by main(), read by objective()
_G_TOKENIZER = None


def _assign_trial_dir() -> str:
    """Scan existing directories: resume incomplete trials, else create new.

    This decouples directory assignment from Optuna trial numbering so
    that killed-and-restarted trials resume from their existing checkpoint
    instead of being abandoned when Optuna assigns a new trial number.
    """
    os.makedirs(PHASE3_DIR, exist_ok=True)
    existing = sorted(
        [d for d in os.listdir(PHASE3_DIR) if d.startswith("trial_")],
        key=lambda x: int(x.split("_")[1]),
    )
    for d in existing:
        full = os.path.join(PHASE3_DIR, d)
        resume = os.path.join(full, "basemodel_resume.pt")
        result = os.path.join(full, "result.json")
        if os.path.exists(resume) and not os.path.exists(result):
            # Also verify config.json exists (trial was actually started)
            if os.path.exists(os.path.join(full, "config.json")):
                print(f"  Found incomplete trial: {d} — will resume")
                return full
    # All complete or no resume checkpoints → new trial
    next_num = len(existing)
    return os.path.join(PHASE3_DIR, f"trial_{next_num:03d}")


def objective(trial: optuna.Trial) -> float:
    global _G_TOKENIZER
    tdir = _assign_trial_dir()
    os.makedirs(tdir, exist_ok=True)

    params = sample_params(trial)
    # If resuming an existing dir, reload its original config (don't overwrite)
    config_path = os.path.join(tdir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            params = json.load(f)
    else:
        with open(config_path, "w") as f:
            json.dump(params, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Map trial number for logging
    t_number = trial.number
    dir_name = os.path.basename(tdir)

    print(f"\n{'='*60}")
    print(f"Trial {t_number:03d} (dir={dir_name}) -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Params: dim={params['dim']} depth={params['depth']} heads={params['heads']}/"
          f"{params['num_kv_heads']} dsa={params['dsa_windows']}")
    print(f"        lr={params['lr']:.2e} wd={params['weight_decay']:.2e} "
          f"drop={params['dropout']} div={params['diversity_weight']} col={params['collapse_weight']}")
    print(f"{'='*60}")

    t0 = time.time()
    val_ce, eval_metrics = train_basemodel(params, tdir, _G_TOKENIZER, device)
    elapsed = time.time() - t0

    trial.set_user_attr("trial_dir", tdir)
    trial.set_user_attr("dir_name", dir_name)
    trial.set_user_attr("elapsed_min", round(elapsed / 60, 1))
    trial.set_user_attr("mape", eval_metrics["mape"])
    trial.set_user_attr("da", eval_metrics["da"])
    trial.set_user_attr("mae", eval_metrics["mae"])
    trial.set_user_attr("rmse", eval_metrics["rmse"])

    result_path = os.path.join(tdir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            r = json.load(f)
        trial.set_user_attr("epoch_stopped", r.get("epoch_stopped", -1))

    print(f"  val_ce={val_ce:.6f}  MAPE={eval_metrics['mape']:.4f}%  "
          f"DA={eval_metrics['da']:.2f}%  time={elapsed/60:.1f} min")
    return val_ce


# ──────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────

def export_summary(study: optuna.Study):
    import csv
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "value": t.value, **t.params}
        # Flatten dsa_windows
        if "dsa_windows" in row:
            row["dsa_windows"] = str(row["dsa_windows"])
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
    os.makedirs(PHASE3_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted(rows, key=lambda r: r["value"])
    print(f"\nTop-10 by val_ce:")
    for r in ranked[:10]:
        print(f"  Trial {r['trial']:03d}  val_ce={r['value']:.6f}  "
              f"dim={r.get('dim','?')}  depth={r.get('depth','?')}")
    print(f"Summary: {SUMMARY_CSV}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    global _G_TOKENIZER

    if CLEAN_START and os.path.exists(PHASE3_DIR):
        import shutil; shutil.rmtree(PHASE3_DIR)
    os.makedirs(PHASE3_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 3 -- BaseModel Architecture HPO")
    print(f"  Output:    {PHASE3_DIR}")
    print(f"  Trials:    {N_TRIALS}")
    print(f"  Device:    {device}")
    if device.type == "cuda":
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print()

    # Load tokenizer once
    print("Loading fixed tokenizer...")
    _G_TOKENIZER = _load_tokenizer(device)

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=f"sqlite:///{STUDY_DB}",
        direction="minimize", load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\nPhase 3 complete.")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best val_ce: {study.best_trial.value:.6f}")
    print(f"  Best params: {json.dumps(study.best_trial.params, indent=4)}")
    export_summary(study)


if __name__ == "__main__":
    main()
