"""Phase 4: Auxiliary Components HPO.

With tokenizer (P2 trial 015) and backbone (P3 trial 047) fixed, searches:
  - num_latent_tokens, latent_reasoner_depth, latent_cross_heads
  - num_factor_tokens

Each trial trains + 1-step downstream eval.  val_ce is Optuna objective;
MAPE recorded for final selection.

Usage:
    python -m hpo.phase4_aux
"""

from __future__ import annotations

import json, os, time
from contextlib import nullcontext
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import DataConfig
from data_processor import get_dataloaders
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT
from reproducibility import set_global_seed

# ── Config ──
N_TRIALS = 30
STUDY_NAME = "phase4_aux"
CLEAN_START = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE4_DIR = os.path.join(PROJECT_ROOT, "trials", "phase4_aux")
STUDY_DB = os.path.join(PHASE4_DIR, "study.db")
SUMMARY_CSV = os.path.join(PHASE4_DIR, "summary.csv")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.pt")
TOKENIZER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer_config.json")
VAL_CACHE = os.path.join(PROJECT_ROOT, "dataset_val.pt")

TOKENIZER_BITS = 10
TOKENIZER_VOCAB = 1 << TOKENIZER_BITS

# Batch size — edit for your GPU
BASEMODEL_BATCH_SIZE = 128   # RTX 4090: 128, RTX 4060: 16
EVAL_BATCH_SIZE = 128

# ── Fixed backbone (Phase 3 trial 047) ──
BACKBONE = {
    "dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
    "dsa_windows": [None, 512, 512],
    "position_encoding": "rope", "rope_base": 10000.0,
    "dropout": 0.1323, "lr": 1.08e-3, "weight_decay": 1.4e-5,
    "batch_size": BASEMODEL_BATCH_SIZE, "accumulation_steps": 2, "grad_clip": 0.3,
    "early_stop_patience": 5, "max_epochs": 30,
    "diversity_weight": 0.156, "collapse_weight": 1.36e-5,
    "use_revin": False,
}


# ── Search space ──
def sample_params(trial: optuna.Trial) -> dict:
    return {
        "num_latent_tokens": trial.suggest_categorical("num_latent_tokens", [8, 16, 32]),
        "latent_reasoner_depth": trial.suggest_categorical("latent_reasoner_depth", [2, 4, 6]),
        "latent_cross_heads": trial.suggest_categorical("latent_cross_heads", [2, 4]),
        "num_factor_tokens": trial.suggest_categorical("num_factor_tokens", [0, 2, 4, 8]),
    }


# ── Helpers ──
def _choose_amp_dtype(device):
    if device.type != "cuda": return None
    if torch.cuda.is_bf16_supported(): return torch.bfloat16
    return torch.float16

def _autocast_ctx(amp, dt):
    if not amp: return nullcontext()
    try: return torch.amp.autocast(device_type="cuda", dtype=dt)
    except Exception: return torch.cuda.amp.autocast(dtype=dt)

def _load_tokenizer(device):
    ckpt = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg and os.path.exists(TOKENIZER_CONFIG_PATH):
        with open(TOKENIZER_CONFIG_PATH) as f: cfg = json.load(f)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tok.load_state_dict(ckpt["model_state_dict"], strict=False)
    tok.eval(); tok.requires_grad_(False)
    return tok

def _prepare_batch(features, time_features, tokenizer, device, non_blocking,
                   encoding_coarse=None, encoding_fine=None):
    if encoding_coarse is not None and encoding_fine is not None:
        idx_c = encoding_coarse.to(device, non_blocking=non_blocking)
        idx_f = encoding_fine.to(device, non_blocking=non_blocking)
    else:
        tk_dev = next(tokenizer.parameters()).device
        f_on_dev = features.to(tk_dev, non_blocking=non_blocking)
        with torch.no_grad(): idx_c, idx_f = tokenizer.encode(f_on_dev)
        del f_on_dev
        if tk_dev != device:
            idx_c = idx_c.to(device, non_blocking=non_blocking)
            idx_f = idx_f.to(device, non_blocking=non_blocking)
    return {
        "input_coarse": idx_c[:, :-1].long(), "input_fine": idx_f[:, :-1].long(),
        "target_coarse": idx_c[:, 1:].long(), "target_fine": idx_f[:, 1:].long(),
        "t_min": time_features["minute"][:, :-1].to(device, non_blocking=non_blocking).long(),
        "t_day": time_features["day"][:, :-1].to(device, non_blocking=non_blocking).long(),
        "t_month": time_features["month"][:, :-1].to(device, non_blocking=non_blocking).long(),
        "t_year": time_features["year"][:, :-1].to(device, non_blocking=non_blocking).long(),
    }

def _unpack_batch(batch_data):
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 3:
        return batch_data[0], batch_data[2], batch_data[3]
    return batch_data[0], None, None


# ── Downstream eval ──
_VAL_DATA = None

def _load_val_data():
    payload = torch.load(VAL_CACHE, map_location="cpu", weights_only=False)
    features = payload["features"]
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)
    tf = {}
    for key in ("minute", "day", "month", "year"):
        t = payload["time_features"][key]
        tf[key] = t if isinstance(t, torch.Tensor) else torch.as_tensor(t, dtype=torch.long)
    ss = payload["seq_stats"]
    N = len(ss)
    means = torch.from_numpy(np.array([np.asarray(s["mean"], dtype=np.float32) for s in ss]))
    stds  = torch.from_numpy(np.array([np.asarray(s["std"],  dtype=np.float32) for s in ss]))
    return features, tf, means, stds

def _get_val_data():
    global _VAL_DATA
    if _VAL_DATA is None:
        _VAL_DATA = _load_val_data()
    return _VAL_DATA

@torch.inference_mode()
def _run_downstream_eval(model, tokenizer, device):
    features, tf, means, stds = _get_val_data()
    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    N = features.shape[0]
    all_preds, all_actuals = [], []
    for start in tqdm(range(0, N, EVAL_BATCH_SIZE), desc="  Eval", leave=False):
        end = min(start + EVAL_BATCH_SIZE, N)
        idx = np.arange(start, end)
        bf = features[idx].to(device, non_blocking=True)
        bm = means[idx].to(device, non_blocking=True)
        bs = stds[idx].to(device, non_blocking=True)
        an = bf[:, 1023, 0]
        ic, i_f = tokenizer.encode(bf[:, :1023, :])
        t_min = tf["minute"][idx][:, :1023].to(device, non_blocking=True).long()
        t_day = tf["day"][idx][:, :1023].to(device, non_blocking=True).long()
        t_mon = tf["month"][idx][:, :1023].to(device, non_blocking=True).long()
        t_yr  = tf["year"][idx][:, :1023].to(device, non_blocking=True).long()
        with _autocast_ctx(use_amp, amp_dtype):
            lc, lf, _ = model(ic, i_f, t_min, t_day, t_mon, t_yr, last_only=True)
        pc = lc[:, -1, :].float().argmax(dim=-1)
        pf = lf[:, -1, :].float().argmax(dim=-1)
        dec = tokenizer.decode(pc.unsqueeze(1), pf.unsqueeze(1))
        pred_norm = dec[:, 0, 0]
        plr = pred_norm * bs[:, 0] + bm[:, 0]
        alr = an * bs[:, 0] + bm[:, 0]
        all_preds.append(plr.cpu()); all_actuals.append(alr.cpu())
        del bf, bm, bs, ic, i_f, lc, lf, dec

    preds = torch.cat(all_preds).numpy().astype(np.float64)
    actuals = torch.cat(all_actuals).numpy().astype(np.float64)
    fin = np.isfinite(preds) & np.isfinite(actuals)
    preds, actuals = preds[fin], actuals[fin]
    pr = np.exp(np.clip(preds, -50, 50)); ar = np.exp(np.clip(actuals, -50, 50))
    mape = float(np.mean(np.abs((pr - ar) / np.maximum(np.abs(ar), 1e-4))) * 100)
    ps = np.where(preds >= 0, 1, -1); as_ = np.where(actuals >= 0, 1, -1)
    da = float(np.mean(ps == as_) * 100)
    err = preds - actuals
    return {"mape": round(mape, 6), "da": round(da, 4),
            "mae": round(float(np.mean(np.abs(err))), 6),
            "rmse": round(float(np.sqrt(np.mean(err*err))), 6),
            "num_samples": int(len(preds))}


# ── Training ──
def train_basemodel(aux: dict, tdir: str, tokenizer, device):
    bp = {**BACKBONE, **aux}
    vocab = TOKENIZER_VOCAB
    patience = bp["early_stop_patience"]
    max_epochs = bp["max_epochs"]

    result_path = os.path.join(tdir, "result.json")
    resume_path = os.path.join(tdir, "basemodel_resume.pt")
    ckpt_path = os.path.join(tdir, "basemodel.pt")
    hist_path = os.path.join(tdir, "basemodel_history.json")
    os.makedirs(tdir, exist_ok=True)

    if os.path.exists(result_path):
        with open(result_path) as f: r = json.load(f)
        print(f"  Already done. val_ce={r['best_val_ce']:.4f}")
        return r["best_val_ce"], r.get("downstream", {})

    set_global_seed(42, deterministic=True)
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=bp["batch_size"], include_demo=False,
        loader_overrides={"num_workers": 0, "persistent_workers": False, "pin_memory": True},
    )

    model = KronosReasoningGPT(
        dim=bp["dim"], depth=bp["depth"], heads=bp["heads"],
        num_kv_heads=bp["num_kv_heads"], dsa_windows=bp["dsa_windows"],
        dropout=bp["dropout"], vocab_size_coarse=vocab, vocab_size_fine=vocab,
        position_encoding=bp["position_encoding"], rope_base=bp["rope_base"],
        num_latent_tokens=bp["num_latent_tokens"],
        latent_reasoner_depth=bp["latent_reasoner_depth"],
        latent_cross_heads=bp["latent_cross_heads"],
        num_factor_tokens=bp["num_factor_tokens"],
        use_revin=bp["use_revin"],
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

    start_epoch = 0; best_val_ce = float("inf"); patience_counter = 0
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
            with open(hist_path) as f: history = json.load(f)
        print(f"  Resume epoch {start_epoch}")

    print(f"  Model: {total_p:,} params  latent_tokens={bp['num_latent_tokens']} "
          f"latent_depth={bp['latent_reasoner_depth']} factor={bp['num_factor_tokens']}")

    stopped_epoch = max_epochs
    for epoch in range(start_epoch, max_epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        tl, bd = 0.0, 0
        for bi, batch_data in enumerate(
            tqdm(train_loader, desc=f"  BM {epoch+1}/{max_epochs}", leave=False), start=1
        ):
            fts, tfs, encs = _unpack_batch(batch_data)
            batch = _prepare_batch(fts, tfs, tokenizer, device, non_blocking=True,
                                   encoding_coarse=encs["idx_coarse"] if encs else None,
                                   encoding_fine=encs["idx_fine"] if encs else None)
            del fts, tfs, encs
            with _autocast_ctx(use_amp, amp_dtype):
                lc, lf, ls = model(batch["input_coarse"], batch["input_fine"],
                                   batch["t_min"], batch["t_day"], batch["t_month"], batch["t_year"])
                ce = criterion(lc.reshape(-1, vocab), batch["target_coarse"].reshape(-1))
                ce = ce + criterion(lf.reshape(-1, vocab), batch["target_fine"].reshape(-1))
                lat = torch.tensor(0.0, device=device)
                if ls is not None and ls.shape[0] >= 2:
                    k, B, N, C = ls.shape
                    lat = bp["diversity_weight"] * torch.exp(-(ls[1:]-ls[:-1]).pow(2).sum(-1).sqrt().mean())
                    lat = lat + bp["collapse_weight"] * torch.exp(-ls.reshape(k, B*N, C).var(dim=1).mean())
                sl = (ce + lat) / acc
            if not torch.isfinite(sl): opt.zero_grad(set_to_none=True); del batch, lc, lf, ls, ce, lat, sl; continue
            scaler.scale(sl).backward(); tl += (ce + lat).item(); bd += 1
            if bi % acc == 0 or bi == len(train_loader):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), bp["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            del batch, lc, lf, ls, ce, lat, sl

        model.eval(); vt, vb = 0.0, 0
        with torch.no_grad():
            for batch_data in val_loader:
                fts, tfs, encs = _unpack_batch(batch_data)
                batch = _prepare_batch(fts, tfs, tokenizer, device, non_blocking=True,
                                       encoding_coarse=encs["idx_coarse"] if encs else None,
                                       encoding_fine=encs["idx_fine"] if encs else None)
                del fts, tfs, encs
                with _autocast_ctx(use_amp, amp_dtype):
                    lc, lf, _ = model(batch["input_coarse"], batch["input_fine"],
                                      batch["t_min"], batch["t_day"], batch["t_month"], batch["t_year"])
                    ce = criterion(lc.reshape(-1, vocab), batch["target_coarse"].reshape(-1))
                    ce = ce + criterion(lf.reshape(-1, vocab), batch["target_fine"].reshape(-1))
                vt += ce.item(); vb += 1; del batch, lc, lf, ce
        avg_val = vt / max(vb, 1); sched.step()
        history["train_loss"].append(tl / max(bd, 1))
        history["val_ce"].append(avg_val); history["lr"].append(opt.param_groups[0]["lr"])
        if device.type == "cuda": torch.cuda.empty_cache()

        if avg_val < best_val_ce: best_val_ce = avg_val; patience_counter = 0
        else: patience_counter += 1
        print(f"  BM epoch {epoch+1:2d}/{max_epochs}  train={history['train_loss'][-1]:.4f}  "
              f"val_ce={avg_val:.4f}  best={best_val_ce:.4f}  patience={patience_counter}/{patience}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "best_val_ce": best_val_ce, "patience_counter": patience_counter}, resume_path)
        with open(hist_path, "w") as f: json.dump(history, f, indent=2)
        if patience_counter >= patience: stopped_epoch = epoch + 1; break
        if avg_val < best_val_ce:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_ce": avg_val}, ckpt_path)

    # Load best
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    eval_metrics = _run_downstream_eval(model, tokenizer, device)
    result = {"best_val_ce": round(best_val_ce, 6), "epoch_stopped": stopped_epoch,
              "max_epochs": max_epochs, "params": bp, "downstream": eval_metrics}
    with open(result_path, "w") as f: json.dump(result, f, indent=2)
    if os.path.exists(resume_path): os.remove(resume_path)
    print(f"  MAPE={eval_metrics['mape']:.4f}%  DA={eval_metrics['da']:.2f}%")
    return best_val_ce, eval_metrics


# ── Resume logic (same as Phase 3) ──
def _assign_trial_dir():
    os.makedirs(PHASE4_DIR, exist_ok=True)
    existing = sorted([d for d in os.listdir(PHASE4_DIR) if d.startswith("trial_")],
                      key=lambda x: int(x.split("_")[1]))
    for d in existing:
        full = os.path.join(PHASE4_DIR, d)
        resume = os.path.join(full, "basemodel_resume.pt")
        result = os.path.join(full, "result.json")
        if os.path.exists(resume) and not os.path.exists(result):
            if os.path.exists(os.path.join(full, "config.json")):
                print(f"  Found incomplete trial: {d} — will resume")
                return full
    return os.path.join(PHASE4_DIR, f"trial_{len(existing):03d}")


# ── Optuna ──
_G_TOKENIZER = None

def objective(trial: optuna.Trial) -> float:
    global _G_TOKENIZER
    tdir = _assign_trial_dir()
    os.makedirs(tdir, exist_ok=True)

    aux = sample_params(trial)
    config_path = os.path.join(tdir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f: aux = json.load(f)
    else:
        with open(config_path, "w") as f: json.dump(aux, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Trial {trial.number:03d} (dir={os.path.basename(tdir)})")
    print(f"Params: latent_t={aux['num_latent_tokens']} latent_d={aux['latent_reasoner_depth']} "
          f"cross_h={aux['latent_cross_heads']} factor={aux['num_factor_tokens']}")
    print(f"{'='*60}")

    t0 = time.time()
    val_ce, eval_metrics = train_basemodel(aux, tdir, _G_TOKENIZER, device)
    elapsed = time.time() - t0

    trial.set_user_attr("trial_dir", tdir)
    trial.set_user_attr("dir_name", os.path.basename(tdir))
    trial.set_user_attr("elapsed_min", round(elapsed / 60, 1))
    trial.set_user_attr("mape", eval_metrics["mape"])
    trial.set_user_attr("da", eval_metrics["da"])
    trial.set_user_attr("mae", eval_metrics["mae"])
    trial.set_user_attr("rmse", eval_metrics["rmse"])

    print(f"  val_ce={val_ce:.6f}  MAPE={eval_metrics['mape']:.4f}%  "
          f"DA={eval_metrics['da']:.2f}%  time={elapsed/60:.1f} min")
    return val_ce


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
    os.makedirs(PHASE4_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    ranked = sorted(rows, key=lambda r: r["value"])
    print(f"\nTop-10 by val_ce:")
    for r in ranked[:10]:
        print(f"  Trial {r['trial']:03d}  val_ce={r['value']:.6f}  "
              f"MAPE={r.get('mape','?')}  lt={r.get('num_latent_tokens','?')}  factor={r.get('num_factor_tokens','?')}")
    print(f"Summary: {SUMMARY_CSV}")


def main():
    global _G_TOKENIZER
    if CLEAN_START and os.path.exists(PHASE4_DIR):
        import shutil; shutil.rmtree(PHASE4_DIR)
    os.makedirs(PHASE4_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 4 -- Auxiliary Components HPO")
    print(f"  Backbone: dim=384 depth=3 heads=4/1 dsa=[None,512,512]")
    print(f"  Output:   {PHASE4_DIR}")
    print(f"  Trials:   {N_TRIALS}")
    print(f"  Device:   {device}")
    if device.type == "cuda":
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print()

    print("Loading tokenizer...")
    _G_TOKENIZER = _load_tokenizer(device)

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=f"sqlite:///{STUDY_DB}",
        direction="minimize", load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\nPhase 4 complete.")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best val_ce: {study.best_trial.value:.6f}")
    print(f"  Best params: {json.dumps(study.best_trial.params, indent=4)}")
    export_summary(study)


if __name__ == "__main__":
    main()
