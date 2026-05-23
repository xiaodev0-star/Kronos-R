"""Phase 2 Supplement: lower embedding_dim boundary.

Top-5 trials all hit embedding_dim=48 (search lower bound).
Tests embedding_dim ∈ [16, 24, 32, 48] to find the true optimum.

Reuses all training infrastructure from phase2_tokenizer.
Saves under trials/phase2_tokenizer_sup/ (separate from main).

Usage:
    python -m hpo.phase2_sup
"""

from __future__ import annotations

import json, os, sys, time
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import optuna
import torch

# Reuse everything from phase2_tokenizer
from hpo.phase2_tokenizer import (
    train_tokenizer, train_basemodel,
    TOKENIZER_BATCH_SIZE, BASEMODEL_BATCH_SIZE,
    TOKENIZER_FIXED, BASEMODEL_PARAMS,
)

# ── Override config ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE2_SUP_DIR = os.path.join(PROJECT_ROOT, "trials", "phase2_tokenizer_sup")
STUDY_DB = os.path.join(PHASE2_SUP_DIR, "study.db")
SUMMARY_CSV = os.path.join(PHASE2_SUP_DIR, "summary.csv")

BITS = 10
N_TRIALS = 20
STUDY_NAME = "phase2_tokenizer_sup"
CLEAN_START = False

# Narrower, targeted search — embedding_dim is the focus
SEARCH_SPACE = {
    "hidden_dim":         [128, 192, 256],           # keep proven good range
    "embedding_dim":      [16, 24, 32, 48],          # ← lower boundary focus
    "bsq_commitment_cost": (0.05, 0.25),              # strongest signal, shifted up
    "learning_rate":       (5e-5, 3e-4),
}

FIXED_ENTROPY_WEIGHT = 0.01  # r=0.001 — no impact, fix to low value


def trial_dir(trial_number: int) -> str:
    return os.path.join(PHASE2_SUP_DIR, f"trial_{trial_number:03d}")


def sample_params(trial: optuna.Trial) -> dict:
    hd = trial.suggest_categorical("hidden_dim", SEARCH_SPACE["hidden_dim"])
    ed_raw = trial.suggest_categorical("embedding_dim", SEARCH_SPACE["embedding_dim"])
    ed = min(ed_raw, hd)
    commit = trial.suggest_float("bsq_commitment_cost", *SEARCH_SPACE["bsq_commitment_cost"], log=True)
    lr     = trial.suggest_float("lr_tokenizer",        *SEARCH_SPACE["learning_rate"],       log=True)
    return {
        **TOKENIZER_FIXED,
        "bits_per_quantizer": BITS,
        "hidden_dim": hd, "embedding_dim_raw": ed_raw, "embedding_dim": ed,
        "bsq_commitment_cost": round(commit, 6),
        "bsq_entropy_weight":  FIXED_ENTROPY_WEIGHT,
        "learning_rate": round(lr, 8),
        "batch_size": TOKENIZER_BATCH_SIZE,
    }


def objective(trial: optuna.Trial) -> float:
    t_number = trial.number
    tdir = trial_dir(t_number)
    os.makedirs(tdir, exist_ok=True)

    params = sample_params(trial)
    with open(os.path.join(tdir, "config.json"), "w") as f:
        json.dump(params, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Trial {t_number:03d} -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Params: hd={params['hidden_dim']} ed={params['embedding_dim']} "
          f"commit={params['bsq_commitment_cost']} lr={params['learning_rate']} "
          f"(entropy={FIXED_ENTROPY_WEIGHT} fixed)")
    print(f"{'='*60}")

    train_tokenizer(params, tdir, device)
    t0 = time.time()
    val_ce = train_basemodel(tdir, device)
    elapsed = time.time() - t0

    trial.set_user_attr("trial_dir", tdir)
    trial.set_user_attr("elapsed_min", round(elapsed / 60, 1))
    trial.set_user_attr("actual_embedding_dim", params["embedding_dim"])

    ckpt_path = os.path.join(tdir, "basemodel_resume.pt")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"  Score (val_ce): {val_ce:.6f}  time: {elapsed/60:.1f} min")
    return val_ce


def export_summary(study: optuna.Study):
    import csv
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
    os.makedirs(PHASE2_SUP_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    sorted_rows = sorted(rows, key=lambda r: r["value"])
    print(f"\nTop-10 by val_ce:")
    for r in sorted_rows[:10]:
        print(f"  Trial {r['trial']:03d}  val_ce={r['value']:.6f}  "
              f"hd={r.get('hidden_dim','?')}  ed={r.get('embedding_dim','?')}  "
              f"commit={r.get('bsq_commitment_cost','?')}")
    print(f"Summary: {SUMMARY_CSV}")


def main():
    if CLEAN_START and os.path.exists(PHASE2_SUP_DIR):
        import shutil; shutil.rmtree(PHASE2_SUP_DIR)
    os.makedirs(PHASE2_SUP_DIR, exist_ok=True)

    print(f"Phase 2 Supplement — Lower embedding_dim ({SEARCH_SPACE['embedding_dim']})")
    print(f"  Output: {PHASE2_SUP_DIR}")
    print(f"  Trials: {N_TRIALS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print()

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=f"sqlite:///{STUDY_DB}",
        direction="minimize", load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\nPhase 2 Supplement complete.")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best val_ce: {study.best_trial.value:.6f}")
    print(f"  Best params: {json.dumps(study.best_trial.params, indent=4)}")
    export_summary(study)


if __name__ == "__main__":
    main()
