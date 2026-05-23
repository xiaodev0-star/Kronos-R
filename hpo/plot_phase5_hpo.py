# -*- coding: utf-8 -*-
"""Phase 5 HPO plots: DA, BalAcc, MAPE trajectories + best-epoch comparison."""

import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(_PROJECT_ROOT, "trials", "phase5_da")
PLOT_PATH = os.path.join(_PROJECT_ROOT, "trials", "phase5_hpo_plots.png")

METHODS = ["ce_best", "expo_best", "dpo_best", "rsft_best"]
LABELS = {"ce_best": "CE", "expo_best": "ExPO", "dpo_best": "DPO", "rsft_best": "RSFT"}
COLORS = {"ce_best": "#999999", "expo_best": "#E69F00", "dpo_best": "#0072B2", "rsft_best": "#009E73"}
MARKERS = {"ce_best": "s", "expo_best": "^", "dpo_best": "o", "rsft_best": "D"}

os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)


def load_histories():
    histories = {}
    final_vals = {}
    for m in METHODS:
        path = os.path.join(OUT_DIR, m, "history.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                histories[m] = json.load(f)
        result_path = os.path.join(OUT_DIR, m, "result.json")
        if os.path.exists(result_path):
            with open(result_path, encoding="utf-8") as f:
                r = json.load(f)
                final_vals[m] = r
    return histories, final_vals


def plot_all():
    histories, final_vals = load_histories()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Phase 5 HPO: Token-Space DA Post-Training — Best Config Comparison",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── 1. DA trajectory ──
    ax = axes[0, 0]
    for m in METHODS:
        if m not in histories: continue
        epochs = [r["epoch"] for r in histories[m]]
        da = [r["val"]["direction_accuracy"] for r in histories[m]]
        ax.plot(epochs, da, color=COLORS[m], marker=MARKERS[m], markersize=5,
                linewidth=2, label=LABELS[m])
    ax.axhline(0.50, color="black", linestyle="--", linewidth=0.8, alpha=0.4, label="random")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Direction Accuracy")
    ax.set_title("DA Trajectory"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    ax.set_ylim(0.48, 0.54)

    # ── 2. BalAcc trajectory ──
    ax = axes[0, 1]
    for m in METHODS:
        if m not in histories: continue
        epochs = [r["epoch"] for r in histories[m]]
        ba = [r["val"]["balanced_accuracy"] for r in histories[m]]
        ax.plot(epochs, ba, color=COLORS[m], marker=MARKERS[m], markersize=5,
                linewidth=2, label=LABELS[m])
    ax.axhline(0.50, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Balanced Accuracy")
    ax.set_title("BalAcc Trajectory"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    ax.set_ylim(0.48, 0.54)

    # ── 3. MAPE trajectory ──
    ax = axes[0, 2]
    for m in METHODS:
        if m not in histories: continue
        epochs = [r["epoch"] for r in histories[m]]
        mape = [r["val"]["mape"] for r in histories[m]]
        ax.plot(epochs, mape, color=COLORS[m], marker=MARKERS[m], markersize=5,
                linewidth=2, label=LABELS[m])
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAPE")
    ax.set_title("MAPE Trajectory"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # ── 4. Best DA / BalAcc bar chart ──
    ax = axes[1, 0]
    x = np.arange(len(METHODS))
    w = 0.35
    best_da = [max(r["val"]["direction_accuracy"] for r in histories[m]) if m in histories else 0
               for m in METHODS]
    best_ba = [max(r["val"]["balanced_accuracy"] for r in histories[m]) if m in histories else 0
               for m in METHODS]
    bars1 = ax.bar(x - w/2, best_da, w, color=[COLORS[m] for m in METHODS],
                   edgecolor="white", linewidth=0.8, label="Best DA")
    bars2 = ax.bar(x + w/2, best_ba, w, color=[COLORS[m] for m in METHODS],
                   edgecolor="white", linewidth=0.8, alpha=0.5, label="Best BalAcc")
    for bar, val in zip(bars1, best_da):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontsize=7, fontweight="bold")
    for bar, val in zip(bars2, best_ba):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylabel("Accuracy"); ax.set_title("Best Epoch DA vs BalAcc")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0.48, 0.55)

    # ── 5. MAPE bar chart ──
    ax = axes[1, 1]
    best_mape = []
    for m in METHODS:
        if m not in histories: best_mape.append(0); continue
        # MAPE at best-DA epoch
        ba_list = [r["val"]["balanced_accuracy"] for r in histories[m]]
        best_idx = ba_list.index(max(ba_list))
        best_mape.append(histories[m][best_idx]["val"]["mape"])
    bars = ax.bar(x, best_mape, color=[COLORS[m] for m in METHODS],
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, best_mape):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylabel("MAPE"); ax.set_title("MAPE at Best BalAcc Epoch")
    ax.grid(True, alpha=0.2, axis="y")

    # ── 6. Up/Down prediction distribution (final epoch) ──
    ax = axes[1, 2]
    up_preds = []
    down_preds = []
    for m in METHODS:
        if m not in histories: up_preds.append(0); down_preds.append(0); continue
        fv = histories[m][-1]["val"]
        up_preds.append(fv["pred_counts"]["up"])
        down_preds.append(fv["pred_counts"]["down"])
    w2 = 0.35
    ax.bar(x - w2/2, down_preds, w2, color="#D55E00", edgecolor="white",
           linewidth=0.8, label="Down preds")
    ax.bar(x + w2/2, up_preds, w2, color="#56B4E9", edgecolor="white",
           linewidth=0.8, label="Up preds")
    ax.axhline(5942/2, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylabel("Count"); ax.set_title("Final Prediction Distribution (total=5942)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {PLOT_PATH}")
    return PLOT_PATH


if __name__ == "__main__":
    plot_all()
