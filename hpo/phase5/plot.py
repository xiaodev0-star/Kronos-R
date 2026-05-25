# -*- coding: utf-8 -*-
"""Phase 5 DA — Comparison plots (LoRA vs Full-FT, BaseModel + Demo).

Usage::

    python -m hpo.phase5.plot
    python -m hpo.phase5.plot --output my_plot.png
"""

from __future__ import annotations

import argparse, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(_PROJECT_ROOT, "trials", "phase5_da")
FT_DIR  = os.path.join(_PROJECT_ROOT, "trials", "phase5_da_ft")

METHOD_ORDER = ["ce", "expo", "dpo", "rsft", "grpo"]
ALL_METHODS = ["basemodel"] + METHOD_ORDER
LABELS = {"basemodel": "Base", "ce": "CE", "expo": "ExPO",
          "dpo": "DPO", "rsft": "RSFT", "grpo": "GRPO"}
COLORS = {
    "basemodel": "#333333", "ce": "#999999", "expo": "#E69F00",
    "dpo": "#0072B2", "rsft": "#009E73", "grpo": "#CC79A7",
}
MARKERS = {"basemodel": "*", "ce": "s", "expo": "^", "dpo": "o", "rsft": "D", "grpo": "P"}


def _load_dir(base_dir):
    """Load histories, results, demo_eval from a directory."""
    histories = {}
    results = {}
    for method in METHOD_ORDER:
        hist_path = os.path.join(base_dir, method, f"phase5_{method}_history.json")
        result_path = os.path.join(base_dir, method, "result.json")
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                histories[method] = json.load(f)
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results[method] = json.load(f)
    demo_eval = {}
    demo_path = os.path.join(base_dir, "demo_eval_results.json")
    if os.path.exists(demo_path):
        with open(demo_path, "r", encoding="utf-8") as f:
            demo_eval = json.load(f)
    return histories, results, demo_eval


def _int_ticks(ax):
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def _best_ba(method, histories, demo_eval):
    if method == "basemodel":
        return demo_eval.get("val_metrics", {}).get("basemodel", {}).get("balanced_accuracy", 0)
    h = histories.get(method, [])
    return max((r["val"].get("balanced_accuracy", 0) for r in h), default=0)


def _demo_ba(method, demo_eval):
    return demo_eval.get("demo_metrics", {}).get(method, {}).get("balanced_accuracy", 0)


def _best_da(method, histories, demo_eval):
    if method == "basemodel":
        return demo_eval.get("val_metrics", {}).get("basemodel", {}).get("direction_accuracy", 0)
    h = histories.get(method, [])
    return max((r["val"].get("direction_accuracy", 0) for r in h), default=0)


def _demo_da(method, demo_eval):
    return demo_eval.get("demo_metrics", {}).get(method, {}).get("direction_accuracy", 0)


def plot_comparison(output_path=None):
    h_lo, r_lo, d_lo = _load_dir(OUT_DIR)
    h_ft, r_ft, d_ft = _load_dir(FT_DIR)

    available_lo = [m for m in METHOD_ORDER if m in h_lo]
    available_ft = [m for m in METHOD_ORDER if m in h_ft]
    has_demo_lo = bool(d_lo.get("demo_metrics"))
    has_demo_ft = bool(d_ft.get("demo_metrics"))

    if not available_lo and not available_ft:
        print("No results found.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle("Phase 5 DA: LoRA vs Full-FT — Val + Demo",
                 fontsize=14, fontweight="bold", y=0.98)

    # ═══════════════════════════════════════════════
    # Row 0, Col 0: BalAcc Trajectory (Val) — LoRA solid, FT dashed
    # ═══════════════════════════════════════════════
    ax = axes[0, 0]
    for m in available_lo:
        eps = [r["epoch"] for r in h_lo[m]]
        ba = [r["val"]["balanced_accuracy"] for r in h_lo[m]]
        ax.plot(eps, ba, color=COLORS[m], marker=MARKERS[m], markersize=5,
                linewidth=2, linestyle="-", label=f"{LABELS[m]} LoRA")
    for m in available_ft:
        eps = [r["epoch"] for r in h_ft[m]]
        ba = [r["val"]["balanced_accuracy"] for r in h_ft[m]]
        ax.plot(eps, ba, color=COLORS[m], marker=MARKERS[m], markersize=4,
                linewidth=1.5, linestyle="--", alpha=0.7, label=f"{LABELS[m]} FT")
    # BaseModel reference
    bm_val_ba = _best_ba("basemodel", h_lo, d_lo)
    if bm_val_ba:
        ax.axhline(bm_val_ba, color=COLORS["basemodel"], linestyle=":",
                   linewidth=1.0, alpha=0.6, label=f"Base ({bm_val_ba:.4f})")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Balanced Accuracy")
    ax.set_title("BalAcc Trajectory (Val)"); ax.legend(fontsize=5.5, ncol=2); ax.grid(True, alpha=0.25)
    _int_ticks(ax)

    # ═══════════════════════════════════════════════
    # Row 0, Col 1: DA Trajectory (Val)
    # ═══════════════════════════════════════════════
    ax = axes[0, 1]
    for m in available_lo:
        eps = [r["epoch"] for r in h_lo[m]]
        da = [r["val"]["direction_accuracy"] for r in h_lo[m]]
        ax.plot(eps, da, color=COLORS[m], marker=MARKERS[m], markersize=5,
                linewidth=2, linestyle="-", label=f"{LABELS[m]} LoRA")
    for m in available_ft:
        eps = [r["epoch"] for r in h_ft[m]]
        da = [r["val"]["direction_accuracy"] for r in h_ft[m]]
        ax.plot(eps, da, color=COLORS[m], marker=MARKERS[m], markersize=4,
                linewidth=1.5, linestyle="--", alpha=0.7, label=f"{LABELS[m]} FT")
    bm_val_da = _best_da("basemodel", h_lo, d_lo)
    if bm_val_da:
        ax.axhline(bm_val_da, color=COLORS["basemodel"], linestyle=":",
                   linewidth=1.0, alpha=0.6, label=f"Base ({bm_val_da:.4f})")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Direction Accuracy")
    ax.set_title("DA Trajectory (Val)"); ax.legend(fontsize=5.5, ncol=2); ax.grid(True, alpha=0.25)
    _int_ticks(ax)

    # ═══════════════════════════════════════════════
    # Row 0, Col 2: Demo BalAcc — LoRA vs FT grouped bars
    # ═══════════════════════════════════════════════
    ax = axes[0, 2]
    x = np.arange(len(METHOD_ORDER))
    w = 0.25
    lo_vals = [_demo_ba(m, d_lo) for m in METHOD_ORDER]
    ft_vals = [_demo_ba(m, d_ft) for m in METHOD_ORDER]
    bm_demo_ba = _demo_ba("basemodel", d_lo)

    ax.bar(x - w, lo_vals, w, color=[COLORS[m] for m in METHOD_ORDER],
           edgecolor="white", linewidth=0.5, label="LoRA Demo BalAcc")
    ax.bar(x, ft_vals, w, color=[COLORS[m] for m in METHOD_ORDER],
           edgecolor="black", linewidth=0.8, alpha=0.5, hatch="///", label="FT Demo BalAcc")
    if bm_demo_ba:
        ax.axhline(bm_demo_ba, color=COLORS["basemodel"], linestyle="--",
                   linewidth=1.2, alpha=0.8, label=f"BaseModel ({bm_demo_ba:.4f})")
    for i, (lv, fv) in enumerate(zip(lo_vals, ft_vals)):
        ax.text(i - w, lv + 0.002, f"{lv:.4f}", ha="center", fontsize=6, fontweight="bold", rotation=90)
        ax.text(i, fv + 0.002, f"{fv:.4f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Balanced Accuracy"); ax.set_title("Demo BalAcc: LoRA vs Full-FT")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.2, axis="y")

    # ═══════════════════════════════════════════════
    # Row 1, Col 0: Demo DA — LoRA vs FT grouped bars
    # ═══════════════════════════════════════════════
    ax = axes[1, 0]
    lo_da_vals = [_demo_da(m, d_lo) for m in METHOD_ORDER]
    ft_da_vals = [_demo_da(m, d_ft) for m in METHOD_ORDER]
    bm_demo_da = _demo_da("basemodel", d_lo)

    ax.bar(x - w, lo_da_vals, w, color=[COLORS[m] for m in METHOD_ORDER],
           edgecolor="white", linewidth=0.5, label="LoRA Demo DA")
    ax.bar(x, ft_da_vals, w, color=[COLORS[m] for m in METHOD_ORDER],
           edgecolor="black", linewidth=0.8, alpha=0.5, hatch="///", label="FT Demo DA")
    if bm_demo_da:
        ax.axhline(bm_demo_da, color=COLORS["basemodel"], linestyle="--",
                   linewidth=1.2, alpha=0.8, label=f"BaseModel ({bm_demo_da:.4f})")
    for i, (lv, fv) in enumerate(zip(lo_da_vals, ft_da_vals)):
        ax.text(i - w, lv + 0.002, f"{lv:.4f}", ha="center", fontsize=6, fontweight="bold", rotation=90)
        ax.text(i, fv + 0.002, f"{fv:.4f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Direction Accuracy"); ax.set_title("Demo DA: LoRA vs Full-FT")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.2, axis="y")

    # ═══════════════════════════════════════════════
    # Row 1, Col 1: Val vs Demo gap (overfitting indicator)
    # ═══════════════════════════════════════════════
    ax = axes[1, 1]
    lo_gaps = []; ft_gaps = []
    for m in METHOD_ORDER:
        v_ba = _best_ba(m, h_lo, d_lo) if m in h_lo else 0
        d_ba = _demo_ba(m, d_lo)
        lo_gaps.append(d_ba - v_ba)
        v_ba_ft = _best_ba(m, h_ft, d_ft) if m in h_ft else 0
        d_ba_ft = _demo_ba(m, d_ft)
        ft_gaps.append(d_ba_ft - v_ba_ft)

    ax.bar(x - w/2, lo_gaps, w, color=[COLORS[m] for m in METHOD_ORDER],
           edgecolor="white", linewidth=0.5, label="LoRA (Demo-Val)")
    ax.bar(x + w/2, ft_gaps, w, color=[COLORS[m] for m in METHOD_ORDER],
           edgecolor="black", linewidth=0.8, alpha=0.5, hatch="///", label="FT (Demo-Val)")
    ax.axhline(0, color="black", linewidth=0.8)
    # Shade overfitting zone
    ax.axhspan(-0.03, -0.005, alpha=0.08, color="red")
    for i, (lg, fg) in enumerate(zip(lo_gaps, ft_gaps)):
        ax.text(i - w/2, lg + (0.002 if lg >= 0 else -0.006), f"{lg:+.3f}", ha="center", fontsize=6, rotation=90)
        ax.text(i + w/2, fg + (0.002 if fg >= 0 else -0.006), f"{fg:+.3f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Demo - Val BalAcc"); ax.set_title("Overfitting Check (Demo-Val Gap)")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.2, axis="y")

    # ═══════════════════════════════════════════════
    # Row 1, Col 2: Prediction distribution (LoRA final epoch, %)
    # ═══════════════════════════════════════════════
    ax = axes[1, 2]
    up_pcts = []; down_pcts = []
    for m in METHOD_ORDER:
        if m in h_lo:
            preds = h_lo[m][-1]["val"].get("pred_counts", {})
        else:
            preds = {}
        total = preds.get("up", 0) + preds.get("down", 0) + preds.get("flat", 0)
        up_pcts.append(preds.get("up", 0) / max(1, total) * 100)
        down_pcts.append(preds.get("down", 0) / max(1, total) * 100)

    x2 = np.arange(len(METHOD_ORDER))
    w2 = 0.35
    ax.bar(x2 - w2/2, down_pcts, w2, color="#D55E00", edgecolor="white", linewidth=0.8, label="Down %")
    ax.bar(x2 + w2/2, up_pcts, w2, color="#56B4E9", edgecolor="white", linewidth=0.8, label="Up %")
    for i in range(len(METHOD_ORDER)):
        ax.text(i - w2/2, down_pcts[i] + 1, f"{down_pcts[i]:.0f}%", ha="center", fontsize=7, color="#D55E00")
        ax.text(i + w2/2, up_pcts[i] + 1, f"{up_pcts[i]:.0f}%", ha="center", fontsize=7, color="#0072B2")
    ax.set_xticks(x2); ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Prediction Share (%)"); ax.set_title("Pred Distribution (LoRA Val, final)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 80)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_path or os.path.join(OUT_DIR, "phase5_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 DA — Plot comparison")
    p.add_argument("--output", type=str, default="",
                   help="Output path for plot PNG")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_comparison(args.output if args.output else None)
