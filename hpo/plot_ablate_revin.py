"""RevIN ablation comparison plot.

Usage:
    python -m hpo.plot_ablate_revin
"""

import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATE_DIR = os.path.join(PROJECT_ROOT, "trials", "ablate_revin")
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "trials", "ablate_revin", "comparison.png")


def load():
    results = {}
    for label, fname in [("ON", "revin_on"), ("OFF", "revin_off")]:
        path = os.path.join(ABLATE_DIR, fname, "result.json")
        if os.path.exists(path):
            with open(path) as f:
                results[label] = json.load(f)
    return results


def plot(results):
    metrics = ["mape", "da", "mae", "rmse"]
    labels = ["MAPE (%)", "DA (%)", "MAE", "RMSE"]
    on_vals = [results["ON"]["downstream"][m] for m in metrics]
    off_vals = [results["OFF"]["downstream"][m] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("RevIN Ablation: ON vs OFF", fontsize=14, fontweight="bold")

    # Panel 1: Grouped bar chart
    ax = axes[0]
    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, on_vals, w, color="#2196F3", alpha=0.85, label="RevIN=ON")
    bars2 = ax.bar(x + w/2, off_vals, w, color="#FF9800", alpha=0.85, label="RevIN=OFF")
    for bar, val in zip(bars1, on_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(on_vals)*0.01,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold", color="#1565C0")
    for bar, val in zip(bars2, off_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(off_vals)*0.01,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold", color="#E65100")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("A. 1-Step Prediction Metrics", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # Panel 2: MAPE delta and val_ce comparison
    ax = axes[1]
    mape_delta = off_vals[0] - on_vals[0]

    # MAPE comparison
    ax.barh(1, on_vals[0], color="#2196F3", alpha=0.85, label="RevIN=ON")
    ax.barh(2, off_vals[0], color="#FF9800", alpha=0.85, label="RevIN=OFF")

    # Annotate
    ax.text(on_vals[0] + 0.002, 1, f"{on_vals[0]:.4f}%", va="center", fontsize=11, fontweight="bold", color="#1565C0")
    ax.text(off_vals[0] + 0.002, 2, f"{off_vals[0]:.4f}%", va="center", fontsize=11, fontweight="bold", color="#E65100")

    ax.set_yticks([1, 2])
    ax.set_yticklabels(["RevIN=ON", "RevIN=OFF"], fontsize=11)
    ax.set_xlabel("MAPE (%)", fontsize=10)
    delta_str = f"+{mape_delta:.4f}pp" if mape_delta > 0 else f"{mape_delta:.4f}pp"
    ax.set_title(f"B. MAPE Comparison  (delta = {delta_str})", fontsize=12, fontweight="bold")

    # Add val_ce annotation
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.scatter([results["ON"]["best_val_ce"]], [1], marker="|", s=200, color="#1565C0", linewidth=2)
    ax2.scatter([results["OFF"]["best_val_ce"]], [2], marker="|", s=200, color="#E65100", linewidth=2)
    ax2.set_xlabel("val_ce (tick marks)", fontsize=9, color="gray")
    ax2.tick_params(axis="x", labelcolor="gray")

    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PNG}")


def main():
    results = load()
    if len(results) < 2:
        print("Need both revin_on and revin_off results.")
        return
    plot(results)


if __name__ == "__main__":
    main()
