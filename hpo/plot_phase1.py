"""Phase 1 plots — codebook size vs model learnability + downstream metrics.

Usage:
    python -m hpo.plot_phase1
"""

import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "trials", "phase1_bits_search")
RESULTS_SUP_DIR = os.path.join(PROJECT_ROOT, "trials", "phase1_bits_search_sup")
EVAL_DIR = os.path.join(PROJECT_ROOT, "trials", "phase1_evaluate")
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "trials", "phase1_plots.png")
OUTPUT_EVAL_PNG = os.path.join(PROJECT_ROOT, "trials", "phase1_plots.png")


def load_results():
    results = []
    for bits in range(6, 13):
        rpath = os.path.join(RESULTS_DIR, f"bits_{bits:02d}", "result.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                results.append(json.load(f))
    for bits in range(3, 6):
        rpath = os.path.join(RESULTS_SUP_DIR, f"bits_{bits:02d}", "result.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                results.append(json.load(f))
    return sorted(results, key=lambda r: r["bits"])


def load_eval_results():
    """Load 1-step evaluation results (with bootstrap std)."""
    results = []
    for bits in range(3, 13):
        epath = os.path.join(EVAL_DIR, f"bits_{bits:02d}.json")
        if os.path.exists(epath):
            with open(epath) as f:
                results.append(json.load(f))
    return sorted(results, key=lambda r: r["bits"])


def plot(results):
    """Original 4-panel Phase 1 plot (unchanged)."""
    bits_arr = np.array([r["bits"] for r in results])
    vocabs    = np.array([r["vocab_size"] for r in results])
    val_ce    = np.array([r["best_val_ce"] for r in results])
    c_util    = np.array([r["token_metrics"]["coarse"]["utilization"] for r in results])
    f_util    = np.array([r["token_metrics"]["fine"]["utilization"] for r in results])
    c_dead    = np.array([r["token_metrics"]["coarse"]["dead_tokens"] for r in results])
    f_dead    = np.array([r["token_metrics"]["fine"]["dead_tokens"] for r in results])
    c_used    = vocabs - c_dead
    f_used    = vocabs - f_dead

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Phase 1: Codebook Size vs Model Learnability", fontsize=14, fontweight="bold")

    # ── Panel A: Absolute token usage ──
    ax = axes[0, 0]
    x = np.arange(len(bits_arr))
    w = 0.30
    ax.bar(x, vocabs, w * 2.2, color="#E0E0E0", edgecolor="#BDBDBD", linewidth=0.5, label="vocab_size (available)")
    ax.bar(x - w/2, c_used, w, color="#4CAF50", alpha=0.85, label="coarse used")
    ax.bar(x + w/2, f_used, w, color="#FF9800", alpha=0.85, label="fine used")
    for i in range(len(bits_arr)):
        ax.text(x[i] - w/2, c_used[i] + max(vocabs)*0.015, str(c_used[i]),
                ha="center", fontsize=7.5, fontweight="bold", color="#2E7D32")
        ax.text(x[i] + w/2, f_used[i] + max(vocabs)*0.015, str(f_used[i]),
                ha="center", fontsize=7.5, fontweight="bold", color="#E65100")
    ax.set_xticks(x)
    ax.set_xticklabels([f"bits={b}\nvocab={2**b}" for b in bits_arr], fontsize=7.5)
    ax.set_ylabel("Number of Tokens", fontsize=11)
    ax.set_title("A. How Many Tokens Can the Model Actually Use?", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(vocabs) * 1.18)

    # ── Panel B: Dead token waste ratio ──
    ax = axes[0, 1]
    c_dead_pct = c_dead / vocabs * 100
    f_dead_pct = f_dead / vocabs * 100
    ax.plot(bits_arr, c_dead_pct, "s-", color="#E91E63", linewidth=2.2, markersize=8,
            markerfacecolor="white", markeredgewidth=2, label="coarse dead %")
    ax.plot(bits_arr, f_dead_pct, "^-", color="#9C27B0", linewidth=2.2, markersize=8,
            markerfacecolor="white", markeredgewidth=2, label="fine dead %")
    for i in range(len(bits_arr)):
        ax.annotate(f"{c_dead_pct[i]:.0f}%", (bits_arr[i], c_dead_pct[i]),
                    textcoords="offset points", xytext=(0, -14),
                    fontsize=7.5, ha="center", color="#C62828")
        ax.annotate(f"{f_dead_pct[i]:.0f}%", (bits_arr[i], f_dead_pct[i]),
                    textcoords="offset points", xytext=(0, -14),
                    fontsize=7.5, ha="center", color="#6A1B9A")
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(bits_arr[-1] - 0.3, 52, "50% dead", fontsize=8, color="red", alpha=0.7, ha="right")
    ax.set_xlabel("bits_per_quantizer", fontsize=11)
    ax.set_ylabel("Dead Tokens (% of vocab)", fontsize=11)
    ax.set_title("B. Vocab Waste — % of Tokens Never Predicted", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 105)

    # ── Panel C: val_ce ──
    ax = axes[1, 0]
    ax.plot(bits_arr, val_ce, "o-", color="#2196F3", linewidth=2.5, markersize=10,
            markerfacecolor="white", markeredgewidth=2.5)
    for i in range(len(bits_arr)):
        ax.annotate(f"{val_ce[i]:.2f}", (bits_arr[i], val_ce[i]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=8, ha="center", fontweight="bold", color="#1565C0")
    ax.set_xlabel("bits_per_quantizer", fontsize=11)
    ax.set_ylabel("Validation CE Loss", fontsize=11)
    ax.set_title("C. Prediction Cost of Larger Codebook (val_ce)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax2_c = ax.twiny()
    ax2_c.set_xlim(ax.get_xlim())
    ax2_c.set_xticks(bits_arr)
    ax2_c.set_xticklabels([f"v={2**b}" for b in bits_arr], fontsize=7)
    ax2_c.set_xlabel("vocabulary size", fontsize=9)

    # ── Panel D: Efficiency frontier ──
    ax = axes[1, 1]
    avg_used = (c_used + f_used) / 2
    for i in range(len(bits_arr)):
        ax.scatter(avg_used[i], val_ce[i], s=vocabs[i]*0.5 + 80,
                   color=plt.cm.viridis(i / (len(bits_arr) - 1)),
                   edgecolors="black", linewidth=0.8, zorder=5)
        ax.annotate(f"bits={bits_arr[i]}", (avg_used[i], val_ce[i]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, fontweight="bold")
    ax.set_xlabel("Average Tokens Actually Used (coarse + fine) / 2", fontsize=11)
    ax.set_ylabel("Validation CE Loss", fontsize=11)
    ax.set_title("D. Efficiency Frontier: More Tokens → Higher Cost", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PNG}")


def plot_downstream(phase1_results, eval_results):
    """New plot: downstream 1-step MAPE & DA with bootstrap error bars."""
    if not eval_results:
        print("No evaluation results found, skipping downstream plot.")
        return

    p1_by_bits = {r["bits"]: r for r in phase1_results}
    ev_by_bits = {r["bits"]: r for r in eval_results}

    bits_all = sorted(ev_by_bits.keys())
    bits_arr = np.array(bits_all)
    vocabs = np.array([1 << b for b in bits_all])

    # Phase 1 metrics
    c_used = np.array([p1_by_bits[b]["vocab_size"] - p1_by_bits[b]["token_metrics"]["coarse"]["dead_tokens"]
                       for b in bits_all], dtype=float)
    f_used = np.array([p1_by_bits[b]["vocab_size"] - p1_by_bits[b]["token_metrics"]["fine"]["dead_tokens"]
                       for b in bits_all], dtype=float)
    c_dead_pct = np.array([
        p1_by_bits[b]["token_metrics"]["coarse"]["dead_tokens"] / p1_by_bits[b]["vocab_size"] * 100
        for b in bits_all])
    f_dead_pct = np.array([
        p1_by_bits[b]["token_metrics"]["fine"]["dead_tokens"] / p1_by_bits[b]["vocab_size"] * 100
        for b in bits_all])
    val_ce = np.array([p1_by_bits[b]["best_val_ce"] for b in bits_all])

    # Downstream metrics with bootstrap std
    mape_vals = np.array([ev_by_bits[b]["mape"] for b in bits_all])
    mape_stds = np.array([ev_by_bits[b].get("mape_std", 0) for b in bits_all])
    da_vals = np.array([ev_by_bits[b]["da"] for b in bits_all])
    da_stds = np.array([ev_by_bits[b].get("da_std", 0) for b in bits_all])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Phase 1: Bits Search — Token Health & Downstream Prediction",
                 fontsize=14, fontweight="bold")

    # ── Panel A: Absolute token usage ──
    ax = axes[0, 0]
    x = np.arange(len(bits_all))
    w = 0.30
    ax.bar(x, vocabs, w * 2.2, color="#E0E0E0", edgecolor="#BDBDBD", linewidth=0.5, label="vocab_size")
    ax.bar(x - w/2, c_used, w, color="#4CAF50", alpha=0.85, label="coarse used")
    ax.bar(x + w/2, f_used, w, color="#FF9800", alpha=0.85, label="fine used")
    # Only annotate coarse/fine used values when bars are tall enough to avoid overlap
    for i in range(len(bits_all)):
        if c_used[i] > max(vocabs) * 0.08:
            ax.text(x[i] - w/2, c_used[i] + max(vocabs)*0.015, str(int(c_used[i])),
                    ha="center", fontsize=7, fontweight="bold", color="#2E7D32")
        if f_used[i] > max(vocabs) * 0.08:
            ax.text(x[i] + w/2, f_used[i] + max(vocabs)*0.015, str(int(f_used[i])),
                    ha="center", fontsize=7, fontweight="bold", color="#E65100")
    ax.set_xticks(x)
    ax.set_xticklabels([f"b={b}" for b in bits_all], fontsize=8)
    ax.set_ylabel("Tokens", fontsize=10)
    ax.set_title("A. Token Usage", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(vocabs) * 1.18)

    # ── Panel B: Dead token % ──
    ax = axes[0, 1]
    ax.plot(bits_arr, c_dead_pct, "s-", color="#E91E63", linewidth=2.2, markersize=8,
            markerfacecolor="white", markeredgewidth=2, label="coarse dead %")
    ax.plot(bits_arr, f_dead_pct, "^-", color="#9C27B0", linewidth=2.2, markersize=8,
            markerfacecolor="white", markeredgewidth=2, label="fine dead %")
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel("bits_per_quantizer", fontsize=10)
    ax.set_ylabel("Dead Tokens (%)", fontsize=10)
    ax.set_title("B. Vocab Waste", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 105)

    # ── Panel C: val_ce ──
    ax = axes[0, 2]
    ax.plot(bits_arr, val_ce, "o-", color="#2196F3", linewidth=2.5, markersize=10,
            markerfacecolor="white", markeredgewidth=2.5)
    for i in range(len(bits_all)):
        ax.annotate(f"{val_ce[i]:.2f}", (bits_arr[i], val_ce[i]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=7.5, ha="center", fontweight="bold", color="#1565C0")
    ax.set_xlabel("bits_per_quantizer", fontsize=10)
    ax.set_ylabel("val_ce", fontsize=10)
    ax.set_title("C. Next-Token CE Loss", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)

    # ── Panel D: 1-step MAPE with error bars ──
    ax = axes[1, 0]
    # Find best (lowest MAPE)
    best_idx = np.argmin(mape_vals)
    colors = []
    for i in range(len(bits_all)):
        if i == best_idx:
            colors.append("#2E7D32")  # best: dark green
        elif mape_vals[i] <= mape_vals[best_idx] + mape_stds[best_idx] + mape_stds[i]:
            colors.append("#FFC107")  # within 1 combined std: amber
        else:
            colors.append("#B0BEC5")  # rest: grey
    ax.bar(bits_arr, mape_vals, color=colors, edgecolor="white", linewidth=0.5, width=0.55)
    ax.errorbar(bits_arr, mape_vals, yerr=mape_stds, fmt="none", ecolor="#333333",
                capsize=4, elinewidth=1.2, capthick=1.2)
    # Annotate only best and a few key points to avoid overlap
    annotate_indices = {best_idx}
    if len(bits_all) > 6:
        annotate_indices.update([0, len(bits_all)//2, len(bits_all)-1])
    for i in range(len(bits_all)):
        if i in annotate_indices:
            ax.text(bits_arr[i], mape_vals[i] + mape_stds[i] + 0.003,
                    f"{mape_vals[i]:.2f}±{mape_stds[i]:.2f}",
                    ha="center", fontsize=7.5, fontweight="bold",
                    color="#1B5E20" if i == best_idx else "#37474F")
    ax.set_xlabel("bits_per_quantizer", fontsize=10)
    ax.set_ylabel("MAPE (%)", fontsize=10)
    ax.set_title("D. ★ 1-Step Close-Ratio MAPE", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # ── Panel E: 1-step DA with error bars ──
    ax = axes[1, 1]
    best_idx_da = np.argmax(da_vals)
    colors_da = []
    for i in range(len(bits_all)):
        if i == best_idx_da:
            colors_da.append("#2E7D32")
        elif da_vals[i] >= da_vals[best_idx_da] - da_stds[best_idx_da] - da_stds[i]:
            colors_da.append("#FFC107")
        else:
            colors_da.append("#B0BEC5")
    ax.bar(bits_arr, da_vals, color=colors_da, edgecolor="white", linewidth=0.5, width=0.55)
    ax.errorbar(bits_arr, da_vals, yerr=da_stds, fmt="none", ecolor="#333333",
                capsize=4, elinewidth=1.2, capthick=1.2)
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.4, linewidth=1, label="random (50%)")
    # Annotate only best and a few key points to avoid overlap
    annotate_indices_da = {best_idx_da}
    if len(bits_all) > 6:
        annotate_indices_da.update([0, len(bits_all)//2, len(bits_all)-1])
    for i in range(len(bits_all)):
        if i in annotate_indices_da:
            ax.text(bits_arr[i], da_vals[i] + da_stds[i] + 0.10,
                    f"{da_vals[i]:.2f}±{da_stds[i]:.2f}",
                    ha="center", fontsize=7.5, fontweight="bold",
                    color="#1B5E20" if i == best_idx_da else "#37474F")
    ax.set_xlabel("bits_per_quantizer", fontsize=10)
    ax.set_ylabel("DA (%)", fontsize=10)
    ax.set_title("E. ★ 1-Step Direction Accuracy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    # ── Panel F: MAPE vs dead token % (reveals correlation or lack thereof) ──
    ax = axes[1, 2]
    avg_dead = (c_dead_pct + f_dead_pct) / 2
    # Cap marker size so large vocab doesn't overflow the panel
    sizes = np.clip(vocabs * 0.15 + 60, 80, 350)
    for i in range(len(bits_all)):
        ax.scatter(avg_dead[i], mape_vals[i], s=sizes[i],
                   color=plt.cm.RdYlGn_r(mape_vals[i] / max(mape_vals)),
                   edgecolors="black", linewidth=0.8, zorder=5)
        # Alternate label placement to reduce overlap
        offset_x = 10 if i % 2 == 0 else -18
        offset_y = 6 if i % 2 == 0 else -10
        ha = "left" if i % 2 == 0 else "right"
        ax.annotate(f"b={bits_all[i]}", (avg_dead[i], mape_vals[i]),
                    textcoords="offset points", xytext=(offset_x, offset_y),
                    fontsize=8, fontweight="bold", ha=ha)
    ax.set_xlabel("Avg Dead Token %", fontsize=10)
    ax.set_ylabel("1-Step MAPE (%)", fontsize=10)
    ax.set_title("F. MAPE vs Token Death Rate", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    # Annotate the paradox using a point that isn't at the extreme edge
    paradox_idx = max(len(bits_all) - 4, 0)
    ax.annotate("Paradox:\nmore dead tokens →\nlower MAPE?",
                xy=(avg_dead[paradox_idx], mape_vals[paradox_idx]),
                xytext=(avg_dead[paradox_idx] + 12, mape_vals[paradox_idx] + 0.06),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=7.5, color="gray")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_EVAL_PNG), exist_ok=True)
    fig.savefig(OUTPUT_EVAL_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_EVAL_PNG}")


def main():
    results = load_results()
    if not results:
        print("No Phase 1 results found.")
        return
    print(f"Loaded {len(results)} bits of Phase 1 results")

    eval_results = load_eval_results()
    if eval_results:
        print(f"Loaded {len(eval_results)} bits of downstream eval results")
        plot_downstream(results, eval_results)
    else:
        print("No downstream eval results yet. Run hpo.evaluate_phase1_bits first.")


if __name__ == "__main__":
    main()
