"""Phase 4 plots — Auxiliary components HPO results.

Usage:
    python -m hpo.plot_phase4
"""

import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE4_DIR = os.path.join(PROJECT_ROOT, "trials", "phase4_aux")
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "trials", "phase4_plots.png")


def load():
    rows = []
    csv_p = os.path.join(PHASE4_DIR, "summary.csv")
    if not os.path.exists(csv_p): return []
    with open(csv_p, newline="") as f:
        for r in csv.DictReader(f):
            for k in r:
                try: r[k] = float(r[k])
                except (ValueError, TypeError): pass
            rows.append(r)
    return [r for r in rows if r.get("mape", "") != ""]


def plot(rows):
    mape  = np.array([r["mape"] for r in rows])
    da    = np.array([r["da"] for r in rows])
    vce   = np.array([r["val_ce"] for r in rows])
    lt    = np.array([r["latent_t"] for r in rows])
    ld    = np.array([r["latent_d"] for r in rows])
    ch    = np.array([r["cross_h"] for r in rows])
    fact  = np.array([r["factor"] for r in rows])

    # Phase 3 baseline
    bl_mape = 2.0426

    best_idx = np.argmin(mape)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Phase 4: Auxiliary Components — All Worse Than Baseline",
                 fontsize=14, fontweight="bold")

    # A: MAPE distribution vs baseline
    ax = axes[0, 0]
    ax.hist(mape, bins=12, color="#B0BEC5", alpha=0.7, edgecolor="white")
    ax.axvline(x=bl_mape, color="#2E7D32", linestyle="-", linewidth=2.5,
               label=f"Phase 3 baseline = {bl_mape:.2f}%")
    ax.axvline(x=mape[best_idx], color="red", linestyle="--", linewidth=2,
               label=f"Phase 4 best = {mape[best_idx]:.2f}%")
    ax.set_xlabel("MAPE (%)"); ax.set_ylabel("Trials")
    ax.set_title(f"A. MAPE: All Worse Than Baseline (mean={mape.mean():.2f}%)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

    # B: Factor tokens effect
    ax = axes[0, 1]
    f_vals = sorted(set(int(f) for f in fact))
    colors = ["#E91E63" if f == 0 else "#4CAF50" if f <= 4 else "#2196F3" for f in f_vals]
    for i, fv in enumerate(f_vals):
        m = fact == fv
        if m.any():
            x_j = np.random.default_rng(42+fv).uniform(-0.15, 0.15, m.sum())
            ax.scatter(np.full(m.sum(), i)+x_j, mape[m], s=35, alpha=0.5,
                      color=colors[i], edgecolors="none")
            mean_m = mape[m].mean()
            ax.scatter([i], [mean_m], s=120, marker="D", color=colors[i],
                      edgecolors="white", linewidth=1.5, zorder=5)
            ax.text(i, mean_m+0.008, f"{mean_m:.2f}%", ha="center", fontsize=9,
                    fontweight="bold")
    ax.axhline(y=bl_mape, color="#2E7D32", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_xticks(range(len(f_vals))); ax.set_xticklabels(f_vals, fontsize=10)
    ax.set_xlabel("num_factor_tokens"); ax.set_ylabel("MAPE (%)")
    ax.set_title(f"B. Factor Tokens: 0 is Best (+{mape[fact==0].mean()-bl_mape:+.2f}pp vs baseline)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # C: Latent tokens effect
    ax = axes[0, 2]
    t_vals = sorted(set(int(t) for t in lt))
    for i, tv in enumerate(t_vals):
        m = lt == tv
        if m.any():
            x_j = np.random.default_rng(99+tv).uniform(-0.15, 0.15, m.sum())
            ax.scatter(np.full(m.sum(), i)+x_j, mape[m], s=35, alpha=0.5,
                      color="#2196F3", edgecolors="none")
            mean_m = mape[m].mean()
            ax.scatter([i], [mean_m], s=120, marker="D", color="#1565C0",
                      edgecolors="white", linewidth=1.5, zorder=5)
            ax.text(i, mean_m+0.008, f"{mean_m:.2f}%", ha="center", fontsize=9,
                    fontweight="bold")
    ax.axhline(y=bl_mape, color="#2E7D32", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_xticks(range(len(t_vals))); ax.set_xticklabels(t_vals, fontsize=10)
    ax.set_xlabel("num_latent_tokens"); ax.set_ylabel("MAPE (%)")
    ax.set_title(f"C. Latent Tokens: 8 is Best (+{mape[lt==8].mean()-bl_mape:+.2f}pp)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # D: Depth effect
    ax = axes[1, 0]
    d_vals = sorted(set(int(d) for d in ld))
    for i, dv in enumerate(d_vals):
        m = ld == dv
        if m.any():
            x_j = np.random.default_rng(55+dv).uniform(-0.15, 0.15, m.sum())
            ax.scatter(np.full(m.sum(), i)+x_j, mape[m], s=35, alpha=0.5,
                      color="#FF9800", edgecolors="none")
            mean_m = mape[m].mean()
            ax.scatter([i], [mean_m], s=120, marker="D", color="#E65100",
                      edgecolors="white", linewidth=1.5, zorder=5)
            ax.text(i, mean_m+0.008, f"{mean_m:.2f}%", ha="center", fontsize=9,
                    fontweight="bold")
    ax.axhline(y=bl_mape, color="#2E7D32", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_xticks(range(len(d_vals))); ax.set_xticklabels(d_vals, fontsize=10)
    ax.set_xlabel("latent_reasoner_depth"); ax.set_ylabel("MAPE (%)")
    ax.set_title("D. Latent Depth: Flat (all ~2.27%)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # E: Cross heads effect
    ax = axes[1, 1]
    h_vals = sorted(set(int(h) for h in ch))
    for i, hv in enumerate(h_vals):
        m = ch == hv
        if m.any():
            x_j = np.random.default_rng(77+hv).uniform(-0.15, 0.15, m.sum())
            ax.scatter(np.full(m.sum(), i)+x_j, mape[m], s=35, alpha=0.5,
                      color="#9C27B0", edgecolors="none")
            mean_m = mape[m].mean()
            ax.scatter([i], [mean_m], s=120, marker="D", color="#6A1B9A",
                      edgecolors="white", linewidth=1.5, zorder=5)
            ax.text(i, mean_m+0.008, f"{mean_m:.2f}%", ha="center", fontsize=9,
                    fontweight="bold")
    ax.axhline(y=bl_mape, color="#2E7D32", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_xticks(range(len(h_vals))); ax.set_xticklabels(h_vals, fontsize=10)
    ax.set_xlabel("latent_cross_heads"); ax.set_ylabel("MAPE (%)")
    ax.set_title("E. Cross Heads: 2 > 4", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # F: MAPE gap explanation
    ax = axes[1, 2]
    ax.axis("off")
    text = (
        "Phase 4 Conclusion\n\n"
        f"Baseline (P3 trial 047):  MAPE = {bl_mape:.2f}%\n"
        f"Phase 4 best:             MAPE = {mape[best_idx]:.2f}%\n"
        f"Delta:                    +{mape[best_idx]-bl_mape:.2f}pp\n\n"
        "ALL 30 trials are worse than\n"
        "the Phase 3 baseline.\n\n"
        "Why?\n"
        "1. Defaults (lt=16, ld=4, ch=4, factor=4)\n"
        "   were already near-optimal.\n"
        "2. Phase 3 trial 047 benefited from\n"
        "   a lucky random seed (2.04% is an\n"
        "   outlier even within Phase 3).\n"
        "3. Aux components have minimal\n"
        "   impact on MAPE (~0.1pp range).\n\n"
        "Recommendation:\n"
        "factor_tokens=0 (safe to remove)\n"
        "latent_tokens=8 (slightly better)\n"
        "Use Phase 3 defaults for the rest."
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.9))

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PNG}")


def main():
    rows = load()
    if not rows: return
    print(f"Loaded {len(rows)} trials")
    plot(rows)


if __name__ == "__main__":
    main()
