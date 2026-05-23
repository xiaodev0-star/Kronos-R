"""Phase 3 plots — BaseModel architecture HPO results.

Usage:
    python -m hpo.plot_phase3
"""

import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE3_DIR = os.path.join(PROJECT_ROOT, "trials", "phase3_basemodel")
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "trials", "phase3_plots.png")


def load():
    rows = []
    csv_p = os.path.join(PHASE3_DIR, "summary.csv")
    if not os.path.exists(csv_p):
        print("summary.csv not found"); return []
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
    dim   = np.array([r["dim"] for r in rows])
    depth = np.array([r["depth"] for r in rows])
    heads = np.array([r["heads"] for r in rows])
    kv    = np.array([r["num_kv_heads"] for r in rows])

    best = np.argmin(mape)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Phase 3: BaseModel Architecture HPO — 1-Step Downstream",
                 fontsize=14, fontweight="bold")

    # A: MAPE vs val_ce
    ax = axes[0, 0]
    ax.scatter(vce, mape, c=dim, cmap="plasma", s=60, edgecolors="black", linewidth=0.3, alpha=0.85)
    ax.scatter([vce[best]], [mape[best]], s=200, marker="*", color="red",
               edgecolors="darkred", linewidth=1.5, zorder=10,
               label=f"best t047: {mape[best]:.2f}%")
    corr = np.corrcoef(vce, mape)[0, 1]
    ax.set_title(f"A. val_ce vs MAPE (r={corr:.3f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("val_ce"); ax.set_ylabel("MAPE (%)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8); cbar.set_label("dim", fontsize=8)

    # B: dim × depth heatmap
    ax = axes[0, 1]
    dims = sorted(set(int(d) for d in dim))
    deps = sorted(set(int(d) for d in depth))
    hm = np.full((len(deps), len(dims)), np.nan); cnt = np.zeros_like(hm)
    for i, dp in enumerate(deps):
        for j, dm in enumerate(dims):
            m = (depth == dp) & (dim == dm)
            if m.any(): hm[i, j] = mape[m].mean(); cnt[i, j] = m.sum()
    im = ax.imshow(hm, aspect="auto", cmap="RdYlGn_r", vmin=mape.min(), vmax=mape.max())
    for i in range(len(deps)):
        for j in range(len(dims)):
            if not np.isnan(hm[i, j]):
                ax.text(j, i, f"{hm[i,j]:.2f}\nn={int(cnt[i,j])}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims, fontsize=10)
    ax.set_yticks(range(len(deps))); ax.set_yticklabels(deps, fontsize=10)
    ax.set_xlabel("dim"); ax.set_ylabel("depth")
    ax.set_title("B. MAPE by dim × depth", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="MAPE (%)")

    # C: MAPE distribution
    ax = axes[0, 2]
    ax.hist(mape, bins=14, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(x=mape[best], color="red", linestyle="--", linewidth=2,
               label=f"best={mape[best]:.2f}%")
    ax.axvline(x=np.mean(mape), color="gray", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(mape):.2f}%")
    ax.set_xlabel("MAPE (%)"); ax.set_ylabel("Trials")
    ax.set_title(f"C. MAPE (range={mape.min():.2f}–{mape.max():.2f}%)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

    # D: Top-10 ranking
    ax = axes[1, 0]
    ranked = sorted(rows, key=lambda r: r["mape"])
    top10 = ranked[:10]
    t_labels = [r["trial"].replace("trial_", "") for r in top10]
    t_mape = [r["mape"] for r in top10]
    t_da = [r["da"] for r in top10]
    colors = ["#2E7D32" if i == 0 else "#FFC107" if i < 3 else "#B0BEC5" for i in range(10)]
    x = np.arange(10)
    ax.bar(x - 0.2, t_mape, 0.35, color=colors, edgecolor="white", label="MAPE%")
    ax2 = ax.twinx()
    ax2.plot(x + 0.2, t_da, "o-", color="#E91E63", markersize=7, linewidth=2, label="DA%")
    ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(t_labels, fontsize=7.5)
    ax.set_ylabel("MAPE (%)"); ax2.set_ylabel("DA (%)", color="#E91E63")
    ax2.tick_params(axis="y", labelcolor="#E91E63")
    ax.set_title("D. Top-10 Trials", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7); ax.grid(axis="y", alpha=0.25)

    # E: DA vs MAPE
    ax = axes[1, 1]
    ax.scatter(mape, da, c=dim, cmap="plasma", s=60, edgecolors="black", linewidth=0.3, alpha=0.85)
    ax.scatter([mape[best]], [da[best]], s=200, marker="*", color="red",
               edgecolors="darkred", linewidth=1.5, zorder=10)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="random")
    ax.set_xlabel("MAPE (%)"); ax.set_ylabel("DA (%)")
    ax.set_title(f"E. DA vs MAPE (DA range={da.min():.1f}–{da.max():.1f}%)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8); cbar.set_label("dim", fontsize=8)

    # F: GQA ratio effect
    ax = axes[1, 2]
    ratios = [f"{int(h)}:{int(k)}" for h, k in zip(heads, kv)]
    uniq = sorted(set(ratios), key=lambda x: int(x.split(":")[1]))
    x_pos = []; y_pos = []; colors_f = []
    for i, r in enumerate(uniq):
        m = np.array([ratios[j] == r for j in range(len(ratios))])
        if m.any():
            for v in mape[m]:
                x_pos.append(i + np.random.default_rng().uniform(-0.15, 0.15))
                y_pos.append(v)
                colors_f.append("#4CAF50")
            mean_m = mape[m].mean()
            ax.scatter([i], [mean_m], s=140, marker="D", color="#2E7D32",
                      edgecolors="white", linewidth=1.5, zorder=5)
            ax.text(i, mean_m + 0.008, f"{mean_m:.2f}%", ha="center", fontsize=9,
                    fontweight="bold", color="#1B5E20")
    ax.scatter(x_pos, y_pos, s=35, alpha=0.45, color=colors_f, edgecolors="none")
    ax.set_xticks(range(len(uniq))); ax.set_xticklabels(uniq, fontsize=10)
    ax.set_xlabel("heads : kv_heads (GQA ratio)"); ax.set_ylabel("MAPE (%)")
    ax.set_title("F. GQA Ratio Effect", fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.25)

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
