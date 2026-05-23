"""Phase 6 plots — Rollout Post-Training HPO results (FIXED).

Usage:
    python -m hpo.plot_phase6
"""

import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE6_DIR = os.path.join(PROJECT_ROOT, "trials", "phase6_rollout")
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "trials", "phase6_plots.png")


def load():
    rows = []
    csv_p = os.path.join(PHASE6_DIR, "summary.csv")
    if not os.path.exists(csv_p):
        print(f"CSV not found: {csv_p}")
        return []
    with open(csv_p, newline="") as f:
        for r in csv.DictReader(f):
            # Convert numeric columns
            for k in list(r.keys()):
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
            rows.append(r)
    return rows


def plot(rows):
    N = len(rows)
    pmape   = np.array([r["value"] for r in rows])       # path_mape
    daily_m = np.array([r["daily_mape"] for r in rows])
    topk    = np.array([int(r["oracle_top_k"]) for r in rows])
    temp    = np.array([r["oracle_temp"] for r in rows])
    kl      = np.array([r["kl_weight"] for r in rows])
    lr      = np.array([r["lr"] for r in rows])
    upd     = np.array([int(r["max_updates"]) for r in rows])
    elapsed = np.array([r["elapsed_min"] for r in rows])
    train_ce = np.array([r["train_ce"] for r in rows])

    best_idx = np.argmin(pmape)
    best = rows[best_idx]

    # ── Compute statistics ──
    corr_topk  = np.corrcoef(topk.astype(float), pmape)[0, 1]
    corr_temp  = np.corrcoef(temp, pmape)[0, 1]
    corr_kl    = np.corrcoef(kl, pmape)[0, 1]
    corr_lr    = np.corrcoef(np.log(lr), pmape)[0, 1]
    corr_upd   = np.corrcoef(upd.astype(float), pmape)[0, 1]

    fig = plt.figure(figsize=(22, 13))
    fig.suptitle(f"Phase 6: Rollout Post-Training HPO (Fixed) — {N} trials",
                 fontsize=15, fontweight="bold", y=0.98)

    # ══════════════════════════════════════════════════════════
    # A: Path MAPE histogram
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 1)
    ax.hist(pmape, bins=20, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(x=pmape[best_idx], color="red", linestyle="--", linewidth=2,
               label=f"best={pmape[best_idx]:.2f}%")
    ax.axvline(x=np.mean(pmape), color="gray", linestyle=":", linewidth=1.5,
               label=f"mean={np.mean(pmape):.2f}%")
    ax.set_xlabel("Path MAPE (%)"); ax.set_ylabel("Trials")
    ax.set_title(f"A. Path MAPE Distribution\n"
                 f"(range={pmape.min():.2f}–{pmape.max():.2f}%, σ={np.std(pmape):.3f})",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25)

    # ══════════════════════════════════════════════════════════
    # B: Daily MAPE histogram
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 2)
    ax.hist(daily_m, bins=20, color="#009688", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Daily MAPE (%)"); ax.set_ylabel("Trials")
    ax.set_title(f"B. Daily MAPE Distribution\n"
                 f"(range={daily_m.min():.3f}–{daily_m.max():.3f}%, σ={np.std(daily_m):.4f})",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # ══════════════════════════════════════════════════════════
    # C: oracle_top_k effect (categorical stripplot)
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 3)
    k_vals = sorted(set(int(k) for k in topk))
    colors_k = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
    means_k = []
    for i, kv in enumerate(k_vals):
        m = topk == kv
        if m.any():
            xj = np.random.default_rng(42 + kv).uniform(-0.15, 0.15, m.sum())
            ax.scatter(np.full(m.sum(), i) + xj, pmape[m], s=30, alpha=0.45,
                       color=colors_k[i % len(colors_k)], edgecolors="none")
            mean_m = pmape[m].mean()
            means_k.append(mean_m)
            ax.scatter([i], [mean_m], s=140, marker="D", color=colors_k[i % len(colors_k)],
                       edgecolors="white", linewidth=1.5, zorder=5)
            ax.text(i, mean_m + 0.008, f"{mean_m:.4f}%", ha="center", fontsize=8,
                    fontweight="bold")
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f"K={v}\n(n={int((topk==v).sum())})" for v in k_vals], fontsize=8)
    ax.set_xlabel("oracle_top_k"); ax.set_ylabel("Path MAPE (%)")
    ax.set_title(f"C. oracle_top_k (r={corr_topk:+.3f})\n"
                 f"K=32 best by {max(means_k)-min(means_k):.4f}pp",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # ══════════════════════════════════════════════════════════
    # D: max_updates effect (categorical stripplot)
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 4)
    u_vals = sorted(set(int(u) for u in upd))
    colors_u = ["#FF5722", "#FF9800", "#4CAF50"]
    means_u = []
    for i, uv in enumerate(u_vals):
        m = upd == uv
        if m.any():
            xj = np.random.default_rng(77 + uv).uniform(-0.15, 0.15, m.sum())
            ax.scatter(np.full(m.sum(), i) + xj, pmape[m], s=30, alpha=0.45,
                       color=colors_u[i], edgecolors="none")
            mean_m = pmape[m].mean()
            means_u.append(mean_m)
            ax.scatter([i], [mean_m], s=140, marker="D", color=colors_u[i],
                       edgecolors="white", linewidth=1.5, zorder=5)
            ax.text(i, mean_m + 0.008, f"{mean_m:.4f}%", ha="center", fontsize=8,
                    fontweight="bold")
    ax.set_xticks(range(len(u_vals)))
    ax.set_xticklabels([f"{v} upd\n(n={int((upd==v).sum())})" for v in u_vals], fontsize=8)
    ax.set_xlabel("max_updates"); ax.set_ylabel("Path MAPE (%)")
    ax.set_title(f"D. max_updates (r={corr_upd:+.3f})\n"
                 f"960 best by {max(means_u)-min(means_u):.4f}pp",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # ══════════════════════════════════════════════════════════
    # E: LR vs Path MAPE (scatter, colored by oracle_temp)
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 5)
    sc = ax.scatter(lr, pmape, c=temp, cmap="coolwarm", s=50,
                    edgecolors="black", linewidth=0.3, alpha=0.8)
    ax.scatter([lr[best_idx]], [pmape[best_idx]], s=200, marker="*",
               color="red", edgecolors="darkred", linewidth=1.5, zorder=10)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate"); ax.set_ylabel("Path MAPE (%)")
    ax.set_title(f"E. LR vs Path MAPE (r={corr_lr:+.3f})\n"
                 f"best lr={lr[best_idx]:.2e}",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("oracle_temp", fontsize=7)

    # ══════════════════════════════════════════════════════════
    # F: Temperature vs Path MAPE (scatter, colored by top_k)
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 6)
    sc = ax.scatter(temp, pmape, c=topk, cmap="viridis", s=50,
                    edgecolors="black", linewidth=0.3, alpha=0.8)
    ax.scatter([temp[best_idx]], [pmape[best_idx]], s=200, marker="*",
               color="red", edgecolors="darkred", linewidth=1.5, zorder=10)
    ax.set_xlabel("Oracle Temperature"); ax.set_ylabel("Path MAPE (%)")
    ax.set_title(f"F. Temperature vs Path MAPE (r={corr_temp:+.3f})",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("oracle_top_k", fontsize=7)

    # ══════════════════════════════════════════════════════════
    # G: KL Weight vs Path MAPE
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 7)
    sc = ax.scatter(kl, pmape, c=upd, cmap="RdYlGn", s=50,
                    edgecolors="black", linewidth=0.3, alpha=0.8)
    ax.scatter([kl[best_idx]], [pmape[best_idx]], s=200, marker="*",
               color="red", edgecolors="darkred", linewidth=1.5, zorder=10)
    ax.set_xlabel("KL Weight"); ax.set_ylabel("Path MAPE (%)")
    ax.set_title(f"G. KL Weight vs Path MAPE (r={corr_kl:+.3f})",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("max_updates", fontsize=7)

    # ══════════════════════════════════════════════════════════
    # H: top_k × max_updates interaction heatmap
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 8)
    heatmap = np.full((4, 3), np.nan)
    count_map = np.full((4, 3), 0)
    for i, kv in enumerate(k_vals):
        for j, uv in enumerate(u_vals):
            m = (topk == kv) & (upd == uv)
            if m.sum() > 0:
                heatmap[i, j] = pmape[m].mean()
                count_map[i, j] = m.sum()

    im = ax.imshow(heatmap, cmap="RdYlGn_r", aspect="auto", vmin=pmape.min(), vmax=pmape.max())
    for i in range(4):
        for j in range(3):
            if count_map[i, j] > 0:
                ax.text(j, i, f"{heatmap[i,j]:.3f}\nn={int(count_map[i,j])}",
                        ha="center", va="center", fontsize=8,
                        color="white" if heatmap[i,j] < 4.63 else "black",
                        fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{v}" for v in u_vals], fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"K={v}" for v in k_vals], fontsize=9)
    ax.set_xlabel("max_updates"); ax.set_ylabel("oracle_top_k")
    ax.set_title("H. top_k × max_updates Interaction\n(mean Path MAPE)",
                 fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.85)

    # ══════════════════════════════════════════════════════════
    # I: Top 10 vs Bottom 10 horizontal bar
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 9)
    sorted_idx = np.argsort(pmape)
    top10_idx = sorted_idx[:10]
    bot10_idx = sorted_idx[-10:]
    labels_top = [f"T{int(rows[i]['trial'])}" for i in top10_idx]
    labels_bot = [f"T{int(rows[i]['trial'])}" for i in bot10_idx]
    all_idx = list(top10_idx) + list(bot10_idx)
    all_labels = labels_top + labels_bot
    all_vals = [pmape[i] for i in all_idx]
    colors_bar = (["#4CAF50"] * 10 + ["#F44336"] * 10)

    y_pos = range(20)
    ax.barh(y_pos, all_vals, color=colors_bar, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos); ax.set_yticklabels(all_labels, fontsize=7)
    ax.set_xlabel("Path MAPE (%)")
    ax.axvline(x=np.mean(pmape), color="gray", linestyle=":", alpha=0.7,
               label=f"mean={np.mean(pmape):.3f}%")
    ax.set_title("I. Top 10 (green) vs Bottom 10 (red)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.25)

    # ══════════════════════════════════════════════════════════
    # J: Train CE vs Path MAPE
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 10)
    sc = ax.scatter(train_ce, pmape, c=upd, cmap="plasma", s=40,
                    edgecolors="black", linewidth=0.2, alpha=0.7)
    ax.set_xlabel("Train CE Loss"); ax.set_ylabel("Path MAPE (%)")
    corr_ce = np.corrcoef(train_ce, pmape)[0, 1]
    ax.set_title(f"J. Train CE vs Path MAPE (r={corr_ce:+.3f})",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("max_updates", fontsize=7)

    # ══════════════════════════════════════════════════════════
    # K: Elapsed time distribution
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 11)
    ax.hist(elapsed, bins=20, color="#9C27B0", alpha=0.7, edgecolor="white")
    ax.axvline(x=np.mean(elapsed), color="white", linestyle="--", linewidth=2,
               label=f"mean={np.mean(elapsed):.1f}min")
    ax.set_xlabel("Elapsed (min)"); ax.set_ylabel("Trials")
    ax.set_title(f"K. Trial Duration (total={np.sum(elapsed)/60:.1f}hrs)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25)

    # ══════════════════════════════════════════════════════════
    # L: Conclusion panel
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 12)
    ax.axis("off")

    # Parameter importance ranking
    params_corr = [
        ("max_updates", corr_upd, "960 best"),
        ("lr (log)", corr_lr, "weak, ~1-3e-5 ok"),
        ("oracle_top_k", corr_topk, "K=32 slightly best"),
        ("kl_weight", corr_kl, "weak, ~0.01-0.05 ok"),
        ("oracle_temp", corr_temp, "weak, ~0.5-0.8 ok"),
    ]
    params_corr.sort(key=lambda x: abs(x[1]), reverse=True)

    lines = [
        "Phase 6 HPO — KEY FINDINGS",
        "",
        f"Best Path MAPE: {pmape[best_idx]:.4f}%",
        f"  (trial {int(best['trial'])})",
        f"Best Daily MAPE: {daily_m[best_idx]:.4f}%",
        f"Range: {pmape.max()-pmape.min():.4f}pp across {N} trials",
        "",
        "Optimal config:",
        f"  oracle_top_k = {int(topk[best_idx])}",
        f"  oracle_temp  = {temp[best_idx]:.4f}",
        f"  kl_weight    = {kl[best_idx]:.6f}",
        f"  lr           = {lr[best_idx]:.2e}",
        f"  max_updates  = {int(upd[best_idx])}",
        "",
        "Parameter importance:",
    ]
    for name, corr, note in params_corr:
        lines.append(f"  {name:>16s}: r={corr:+.3f} ({note})")

    lines += [
        "",
        "Key insights:",
        "1. All params have WEAK effect",
        "   (range=0.2pp, no dominant param)",
        "2. max_updates=960 best",
        "   (more training = better path)",
        "3. top_k=32 slightly outperforms",
        "   (larger candidate pool helps)",
        "4. KL weight, temp nearly irrelevant",
        "5. LR in 5e-6 to 3e-5 is safe",
        "",
        f"Total: {np.sum(elapsed)/60:.1f} hrs, {N} trials",
        f"Mean: {np.mean(elapsed):.1f} min/trial",
    ]

    ax.text(0.05, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PNG}")


def main():
    rows = load()
    if not rows:
        print("No trial data found.")
        return
    print(f"Loaded {len(rows)} trials")
    plot(rows)


if __name__ == "__main__":
    main()
