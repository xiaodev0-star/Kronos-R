"""Phase 2 plots — tokenizer HPO results + downstream evaluation.

Usage:
    python -m hpo.plot_phase2
"""

import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE2_DIR = os.path.join(PROJECT_ROOT, "trials", "phase2_tokenizer")
EVAL_CSV = os.path.join(PHASE2_DIR, "eval.csv")
SUP_EVAL_CSV = os.path.join(PROJECT_ROOT, "trials", "phase2_tokenizer_sup", "eval.csv")
SUP_SUMMARY_CSV = os.path.join(PROJECT_ROOT, "trials", "phase2_tokenizer_sup", "summary.csv")
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "trials", "phase2_plots.png")


def _cast_row(row):
    for k in list(row.keys()):
        try:
            row[k] = float(row[k])
        except (ValueError, TypeError):
            pass
    return row


def load_data():
    rows = []
    with open(EVAL_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(_cast_row(r))
    return rows


def load_sup_data():
    """Load supplement data — joins eval.csv (MAPE) with summary.csv (params)."""
    # Load eval metrics
    eval_rows = {}
    has_eval = os.path.exists(SUP_EVAL_CSV)
    if has_eval:
        with open(SUP_EVAL_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r = _cast_row(r)
                eval_rows[int(r["trial"])] = r

    # Load params from summary.csv
    if not os.path.exists(SUP_SUMMARY_CSV):
        return []
    rows = []
    with open(SUP_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r2 = _cast_row(r)
            tnum = int(r2["trial"])
            # summary.csv uses 'value' for val_ce
            if "value" in r2:
                r2["val_ce"] = r2.pop("value")
            # Merge eval metrics if available
            if tnum in eval_rows:
                for k, v in eval_rows[tnum].items():
                    if k not in r2:
                        r2[k] = v
            # Normalize param key names
            for p in ["hidden_dim", "embedding_dim", "bsq_commitment_cost",
                       "bsq_entropy_weight", "learning_rate"]:
                if p in r2 and f"cfg_{p}" not in r2:
                    r2[f"cfg_{p}"] = r2.pop(p)
            rows.append(r2)
    return rows


def plot(rows, sup_rows=None):
    mape     = np.array([r["mape"] for r in rows])
    da       = np.array([r["da"] for r in rows])
    val_ce   = np.array([r.get("val_ce", np.nan) for r in rows])
    hdim     = np.array([r["cfg_hidden_dim"] for r in rows])
    edim     = np.array([r["cfg_embedding_dim"] for r in rows])
    commit   = np.array([r["cfg_bsq_commitment_cost"] for r in rows])
    entropy  = np.array([r["cfg_bsq_entropy_weight"] for r in rows])
    lr       = np.array([r["cfg_learning_rate"] for r in rows])
    util     = np.array([r.get("coarse_utilization", np.nan) for r in rows])
    dead     = np.array([r.get("coarse_dead_tokens", np.nan) for r in rows])

    # Supplement data
    sup_commit = None; sup_edim = None; sup_valce = None; sup_hdim = None
    sup_mape = None; sup_da = None; sup_trial = None
    if sup_rows:
        sup_commit = np.array([r["cfg_bsq_commitment_cost"] for r in sup_rows])
        sup_edim   = np.array([r["cfg_embedding_dim"] for r in sup_rows])
        sup_valce  = np.array([r["val_ce"] for r in sup_rows])
        sup_hdim   = np.array([r["cfg_hidden_dim"] for r in sup_rows])
        if "mape" in sup_rows[0]:
            sup_mape = np.array([r["mape"] for r in sup_rows])
            sup_da   = np.array([r["da"] for r in sup_rows])
            sup_trial = np.array([int(r["trial"]) for r in sup_rows])
    has_sup_mape = sup_mape is not None

    best_idx = np.argmin(mape)
    valid_ce = ~np.isnan(val_ce)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Phase 2: Tokenizer HPO — Downstream 1-Step Prediction",
                 fontsize=14, fontweight="bold")

    # ── A: val_ce vs MAPE (proxy quality check) ──
    ax = axes[0, 0]
    ax.scatter(val_ce[valid_ce], mape[valid_ce], c=commit[valid_ce],
               cmap="RdYlGn_r", s=50, edgecolors="black", linewidth=0.4, alpha=0.8)
    ax.scatter([val_ce[best_idx]], [mape[best_idx]], s=180, marker="*",
               color="red", edgecolors="darkred", linewidth=1.5, zorder=10,
               label=f"best (trial {int(rows[best_idx]['trial'])})")
    if valid_ce.sum() > 2:
        corr = np.corrcoef(val_ce[valid_ce], mape[valid_ce])[0, 1]
        ax.set_title(f"A. val_ce vs MAPE (r={corr:.3f})", fontsize=12, fontweight="bold")
    else:
        ax.set_title("A. val_ce vs MAPE", fontsize=12, fontweight="bold")
    ax.set_xlabel("Validation CE Loss", fontsize=10)
    ax.set_ylabel("1-Step MAPE (%)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cbar.set_label("commitment_cost", fontsize=8)

    # ── B: Commitment cost vs MAPE / val_ce (overlay supplement) ──
    ax = axes[0, 1]
    ax.scatter(commit, mape, c=edim, cmap="viridis", s=60,
               edgecolors="black", linewidth=0.4, alpha=0.8, label="main (MAPE)")
    ax.scatter([commit[best_idx]], [mape[best_idx]], s=180, marker="*",
               color="red", edgecolors="darkred", linewidth=1.5, zorder=10)
    # Overlay supplement
    if sup_commit is not None and len(sup_commit) > 0:
        ax2_b = ax.twinx()
        if has_sup_mape:
            # Use MAPE if available (preferred)
            ax2_b.scatter(sup_commit, sup_mape, c="none", edgecolors="#C62828",
                          linewidth=1.5, marker="^", s=80, alpha=0.9, label="supplement (MAPE)")
            ax2_b.set_ylabel("MAPE % (supplement)", fontsize=9, color="#C62828")
            ax2_b.tick_params(axis="y", labelcolor="#C62828")
            sup_best = np.argmin(sup_mape)
            ax2_b.scatter([sup_commit[sup_best]], [sup_mape[sup_best]], s=150, marker="*",
                          color="#C62828", edgecolors="darkred", linewidth=1.2, zorder=11)
        else:
            ax2_b.scatter(sup_commit, sup_valce, c="none", edgecolors="#E91E63",
                          linewidth=1.2, marker="^", s=70, alpha=0.9, label="supplement (val_ce)")
            ax2_b.set_ylabel("val_ce (supplement)", fontsize=9, color="#E91E63")
            ax2_b.tick_params(axis="y", labelcolor="#E91E63")
            sup_best = np.argmin(sup_valce)
            ax2_b.scatter([sup_commit[sup_best]], [sup_valce[sup_best]], s=150, marker="*",
                          color="#C62828", edgecolors="darkred", linewidth=1.2, zorder=11)
        ax2_b.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("bsq_commitment_cost", fontsize=10)
    ax.set_ylabel("1-Step MAPE (%)", fontsize=10)
    corr_c = np.corrcoef(commit, mape)[0, 1]
    ax.set_title(f"B. Commitment Cost vs MAPE (r={corr_c:.3f})",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=7)
    cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cbar.set_label("embedding_dim", fontsize=8)

    # ── C: MAPE distribution (main + supplement) ──
    ax = axes[0, 2]
    ax.hist(mape, bins=12, color="#2196F3", alpha=0.6, edgecolor="white",
            linewidth=0.5, label="main")
    if has_sup_mape:
        ax.hist(sup_mape, bins=8, color="#E91E63", alpha=0.5, edgecolor="white",
                linewidth=0.5, label="supplement")
    ax.axvline(x=mape[best_idx], color="#1565C0", linestyle="--", linewidth=2,
               label=f"main best={mape[best_idx]:.2f}%")
    if has_sup_mape:
        sup_best = np.min(sup_mape)
        ax.axvline(x=sup_best, color="#C62828", linestyle="--", linewidth=2,
                   label=f"sup best={sup_best:.2f}%")
    all_mape = np.concatenate([mape, sup_mape]) if has_sup_mape else mape
    ax.set_xlabel("1-Step MAPE (%)", fontsize=10)
    ax.set_ylabel("Trials", fontsize=10)
    ax.set_title(f"C. MAPE Distribution (range={all_mape.min():.2f}-{all_mape.max():.2f}%)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    # ── D: embedding_dim effect (main + supplement) ──
    ax = axes[1, 0]
    # Main: edim vs MAPE (boxplot-like scatter)
    ed_vals_main = sorted(set(int(e) for e in edim))
    for i, ed in enumerate(ed_vals_main):
        mask = edim == ed
        if mask.any():
            x_jitter = np.random.default_rng(42 + ed).uniform(-0.15, 0.15, mask.sum())
            ax.scatter(np.full(mask.sum(), i) + x_jitter, mape[mask],
                      s=40, alpha=0.5, color="#4CAF50", edgecolors="none", zorder=3)
            mean_m = mape[mask].mean()
            ax.scatter([i], [mean_m], s=100, marker="D", color="#2E7D32",
                      edgecolors="white", linewidth=1.5, zorder=5)
            ax.text(i, mean_m + 0.013, f"{mean_m:.2f}%", ha="center", fontsize=9,
                    fontweight="bold", color="#1B5E20")

    # Supplement: edim vs MAPE (or val_ce) on right axis
    if sup_edim is not None:
        ax_twin_d = ax.twinx()
        ed_vals_sup = sorted(set(int(e) for e in sup_edim))
        sup_y = sup_mape if has_sup_mape else sup_valce
        sup_y_label = "MAPE% (sup)" if has_sup_mape else "val_ce (sup)"
        for i, ed in enumerate(ed_vals_sup):
            mask = sup_edim == ed
            if mask.any():
                x_jitter = np.random.default_rng(99 + ed).uniform(-0.15, 0.15, mask.sum())
                ax_twin_d.scatter(np.full(mask.sum(), i + 0.5 + len(ed_vals_main)) + x_jitter,
                                 sup_y[mask], s=40, alpha=0.5, color="#E91E63",
                                 edgecolors="none", zorder=3, marker="^")
                mean_s = sup_y[mask].mean()
                ax_twin_d.scatter([i + 0.5 + len(ed_vals_main)], [mean_s], s=100, marker="D",
                                 color="#C62828", edgecolors="white", linewidth=1.5, zorder=5)
                ax_twin_d.text(i + 0.5 + len(ed_vals_main), mean_s + (sup_y.max()-sup_y.min())*0.05,
                              f"{mean_s:.2f}", ha="center", fontsize=7.5, fontweight="bold",
                              color="#C62828")
        ax_twin_d.set_ylabel(sup_y_label, fontsize=9, color="#E91E63")
        ax_twin_d.tick_params(axis="y", labelcolor="#E91E63")

    ax.set_xticks(list(range(len(ed_vals_main))) +
                  [i + 0.5 + len(ed_vals_main) for i in range(len(ed_vals_sup))] if sup_edim is not None else [])
    ax.set_xticklabels([f"main\ned={ed}" for ed in ed_vals_main] +
                       ([f"sup\ned={ed}" for ed in ed_vals_sup] if sup_edim is not None else []),
                       fontsize=7.5)
    ax.set_ylabel("1-Step MAPE (%)", fontsize=10)
    ax.set_title("D. embedding_dim Effect (main + sup)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # ── E: Top-10 trial ranking (main + supplement merged) ──
    ax = axes[1, 1]
    # Merge and rank
    all_trials = [(r["mape"], r["da"], int(r["trial"]), "main") for r in rows]
    if has_sup_mape:
        all_trials += [(sup_mape[i], sup_da[i], int(sup_trial[i]), "sup") for i in range(len(sup_mape))]
    all_trials.sort(key=lambda x: x[0])
    top12 = all_trials[:12]
    t_mape = [t[0] for t in top12]
    t_da   = [t[1] for t in top12]
    t_labels = [f"{'S' if t[3]=='sup' else ''}{t[2]}" for t in top12]
    colors = ["#2E7D32" if i == 0 else "#FFC107" if i < 3 else "#B0BEC5"
              for i in range(len(top12))]
    # Mark supplement with red edge
    edge_colors = ["#C62828" if t[3] == "sup" else "white" for t in top12]
    x = np.arange(len(top12))
    ax.bar(x - 0.2, t_mape, 0.35, color=colors, edgecolor=edge_colors,
           linewidth=1.5, label="MAPE%")
    ax_twin = ax.twinx()
    ax_twin.plot(x + 0.2, t_da, "o-", color="#E91E63", markersize=8, linewidth=2, label="DA%")
    ax_twin.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(t_labels, fontsize=7.5)
    ax.set_ylabel("MAPE (%)", fontsize=10)
    ax_twin.set_ylabel("DA (%)", fontsize=10, color="#E91E63")
    ax_twin.tick_params(axis="y", labelcolor="#E91E63")
    ax.set_title("E. Top-12 All Trials: MAPE + DA (S=supplement)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    ax_twin.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    # ── F: DA distribution (main + supplement) ──
    ax = axes[1, 2]
    ax.hist(da, bins=10, color="#FF9800", alpha=0.5, edgecolor="white",
            linewidth=0.5, label="main")
    if has_sup_mape:
        ax.hist(sup_da, bins=6, color="#E91E63", alpha=0.4, edgecolor="white",
                linewidth=0.5, label="supplement")
    ax.axvline(x=50, color="gray", linestyle="--", linewidth=2, label="random (50%)")
    ax.axvline(x=da[best_idx], color="#E65100", linestyle="--", linewidth=1.5,
               label=f"main best={da[best_idx]:.1f}%")
    if has_sup_mape:
        sup_best_da_idx = np.argmax(sup_da)
        ax.axvline(x=sup_da[sup_best_da_idx], color="#C62828", linestyle="--", linewidth=1.5,
                   label=f"sup best={sup_da[sup_best_da_idx]:.1f}%")
    ax.set_xlabel("Direction Accuracy (%)", fontsize=10)
    ax.set_ylabel("Trials", fontsize=10)
    all_da = np.concatenate([da, sup_da]) if has_sup_mape else da
    ax.set_title(f"F. DA Distribution (range={all_da.min():.1f}-{all_da.max():.1f}%)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PNG}")


def main():
    if not os.path.exists(EVAL_CSV):
        print(f"eval.csv not found at {EVAL_CSV}. Run hpo.evaluate_phase2 first.")
        return
    rows = load_data()
    sup_rows = load_sup_data()
    print(f"Loaded {len(rows)} main trials + {len(sup_rows)} supplement trials")
    plot(rows, sup_rows)


if __name__ == "__main__":
    main()
