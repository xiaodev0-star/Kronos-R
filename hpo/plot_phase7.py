"""Phase 7 plots — CI methods comparison (Sampling vs Training).

Visualises:
  A. Sampling HPO: temperature × num_samples heatmap
  B. Sampling HPO: coverage vs confidence_level
  C. Training HPO: parameter importance
  D. Training HPO: concentration_weight × interval_score_weight interaction
  E. Method comparison: best sampling vs best training
  F. Per-step width/coverage comparison
  G. Path-level interval scores
  H. Key findings panel

Usage:
    python -m hpo.plot_phase7
"""

import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE7_DIR = os.path.join(PROJECT_ROOT, "trials", "phase7_ci")
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "trials", "phase7_plots.png")


def load_sampling():
    """Load CI sampling grid search results."""
    csv_path = os.path.join(PHASE7_DIR, "summary_sampling.csv")
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            for k in list(r.keys()):
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
            rows.append(r)
    return rows


def load_training():
    """Load CI training HPO results."""
    csv_path = os.path.join(PHASE7_DIR, "summary_training.csv")
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            for k in list(r.keys()):
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
            if "value" in r:
                r["avg_interval_score"] = r["value"]
            rows.append(r)
    return rows


def load_smoke_test():
    """Load smoke test results for initial reference points."""
    result_path = os.path.join(PHASE7_DIR, "smoke_test_results.json")
    if not os.path.exists(result_path):
        return None
    with open(result_path) as f:
        return json.load(f)


def plot(sampling_rows, training_rows, smoke_data):
    has_sampling = len(sampling_rows) > 0
    has_training = len(training_rows) > 0

    fig = plt.figure(figsize=(24, 14))
    fig.suptitle(
        f"Phase 7: CI Methods Comparison — Sampling ({len(sampling_rows)} configs)"
        f" vs Training ({len(training_rows)} trials)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ══════════════════════════════════════════════════════════
    # A: Temperature × Num Samples heatmap (best conf, best feed)
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 1)
    if has_sampling:
        sr = sampling_rows
        temps = sorted(set(r["temperature"] for r in sr))
        ns_vals = sorted(set(int(r["num_samples"]) for r in sr))
        heatmap = np.full((len(temps), len(ns_vals)), np.nan)
        count_map = np.full((len(temps), len(ns_vals)), 0)
        for r in sr:
            if r.get("confidence_level") == 0.8 and r.get("feed_mode") == "argmax":
                ti = temps.index(r["temperature"])
                ni = ns_vals.index(int(r["num_samples"]))
                heatmap[ti, ni] = r.get("avg_interval_score", np.nan)
                count_map[ti, ni] += 1

        im = ax.imshow(heatmap, cmap="RdYlGn_r", aspect="auto",
                        vmin=np.nanmin(heatmap) if np.any(np.isfinite(heatmap)) else 0,
                        vmax=np.nanmax(heatmap) if np.any(np.isfinite(heatmap)) else 1)
        for i in range(len(temps)):
            for j in range(len(ns_vals)):
                if np.isfinite(heatmap[i, j]):
                    ax.text(j, i, f"{heatmap[i,j]:.4f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if heatmap[i,j] < 0.09 else "black")
        ax.set_xticks(range(len(ns_vals)))
        ax.set_xticklabels([str(v) for v in ns_vals], fontsize=8)
        ax.set_yticks(range(len(temps)))
        ax.set_yticklabels([str(v) for v in temps], fontsize=8)
        ax.set_xlabel("num_samples"); ax.set_ylabel("temperature")
        ax.set_title("A. Temp × N Heatmap (IS, C=0.80, argmax)",
                     fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85)
    else:
        ax.text(0.5, 0.5, "No sampling data", ha="center", va="center")
        ax.set_title("A. Sampling HPO (pending)")

    # ══════════════════════════════════════════════════════════
    # B: Coverage vs Confidence Level (calibration curve)
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 2)
    if has_sampling:
        confs = sorted(set(r["confidence_level"] for r in sr))
        colors = plt.cm.viridis(np.linspace(0, 1, len(confs)))
        for ci, conf in enumerate(confs):
            subset = [r for r in sr if r["confidence_level"] == conf
                      and r.get("feed_mode") == "argmax"]
            if subset:
                covs = [r["coverage"] for r in subset]
                iss = [r["avg_interval_score"] for r in subset]
                ax.scatter(covs, iss, c=[colors[ci]], label=f"C={conf:.0%}",
                          s=30, alpha=0.6, edgecolors="none")
        ax.plot([0, 1], [0, 0], 'k--', alpha=0.2, linewidth=1)
        ax.set_xlabel("Empirical Coverage"); ax.set_ylabel("Interval Score")
        ax.set_title("B. Calibration: Coverage vs IS", fontsize=10, fontweight="bold")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("B. Calibration (pending)")

    # ══════════════════════════════════════════════════════════
    # C: Training HPO — concentration_weight effect
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 3)
    if has_training:
        tr = training_rows
        conc_vals = sorted(set(r.get("concentration_weight", 1.0) for r in tr))
        means_c, stds_c = [], []
        for cv in conc_vals:
            subset = [r["avg_interval_score"] for r in tr
                      if r.get("concentration_weight") == cv]
            if subset:
                means_c.append(np.mean(subset))
                stds_c.append(np.std(subset))
            else:
                means_c.append(np.nan)
                stds_c.append(np.nan)
        for i, cv in enumerate(conc_vals):
            subset = [r["avg_interval_score"] for r in tr
                      if r.get("concentration_weight") == cv]
            xj = np.random.default_rng(42 + i).uniform(-0.08, 0.08, len(subset))
            ax.scatter(np.full(len(subset), i) + xj, subset, s=20, alpha=0.4,
                      color="#2196F3", edgecolors="none")
        ax.errorbar(range(len(conc_vals)), means_c, yerr=stds_c,
                    fmt='D', color="#E91E63", capsize=4, markersize=8,
                    linewidth=2, zorder=5)
        ax.set_xticks(range(len(conc_vals)))
        ax.set_xticklabels([str(v) for v in conc_vals], fontsize=8)
        ax.set_xlabel("concentration_weight"); ax.set_ylabel("Interval Score")
        ax.set_title("C. Concentration Weight Effect", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No training data", ha="center", va="center")
        ax.set_title("C. Training HPO (pending)")

    # ══════════════════════════════════════════════════════════
    # D: Training — interval_score_weight effect
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 4)
    if has_training:
        isw_vals = sorted(set(r.get("interval_score_weight", 0.0) for r in tr))
        means_w, stds_w = [], []
        for iw in isw_vals:
            subset = [r["avg_interval_score"] for r in tr
                      if r.get("interval_score_weight") == iw]
            if subset:
                means_w.append(np.mean(subset))
                stds_w.append(np.std(subset))
            else:
                means_w.append(np.nan)
                stds_w.append(np.nan)
        for i, iw in enumerate(isw_vals):
            subset = [r["avg_interval_score"] for r in tr
                      if r.get("interval_score_weight") == iw]
            xj = np.random.default_rng(77 + i).uniform(-0.08, 0.08, len(subset))
            ax.scatter(np.full(len(subset), i) + xj, subset, s=20, alpha=0.4,
                      color="#4CAF50", edgecolors="none")
        ax.errorbar(range(len(isw_vals)), means_w, yerr=stds_w,
                    fmt='D', color="#FF5722", capsize=4, markersize=8,
                    linewidth=2, zorder=5)
        ax.set_xticks(range(len(isw_vals)))
        ax.set_xticklabels([str(v) for v in isw_vals], fontsize=8)
        ax.set_xlabel("interval_score_weight"); ax.set_ylabel("Interval Score")
        ax.set_title("D. Interval Score Weight Effect", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("D. Training HPO (pending)")

    # ══════════════════════════════════════════════════════════
    # E: Method Comparison — best Sampling vs best Training
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 5)
    method_names = []
    method_is = []
    method_cov = []
    method_width = []
    colors_m = []

    if has_sampling:
        best_s = min(sampling_rows, key=lambda r: r.get("avg_interval_score", 999))
        method_names.append("Sampling\n(best)")
        method_is.append(best_s["avg_interval_score"])
        method_cov.append(best_s.get("coverage", 0))
        method_width.append(best_s.get("avg_width", 0))
        colors_m.append("#2196F3")

    if has_training:
        best_t = min(training_rows, key=lambda r: r.get("avg_interval_score", 999))
        method_names.append("Training\n(best)")
        method_is.append(best_t["avg_interval_score"])
        method_cov.append(best_t.get("coverage", 0))
        method_width.append(best_t.get("avg_width", 0))
        colors_m.append("#E91E63")

    if smoke_data and "model_comparison" in smoke_data:
        for name, r in smoke_data["model_comparison"].items():
            method_names.append(f"{name}\n(smoke)")
            method_is.append(r["avg_interval_score"])
            method_cov.append(r["coverage"])
            method_width.append(r["avg_width"])
            colors_m.append("#9E9E9E")

    if method_names:
        x = np.arange(len(method_names))
        w = 0.25
        ax.bar(x - w, method_is, w, color=colors_m, alpha=0.8, label="Interval Score")
        ax.bar(x, method_cov, w, color=[plt.cm.RdYlGn(c) for c in method_cov],
               alpha=0.8, label="Coverage")
        ax.bar(x + w, method_width, w, color="#FFC107", alpha=0.8, label="Avg Width")
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, fontsize=8)
        ax.set_ylabel("Score")
        ax.set_title("E. Method Comparison", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No comparison data", ha="center", va="center")
        ax.set_title("E. Method Comparison (pending)")

    # ══════════════════════════════════════════════════════════
    # F: Per-step width comparison
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 6)
    steps = list(range(1, 11))
    if smoke_data and "model_comparison" in smoke_data:
        for name, r in smoke_data["model_comparison"].items():
            ax.plot(steps, r.get("per_step_widths", []), 'o-', linewidth=1.5,
                   markersize=5, label=name, alpha=0.8)
    if has_sampling:
        best_s = min(sampling_rows, key=lambda r: r.get("avg_interval_score", 999))
        if "per_step_width" in best_s and len(best_s.get("per_step_width", [])) == len(steps):
            ax.plot(steps, best_s["per_step_width"], 's--', linewidth=2,
                   markersize=6, label="Sampling best", color="#2196F3")
    if has_training:
        best_t = min(training_rows, key=lambda r: r.get("avg_interval_score", 999))
        if "per_step_width" in best_t:
            pw = best_t.get("per_step_width", [])
            if isinstance(pw, list) and len(pw) == len(steps):
                ax.plot(steps, pw, '^--', linewidth=2,
                       markersize=6, label="Training best", color="#E91E63")
    ax.set_xlabel("Step"); ax.set_ylabel("Interval Width")
    ax.set_title("F. Per-Step Width Comparison", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # ══════════════════════════════════════════════════════════
    # G: Per-step coverage comparison
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 7)
    if smoke_data and "model_comparison" in smoke_data:
        for name, r in smoke_data["model_comparison"].items():
            ax.plot(steps, r.get("per_step_coverages", []), 'o-', linewidth=1.5,
                   markersize=5, label=name, alpha=0.8)
    if has_sampling:
        try:
            pc = best_s.get("per_step_coverage", [])
            if isinstance(pc, list) and len(pc) == len(steps):
                ax.plot(steps, pc, 's--', linewidth=2, markersize=6,
                       label="Sampling best", color="#2196F3")
        except Exception: pass
    if has_training:
        try:
            pc = best_t.get("per_step_coverage", [])
            if isinstance(pc, list) and len(pc) == len(steps):
                ax.plot(steps, pc, '^--', linewidth=2, markersize=6,
                       label="Training best", color="#E91E63")
        except Exception: pass
    ax.axhline(y=0.80, color="gray", linestyle=":", alpha=0.5, label="target 80%")
    ax.set_xlabel("Step"); ax.set_ylabel("Coverage")
    ax.set_title("G. Per-Step Coverage Comparison", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # ══════════════════════════════════════════════════════════
    # H: Training — LR vs IS
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 8)
    if has_training:
        lr_vals = np.array([r.get("lr", 1e-5) for r in tr])
        is_vals = np.array([r["avg_interval_score"] for r in tr])
        conc_vals = np.array([r.get("concentration_weight", 1.0) for r in tr])
        sc = ax.scatter(lr_vals, is_vals, c=conc_vals, cmap="coolwarm",
                       s=40, edgecolors="black", linewidth=0.2, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate"); ax.set_ylabel("Interval Score")
        ax.set_title(f"H. LR vs IS (n={len(tr)})", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)
        plt.colorbar(sc, ax=ax, shrink=0.85).set_label("conc_weight", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("H. LR vs IS (pending)")

    # ══════════════════════════════════════════════════════════
    # I: Sampling — feed_mode comparison
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 9)
    if has_sampling:
        feed_modes = sorted(set(r.get("feed_mode", "argmax") for r in sr))
        fm_data = {}
        for fm in feed_modes:
            subset = [r["avg_interval_score"] for r in sr if r.get("feed_mode") == fm]
            fm_data[fm] = subset
        bp = ax.boxplot([fm_data[fm] for fm in feed_modes], labels=feed_modes,
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], ["#2196F3", "#4CAF50"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_ylabel("Interval Score")
        ax.set_title("I. Feed Mode Comparison", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("I. Feed Mode (pending)")

    # ══════════════════════════════════════════════════════════
    # J: Training — ci_top_k effect
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 10)
    if has_training:
        topk_vals = sorted(set(int(r.get("ci_top_k", 32)) for r in tr))
        for i, tk in enumerate(topk_vals):
            subset = [r["avg_interval_score"] for r in tr
                      if int(r.get("ci_top_k", 32)) == tk]
            xj = np.random.default_rng(100 + i).uniform(-0.08, 0.08, len(subset))
            ax.scatter(np.full(len(subset), i) + xj, subset, s=20, alpha=0.4,
                      color="#FF9800", edgecolors="none")
            if subset:
                mean_v = np.mean(subset)
                ax.scatter([i], [mean_v], s=100, marker="D", color="#E91E63",
                          edgecolors="white", linewidth=1.5, zorder=5)
        ax.set_xticks(range(len(topk_vals)))
        ax.set_xticklabels([str(v) for v in topk_vals], fontsize=8)
        ax.set_xlabel("ci_top_k"); ax.set_ylabel("Interval Score")
        ax.set_title("J. Top-K Effect", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("J. Top-K (pending)")

    # ══════════════════════════════════════════════════════════
    # K: Training time distribution
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 11)
    if has_training:
        times = [r.get("elapsed_min", 0) for r in tr if "elapsed_min" in r]
        if times:
            ax.hist(times, bins=15, color="#9C27B0", alpha=0.7, edgecolor="white")
            ax.axvline(x=np.mean(times), color="white", linestyle="--", linewidth=2,
                      label=f"mean={np.mean(times):.1f}min")
            ax.set_xlabel("Elapsed (min)"); ax.set_ylabel("Trials")
            ax.set_title(f"K. Trial Duration ",
                        fontsize=10, fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No timing data", ha="center", va="center")
            ax.set_title("K. Trial Duration")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("K. Duration (pending)")

    # ══════════════════════════════════════════════════════════
    # L: Key Findings panel
    # ══════════════════════════════════════════════════════════
    ax = plt.subplot(3, 4, 12)
    ax.axis("off")

    lines = [
        "Phase 7 CI HPO — KEY FINDINGS",
        "",
    ]

    if has_sampling:
        best_s = min(sampling_rows, key=lambda r: r.get("avg_interval_score", 999))
        lines += [
            "--- CI Sampling (Idea 1) ---",
            f"Best IS: {best_s['avg_interval_score']:.6f}",
            f"  T={best_s['temperature']} N={int(best_s['num_samples'])}",
            f"  C={best_s['confidence_level']} fm={best_s.get('feed_mode','?')}",
            f"  Coverage: {best_s.get('coverage', 0):.4f}",
            f"  Avg Width: {best_s.get('avg_width', 0):.6f}",
            "",
        ]

    if has_training:
        best_t = min(training_rows, key=lambda r: r.get("avg_interval_score", 999))
        lines += [
            "--- CI Training (Idea 2) ---",
            f"Best IS: {best_t['avg_interval_score']:.6f}",
            f"  conc_w={best_t.get('concentration_weight','?')}",
            f"  is_w={best_t.get('interval_score_weight','?')}",
            f"  topk={best_t.get('ci_top_k','?')}",
            f"  lr={best_t.get('lr','?'):.2e}",
            f"  Coverage: {best_t.get('coverage', 0):.4f}",
            f"  Avg Width: {best_t.get('avg_width', 0):.6f}",
            "",
        ]

    if has_sampling and has_training:
        delta_is = best_s["avg_interval_score"] - best_t["avg_interval_score"]
        winner = "Sampling" if delta_is < 0 else "Training"
        lines += [
            "--- Head-to-Head ---",
            f"ΔIS = {delta_is:+.6f} ({winner} wins)",
        ]

    if smoke_data and "train_timing" in smoke_data:
        tt = smoke_data["train_timing"]
        lines += [
            "",
            f"Est. training HPO: {tt['est_total_hpo_hrs']:.1f} hrs",
            f"Est. per trial: {tt['est_time_per_trial_min']:.1f} min",
        ]

    ax.text(0.05, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=7.5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PNG}")


def main():
    sampling_rows = load_sampling()
    training_rows = load_training()
    smoke_data = load_smoke_test()

    print(f"Phase 7 CI plots")
    print(f"  Sampling configs: {len(sampling_rows)}")
    print(f"  Training trials:  {len(training_rows)}")
    print(f"  Smoke data:       {'yes' if smoke_data else 'no'}")

    plot(sampling_rows, training_rows, smoke_data)


if __name__ == "__main__":
    main()
