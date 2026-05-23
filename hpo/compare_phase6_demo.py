"""Compare BaseModel vs Best Phase6 Rollout Model on Demo set — 10-step AR.

Generates:
  - trials/phase6_demo_comparison.json  (raw metrics)
  - trials/phase6_demo_comparison.png   (error accumulation chart)
"""

import json, os, sys
from argparse import Namespace
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from config import DataConfig
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT
from posttrain.rollout.data import RolloutWindowDataset, rollout_collate

# ── Paths ──
TOKENIZER_PATH      = os.path.join(_PROJECT_ROOT, "checkpoints", "tokenizer.pt")
TOKENIZER_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "tokenizer_config.json")
BASEMODEL_PATH       = os.path.join(_PROJECT_ROOT, "checkpoints", "base_model.pt")
ROLLOUT_MODEL_PATH   = os.path.join(_PROJECT_ROOT, "trials", "phase6_rollout", "trial_006", "rollout_model.pt")
OUTPUT_JSON          = os.path.join(_PROJECT_ROOT, "trials", "phase6_demo_comparison.json")
OUTPUT_PNG           = os.path.join(_PROJECT_ROOT, "trials", "phase6_demo_comparison.png")

PREFIX_LEN = 1023
HORIZON = 10
TOKENIZER_VOCAB = 1 << 10

BACKBONE = {
    "dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
    "dsa_windows": [None, 512, 512],
    "position_encoding": "rope", "rope_base": 10000.0,
    "dropout": 0.1323, "use_revin": False, "num_factor_tokens": 0,
}


def _make_rollout_cfg():
    return Namespace(
        prefix_len=PREFIX_LEN, horizon=HORIZON,
        stride_ratio=DataConfig.stride_ratio,
        cache_dir=os.path.join(_PROJECT_ROOT, "posttrain", "rollout", "cache"),
        max_stocks=0, cache_rebuild=False,
    )


def load_tokenizer(device):
    ckpt = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg and os.path.exists(TOKENIZER_CONFIG_PATH):
        with open(TOKENIZER_CONFIG_PATH) as f:
            cfg = json.load(f)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tok.load_state_dict(ckpt["model_state_dict"], strict=False)
    tok.eval()
    tok.requires_grad_(False)
    return tok


def build_model(device):
    bp = BACKBONE
    return KronosReasoningGPT(
        dim=bp["dim"], depth=bp["depth"], heads=bp["heads"],
        num_kv_heads=bp["num_kv_heads"], dsa_windows=bp["dsa_windows"],
        dropout=bp["dropout"], vocab_size_coarse=TOKENIZER_VOCAB,
        vocab_size_fine=TOKENIZER_VOCAB,
        position_encoding=bp["position_encoding"], rope_base=bp["rope_base"],
        use_revin=bp["use_revin"], num_factor_tokens=bp["num_factor_tokens"],
    ).to(device)


def load_basemodel(device):
    model = build_model(device)
    ckpt = torch.load(BASEMODEL_PATH, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def load_rollout_model(device):
    model = build_model(device)
    ckpt = torch.load(ROLLOUT_MODEL_PATH, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate_ar(model, tokenizer, loader, device):
    """Run 10-step AR evaluation, return per-step metrics."""
    model.eval()

    # Accumulators: per-step
    per_step_path_mape = [[] for _ in range(HORIZON)]
    per_step_daily_mape = [[] for _ in range(HORIZON)]
    per_step_pred_returns = [[] for _ in range(HORIZON)]
    per_step_actual_returns = [[] for _ in range(HORIZON)]

    n_batches = 0
    for batch in tqdm(loader, desc="  Eval AR10", leave=False):
        feats  = batch["features"].to(device=device, dtype=torch.float32)
        means  = batch["means"].to(device=device, dtype=torch.float32)
        stds   = batch["stds"].to(device=device, dtype=torch.float32)
        actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
        times  = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}

        B = feats.shape[0]
        if B == 0: continue
        n_batches += 1

        idx_c, idx_f = tokenizer.encode(feats)
        cur_c = idx_c[:, :PREFIX_LEN].clone()
        cur_f = idx_f[:, :PREFIX_LEN].clone()
        actual_rets = actual.cpu()

        pred_rets = []
        for step in range(HORIZON):
            sl = int(cur_c.size(1))
            cur_time = {
                "minute": times["minute"][:, :sl],
                "day":    times["day"][:, :sl],
                "month":  times["month"][:, :sl],
                "year":   times["year"][:, :sl],
            }
            logits_c, logits_f, _ = model(
                cur_c, cur_f,
                cur_time["minute"], cur_time["day"],
                cur_time["month"], cur_time["year"],
                last_only=True,
            )
            pc = logits_c[:, -1, :].argmax(dim=-1)
            pf = logits_f[:, -1, :].argmax(dim=-1)
            dec = tokenizer.decode(pc.unsqueeze(1), pf.unsqueeze(1))
            pred_norm = dec[:, 0, 0].cpu().float()
            pred_ret = pred_norm * stds[:, 0].cpu() + means[:, 0].cpu()
            pred_rets.append(pred_ret)

            if step < HORIZON - 1:
                cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
                cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)

        pred_rets = torch.stack(pred_rets, dim=1)  # [B, 10]
        cum_pred   = torch.cumsum(pred_rets.float(), dim=1)
        cum_actual = torch.cumsum(actual_rets.float(), dim=1)

        for step in range(HORIZON):
            # Path MAPE
            pr = torch.exp(torch.clamp(cum_pred[:, step], -20, 20))
            ar = torch.exp(torch.clamp(cum_actual[:, step], -20, 20))
            denom = torch.clamp(torch.abs(ar), min=1e-6)
            valid = torch.isfinite(pr) & torch.isfinite(ar) & (denom > 0)
            if valid.sum() > 0:
                m = (torch.abs(pr[valid] - ar[valid]) / denom[valid]).mean().item() * 100
                per_step_path_mape[step].append(m)

            # Daily MAPE
            dr = torch.exp(torch.clamp(pred_rets[:, step].float(), -20, 20))
            da = torch.exp(torch.clamp(actual_rets[:, step].float(), -20, 20))
            denom_d = torch.clamp(torch.abs(da), min=1e-6)
            valid_d = torch.isfinite(dr) & torch.isfinite(da) & (denom_d > 0)
            if valid_d.sum() > 0:
                md = (torch.abs(dr[valid_d] - da[valid_d]) / denom_d[valid_d]).mean().item() * 100
                per_step_daily_mape[step].append(md)

            # Store for distribution
            per_step_pred_returns[step].append(pred_rets[:, step].numpy())
            per_step_actual_returns[step].append(actual_rets[:, step].numpy())

    # Aggregate
    result = {
        "num_batches": n_batches,
        "horizon": HORIZON,
    }
    for step in range(HORIZON):
        vals = per_step_path_mape[step]
        result[f"step_{step+1}_path_mape_mean"] = float(np.mean(vals)) if vals else float("nan")
        result[f"step_{step+1}_path_mape_std"]  = float(np.std(vals)) if vals else float("nan")
        result[f"step_{step+1}_num_samples"] = len(vals)

        dvals = per_step_daily_mape[step]
        result[f"step_{step+1}_daily_mape_mean"] = float(np.mean(dvals)) if dvals else float("nan")

        if per_step_pred_returns[step]:
            all_pred = np.concatenate(per_step_pred_returns[step])
            all_act = np.concatenate(per_step_actual_returns[step])
            result[f"step_{step+1}_pred_mean"] = float(np.mean(all_pred))
            result[f"step_{step+1}_actual_mean"] = float(np.mean(all_act))
            result[f"step_{step+1}_pred_std"] = float(np.std(all_pred))
            result[f"step_{step+1}_actual_std"] = float(np.std(all_act))

    # Overall metrics
    all_path = []
    for step_vals in per_step_path_mape:
        all_path.extend(step_vals)
    result["path_mape_overall"] = float(np.mean(all_path)) if all_path else float("nan")

    all_daily = []
    for step_vals in per_step_daily_mape:
        all_daily.extend(step_vals)
    result["daily_mape_overall"] = float(np.mean(all_daily)) if all_daily else float("nan")

    return result


def plot_comparison(base_metrics, rollout_metrics):
    """Create multi-panel comparison chart."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Phase 6: BaseModel vs Rollout (Oracle-Guided) — Demo Set 10-Step AR",
                 fontsize=13, fontweight="bold")

    steps = list(range(1, HORIZON + 1))
    base_path   = [base_metrics[f"step_{s}_path_mape_mean"] for s in steps]
    base_path_s = [base_metrics[f"step_{s}_path_mape_std"] for s in steps]
    rollout_path   = [rollout_metrics[f"step_{s}_path_mape_mean"] for s in steps]
    rollout_path_s = [rollout_metrics[f"step_{s}_path_mape_std"] for s in steps]

    base_daily   = [base_metrics[f"step_{s}_daily_mape_mean"] for s in steps]
    rollout_daily   = [rollout_metrics[f"step_{s}_daily_mape_mean"] for s in steps]

    # ═══ A: Path MAPE accumulation ═══
    ax = axes[0, 0]
    ax.errorbar(steps, base_path, yerr=base_path_s, marker="o", linewidth=2,
                markersize=7, capsize=3, label="BaseModel", color="#2196F3")
    ax.errorbar(steps, rollout_path, yerr=rollout_path_s, marker="s", linewidth=2,
                markersize=7, capsize=3, label="Rollout (Phase6)", color="#E91E63")
    ax.set_xlabel("AR Step"); ax.set_ylabel("Path MAPE (%)")
    ax.set_title("A. Cumulative Path MAPE by Step", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.set_xticks(steps)

    # Annotate improvement
    for s in steps:
        diff = rollout_path[s-1] - base_path[s-1]
        if abs(diff) > 0.001:
            color = "green" if diff < 0 else "red"
            ax.annotate(f"{diff:+.2f}", (s, rollout_path[s-1]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, color=color, ha="center")

    # ═══ B: Daily MAPE ═══
    ax = axes[0, 1]
    ax.plot(steps, base_daily, marker="o", linewidth=2, markersize=7,
            label="BaseModel", color="#2196F3")
    ax.plot(steps, rollout_daily, marker="s", linewidth=2, markersize=7,
            label="Rollout (Phase6)", color="#E91E63")
    ax.set_xlabel("AR Step"); ax.set_ylabel("Daily MAPE (%)")
    ax.set_title("B. Daily (1-Step) MAPE by Step", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.set_xticks(steps)

    # ═══ C: Path MAPE difference (Rollout - Base) ═══
    ax = axes[0, 2]
    diffs = [rollout_path[s-1] - base_path[s-1] for s in steps]
    colors_bar = ["#4CAF50" if d < 0 else "#F44336" for d in diffs]
    ax.bar(steps, diffs, color=colors_bar, alpha=0.8, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xlabel("AR Step"); ax.set_ylabel("Δ Path MAPE (pp)")
    ax.set_title("C. Rollout − BaseModel (negative = improvement)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(steps)
    for s, d in zip(steps, diffs):
        ax.text(s, d + (0.01 if d >= 0 else -0.03), f"{d:+.2f}", ha="center", fontsize=8,
                fontweight="bold", color=colors_bar[s-1])

    # ═══ D: Predicted vs Actual mean return by step ═══
    ax = axes[1, 0]
    base_pred_mean = [base_metrics.get(f"step_{s}_pred_mean", np.nan) for s in steps]
    base_act_mean  = [base_metrics.get(f"step_{s}_actual_mean", np.nan) for s in steps]
    rollout_pred_mean = [rollout_metrics.get(f"step_{s}_pred_mean", np.nan) for s in steps]

    ax.plot(steps, base_act_mean, marker="D", linewidth=2.5, markersize=8,
            label="Actual Return", color="black", linestyle="--")
    ax.plot(steps, base_pred_mean, marker="o", linewidth=1.5, markersize=6,
            label="BaseModel Pred", color="#2196F3")
    ax.plot(steps, rollout_pred_mean, marker="s", linewidth=1.5, markersize=6,
            label="Rollout Pred", color="#E91E63")
    ax.set_xlabel("AR Step"); ax.set_ylabel("Mean Log Return")
    ax.set_title("D. Mean Predicted vs Actual Return", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_xticks(steps)

    # ═══ E: Summary table ═══
    ax = axes[1, 1]
    ax.axis("off")

    bp = base_metrics["path_mape_overall"]
    rp = rollout_metrics["path_mape_overall"]
    bd = base_metrics["daily_mape_overall"]
    rd = rollout_metrics["daily_mape_overall"]
    improvement_path = bp - rp
    improvement_daily = bd - rd

    lines = [
        "Demo Set 10-Step AR — Summary",
        "",
        f"BaseModel Path MAPE:  {bp:.4f}%",
        f"Rollout   Path MAPE:  {rp:.4f}%",
        f"Improvement:          {improvement_path:+.4f}pp",
        f"                      ({improvement_path/bp*100:+.1f}% relative)",
        "",
        f"BaseModel Daily MAPE: {bd:.4f}%",
        f"Rollout   Daily MAPE: {rd:.4f}%",
        f"Improvement:          {improvement_daily:+.4f}pp",
        "",
        f"BaseModel batches:    {base_metrics['num_batches']}",
        f"Rollout   batches:    {rollout_metrics['num_batches']}",
        "",
        "Best Phase6 config:",
        "  trial_006",
        "  oracle_top_k=8, temp=0.52",
        "  kl=0.029, lr=2.1e-5, upd=480",
    ]
    color_text = "green" if improvement_path > 0 else "red"
    ax.text(0.05, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.9),
            color=color_text)

    # ═══ F: Path MAPE growth rate comparison ═══
    ax = axes[1, 2]
    # Relative growth: step_k / step_1
    base_rel = [base_path[s-1] / base_path[0] for s in steps]
    rollout_rel = [rollout_path[s-1] / rollout_path[0] for s in steps]
    ax.plot(steps, base_rel, marker="o", linewidth=2, markersize=7,
            label="BaseModel", color="#2196F3")
    ax.plot(steps, rollout_rel, marker="s", linewidth=2, markersize=7,
            label="Rollout (Phase6)", color="#E91E63")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("AR Step"); ax.set_ylabel("Relative Path MAPE (× step 1)")
    ax.set_title("F. Error Accumulation Rate\n(Path MAPE relative to step 1)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.set_xticks(steps)

    # Annotate final ratio
    ax.annotate(f"{base_rel[-1]:.2f}×", (10, base_rel[-1]),
                textcoords="offset points", xytext=(0, 10),
                fontsize=9, color="#2196F3", fontweight="bold", ha="center")
    ax.annotate(f"{rollout_rel[-1]:.2f}×", (10, rollout_rel[-1]),
                textcoords="offset points", xytext=(0, -15),
                fontsize=9, color="#E91E63", fontweight="bold", ha="center")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PNG}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer once
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(device)

    # Load BaseModel
    print("Loading BaseModel...")
    base_model = load_basemodel(device)
    print(f"  {sum(p.numel() for p in base_model.parameters()):,} params")

    # Load Rollout model
    print(f"Loading Rollout model from {ROLLOUT_MODEL_PATH}...")
    rollout_model = load_rollout_model(device)

    # Build Demo dataset
    print("\nBuilding Demo dataset...")
    cfg = _make_rollout_cfg()
    demo_ds = RolloutWindowDataset("demo", cfg=cfg, max_samples=0, seed=42)
    print(f"  Demo windows: {len(demo_ds)}")

    demo_loader = torch.utils.data.DataLoader(
        demo_ds, batch_size=8, shuffle=False,
        collate_fn=rollout_collate,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # Evaluate BaseModel
    print("\n" + "=" * 60)
    print("Evaluating BaseModel on Demo set (10-step AR)...")
    print("=" * 60)
    base_metrics = evaluate_ar(base_model, tokenizer, demo_loader, device)
    base_metrics["model"] = "BaseModel"
    print(f"  Path MAPE: {base_metrics['path_mape_overall']:.4f}%")
    print(f"  Daily MAPE: {base_metrics['daily_mape_overall']:.4f}%")
    for s in range(1, HORIZON + 1):
        print(f"    Step {s:2d}: path_mape={base_metrics[f'step_{s}_path_mape_mean']:.4f}%  "
              f"daily={base_metrics[f'step_{s}_daily_mape_mean']:.4f}%")

    # Evaluate Rollout model
    print("\n" + "=" * 60)
    print("Evaluating Rollout model on Demo set (10-step AR)...")
    print("=" * 60)
    rollout_metrics = evaluate_ar(rollout_model, tokenizer, demo_loader, device)
    rollout_metrics["model"] = "Phase6_Rollout_trial006"
    print(f"  Path MAPE: {rollout_metrics['path_mape_overall']:.4f}%")
    print(f"  Daily MAPE: {rollout_metrics['daily_mape_overall']:.4f}%")
    for s in range(1, HORIZON + 1):
        print(f"    Step {s:2d}: path_mape={rollout_metrics[f'step_{s}_path_mape_mean']:.4f}%  "
              f"daily={rollout_metrics[f'step_{s}_daily_mape_mean']:.4f}%")

    # Comparison
    improvement = base_metrics["path_mape_overall"] - rollout_metrics["path_mape_overall"]
    print(f"\n{'='*60}")
    print(f"Improvement: {improvement:+.4f}pp ({improvement/base_metrics['path_mape_overall']*100:+.1f}%)")
    print(f"{'='*60}")

    # Save JSON
    comparison = {
        "base_model": base_metrics,
        "rollout_model": rollout_metrics,
        "improvement_path_mape_pp": improvement,
        "improvement_relative_pct": float(improvement / base_metrics["path_mape_overall"] * 100),
        "best_trial": "trial_006",
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_JSON}")

    # Plot
    plot_comparison(base_metrics, rollout_metrics)


if __name__ == "__main__":
    main()
