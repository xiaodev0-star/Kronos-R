"""Visualize 16 random stock predictions vs actual — check for zero-collapse."""
import os, sys, random, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace

from evaluate_predictions import load_model
from posttrain.rollout.data import RolloutWindowDataset, rollout_collate, resolve_project_path
from posttrain.rollout.train_rollout import _amp_dtype, _autocast_context, _move_batch, _encode_features

BEST_CKPT = "trials/phase8_star_cast_v4/Phase2-refine/trial_009/star_cast_model.pt"
N_STOCKS = 16; HORIZON = 10; PLOT_COLS = 4
os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high"); torch.cuda.empty_cache()
amp_dtype = _amp_dtype("bfloat16"); amp_enabled = True

print("Loading model...")
model, tokenizer = load_model(device=device, checkpoint_path=resolve_project_path(BEST_CKPT), strict_checkpoint_compat=False)
tokenizer.eval(); tokenizer.requires_grad_(False); model.eval()

cfg = Namespace(prefix_len=1023, horizon=HORIZON, stride_ratio=0.5,
    cache_dir=resolve_project_path("posttrain/rollout/cache"), max_stocks=0, cache_rebuild=False, mape_eps=1e-4)
print("Loading demo dataset...")
demo_dataset = RolloutWindowDataset("val", cfg=cfg, max_samples=0, seed=999)
print(f"Demo windows: {len(demo_dataset)}")

# Pick random indices
rng = random.Random(42)
indices = rng.sample(range(len(demo_dataset)), min(N_STOCKS, len(demo_dataset)))

@torch.inference_mode()
def predict_one(item):
    """Single-sample autoregressive prediction."""
    # Build batch of size 1 using rollout_collate
    batch = rollout_collate([item])
    batch = _move_batch(batch, device)
    idx_c, idx_f = _encode_features(tokenizer, batch["features"])
    cur_c = idx_c[:, :1023].clone(); cur_f = idx_f[:, :1023].clone()
    preds = []
    for step in range(HORIZON):
        sl = int(cur_c.size(1))
        ct = {k: v[:, :sl] for k, v in batch["time"].items()}
        with _autocast_context(device, amp_enabled, amp_dtype):
            lc, lf, _ = model(cur_c, cur_f, ct["minute"], ct["day"], ct["month"], ct["year"], last_only=True)
        pc = lc[:, -1, :].argmax(dim=-1); pf = lf[:, -1, :].argmax(dim=-1)
        dec = tokenizer.decode(pc.unsqueeze(1), pf.unsqueeze(1))
        ret = dec[:, 0, 0].cpu().float() * batch["stds"][:, 0].cpu() + batch["means"][:, 0].cpu()
        preds.append(ret)
        if step < HORIZON - 1:
            cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
            cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)
    pred = torch.stack(preds, dim=1)[0].numpy()
    actual = item["actual_returns"][:HORIZON].numpy()
    symbol = item.get("symbol", "?")
    return pred, actual, str(symbol)

print(f"Predicting {len(indices)} stocks...")
results = []
for idx in tqdm(indices):
    item = demo_dataset[idx]
    pred, actual, sym = predict_one(item)
    results.append((pred, actual, sym))

# ═══════════════ Analysis ═══════════════
all_preds = np.concatenate([r[0] for r in results])
all_actuals = np.concatenate([r[1] for r in results])

print(f"\n{'='*55}")
print(f"Zero-collapse analysis ({len(results)} stocks x {HORIZON} steps = {len(all_preds)} preds)")
print(f"{'='*55}")

abs_pred = np.abs(all_preds)
print(f"\n|Pred| statistics:")
print(f"  Mean={np.mean(abs_pred):.6f}  Median={np.median(abs_pred):.6f}  Std={np.std(abs_pred):.6f}")
for pct in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{pct}={np.percentile(abs_pred, pct):.6f}")

for t in [0.001, 0.002, 0.005, 0.01]:
    print(f"  |pred|<{t:.3f}: {np.mean(abs_pred<t)*100:.1f}%")

pred_sign = np.sign(all_preds + 1e-10)
actual_sign = np.sign(all_actuals + 1e-10)
da = np.mean(pred_sign == actual_sign) * 100
print(f"\nDirection Accuracy: {da:.1f}%")

conf = abs_pred > 0.005
if conf.sum() > 0:
    act_da = np.mean(pred_sign[conf] == actual_sign[conf]) * 100
    act_ratio = conf.mean() * 100
    print(f"Actionable DA (|pred|>0.5%): {act_da:.1f}% (ratio={act_ratio:.1f}%)")

# Regression
from scipy import stats as scistats
slope, intercept, r_val, p_val, _ = scistats.linregress(all_actuals, all_preds)
print(f"\npred = {slope:.4f} * actual + {intercept:.6f}   R^2={r_val**2:.4f}")

# ═══════════════ Plot ═══════════════
PLOT_ROWS = 4
fig, axes = plt.subplots(PLOT_ROWS, PLOT_COLS, figsize=(20, 17))
fig.suptitle(f"STAR-CAST V4 (8-2) — 10-Step Predictions vs Actual (16 Random Stocks)\n"
             f"DA={da:.1f}% | ActDA={act_da:.1f}% | ActRatio={act_ratio:.1f}% | "
             f"Pred-Act R²={r_val**2:.3f}",
             fontsize=13, fontweight="bold")

days = np.arange(1, HORIZON+1)
for i, (pred, actual, sym) in enumerate(results):
    row, col = divmod(i, PLOT_COLS)
    ax = axes[row, col]
    cum_pred = np.cumsum(pred); cum_actual = np.cumsum(actual)

    # Per-step bars
    w = 0.35; x = np.arange(HORIZON)
    ax.bar(x-w/2, pred, w, color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.2, label="Pred")
    ax.bar(x+w/2, actual, w, color="#FF5722", alpha=0.85, edgecolor="white", linewidth=0.2, label="Actual")

    # Cumulative line
    ax2 = ax.twinx()
    ax2.plot(days, cum_pred, "o-", color="#0D47A1", lw=2.0, ms=4, label="Cum Pred")
    ax2.plot(days, cum_actual, "s--", color="#BF360C", lw=2.0, ms=4, label="Cum Actual")

    ax.set_title(f"{sym}", fontsize=9, fontweight="bold")
    ax.set_xticks(range(HORIZON)); ax.set_xticklabels([f"{d}" for d in days], fontsize=7)
    ax.axhline(y=0, color="#888", lw=0.5, ls=":")
    ax.set_ylabel("Log-Return", fontsize=7); ax2.set_ylabel("Cum.", fontsize=7)
    ax.tick_params(labelsize=6); ax2.tick_params(labelsize=6)

    # Show da for this stock
    s_da = np.mean(np.sign(pred+1e-10) == np.sign(actual+1e-10))*100
    ax.text(0.98, 0.95, f"DA={s_da:.0f}%", transform=ax.transAxes, fontsize=7,
            ha="right", va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7))

    if i == 0:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, fontsize=7, loc="upper left", ncol=2)

plt.tight_layout(rect=[0,0,1,0.96])
p1 = "outputs/starcast_v4_16stocks.png"
fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {p1}")

# ═══════════════ Distribution plot ═══════════════
fig2, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(14, 5))

ax_h.hist(all_preds, bins=70, color="#2196F3", alpha=0.75, edgecolor="white", lw=0.3)
ax_h.axvline(x=0, color="red", ls="--", lw=1)
ax_h.axvline(x=0.005, color="orange", ls=":", lw=1)
ax_h.axvline(x=-0.005, color="orange", ls=":", lw=1)
ax_h.set_title(f"Predicted Return Distribution\nMean={np.mean(all_preds):.4f} Median={np.median(all_preds):.4f} Std={np.std(all_preds):.4f}", fontsize=11)
ax_h.set_xlabel("Predicted Log-Return"); ax_h.set_ylabel("Count")

ax_s.scatter(all_actuals, all_preds, alpha=0.35, s=12, color="#2196F3", edgecolors="none")
ax_s.axhline(y=0, color="gray", ls=":", lw=0.5); ax_s.axvline(x=0, color="gray", ls=":", lw=0.5)
xx = np.linspace(all_actuals.min(), all_actuals.max(), 100)
ax_s.plot(xx, slope*xx+intercept, "r--", lw=1.5, label=f"Fit (R²={r_val**2:.3f})")
ax_s.plot(xx, xx, "g:", lw=0.8, alpha=0.5, label="Perfect")
ax_s.set_title(f"Predicted vs Actual\nslope={slope:.3f} intercept={intercept:.5f}", fontsize=11)
ax_s.set_xlabel("Actual Log-Return"); ax_s.set_ylabel("Predicted Log-Return")
ax_s.legend(fontsize=8)

plt.tight_layout()
p2 = "outputs/starcast_v4_distribution.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {p2}")
print("\nDone!")
