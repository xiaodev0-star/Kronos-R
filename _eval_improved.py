"""Evaluate BaseModel vs Improved STAR-CAST on Demo dataset."""
import os, sys, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace

from evaluate_predictions import load_model
from posttrain.rollout.data import RolloutWindowDataset, rollout_collate, resolve_project_path
from posttrain.rollout.train_rollout import (
    _amp_dtype, _autocast_context, _move_batch, _encode_features, compute_rollout_metrics,
)
from reproducibility import set_global_seed

set_global_seed(42, deterministic=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

amp_dtype = _amp_dtype("bfloat16")
amp_enabled = device.type == "cuda"
actionable_threshold = 0.005  # 0.5%

cfg = Namespace(
    prefix_len=1023, horizon=10, stride_ratio=0.5,
    cache_dir=resolve_project_path("posttrain/rollout/cache"),
    max_stocks=0, cache_rebuild=False,
    mape_eps=1e-4,
)

print("Loading Demo dataset...")
demo_dataset = RolloutWindowDataset("val", cfg=cfg, max_samples=0, seed=999)
print(f"Demo windows={len(demo_dataset)}")

loader_kwargs = {"num_workers": 0, "pin_memory": True, "collate_fn": rollout_collate}
demo_loader = DataLoader(demo_dataset, batch_size=2, shuffle=False, drop_last=False, **loader_kwargs)

@torch.inference_mode()
def eval_checkpoint(ckpt_path, label, max_batches=200):
    """Evaluate a checkpoint with autoregressive 10-step rollout."""
    print(f"\nLoading: {label}")
    torch.cuda.empty_cache()
    model, tokenizer = load_model(device=device, checkpoint_path=ckpt_path, strict_checkpoint_compat=False)
    tokenizer.eval(); tokenizer.requires_grad_(False)
    model.eval()

    all_pred, all_actual = [], []
    n_batches = 0
    for batch in tqdm(demo_loader, desc=f"Eval {label}", total=min(max_batches, len(demo_loader))):
        batch = _move_batch(batch, device)
        idx_c, idx_f = _encode_features(tokenizer, batch["features"])
        cur_c, cur_f = idx_c[:, :1023].clone(), idx_f[:, :1023].clone()
        pred_rets = []
        for step in range(10):
            sl = cur_c.size(1)
            cur_time = {k: v[:, :sl] for k, v in batch["time"].items()}
            with _autocast_context(device, amp_enabled, amp_dtype):
                logits_c, logits_f, _ = model(cur_c, cur_f,
                    cur_time["minute"], cur_time["day"],
                    cur_time["month"], cur_time["year"], last_only=True)
            pc = logits_c[:, -1, :].argmax(dim=-1)
            pf = logits_f[:, -1, :].argmax(dim=-1)
            dec = tokenizer.decode(pc.unsqueeze(1), pf.unsqueeze(1))
            ret = dec[:, 0, 0].cpu().float() * batch["stds"][:, 0].cpu() + batch["means"][:, 0].cpu()
            pred_rets.append(ret)
            if step < 9:
                cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
                cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)
        all_pred.append(torch.stack(pred_rets, dim=1))
        all_actual.append(batch["actual_returns"].cpu())
        n_batches += 1
        if n_batches >= max_batches: break

    pred = torch.cat(all_pred, dim=0).numpy()
    actual = torch.cat(all_actual, dim=0).numpy()
    metrics = compute_rollout_metrics(pred, actual, mape_eps=float(cfg.mape_eps))

    # Actionable DA
    pred_t = torch.from_numpy(pred)
    actual_t = torch.from_numpy(actual)
    pred_sign = (pred_t >= 0).float() * 2 - 1
    actual_sign = (actual_t >= 0).float() * 2 - 1
    conf_mask = torch.abs(pred_t) > actionable_threshold
    if conf_mask.sum() > 0:
        metrics["actionable_da"] = float((pred_sign[conf_mask] == actual_sign[conf_mask]).float().mean().item() * 100)
        metrics["actionable_ratio"] = float(conf_mask.float().mean().item() * 100)
    else:
        metrics["actionable_da"] = 0.0; metrics["actionable_ratio"] = 0.0

    # Step 1 actionable DA
    conf_s1 = torch.abs(pred_t[:, 0]) > actionable_threshold
    if conf_s1.sum() > 0:
        metrics["step1_actionable_da"] = float((pred_sign[conf_s1, 0] == actual_sign[conf_s1, 0]).float().mean().item() * 100)
        metrics["step1_actionable_ratio"] = float(conf_s1.float().mean().item() * 100)
    else:
        metrics["step1_actionable_da"] = 0.0; metrics["step1_actionable_ratio"] = 0.0

    # Step 10 actionable DA
    conf_s10 = torch.abs(pred_t[:, 9]) > actionable_threshold
    if conf_s10.sum() > 0:
        metrics["step10_actionable_da"] = float((pred_sign[conf_s10, 9] == actual_sign[conf_s10, 9]).float().mean().item() * 100)
        metrics["step10_actionable_ratio"] = float(conf_s10.float().mean().item() * 100)
    else:
        metrics["step10_actionable_da"] = 0.0; metrics["step10_actionable_ratio"] = 0.0

    del model; torch.cuda.empty_cache()
    return metrics

# ── Evaluate both models ──
base_path = resolve_project_path("checkpoints/base_model.pt")
improved_path = resolve_project_path("checkpoints/post_train_star_cast/star_cast_improved_test.pt")
# Also evaluate the previous best (nosched-step160)
prev_best_path = resolve_project_path("checkpoints/post_train_star_cast/star_cast_nosched-step160.pt")

base_m = eval_checkpoint(base_path, "BaseModel")
improved_m = eval_checkpoint(improved_path, "Improved-120step")
prev_best_m = eval_checkpoint(prev_best_path, "Previous-Best-160step")

# ── Print comparison ──
print("\n" + "=" * 90)
print(f"{'Metric':<30} {'BaseModel':>14} {'PrevBest':>14} {'Improved-120':>14} {'Delta(B→I)':>14}")
print("-" * 90)

rows = [
    ("Path MAPE (%)", "path_mape", "{:.4f}"),
    ("Daily MAPE (%)", "mape", "{:.4f}"),
    ("DA (%)", "da", "{:.2f}"),
    ("Actionable DA (%)", "actionable_da", "{:.2f}"),
    ("Actionable Ratio (%)", "actionable_ratio", "{:.2f}"),
    ("Step 1 Act DA (%)", "step1_actionable_da", "{:.2f}"),
    ("Step 1 Act Ratio (%)", "step1_actionable_ratio", "{:.2f}"),
    ("Step 10 Act DA (%)", "step10_actionable_da", "{:.2f}"),
    ("Step 10 Act Ratio (%)", "step10_actionable_ratio", "{:.2f}"),
    ("Path MAE", "path_mae", "{:.6f}"),
    ("Path RMSE", "path_rmse", "{:.6f}"),
    ("Num Samples", "num_samples", "{:.0f}"),
]

for name, key, fmt in rows:
    b = base_m.get(key, float("nan"))
    pb = prev_best_m.get(key, float("nan"))
    i = improved_m.get(key, float("nan"))
    delta = i - b if not (isinstance(i, float) and np.isnan(i)) and not (isinstance(b, float) and np.isnan(b)) else float("nan")
    b_s = fmt.format(b) if not (isinstance(b, float) and np.isnan(b)) else "N/A"
    pb_s = fmt.format(pb) if not (isinstance(pb, float) and np.isnan(pb)) else "N/A"
    i_s = fmt.format(i) if not (isinstance(i, float) and np.isnan(i)) else "N/A"
    d_s = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
    if "MAPE" in name and not np.isnan(delta):
        arrow = "↓ BETTER" if delta < 0 else "↑ worse"
        d_s += f" {arrow}"
    elif "DA" in name and not np.isnan(delta):
        arrow = "↑ BETTER" if delta > 0 else "↓ worse"
        d_s += f" {arrow}"
    print(f"{name:<30} {b_s:>14} {pb_s:>14} {i_s:>14} {d_s:>14}")

# Per-step comparison
print("\n" + "=" * 90)
print(f"{'Per-Step Path MAPE':^90}")
print("-" * 90)
print(f"{'Step':<10} {'BaseModel':>14} {'PrevBest':>14} {'Improved':>14} {'Delta(B→I)':>18}")
print("-" * 90)
for i, (bs, ps, ims) in enumerate(zip(
    base_m.get("per_step", []),
    prev_best_m.get("per_step", []),
    improved_m.get("per_step", []),
)):
    bp = bs["path_mape"]; pp = ps["path_mape"]; ip = ims["path_mape"]
    delta = ip - bp
    print(f"{i+1:<10} {bp:>14.4f} {pp:>14.4f} {ip:>14.4f} {delta:>+17.4f}")

print("=" * 90)
print(f"\nBaseModel path_mape: {base_m['path_mape']:.4f}%")
print(f"PrevBest path_mape:   {prev_best_m['path_mape']:.4f}%")
print(f"Improved path_mape:   {improved_m['path_mape']:.4f}%")
print(f"Delta (Improved - BaseModel): {improved_m['path_mape'] - base_m['path_mape']:+.4f}pp")
print(f"Delta (Improved - PrevBest):  {improved_m['path_mape'] - prev_best_m['path_mape']:+.4f}pp")
