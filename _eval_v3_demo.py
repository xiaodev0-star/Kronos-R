"""Quick Demo eval of V3 HPO best trial."""
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
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.cuda.empty_cache()

amp_dtype = _amp_dtype("bfloat16")
amp_enabled = device.type == "cuda"

cfg = Namespace(
    prefix_len=1023, horizon=10, stride_ratio=0.5,
    cache_dir=resolve_project_path("posttrain/rollout/cache"),
    max_stocks=0, cache_rebuild=False, mape_eps=1e-4,
)

# Load Demo dataset (full, no limit on stocks)
demo_dataset = RolloutWindowDataset("val", cfg=cfg, max_samples=0, seed=999)
loader_kwargs = {"num_workers": 0, "pin_memory": True, "collate_fn": rollout_collate}
demo_loader = DataLoader(demo_dataset, batch_size=2, shuffle=False, drop_last=False, **loader_kwargs)
print(f"Demo windows: {len(demo_dataset)} (max 400 eval batches)")

THRESHOLD = 0.005

@torch.inference_mode()
def eval_checkpoint(ckpt_path, label, max_batches=400):
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
    metrics = compute_rollout_metrics(pred, actual, mape_eps=1e-4)

    # Actionable DA
    pred_t = torch.from_numpy(pred)
    actual_t = torch.from_numpy(actual)
    pred_sign = (pred_t >= 0).float() * 2 - 1
    actual_sign = (actual_t >= 0).float() * 2 - 1
    conf_mask = torch.abs(pred_t) > THRESHOLD
    if conf_mask.sum() > 0:
        metrics["actionable_da"] = float((pred_sign[conf_mask] == actual_sign[conf_mask]).float().mean().item() * 100)
        metrics["actionable_ratio"] = float(conf_mask.float().mean().item() * 100)
    else:
        metrics["actionable_da"] = 0.0; metrics["actionable_ratio"] = 0.0

    # Per-step actionable DA
    metrics["step_act_da"] = []
    metrics["step_act_ratio"] = []
    for step in range(10):
        cs = torch.abs(pred_t[:, step]) > THRESHOLD
        if cs.sum() > 0:
            metrics["step_act_da"].append(float((pred_sign[cs, step] == actual_sign[cs, step]).float().mean().item() * 100))
            metrics["step_act_ratio"].append(float(cs.float().mean().item() * 100))
        else:
            metrics["step_act_da"].append(0.0)
            metrics["step_act_ratio"].append(0.0)

    del model; torch.cuda.empty_cache()
    return metrics

# Evaluate best trial and baselines
best_path = resolve_project_path("trials/phase8_star_cast_v3/trial_004/star_cast_model.pt")
base_path = resolve_project_path("checkpoints/base_model.pt")
prev_best_path = resolve_project_path("checkpoints/post_train_star_cast/star_cast_nosched-step160.pt")

base_m = eval_checkpoint(base_path, "BaseModel")
best_m = eval_checkpoint(best_path, "V3-Best-Trial004")
prev_m = eval_checkpoint(prev_best_path, "PrevBest-160step")

# Print comparison
print("\n" + "=" * 95)
print(f"{'Metric':<32} {'BaseModel':>14} {'PrevBest':>14} {'V3-Best':>14} {'Delta(B→V3)':>14}")
print("-" * 95)

rows = [
    ("Path MAPE (%)", "path_mape", "{:.4f}"),
    ("Daily MAPE (%)", "mape", "{:.4f}"),
    ("DA (%)", "da", "{:.2f}"),
    ("Path DA (%)", "path_da", "{:.2f}"),
    ("Actionable DA (%)", "actionable_da", "{:.2f}"),
    ("Actionable Ratio (%)", "actionable_ratio", "{:.2f}"),
    ("Path MAE", "path_mae", "{:.6f}"),
    ("Path RMSE", "path_rmse", "{:.6f}"),
    ("Pred Up Ratio (%)", "pred_up_ratio", "{:.2f}"),
]

for name, key, fmt in rows:
    b = base_m.get(key, float("nan"))
    pb = prev_m.get(key, float("nan"))
    v3 = best_m.get(key, float("nan"))
    delta = v3 - b if not np.isnan(v3) and not np.isnan(b) else float("nan")
    b_s = fmt.format(b) if not (isinstance(b, float) and np.isnan(b)) else "N/A"
    pb_s = fmt.format(pb) if not (isinstance(pb, float) and np.isnan(pb)) else "N/A"
    v3_s = fmt.format(v3) if not (isinstance(v3, float) and np.isnan(v3)) else "N/A"
    d_s = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
    print(f"{name:<32} {b_s:>14} {pb_s:>14} {v3_s:>14} {d_s:>14}")

# Per-step actionable DA
print(f"\n{'=' * 95}")
print(f"{'Per-Step Actionable DA':^95}")
print("-" * 95)
print(f"{'Step':<10} {'B:ActDA':>10} {'B:Ratio':>10} {'PB:ActDA':>10} {'PB:Ratio':>10} {'V3:ActDA':>10} {'V3:Ratio':>10}")
print("-" * 95)
for i in range(10):
    b_ad = base_m["step_act_da"][i]; b_ar = base_m["step_act_ratio"][i]
    p_ad = prev_m["step_act_da"][i]; p_ar = prev_m["step_act_ratio"][i]
    v_ad = best_m["step_act_da"][i]; v_ar = best_m["step_act_ratio"][i]
    print(f"{i+1:<10} {b_ad:>10.2f} {b_ar:>10.1f} {p_ad:>10.2f} {p_ar:>10.1f} {v_ad:>10.2f} {v_ar:>10.1f}")

# Per-step Path MAPE
print(f"\n{'=' * 95}")
print(f"{'Per-Step Path MAPE':^95}")
print("-" * 95)
print(f"{'Step':<10} {'BaseModel':>14} {'PrevBest':>14} {'V3-Best':>14} {'Delta(B→V3)':>14}")
print("-" * 95)
for i, (bs, ps, vs) in enumerate(zip(
    base_m.get("per_step", []),
    prev_m.get("per_step", []),
    best_m.get("per_step", []),
)):
    bp = bs["path_mape"]; pp = ps["path_mape"]; vp = vs["path_mape"]
    print(f"{i+1:<10} {bp:>14.4f} {pp:>14.4f} {vp:>14.4f} {vp-bp:>+13.4f}")

print("=" * 95)
print(f"\nV3 Best (Trial 004) params: timidity_w=1.03, oracle_mag=3.99, sharpening=0.933")
print(f"BaseModel path_mape:  {base_m['path_mape']:.4f}%")
print(f"PrevBest path_mape:   {prev_m['path_mape']:.4f}%")
print(f"V3 Best path_mape:    {best_m['path_mape']:.4f}%")
