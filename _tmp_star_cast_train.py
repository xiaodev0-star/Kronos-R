"""STAR-CAST training: GA=16, no LR schedule, 240 steps, ckpt every 80."""
import os, sys, json, time, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_predictions import load_model
from model.lora import trainable_parameter_summary
from posttrain.rollout.data import RolloutWindowDataset, rollout_cache_path, rollout_collate, resolve_project_path
from posttrain.rollout.train_star_cast import train_star_cast_step, _save_star_cast_checkpoint
from posttrain.rollout.train_rollout import (_autocast_context, _amp_dtype, _as_bool,
    _cfg_to_dict, _cuda_peak_memory_stats, _move_batch, _encode_features,
    _configure_trainable, _build_optimizer, _write_history, evaluate_model)
from reproducibility import set_global_seed
from argparse import Namespace

cfg = Namespace(
    random_seed=42, deterministic=False,
    output_dir=resolve_project_path("checkpoints/post_train_star_cast"),
    checkpoint_path=resolve_project_path("checkpoints/base_model.pt"),
    save_name="star_cast_nosched.pt", save_epoch_checkpoints=True,
    prefix_len=1023, horizon=10, stride_ratio=0.5,
    cache_dir=resolve_project_path("posttrain/rollout/cache"), cache_rebuild=False,
    max_stocks=0, max_train_samples=0, max_val_samples=0,
    epochs=1, batch_size=2, eval_batch_size=8,
    accumulation_steps=16, num_workers=0,
    learning_rate=1.98e-5, weight_decay=1e-4, grad_clip=0.5,
    max_train_updates=240, progress_interval=20, checkpoint_interval=80,
    neftune_alpha=5.78, num_trajectories=4, exploration_temperature=0.517,
    top_k_expected_return=16,
    asymmetric_alpha=3.0, asymmetric_beta=10.0,
    path_asymmetric_alpha=4.0, path_asymmetric_beta=15.0,
    step_asym_weight=1.0, path_asym_weight=1.5, star_ce_weight=0.174,
    timidity_penalty_weight=2.0, timidity_ratio_threshold=0.5,
    oracle_magnitude_penalty=2.0, prob_sharpening_temp=0.5,
    actionable_da_threshold=0.005,
    freeze_backbone=False, trainable_scope="all",
    use_gradient_checkpointing=False,
    use_amp=True, amp_dtype="bfloat16", use_tf32=True, mape_eps=1e-4,
)

set_global_seed(int(cfg.random_seed), deterministic=bool(cfg.deterministic))
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
amp_dtype = _amp_dtype(cfg.amp_dtype)
amp_enabled = bool(cfg.use_amp and device.type == "cuda" and amp_dtype is not None)

train_dataset = RolloutWindowDataset("train", cfg=cfg, max_samples=0, seed=42)
val_dataset = RolloutWindowDataset("val", cfg=cfg, max_samples=0, seed=59)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False,
                           num_workers=0, pin_memory=True, collate_fn=rollout_collate)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False,
                         num_workers=0, pin_memory=True, collate_fn=rollout_collate)

model, tokenizer = load_model(device=device, checkpoint_path=cfg.checkpoint_path, strict_checkpoint_compat=False)
tokenizer.eval(); tokenizer.requires_grad_(False)
param_groups = _configure_trainable(model, cfg)
optimizer, _ = _build_optimizer(param_groups, cfg, device)
scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

print(f"Train windows={len(train_dataset)}, val={len(val_dataset)}")
print(f"GA={cfg.accumulation_steps}, effective_batch=2x{cfg.accumulation_steps}=32")
print(f"No LR schedule, constant lr={cfg.learning_rate}")
print(f"240 updates = {240*cfg.accumulation_steps} batches")

history = []; best_score = -float("inf"); best_metrics = None
best_path = os.path.join(cfg.output_dir, cfg.save_name)
updates = 0; t0 = time.time()

for epoch in range(int(cfg.epochs)):
    model.train(); optimizer.zero_grad(set_to_none=True)
    epoch_totals = {"total_loss": 0., "step_asym": 0., "path_asym": 0., "star_ce": 0., "golden_rate": 0.}
    batches = 0
    pbar = tqdm(train_loader, desc=f"STAR-CAST GA=16")

    for batch_idx, raw_batch in enumerate(pbar, start=1):
        batch = _move_batch(raw_batch, device)
        loss, stats = train_star_cast_step(model=model, tokenizer=tokenizer, batch=batch, cfg=cfg,
                                            device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        scaled_loss = loss / int(cfg.accumulation_steps)
        scaler.scale(scaled_loss).backward()

        if batch_idx % int(cfg.accumulation_steps) == 0 or batch_idx == len(train_loader):
            if float(cfg.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], max_norm=float(cfg.grad_clip))
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

            ci = int(getattr(cfg, "checkpoint_interval", 0))
            if ci > 0 and updates % ci == 0:
                step_path = os.path.join(cfg.output_dir, f"{os.path.splitext(cfg.save_name)[0]}-step{updates}.pt")
                _save_star_cast_checkpoint(step_path, model, tokenizer, cfg, {"updates": updates}, history)
                print(f"  [Checkpoint] saved step {updates}")

        for key in epoch_totals: epoch_totals[key] += float(stats.get(key, 0.))
        batches += 1
        if batch_idx % int(cfg.progress_interval) == 0:
            pbar.set_postfix({"loss": f"{stats['total_loss']:.4f}", "golden": f"{stats['golden_rate']:.2f}"})
        if int(cfg.max_train_updates) > 0 and updates >= int(cfg.max_train_updates): break

    train_row = {k: v / max(1, batches) for k, v in epoch_totals.items()}
    val_metrics = evaluate_model(model, tokenizer, val_loader, cfg, device, amp_enabled, amp_dtype)
    score = -float(val_metrics.get("path_mape", val_metrics.get("mape", float("inf"))))
    row = {"epoch": int(epoch + 1), "updates": int(updates), "train": train_row, "val": val_metrics, "memory": _cuda_peak_memory_stats(device)}
    history.append(row)
    print(json.dumps(row, indent=2, ensure_ascii=False))
    if score > best_score:
        best_score = score; best_metrics = val_metrics
        _save_star_cast_checkpoint(best_path, model, tokenizer, cfg, val_metrics, history)
    if int(cfg.max_train_updates) > 0 and updates >= int(cfg.max_train_updates): break

elapsed = time.time() - t0
print(f"DONE in {elapsed/60:.1f} min! Best path_mape: {best_metrics.get('path_mape', 'N/A')}")
