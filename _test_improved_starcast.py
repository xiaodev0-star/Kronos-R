"""Quick 120-step test of improved STAR-CAST — TRAIN ONLY, no eval during training."""
import os, sys, math, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace

from evaluate_predictions import load_model
from posttrain.rollout.data import RolloutWindowDataset, rollout_collate, resolve_project_path
from posttrain.rollout.train_star_cast import train_star_cast_step, _save_star_cast_checkpoint
from posttrain.rollout.train_rollout import (
    _autocast_context, _amp_dtype, _move_batch,
    _configure_trainable, _build_optimizer,
)
from reproducibility import set_global_seed

# ── Config ──
cfg = Namespace(
    random_seed=42, deterministic=False,
    output_dir=resolve_project_path("checkpoints/post_train_star_cast"),
    checkpoint_path=resolve_project_path("checkpoints/base_model.pt"),
    save_name="star_cast_improved_test.pt", save_epoch_checkpoints=False,
    prefix_len=1023, horizon=10, stride_ratio=0.5,
    cache_dir=resolve_project_path("posttrain/rollout/cache"), cache_rebuild=False,
    max_stocks=0, max_train_samples=0, max_val_samples=0,
    epochs=1, batch_size=2, eval_batch_size=2,
    accumulation_steps=16, num_workers=0,
    learning_rate=1.98e-5, weight_decay=1e-4, grad_clip=0.5,
    max_train_updates=120, progress_interval=20, checkpoint_interval=60,
    neftune_alpha=5.78, num_trajectories=4, exploration_temperature=0.517,
    top_k_expected_return=16,
    asymmetric_alpha=3.0, asymmetric_beta=10.0,
    path_asymmetric_alpha=4.0, path_asymmetric_beta=15.0,
    step_asym_weight=1.0, path_asym_weight=1.5, star_ce_weight=0.174,
    # ── Improved params ──
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
torch.cuda.empty_cache()

amp_dtype = _amp_dtype(cfg.amp_dtype)
amp_enabled = bool(cfg.use_amp and device.type == "cuda" and amp_dtype is not None)

# ── Data ──
print("Loading datasets...")
train_dataset = RolloutWindowDataset("train", cfg=cfg, max_samples=0, seed=42)
print(f"Train windows={len(train_dataset)}")

loader_kwargs = {"num_workers": 0, "pin_memory": True, "collate_fn": rollout_collate}
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False, **loader_kwargs)

# ── Model ──
print("Loading base model...")
model, tokenizer = load_model(device=device, checkpoint_path=cfg.checkpoint_path, strict_checkpoint_compat=False)
tokenizer.eval(); tokenizer.requires_grad_(False)

param_groups = _configure_trainable(model, cfg)
optimizer, opt_kwargs = _build_optimizer(param_groups, cfg, device)
scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

print(f"Device: {device}, amp={amp_enabled}, amp_dtype={cfg.amp_dtype}")
print(f"Timidity_w={cfg.timidity_penalty_weight}, oracle_mag={cfg.oracle_magnitude_penalty}, "
      f"sharpening={cfg.prob_sharpening_temp}")

# ── Training ──
print(f"\n=== Training {cfg.max_train_updates} steps ===")
model.train()
optimizer.zero_grad(set_to_none=True)
updates = 0
microbatch = 0
history = []
t0 = time.time()

pbar = tqdm(total=cfg.max_train_updates, desc="STAR-CAST")
while updates < cfg.max_train_updates:
    for batch in train_loader:
        if updates >= cfg.max_train_updates:
            break
        batch = _move_batch(batch, device)
        try:
            loss, stats = train_star_cast_step(
                model=model, tokenizer=tokenizer, batch=batch, cfg=cfg,
                device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at update {updates}, skipping batch")
            torch.cuda.empty_cache()
            continue

        scaled = loss / cfg.accumulation_steps
        scaler.scale(scaled).backward()
        microbatch += 1

        if microbatch % cfg.accumulation_steps == 0:
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], cfg.grad_clip,
                )
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1; pbar.update(1)
            pbar.set_postfix(
                loss=f"{stats['total_loss']:.4f}",
                golden=f"{stats['golden_rate']:.2f}",
                step_asym=f"{stats['step_asym']:.4f}",
                star_ce=f"{stats['star_ce']:.4f}",
            )

            if updates % cfg.checkpoint_interval == 0:
                path = os.path.join(cfg.output_dir, f"star_cast_improved-step{updates}.pt")
                _save_star_cast_checkpoint(path, model, tokenizer, cfg, {"updates": updates}, history)
                print(f"\n  [Saved step {updates}]")

pbar.close()
elapsed = time.time() - t0
print(f"\nTraining done in {elapsed/60:.1f} min ({elapsed/updates:.1f}s/step)")

# ── Save final ──
final_path = os.path.join(cfg.output_dir, cfg.save_name)
_save_star_cast_checkpoint(final_path, model, tokenizer, cfg, {"updates": updates}, history)
print(f"Final model: {final_path}")
