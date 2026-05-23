# -*- coding: utf-8 -*-
"""Phase 5 DA: Token-space DPO / ExPO / RSFT post-training comparison.

Fixes over the previous (deleted) v2:
  - Optimizes token log-probabilities (not direction_head logits).
    Winner/loser pairs live in token space, so the loss should too.
  - Reference model = frozen P3 base (useful token logits, unlike a random
    direction_head).
  - LoRA (rank=8) on attention + output-head projections keeps the backbone
    frozen, making the comparison fair and avoiding spurious full-model drift.
  - Evaluates with the production protocol (argmax-token → decode → sign)
    AND monitors MAPE so we know whether direction gains trade off regression.
  - Methods share identical candidate sampling (192, temp=1.0).

Methods (all LoRA, same base):
  T1. CE       – standard next-token cross-entropy (baseline)
  T2. ExPO     – ExPO regression on token log-probabilities
  T3. DPO      – DPO sigmoid loss on token log-probabilities
  T4. RSFT     – rejection-sampled best token → CE

Usage:
    python -m hpo.phase5_da
"""

from __future__ import annotations

import copy, json, os, sys, warnings
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from config import DataConfig, ModelConfig
from model.kronos_reasoning import KronosReasoningGPT
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.lora import inject_lora, has_lora_layers, lora_state_dict
from reproducibility import set_global_seed

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════

LABEL_DOWN, LABEL_FLAT, LABEL_UP = 0, 1, 2
IGNORE_INDEX = -100

SEED = 42
P3_CKPT = os.path.join(_PROJECT_ROOT, "trials", "phase3_basemodel", "trial_047", "basemodel.pt")
TOK_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "tokenizer.pt")
OUT_DIR  = os.path.join(_PROJECT_ROOT, "trials", "phase5_da")
VOCAB = 1 << 10

P3 = {"dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
      "dsa_windows": [None, 512, 512], "dropout": 0.1323}

# ── Shared config ──
CFG = {
    "epochs": 10, "batch_size": 8, "accumulation_steps": 1,
    "lr": 5e-5, "weight_decay": 1e-4, "grad_clip": 1.0,
    "num_candidates": 192, "temperature": 1.0,
    "direction_bonus": 1.0, "error_weight": 0.25,
    "score_margin": 0.05, "include_gold": True,
    "dpo_beta": 0.5,
    "expo_reference_weight": 0.6,
    "label_mode": "global_median", "epsilon_scale": 0.5,
    "min_epsilon": 1e-5, "flat_policy": "ignore",
    "progress_interval": 50,
}

# ── LoRA ──
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.05
# NOTE: do NOT include "out_proj" — it matches nn.MultiheadAttention.out_proj
# inside LatentReasoner, whose forward() accesses .weight directly.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj")


# ═══════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════

def _ac(device, enabled, dtype):
    if device.type != "cuda" or not enabled or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


# ═══════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════

def _load_tokenizer(device):
    ckpt = torch.load(TOK_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tok.load_state_dict(ckpt["model_state_dict"], strict=False)
    tok.eval(); tok.requires_grad_(False)
    return tok


def _build_model(device):
    return KronosReasoningGPT(
        dim=P3["dim"], depth=P3["depth"], heads=P3["heads"],
        num_kv_heads=P3["num_kv_heads"], dsa_windows=P3["dsa_windows"],
        dropout=P3["dropout"], vocab_size_coarse=VOCAB, vocab_size_fine=VOCAB,
        num_latent_tokens=ModelConfig.num_latent_tokens,
        latent_reasoner_depth=ModelConfig.latent_reasoner_depth,
        latent_cross_heads=ModelConfig.latent_cross_heads,
        position_encoding="rope", rope_base=10000.0,
        max_len=ModelConfig.max_len,
        horizon_tokens=ModelConfig.horizon_tokens,
        horizon_decoder_depth=ModelConfig.horizon_decoder_depth,
        horizon_decoder_heads=ModelConfig.horizon_decoder_heads,
        use_revin=ModelConfig.use_revin, revin_affine=ModelConfig.revin_affine,
        revin_eps=ModelConfig.revin_eps, num_factor_tokens=ModelConfig.num_factor_tokens,
    ).to(device)


def _load_base_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict")
    if sd is not None:
        model.load_state_dict(sd, strict=False)
    return model


def build_trainable_model(device):
    """Build model with P3 weights + LoRA adapters.  Only LoRA params are trainable."""
    model = _build_model(device)
    _load_base_weights(model, P3_CKPT)
    model.eval()
    injected = inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA,
                           dropout=LORA_DROPOUT, target_keywords=LORA_TARGETS,
                           freeze_base=True)
    print(f"LoRA injected into {len(injected)} layers: {injected}")
    return model


def build_ref_model(device):
    """Frozen P3 base model – NO LoRA."""
    model = _build_model(device)
    _load_base_weights(model, P3_CKPT)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ═══════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════

def _seq_arrays(cache_payload):
    means, stds = [], []
    for s in cache_payload["seq_stats"]:
        means.append(np.asarray(s["mean"], dtype=np.float32))
        stds.append(np.asarray(s["std"], dtype=np.float32))
    arr_m = np.stack(means, axis=0); arr_s = np.stack(stds, axis=0)
    cache_payload["_ma"] = arr_m; cache_payload["_sa"] = arr_s
    return arr_m, arr_s


def _get_seq_arrays(cache_payload):
    ma = cache_payload.get("_ma"); sa = cache_payload.get("_sa")
    if ma is None or sa is None:
        ma, sa = _seq_arrays(cache_payload)
    return ma, sa


def _denorm_last_returns(cache_payload):
    features = cache_payload["features"]
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)
    means, stds = _get_seq_arrays(cache_payload)
    norm_last = features[:, -1, 0].cpu().numpy().astype(np.float64)
    return norm_last * stds[:, 0].astype(np.float64) + means[:, 0].astype(np.float64)


class DirectionDataset(Dataset):
    """Yields 1023-step history + 1024th-step label for next-day prediction."""

    def __init__(self, cache_payload, indices, real_returns, global_epsilon, flat_policy):
        self.cache = cache_payload
        self.indices = np.asarray(indices, dtype=np.int64)
        self.real_returns = np.asarray(real_returns, dtype=np.float64)
        self.epsilon = float(global_epsilon)
        self.flat_policy = str(flat_policy).strip().lower()

        self.features = self.cache["features"]
        if not isinstance(self.features, torch.Tensor):
            self.features = torch.as_tensor(self.features, dtype=torch.float32)
            self.cache["features"] = self.features
        self.seq_means, self.seq_stds = _get_seq_arrays(self.cache)
        self.has_encoded = (
            isinstance(self.cache.get("encoded_indices_coarse"), torch.Tensor)
            and isinstance(self.cache.get("encoded_indices_fine"), torch.Tensor))
        self.samples = []; self.class_counts = np.zeros(3, dtype=np.int64)
        self._build()

    def _build(self):
        idx = self.indices
        if len(idx) == 0: return
        returns = self.real_returns[idx]
        finite = np.isfinite(returns); idx = idx[finite]; returns = returns[finite]
        n = len(idx)
        if n == 0: return
        epsilons = np.full(n, self.epsilon, dtype=np.float64)
        labels = np.full(n, LABEL_FLAT, dtype=np.int64)
        labels[returns > epsilons] = LABEL_UP
        labels[returns < -epsilons] = LABEL_DOWN
        loss_labels = labels.copy()
        if self.flat_policy == "ignore":
            loss_labels[labels == LABEL_FLAT] = IGNORE_INDEX
        self.samples = [(int(idx[i]), int(labels[i]), int(loss_labels[i]),
                         float(returns[i])) for i in range(n)]
        for _, lbl, _, _ in self.samples:
            self.class_counts[lbl] += 1

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        sample_idx, label, loss_label, real_return = self.samples[i]
        # time features: first 1023 steps (trimmed to match model input)
        time_features = {key: value[sample_idx, :-1].to(dtype=torch.long)
                         for key, value in self.cache["time_features"].items()}
        seq_stat = self.cache["seq_stats"][sample_idx]
        result = {
            "label": int(label), "loss_label": int(loss_label),
            "real_return": float(real_return),
            "prompt_mean": float(seq_stat["mean"][0]),
            "prompt_std": float(seq_stat["std"][0]),
            "time": time_features,
        }
        if self.has_encoded:
            result["idx_c"] = self.cache["encoded_indices_coarse"][sample_idx].long()
            result["idx_f"] = self.cache["encoded_indices_fine"][sample_idx].long()
        else:
            result["features_full"] = self.features[sample_idx].to(dtype=torch.float32)
        return result


def collate_fn(batch):
    has_enc = all("idx_c" in item for item in batch)
    result = {
        "time": {key: torch.stack([item["time"][key] for item in batch], dim=0).long()
                 for key in ("minute", "day", "month", "year")},
        "labels": torch.as_tensor([item["label"] for item in batch], dtype=torch.long),
        "loss_labels": torch.as_tensor([item["loss_label"] for item in batch], dtype=torch.long),
        "real_returns": torch.as_tensor([item["real_return"] for item in batch], dtype=torch.float32),
        "prompt_means": torch.as_tensor([item["prompt_mean"] for item in batch], dtype=torch.float32),
        "prompt_stds": torch.as_tensor([item["prompt_std"] for item in batch], dtype=torch.float32),
    }
    if has_enc:
        result["idx_c"] = torch.stack([item["idx_c"] for item in batch], dim=0)
        result["idx_f"] = torch.stack([item["idx_f"] for item in batch], dim=0)
    else:
        result["features_full"] = torch.stack([item["features_full"] for item in batch], dim=0)
    return result


def _get_dataloaders():
    train_payload = torch.load(os.path.join(_PROJECT_ROOT, "dataset_train.pt"),
                               map_location="cpu", weights_only=False)
    val_payload = torch.load(os.path.join(_PROJECT_ROOT, "dataset_val.pt"),
                             map_location="cpu", weights_only=False)
    train_returns = _denorm_last_returns(train_payload)
    val_returns = _denorm_last_returns(val_payload)
    train_abs = np.abs(train_returns)
    train_abs = train_abs[np.isfinite(train_abs)]
    eps = max(CFG["min_epsilon"], float(np.median(train_abs)) * CFG["epsilon_scale"])
    print(f"Global epsilon = {eps:.6f}  (median |r| = {np.median(train_abs):.6f})")

    train_ds = DirectionDataset(train_payload, np.arange(len(train_returns), dtype=np.int64),
                                train_returns, eps, CFG["flat_policy"])
    val_ds = DirectionDataset(val_payload, np.arange(len(val_returns), dtype=np.int64),
                              val_returns, eps, "class")
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")
    print(f"Train class dist: down={train_ds.class_counts[0]} flat={train_ds.class_counts[1]} up={train_ds.class_counts[2]}")
    print(f"Val   class dist: down={val_ds.class_counts[0]} flat={val_ds.class_counts[1]} up={val_ds.class_counts[2]}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                            collate_fn=collate_fn)
    return train_loader, val_loader, eps


# ═══════════════════════════════════════════════
# Token helpers
# ═══════════════════════════════════════════════

@torch.no_grad()
def _sample_tokens(logits, temperature, num_samples):
    temp = max(float(temperature), 1e-5)
    safe = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    probs = torch.softmax(safe / temp, dim=-1)
    return torch.multinomial(probs, num_samples=max(1, int(num_samples)), replacement=True)


@torch.no_grad()
def _token_returns(tokenizer, c_idx, f_idx, means, stds):
    decoded = tokenizer.decode(c_idx, f_idx)
    log_ret_norm = decoded[..., 0]
    dev = log_ret_norm.device
    means = means.to(dev); stds = stds.to(dev)
    if means.ndim == 1: mean_col = means
    else: mean_col = means[:, 0]
    if stds.ndim == 1: std_col = stds
    else: std_col = stds[:, 0]
    while mean_col.ndim < log_ret_norm.ndim:
        mean_col = mean_col.unsqueeze(-1); std_col = std_col.unsqueeze(-1)
    return log_ret_norm * std_col + mean_col


@torch.no_grad()
def _token_direction(tokenizer, c_idx, f_idx, means, stds):
    ret = _token_returns(tokenizer, c_idx, f_idx, means, stds)
    return torch.where(ret > 0, torch.tensor(LABEL_UP, device=ret.device, dtype=torch.long),
                       torch.tensor(LABEL_DOWN, device=ret.device, dtype=torch.long))


@torch.no_grad()
def _encode(tokenizer, features):
    idx_c, idx_f = tokenizer.encode(features)
    return idx_c.long(), idx_f.long()


def _candidate_logp(logits_c, logits_f, cand_c, cand_f):
    """Token log-probability of a specific (coarse, fine) pair."""
    logp_c = F.log_softmax(logits_c.float(), dim=-1)
    logp_f = F.log_softmax(logits_f.float(), dim=-1)
    return (logp_c.gather(1, cand_c.long().unsqueeze(1)).squeeze(1)
            + logp_f.gather(1, cand_f.long().unsqueeze(1)).squeeze(1))


@torch.no_grad()
def _build_winner_loser(tokenizer, ref_logits_c, ref_logits_f, batch, cfg):
    """Sample candidates from reference token logits, score, pick winner/loser.

    Returns winner/loser token indices + valid_pair mask.
    """
    B = ref_logits_c.size(0)
    sampled_c = _sample_tokens(ref_logits_c, cfg["temperature"], cfg["num_candidates"])
    sampled_f = _sample_tokens(ref_logits_f, cfg["temperature"], cfg["num_candidates"])

    # Optionally include the ground-truth token as a candidate
    if cfg["include_gold"]:
        gold_c = batch["idx_c_full"][:, -1]
        gold_f = batch["idx_f_full"][:, -1]
        sampled_c = torch.cat([sampled_c, gold_c.long().unsqueeze(1)], dim=1)
        sampled_f = torch.cat([sampled_f, gold_f.long().unsqueeze(1)], dim=1)

    cand_dir = _token_direction(tokenizer, sampled_c, sampled_f,
                                batch["prompt_means"], batch["prompt_stds"])
    cand_ret = _token_returns(tokenizer, sampled_c, sampled_f,
                              batch["prompt_means"], batch["prompt_stds"])

    valid_label = batch["loss_labels"] != IGNORE_INDEX
    real = batch["real_returns"].to(device=cand_ret.device, dtype=cand_ret.dtype)
    eps = torch.tensor(1e-6, device=cand_ret.device, dtype=cand_ret.dtype)
    err_scale = torch.maximum(real.abs(), eps)
    while real.ndim < cand_ret.ndim:
        real = real.unsqueeze(-1); err_scale = err_scale.unsqueeze(-1)
    norm_err = torch.nan_to_num((cand_ret - real).abs() / err_scale,
                                nan=1e6, posinf=1e6, neginf=1e6)

    labels_2d = batch["labels"].unsqueeze(1)
    dir_correct = cand_dir.eq(labels_2d) & valid_label.unsqueeze(1)
    scores = (dir_correct.to(dtype=cand_ret.dtype) * cfg["direction_bonus"]
              - norm_err * cfg["error_weight"])
    scores = scores.masked_fill(~valid_label.unsqueeze(1), float("-inf"))

    winner_score, winner_idx = scores.max(dim=1)
    loser_score, loser_idx = scores.min(dim=1)
    margin = winner_score - loser_score
    valid_pair = (valid_label & torch.isfinite(winner_score) & torch.isfinite(loser_score)
                  & (winner_idx != loser_idx) & (margin >= cfg["score_margin"]))

    rows = torch.arange(B, device=sampled_c.device)
    return (sampled_c[rows, winner_idx], sampled_f[rows, winner_idx],
            sampled_c[rows, loser_idx], sampled_f[rows, loser_idx],
            valid_pair, cand_dir[rows, winner_idx], cand_dir[rows, loser_idx])


# ═══════════════════════════════════════════════
# Batch I/O
# ═══════════════════════════════════════════════

def _move_batch(batch, device):
    result = {
        "time": {key: value.to(device=device, dtype=torch.long, non_blocking=True)
                 for key, value in batch["time"].items()},
        "labels": batch["labels"].to(device=device, dtype=torch.long, non_blocking=True),
        "loss_labels": batch["loss_labels"].to(device=device, dtype=torch.long, non_blocking=True),
        "real_returns": batch["real_returns"].to(device=device, dtype=torch.float32, non_blocking=True),
        "prompt_means": batch["prompt_means"].to(device=device, dtype=torch.float32, non_blocking=True),
        "prompt_stds": batch["prompt_stds"].to(device=device, dtype=torch.float32, non_blocking=True),
    }
    if "features_full" in batch:
        result["features_full"] = batch["features_full"].to(device=device, dtype=torch.float32, non_blocking=True)
    if "idx_c" in batch:
        result["idx_c_full"] = batch["idx_c"].to(device=device, dtype=torch.long, non_blocking=True)
        result["idx_f_full"] = batch["idx_f"].to(device=device, dtype=torch.long, non_blocking=True)
        # Also add the trimmed version for convenience
        result["idx_c"] = result["idx_c_full"][:, :-1]
        result["idx_f"] = result["idx_f_full"][:, :-1]
    return result


def _prepare_inputs(batch):
    """Resolve token indices and time features for the model forward pass."""
    if "idx_c" in batch:
        idx_c, idx_f = batch["idx_c"], batch["idx_f"]
    else:
        idx_c, idx_f = _encode(batch["tokenizer"], batch["features_full"])
        idx_c, idx_f = idx_c[:, :-1], idx_f[:, :-1]
    t = batch["time"]
    t_min, t_day, t_mon, t_yr = t["minute"], t["day"], t["month"], t["year"]
    # Time features are already trimmed to 1023 by the dataset
    return idx_c, idx_f, t_min, t_day, t_mon, t_yr


# ═══════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, tokenizer, loader, device, amp_enabled, amp_dtype):
    """Evaluate using the production protocol: argmax token → decode → sign(return)."""
    model.eval()
    all_preds, all_labels, all_returns = [], [], []
    all_mape, all_pred_returns, all_true_returns = [], [], []

    for raw_batch in tqdm(loader, desc="Eval", leave=False):
        batch = _move_batch(raw_batch, device)
        batch["tokenizer"] = tokenizer
        idx_c, idx_f, t_min, t_day, t_mon, t_yr = _prepare_inputs(batch)

        with _ac(device, amp_enabled, amp_dtype):
            logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr,
                                          last_only=True)

        # v1 protocol: argmax token → decode → sign
        pred_c = logits_c[:, -1, :].float().argmax(dim=-1)
        pred_f = logits_f[:, -1, :].float().argmax(dim=-1)
        pred_ret = _token_returns(tokenizer, pred_c.unsqueeze(1), pred_f.unsqueeze(1),
                                  batch["prompt_means"], batch["prompt_stds"]).squeeze(1)
        pred_dir = torch.where(pred_ret > 0, torch.tensor(LABEL_UP, device=device),
                               torch.tensor(LABEL_DOWN, device=device))

        all_preds.append(pred_dir.cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
        all_returns.append(batch["real_returns"].cpu().numpy())
        all_pred_returns.append(pred_ret.cpu().numpy())
        all_true_returns.append(batch["real_returns"].cpu().numpy())

    if not all_preds:
        return {"num_samples": 0}

    return _compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels),
                            np.concatenate(all_returns),
                            np.concatenate(all_pred_returns), np.concatenate(all_true_returns))


def _compute_metrics(preds, labels, real_returns, pred_returns, true_returns):
    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    real_returns = np.asarray(real_returns, dtype=np.float64)
    pred_returns = np.asarray(pred_returns, dtype=np.float64)
    true_returns = np.asarray(true_returns, dtype=np.float64)

    n = len(labels)
    if n == 0: return {"num_samples": 0}

    true_dir = labels != LABEL_FLAT
    dir_correct = (preds == labels) & true_dir
    dir_acc = float(np.mean(dir_correct[true_dir])) if true_dir.any() else 0.0

    recall_up = float(np.mean(preds[labels == LABEL_UP] == LABEL_UP)) if (labels == LABEL_UP).any() else 0.0
    recall_down = float(np.mean(preds[labels == LABEL_DOWN] == LABEL_DOWN)) if (labels == LABEL_DOWN).any() else 0.0
    bal_acc = float(np.mean([recall_up, recall_down]))

    up_prec = float(np.mean(labels[preds == LABEL_UP] == LABEL_UP)) if (preds == LABEL_UP).any() else 0.0
    down_prec = float(np.mean(labels[preds == LABEL_DOWN] == LABEL_DOWN)) if (preds == LABEL_DOWN).any() else 0.0

    # MAPE (on directional samples only)
    dir_mask = true_dir
    mape = float(np.mean(np.abs((pred_returns[dir_mask] - true_returns[dir_mask])
                                / (np.abs(true_returns[dir_mask]) + 1e-6)))) if dir_mask.any() else 0.0

    # Bucket DA
    bucket_metrics = {}
    abs_ret = np.abs(real_returns)
    if len(abs_ret) >= 3:
        q1, q2 = np.quantile(abs_ret, [1./3., 2./3.])
        for name, mask in [("small", abs_ret <= q1), ("medium", (abs_ret > q1) & (abs_ret <= q2)),
                           ("large", abs_ret > q2)]:
            dir_mask_b = mask & true_dir
            bucket_metrics[f"return_bucket_{name}_accuracy"] = (
                float(np.mean(dir_correct[dir_mask_b])) if dir_mask_b.any() else 0.0)

    pred_counts = {"down": int((preds == LABEL_DOWN).sum()),
                   "flat": int((preds == LABEL_FLAT).sum()),
                   "up": int((preds == LABEL_UP).sum())}
    class_counts = {"down": int((labels == LABEL_DOWN).sum()),
                    "flat": int((labels == LABEL_FLAT).sum()),
                    "up": int((labels == LABEL_UP).sum())}

    return {
        "num_samples": n,
        "direction_accuracy": dir_acc,
        "balanced_accuracy": bal_acc,
        "up_precision": up_prec, "down_precision": down_prec,
        "recall_up": recall_up, "recall_down": recall_down,
        "mape": mape,
        "class_counts": class_counts,
        "pred_counts": pred_counts,
        **bucket_metrics,
    }


# ═══════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════

def _run_training(technique_name, device, loss_fn, needs_ref_model=False):
    out_dir = os.path.join(OUT_DIR, technique_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\nPhase 5 DA: {technique_name}\n{'='*60}")

    train_loader, val_loader, eps = _get_dataloaders()

    tokenizer = _load_tokenizer(device)
    model = build_trainable_model(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    ref_model = build_ref_model(device) if needs_ref_model else None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG["lr"], weight_decay=CFG["weight_decay"],
        fused=True if device.type == "cuda" else False)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    amp_enabled = device.type == "cuda" and amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    epochs = CFG["epochs"]
    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=CFG["lr"] * 0.01)

    history = []
    best_score = -float("inf")
    best_path = os.path.join(out_dir, f"phase5_da_{technique_name}.pt")
    updates = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0; batches = 0
        pbar = tqdm(train_loader, desc=f"  {technique_name} epoch {epoch+1}/{epochs}")

        for step, raw_batch in enumerate(pbar, start=1):
            batch = _move_batch(raw_batch, device)
            batch["tokenizer"] = tokenizer
            loss = loss_fn(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], CFG["grad_clip"])
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates <= warmup_steps:
                scheduler.step()
            else:
                cosine.step()

            batches += 1; total_loss += loss.detach().cpu().item()
            if step % CFG["progress_interval"] == 0:
                pbar.set_postfix({"loss": f"{total_loss/max(1,batches):.4f}", "upd": updates})

        val_metrics = evaluate(model, tokenizer, val_loader, device, amp_enabled, amp_dtype)
        record = {"epoch": epoch+1, "updates": updates,
                  "train_loss": total_loss/max(1,batches), "val": val_metrics}
        score = val_metrics.get("balanced_accuracy", val_metrics.get("direction_accuracy", 0.0))
        record["selection_score"] = float(score)
        history.append(record)

        print(f"  DA={val_metrics.get('direction_accuracy','?'):.4f}  "
              f"BalAcc={val_metrics.get('balanced_accuracy','?'):.4f}  "
              f"MAPE={val_metrics.get('mape','?'):.4f}  "
              f"PredDist={val_metrics.get('pred_counts',{})}")

        if float(score) > best_score:
            best_score = float(score)
            save_dict = {
                "technique": technique_name, "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "lora_state_dict": lora_state_dict(model),
                "tokenizer_state_dict": tokenizer.state_dict(),
                "model_config": P3, "lora_config": {"rank": LORA_RANK, "alpha": LORA_ALPHA,
                                                     "dropout": LORA_DROPOUT, "targets": LORA_TARGETS},
                "metrics": val_metrics, "history": history,
            }
            torch.save(save_dict, best_path)
            print(f"  Saved best: {best_path}")

    result = {"technique": technique_name, "best_score": best_score,
              "best_epoch": max((i for i, r in enumerate(history)
                                 if r.get("selection_score", -float("inf")) == best_score), default=-1)+1,
              "final_val": history[-1]["val"] if history else {}}
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, f"phase5_da_{technique_name}_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"  {technique_name} done. Best: {best_score:.4f}")
    return result


# ═══════════════════════════════════════════════
# T1: CE — standard next-token cross-entropy
# ═══════════════════════════════════════════════

def _loss_ce(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = _prepare_inputs(batch)
    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    target_c = batch["idx_c_full"][:, -1]
    target_f = batch["idx_f_full"][:, -1]
    return (F.cross_entropy(logits_c[:, -1, :].float(), target_c)
            + F.cross_entropy(logits_f[:, -1, :].float(), target_f))


def run_ce(device):
    return _run_training("ce", device, _loss_ce, needs_ref_model=False)


# ═══════════════════════════════════════════════
# T2: ExPO — regression on token log-probabilities
# ═══════════════════════════════════════════════

def _loss_expo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = _prepare_inputs(batch)

    # Policy logits
    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_last_c = logits_c[:, -1, :]
    pi_last_f = logits_f[:, -1, :]

    # Reference logits (frozen P3, no LoRA)
    with torch.no_grad():
        with _ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_last_c = ref_lc[:, -1, :]
        ref_last_f = ref_lf[:, -1, :]

    # Sample winner/loser from reference
    w_c, w_f, l_c, l_f, valid_pair, w_dir, l_dir = _build_winner_loser(
        tokenizer, ref_last_c, ref_last_f, batch, CFG)

    # ExPO regression: push σ(θ_win - θ_lose) toward target
    theta_win = _candidate_logp(pi_last_c, pi_last_f, w_c, w_f)
    theta_lose = _candidate_logp(pi_last_c, pi_last_f, l_c, l_f)
    with torch.no_grad():
        ref_win = _candidate_logp(ref_last_c, ref_last_f, w_c, w_f)
        ref_lose = _candidate_logp(ref_last_c, ref_last_f, l_c, l_f)
        ref_pref = torch.sigmoid(ref_win - ref_lose)

    theta_pref = torch.sigmoid(theta_win - theta_lose)
    lam = max(0.0, min(1.0, CFG["expo_reference_weight"]))
    target_pref = (lam * ref_pref + (1.0 - lam)).clamp(0.0, 1.0)

    per_row = (theta_pref - target_pref).pow(2)
    vw = valid_pair.to(dtype=per_row.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (per_row * vw).sum() / vw.sum().clamp_min(1.0)


def run_expo(device):
    return _run_training("expo", device, _loss_expo, needs_ref_model=True)


# ═══════════════════════════════════════════════
# T3: DPO — sigmoid loss on token log-probabilities
# ═══════════════════════════════════════════════

def _loss_dpo(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = _prepare_inputs(batch)

    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
    pi_last_c = logits_c[:, -1, :]
    pi_last_f = logits_f[:, -1, :]

    with torch.no_grad():
        with _ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_last_c = ref_lc[:, -1, :]
        ref_last_f = ref_lf[:, -1, :]

    w_c, w_f, l_c, l_f, valid_pair, _, _ = _build_winner_loser(
        tokenizer, ref_last_c, ref_last_f, batch, CFG)

    # DPO: -log σ(β * ((π_win-π_lose) - (ref_win-ref_lose)))
    pi_win = _candidate_logp(pi_last_c, pi_last_f, w_c, w_f)
    pi_lose = _candidate_logp(pi_last_c, pi_last_f, l_c, l_f)
    with torch.no_grad():
        ref_win = _candidate_logp(ref_last_c, ref_last_f, w_c, w_f)
        ref_lose = _candidate_logp(ref_last_c, ref_last_f, l_c, l_f)

    log_ratio = (pi_win - pi_lose) - (ref_win - ref_lose)
    per_row = -F.logsigmoid(CFG["dpo_beta"] * log_ratio)
    vw = valid_pair.to(dtype=per_row.dtype)
    if vw.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (per_row * vw).sum() / vw.sum().clamp_min(1.0)


def run_dpo(device):
    return _run_training("dpo", device, _loss_dpo, needs_ref_model=True)


# ═══════════════════════════════════════════════
# T4: RSFT — rejection-sampled best token → CE
# ═══════════════════════════════════════════════

def _loss_rsft(model, ref_model, batch, tokenizer, device, amp_enabled, amp_dtype):
    idx_c, idx_f, t_min, t_day, t_mon, t_yr = _prepare_inputs(batch)

    # Reference sampling
    with torch.no_grad():
        with _ac(device, amp_enabled, amp_dtype):
            ref_lc, ref_lf, _ = ref_model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        ref_last_c = ref_lc[:, -1, :]
        ref_last_f = ref_lf[:, -1, :]

    B = ref_last_c.size(0)
    sampled_c = _sample_tokens(ref_last_c, CFG["temperature"], CFG["num_candidates"])
    sampled_f = _sample_tokens(ref_last_f, CFG["temperature"], CFG["num_candidates"])

    if CFG["include_gold"]:
        gold_c = batch["idx_c_full"][:, -1]
        gold_f = batch["idx_f_full"][:, -1]
        sampled_c = torch.cat([sampled_c, gold_c.long().unsqueeze(1)], dim=1)
        sampled_f = torch.cat([sampled_f, gold_f.long().unsqueeze(1)], dim=1)

    cand_dir = _token_direction(tokenizer, sampled_c, sampled_f,
                                batch["prompt_means"], batch["prompt_stds"])
    cand_ret = _token_returns(tokenizer, sampled_c, sampled_f,
                              batch["prompt_means"], batch["prompt_stds"])

    valid_label = batch["loss_labels"] != IGNORE_INDEX
    real = batch["real_returns"].to(device=cand_ret.device, dtype=cand_ret.dtype)
    err_scale = torch.maximum(real.abs(), torch.tensor(1e-6, device=cand_ret.device))
    while real.ndim < cand_ret.ndim:
        real = real.unsqueeze(-1); err_scale = err_scale.unsqueeze(-1)
    norm_err = torch.nan_to_num((cand_ret - real).abs() / err_scale, nan=1e6, posinf=1e6)

    labels_2d = batch["labels"].unsqueeze(1)
    dir_correct = cand_dir.eq(labels_2d) & valid_label.unsqueeze(1)
    scores = (dir_correct.to(dtype=cand_ret.dtype) * CFG["direction_bonus"]
              - norm_err * CFG["error_weight"])
    scores = scores.masked_fill(~valid_label.unsqueeze(1), float("-inf"))

    _, best_idx = scores.max(dim=1)
    rows = torch.arange(B, device=sampled_c.device)
    best_c = sampled_c[rows, best_idx]
    best_f = sampled_f[rows, best_idx]
    best_valid = valid_label & scores.max(dim=1).values.isfinite()

    # CE on best candidate
    with _ac(device, amp_enabled, amp_dtype):
        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)

    if best_valid.any():
        return (F.cross_entropy(logits_c[best_valid, -1, :].float(), best_c[best_valid].long())
                + F.cross_entropy(logits_f[best_valid, -1, :].float(), best_f[best_valid].long()))
    return torch.tensor(0.0, device=device, requires_grad=True)


def run_rsft(device):
    return _run_training("rsft", device, _loss_rsft, needs_ref_model=True)


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 5 DA: Token-space DPO / ExPO / RSFT comparison (LoRA rank={LORA_RANK})")
    print(f"  Device: {device}")
    print(f"  Base: {P3_CKPT}")
    print(f"  Output: {OUT_DIR}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    set_global_seed(SEED, deterministic=False)
    os.makedirs(OUT_DIR, exist_ok=True)

    techniques = [
        ("ce",   run_ce),
        ("expo", run_expo),
        ("dpo",  run_dpo),
        ("rsft", run_rsft),
    ]

    all_results = {}
    for tech_name, run_fn in techniques:
        if os.path.exists(os.path.join(OUT_DIR, tech_name, "result.json")):
            print(f"\nSkip {tech_name} (already done)")
            with open(os.path.join(OUT_DIR, tech_name, "result.json")) as f:
                all_results[tech_name] = json.load(f)
            continue
        try:
            all_results[tech_name] = run_fn(device)
        except Exception as e:
            print(f"\n!!! {tech_name} FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results[tech_name] = {"technique": tech_name, "error": str(e), "final_val": {}}

    # ── Summary ──
    print(f"\n{'='*60}")
    print("Phase 5 DA Summary")
    print(f"{'='*60}")
    header = f"{'Technique':10s}  {'DA':>8s}  {'BalAcc':>8s}  {'MAPE':>8s}  {'UpP':>8s}  {'DnP':>8s}  {'Preds':>20s}"
    print(header)
    print("-" * len(header))
    for tech_name in ["ce", "expo", "dpo", "rsft"]:
        r = all_results.get(tech_name, {})
        fv = r.get("final_val", {})
        preds = fv.get("pred_counts", {})
        pred_str = f"u={preds.get('up',0)} d={preds.get('down',0)} f={preds.get('flat',0)}"
        print(f"  {tech_name:8s}  {fv.get('direction_accuracy',0):8.4f}  "
              f"{fv.get('balanced_accuracy',0):8.4f}  {fv.get('mape',0):8.4f}  "
              f"{fv.get('up_precision',0):8.4f}  {fv.get('down_precision',0):8.4f}  {pred_str}")

    # Save cross-method summary
    cross = {"phase": "5_da", "seed": SEED, "base_checkpoint": P3_CKPT,
             "config": CFG, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
             "results": all_results}
    with open(os.path.join(OUT_DIR, "cross_method_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cross, f, indent=2, ensure_ascii=False)

    # Best method
    best = max(all_results.items(), key=lambda kv: kv[1].get("final_val", {}).get("balanced_accuracy", 0.0))
    print(f"\nBest: {best[0]} (BalAcc={best[1].get('final_val',{}).get('balanced_accuracy',0):.4f})")
    return cross


if __name__ == "__main__":
    main()
