# -*- coding: utf-8 -*-
"""Phase 5 DA — Core shared infrastructure.

Model loading, dataset construction, token helpers, evaluation.
"""

from __future__ import annotations

import json, os, sys, warnings
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from config import DataConfig, ModelConfig
from model.kronos_reasoning import KronosReasoningGPT
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.lora import inject_lora, lora_state_dict, has_lora_layers

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
VOCAB = 1 << 10  # 1024

# P3 base model architecture (trial 047)
P3 = {
    "dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
    "dsa_windows": [None, 512, 512], "dropout": 0.1323,
}

# ═══════════════════════════════════════════════
# Default config
# ═══════════════════════════════════════════════

DEFAULT_CFG = {
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
    # GRPO-specific
    "grpo_group_size": 16,
    "grpo_temperature": 1.2,
    "grpo_clip_eps": 0.2,
    "grpo_kl_weight": 0.02,
}

# ═══════════════════════════════════════════════
# LoRA
# ═══════════════════════════════════════════════

LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.05
LORA_TARGETS = ("q_proj", "k_proj", "v_proj")


# ═══════════════════════════════════════════════
# Autocast helper
# ═══════════════════════════════════════════════

def _ac(device, enabled, dtype):
    if device.type != "cuda" or not enabled or dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


# ═══════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════

def load_tokenizer(device):
    ckpt = torch.load(TOK_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tok.load_state_dict(ckpt["model_state_dict"], strict=False)
    tok.eval()
    tok.requires_grad_(False)
    return tok


def build_model(device):
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


def load_base_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict")
    if sd is not None:
        model.load_state_dict(sd, strict=False)
    return model


def build_trainable_model(device, use_lora=True, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA):
    """Build P3 base model, optionally inject LoRA. Only LoRA params trainable if use_lora=True."""
    model = build_model(device)
    load_base_weights(model, P3_CKPT)
    model.eval()

    if use_lora:
        injected = inject_lora(model, rank=lora_rank, alpha=lora_alpha,
                               dropout=LORA_DROPOUT, target_keywords=LORA_TARGETS,
                               freeze_base=True)
        print(f"  LoRA injected into {len(injected)} layers")
    else:
        for p in model.parameters():
            p.requires_grad = True

    return model


def build_ref_model(device):
    """Frozen P3 base model (no LoRA)."""
    model = build_model(device)
    load_base_weights(model, P3_CKPT)
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
    arr_m = np.stack(means, axis=0)
    arr_s = np.stack(stds, axis=0)
    cache_payload["_ma"] = arr_m
    cache_payload["_sa"] = arr_s
    return arr_m, arr_s


def _get_seq_arrays(cache_payload):
    ma = cache_payload.get("_ma")
    sa = cache_payload.get("_sa")
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
        self.samples = []
        self.class_counts = np.zeros(3, dtype=np.int64)
        self._build()

    def _build(self):
        idx = self.indices
        if len(idx) == 0:
            return
        returns = self.real_returns[idx]
        finite = np.isfinite(returns)
        idx = idx[finite]; returns = returns[finite]
        n = len(idx)
        if n == 0:
            return
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample_idx, label, loss_label, real_return = self.samples[i]
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


def get_dataloaders():
    train_payload = torch.load(os.path.join(_PROJECT_ROOT, "dataset_train.pt"),
                               map_location="cpu", weights_only=False)
    val_payload = torch.load(os.path.join(_PROJECT_ROOT, "dataset_val.pt"),
                             map_location="cpu", weights_only=False)
    train_returns = _denorm_last_returns(train_payload)
    val_returns = _denorm_last_returns(val_payload)
    train_abs = np.abs(train_returns)
    train_abs = train_abs[np.isfinite(train_abs)]
    eps = max(DEFAULT_CFG["min_epsilon"], float(np.median(train_abs)) * DEFAULT_CFG["epsilon_scale"])
    print(f"  Global epsilon = {eps:.6f}  (median |r| = {np.median(train_abs):.6f})")

    train_ds = DirectionDataset(train_payload, np.arange(len(train_returns), dtype=np.int64),
                                train_returns, eps, DEFAULT_CFG["flat_policy"])
    val_ds = DirectionDataset(val_payload, np.arange(len(val_returns), dtype=np.int64),
                              val_returns, eps, "class")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"  Train dist: down={train_ds.class_counts[0]} flat={train_ds.class_counts[1]} up={train_ds.class_counts[2]}")
    print(f"  Val   dist: down={val_ds.class_counts[0]} flat={val_ds.class_counts[1]} up={val_ds.class_counts[2]}")

    train_loader = DataLoader(train_ds, batch_size=DEFAULT_CFG["batch_size"], shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=DEFAULT_CFG["batch_size"], shuffle=False,
                            collate_fn=collate_fn)
    return train_loader, val_loader, eps


# ═══════════════════════════════════════════════
# Token helpers
# ═══════════════════════════════════════════════

@torch.no_grad()
def sample_tokens(logits, temperature, num_samples):
    temp = max(float(temperature), 1e-5)
    safe = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    probs = torch.softmax(safe / temp, dim=-1)
    return torch.multinomial(probs, num_samples=max(1, int(num_samples)), replacement=True)


@torch.no_grad()
def token_returns(tokenizer, c_idx, f_idx, means, stds):
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
def token_direction(tokenizer, c_idx, f_idx, means, stds):
    ret = token_returns(tokenizer, c_idx, f_idx, means, stds)
    return torch.where(ret > 0, torch.tensor(LABEL_UP, device=ret.device, dtype=torch.long),
                       torch.tensor(LABEL_DOWN, device=ret.device, dtype=torch.long))


@torch.no_grad()
def encode(tokenizer, features):
    idx_c, idx_f = tokenizer.encode(features)
    return idx_c.long(), idx_f.long()


def candidate_logp(logits_c, logits_f, cand_c, cand_f):
    """Token log-probability of a specific (coarse, fine) pair."""
    logp_c = F.log_softmax(logits_c.float(), dim=-1)
    logp_f = F.log_softmax(logits_f.float(), dim=-1)
    return (logp_c.gather(1, cand_c.long().unsqueeze(1)).squeeze(1)
            + logp_f.gather(1, cand_f.long().unsqueeze(1)).squeeze(1))


@torch.no_grad()
def build_winner_loser(tokenizer, ref_logits_c, ref_logits_f, batch, cfg):
    """Sample candidates from reference token logits, score, pick winner/loser.

    Returns winner/loser token indices + valid_pair mask.
    """
    B = ref_logits_c.size(0)
    sampled_c = sample_tokens(ref_logits_c, cfg["temperature"], cfg["num_candidates"])
    sampled_f = sample_tokens(ref_logits_f, cfg["temperature"], cfg["num_candidates"])

    if cfg["include_gold"]:
        gold_c = batch["idx_c_full"][:, -1]
        gold_f = batch["idx_f_full"][:, -1]
        sampled_c = torch.cat([sampled_c, gold_c.long().unsqueeze(1)], dim=1)
        sampled_f = torch.cat([sampled_f, gold_f.long().unsqueeze(1)], dim=1)

    cand_dir = token_direction(tokenizer, sampled_c, sampled_f,
                                batch["prompt_means"], batch["prompt_stds"])
    cand_ret = token_returns(tokenizer, sampled_c, sampled_f,
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

def move_batch(batch, device):
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
        result["idx_c"] = result["idx_c_full"][:, :-1]
        result["idx_f"] = result["idx_f_full"][:, :-1]
    return result


def prepare_inputs(batch):
    """Resolve token indices and time features for model forward pass."""
    if "idx_c" in batch:
        idx_c, idx_f = batch["idx_c"], batch["idx_f"]
    else:
        idx_c, idx_f = encode(batch["tokenizer"], batch["features_full"])
        idx_c, idx_f = idx_c[:, :-1], idx_f[:, :-1]
    t = batch["time"]
    t_min, t_day, t_mon, t_yr = t["minute"], t["day"], t["month"], t["year"]
    return idx_c, idx_f, t_min, t_day, t_mon, t_yr


# ═══════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, tokenizer, loader, device, amp_enabled, amp_dtype):
    """Production evaluation: argmax token → decode → sign(return)."""
    model.eval()
    all_preds, all_labels, all_returns = [], [], []
    all_pred_returns, all_true_returns = [], []

    for raw_batch in tqdm(loader, desc="Eval", leave=False):
        batch = move_batch(raw_batch, device)
        batch["tokenizer"] = tokenizer
        idx_c, idx_f, t_min, t_day, t_mon, t_yr = prepare_inputs(batch)

        with _ac(device, amp_enabled, amp_dtype):
            logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr,
                                          last_only=True)

        pred_c = logits_c[:, -1, :].float().argmax(dim=-1)
        pred_f = logits_f[:, -1, :].float().argmax(dim=-1)
        pred_ret = token_returns(tokenizer, pred_c.unsqueeze(1), pred_f.unsqueeze(1),
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

    return compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels),
                           np.concatenate(all_returns),
                           np.concatenate(all_pred_returns), np.concatenate(all_true_returns))


def compute_metrics(preds, labels, real_returns, pred_returns, true_returns):
    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    real_returns = np.asarray(real_returns, dtype=np.float64)
    pred_returns = np.asarray(pred_returns, dtype=np.float64)
    true_returns = np.asarray(true_returns, dtype=np.float64)

    n = len(labels)
    if n == 0:
        return {"num_samples": 0}

    true_dir = labels != LABEL_FLAT
    dir_correct = (preds == labels) & true_dir
    dir_acc = float(np.mean(dir_correct[true_dir])) if true_dir.any() else 0.0

    recall_up = float(np.mean(preds[labels == LABEL_UP] == LABEL_UP)) if (labels == LABEL_UP).any() else 0.0
    recall_down = float(np.mean(preds[labels == LABEL_DOWN] == LABEL_DOWN)) if (labels == LABEL_DOWN).any() else 0.0
    bal_acc = float(np.mean([recall_up, recall_down]))

    up_prec = float(np.mean(labels[preds == LABEL_UP] == LABEL_UP)) if (preds == LABEL_UP).any() else 0.0
    down_prec = float(np.mean(labels[preds == LABEL_DOWN] == LABEL_DOWN)) if (preds == LABEL_DOWN).any() else 0.0

    dir_mask = true_dir
    mape = float(np.mean(np.abs((pred_returns[dir_mask] - true_returns[dir_mask])
                                / (np.abs(true_returns[dir_mask]) + 1e-6)))) if dir_mask.any() else 0.0

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
