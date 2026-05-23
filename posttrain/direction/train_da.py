# -*- coding: utf-8 -*-
"""Post_Train_DA: next-day direction-accuracy EXPO for Kronos-R.

Stage B2 samples candidate next-token pairs from a frozen reference policy,
builds winner/loser preferences from next-day direction and return error,
then fine-tunes the token policy with regression EXPO.
The cache contains fixed 1024-token sequences, so training uses the
first 1023 cached tokens as history and the final cached token's
denormalized log_ret as the next-step direction label.
"""

import argparse
import copy
import json
import math
import os
import warnings
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DataConfig, PostTrainDAConfig, TrainingConfig
from data_processor import AShareDataset, _dataset_source_fingerprint, _resolve_tokenizer_fingerprint
from evaluate_predictions import build_rolling_1d_eval_items, load_model
from model.lora import (
    has_lora_layers,
    inject_lora,
    lora_state_dict,
    save_lora_adapter,
    trainable_parameter_summary,
)
from reproducibility import set_global_seed

warnings.filterwarnings("ignore")

LABEL_DOWN = 0
LABEL_FLAT = 1
LABEL_UP = 2
IGNORE_INDEX = -100
LABEL_NAMES = ("down", "flat", "up")
EXPO_COUNT_KEYS = (
    "valid",
    "pairs",
    "skipped",
    "winner_direction_correct",
    "loser_direction_correct",
    "winner_from_gold",
)
EXPO_SUM_KEYS = (
    "preference_margin",
    "winner_abs_error",
    "loser_abs_error",
    "target_prob",
    "theta_prob",
    "ref_prob",
)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _amp_dtype(name):
    name = str(name).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return None


def _autocast_context(device, enabled, dtype):
    if device.type != "cuda" or not enabled or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _resolve_project_path(path_like):
    path_text = str(path_like or "").strip()
    if not path_text:
        return ""
    expanded = os.path.expanduser(path_text)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(_PROJECT_ROOT, expanded))


def _expected_data_cache_signature(mode):
    return {
        "mode": str(mode),
        "seq_len": int(DataConfig.seq_len),
        "stride_ratio": float(DataConfig.stride_ratio),
        "feature_cols": tuple(DataConfig.feature_cols),
        "random_seed": int(getattr(DataConfig, "random_seed", 42)),
        "train_val_split": float(DataConfig.train_val_split),
        "demo_days": int(max(1, int(getattr(DataConfig, "demo_days", 30)))),
        "demo_ratio": float(DataConfig.demo_ratio),
        "max_stocks": int(DataConfig.max_stocks) if DataConfig.max_stocks else None,
        "source_fingerprint": _dataset_source_fingerprint(_resolve_project_path(DataConfig.data_dir)),
    }


def _expected_encoding_cache_signature():
    tokenizer_path = _resolve_project_path(getattr(TrainingConfig, "tokenizer_path", ""))
    return {
        "tokenizer_path": tokenizer_path or None,
        "tokenizer_fingerprint": _resolve_tokenizer_fingerprint(),
    }


def _drop_cached_encodings(cache_payload, reason):
    if "encoded_indices_coarse" in cache_payload or "encoded_indices_fine" in cache_payload:
        print(f"Warning: drop cached tokenizer encodings ({reason}).")
    cache_payload.pop("encoded_indices_coarse", None)
    cache_payload.pop("encoded_indices_fine", None)


def _validate_cache_payload(cache_payload, mode, cache_path, skip_signature_check=False):
    required_cache_keys = {"features", "sector_ids", "time_features", "seq_stats"}
    missing_cache_keys = sorted(required_cache_keys - set(cache_payload.keys()))
    if missing_cache_keys:
        raise KeyError(f"Cache is missing required keys: {missing_cache_keys}")

    cached_data_sig = cache_payload.get("_data_cache_signature", cache_payload.get("_cache_signature"))
    expected_data_sig = _expected_data_cache_signature(mode)
    if cached_data_sig is None:
        print(f"Warning: cache has no data signature ({cache_path}), fallback to key-only validation.")
    elif cached_data_sig != expected_data_sig:
        mismatched = []
        for key in sorted(set(cached_data_sig) | set(expected_data_sig)):
            cv = cached_data_sig.get(key, "<MISSING>")
            ev = expected_data_sig.get(key, "<MISSING>")
            if cv != ev:
                mismatched.append(f"  {key}: cache={cv!r} vs expected={ev!r}")
        msg = (
            "Cache signature mismatch. Please rebuild cache to match current DataConfig.\n"
            + "\n".join(mismatched)
            + f"\nmode={mode}, cache={cache_path}"
        )
        if skip_signature_check:
            print(f"Warning: {msg}\nContinuing because --skip-cache-signature-check is set.")
        else:
            raise RuntimeError(msg)

    features = cache_payload["features"]
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)
    expected_samples = int(features.shape[0])
    expected_seq_len = int(DataConfig.seq_len)

    coarse = cache_payload.get("encoded_indices_coarse")
    fine = cache_payload.get("encoded_indices_fine")
    if coarse is None or fine is None:
        return

    if not isinstance(coarse, torch.Tensor) or not isinstance(fine, torch.Tensor):
        _drop_cached_encodings(cache_payload, reason="non-tensor encoded indices")
        return
    if coarse.ndim != 2 or fine.ndim != 2:
        _drop_cached_encodings(cache_payload, reason="encoded indices shape is not 2D")
        return
    if int(coarse.shape[0]) != expected_samples or int(fine.shape[0]) != expected_samples:
        _drop_cached_encodings(cache_payload, reason="encoded sample count mismatch")
        return
    if int(coarse.shape[1]) != expected_seq_len or int(fine.shape[1]) != expected_seq_len:
        _drop_cached_encodings(cache_payload, reason="encoded seq_len mismatch")
        return

    cached_encoding_sig = cache_payload.get("_encoding_cache_signature")
    expected_encoding_sig = _expected_encoding_cache_signature()
    if cached_encoding_sig != expected_encoding_sig:
        _drop_cached_encodings(cache_payload, reason="tokenizer fingerprint mismatch")


def _load_cache_payload(cache_path, mode, skip_signature_check=False):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Post_Train_DA cache not found: {cache_path}")
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def _direction_label(real_return, epsilon):
    if real_return > epsilon:
        return LABEL_UP
    if real_return < -epsilon:
        return LABEL_DOWN
    return LABEL_FLAT


def _seq_stats_to_arrays(seq_stats):
    means = []
    stds = []
    for item in seq_stats:
        means.append(np.asarray(item["mean"], dtype=np.float32))
        stds.append(np.asarray(item["std"], dtype=np.float32))
    return np.stack(means, axis=0), np.stack(stds, axis=0)


def _denormalized_last_returns(cache_payload):
    features = cache_payload["features"]
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)
    means, stds = _ensure_seq_arrays(cache_payload)
    norm_last = features[:, -1, 0].cpu().numpy().astype(np.float64)
    return norm_last * stds[:, 0].astype(np.float64) + means[:, 0].astype(np.float64)


def _cache_sample_epsilon(cache_payload, sample_idx, real_returns, label_mode, global_epsilon, cfg):
    if label_mode == "fixed":
        return max(float(cfg.min_epsilon), abs(float(cfg.fixed_epsilon)))
    if label_mode == "rolling_vol":
        features = cache_payload["features"]
        means = cache_payload.get("_seq_means_array")
        stds = cache_payload.get("_seq_stds_array")
        if means is None or stds is None:
            means, stds = _seq_stats_to_arrays(cache_payload["seq_stats"])
            cache_payload["_seq_means_array"] = means
            cache_payload["_seq_stds_array"] = stds
        prefix_norm = features[sample_idx, :-1, 0].cpu().numpy().astype(np.float64)
        prefix_returns = prefix_norm * float(stds[sample_idx, 0]) + float(means[sample_idx, 0])
        window = max(2, int(getattr(cfg, "rolling_vol_window", PostTrainDAConfig.rolling_vol_window)))
        vol = float(np.std(prefix_returns[-window:]))
        return max(float(cfg.min_epsilon), abs(vol) * abs(float(cfg.z_threshold)))
    return float(global_epsilon)


def _ensure_seq_arrays(cache_payload):
    if cache_payload.get("_seq_means_array") is None or cache_payload.get("_seq_stds_array") is None:
        means, stds = _seq_stats_to_arrays(cache_payload["seq_stats"])
        cache_payload["_seq_means_array"] = means
        cache_payload["_seq_stds_array"] = stds
    return cache_payload["_seq_means_array"], cache_payload["_seq_stds_array"]


def _split_cache_indices(num_samples, val_ratio):
    indices = np.arange(int(num_samples), dtype=np.int64)
    if len(indices) < 2:
        raise RuntimeError(f"Cache has too few samples for train/val split: {len(indices)}")
    val_count = int(round(len(indices) * float(val_ratio)))
    val_count = min(max(1, val_count), max(1, len(indices) - 1))
    split_idx = len(indices) - val_count
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    return train_indices, val_indices


class CachedDirectionDataset(Dataset):
    def __init__(
        self,
        cache_payload,
        indices,
        mode,
        real_returns,
        global_epsilon,
        max_samples,
        label_mode,
        fixed_epsilon,
        z_threshold,
        min_epsilon,
        rolling_vol_window,
        flat_policy,
        random_seed,
    ):
        self.cache = cache_payload
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mode = mode
        self.seq_len = int(DataConfig.seq_len)
        self.real_returns = np.asarray(real_returns, dtype=np.float64)
        self.global_epsilon = float(global_epsilon)
        self.max_samples = max(0, int(max_samples))
        self.label_mode = str(label_mode).strip().lower()
        self.fixed_epsilon = float(fixed_epsilon)
        self.z_threshold = float(z_threshold)
        self.min_epsilon = float(min_epsilon)
        self.rolling_vol_window = max(2, int(rolling_vol_window))
        self.flat_policy = str(flat_policy).strip().lower()
        self.random_seed = int(random_seed)
        self.features = self.cache["features"]
        if not isinstance(self.features, torch.Tensor):
            self.features = torch.as_tensor(self.features, dtype=torch.float32)
            self.cache["features"] = self.features
        self.seq_means, self.seq_stds = _ensure_seq_arrays(self.cache)
        self.samples = []
        self.class_counts = np.zeros(3, dtype=np.int64)
        self.loss_class_counts = np.zeros(3, dtype=np.int64)
        self.has_encoded = (
            isinstance(self.cache.get("encoded_indices_coarse"), torch.Tensor)
            and isinstance(self.cache.get("encoded_indices_fine"), torch.Tensor)
        )
        self._build_samples()

    def _compute_rolling_vol_epsilons(self, indices):
        n = len(indices)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        prefix_norm = self.features[indices, :-1, 0].numpy().astype(np.float64)
        means_col = self.seq_means[indices, 0].astype(np.float64)
        stds_col = self.seq_stds[indices, 0].astype(np.float64)
        prefix_returns = prefix_norm * stds_col[:, None] + means_col[:, None]
        window = self.rolling_vol_window
        last_seg = prefix_returns[:, -window:]
        vol = np.std(last_seg, axis=1)
        return np.maximum(self.min_epsilon, np.abs(vol) * self.z_threshold)

    def _build_samples(self):
        sample_indices = self.indices
        if len(sample_indices) == 0:
            return

        returns = self.real_returns[sample_indices]
        finite = np.isfinite(returns)
        if not finite.all():
            sample_indices = sample_indices[finite]
            returns = returns[finite]

        n = len(sample_indices)
        if n == 0:
            return

        if self.label_mode == "fixed":
            epsilons = np.full(n, max(self.min_epsilon, abs(self.fixed_epsilon)), dtype=np.float64)
        elif self.label_mode == "rolling_vol":
            epsilons = self._compute_rolling_vol_epsilons(sample_indices)
        else:
            epsilons = np.full(n, self.global_epsilon, dtype=np.float64)

        labels = np.full(n, LABEL_FLAT, dtype=np.int64)
        labels[returns > epsilons] = LABEL_UP
        labels[returns < -epsilons] = LABEL_DOWN

        loss_labels = labels.copy()
        if self.flat_policy == "ignore":
            loss_labels[labels == LABEL_FLAT] = IGNORE_INDEX

        self.samples = [
            (int(sample_indices[i]), int(labels[i]), int(loss_labels[i]),
             float(returns[i]), float(epsilons[i]))
            for i in range(n)
        ]

        self.class_counts[0] = int((labels == LABEL_DOWN).sum())
        self.class_counts[1] = int((labels == LABEL_FLAT).sum())
        self.class_counts[2] = int((labels == LABEL_UP).sum())

        active = loss_labels != IGNORE_INDEX
        self.loss_class_counts[0] = int((loss_labels[active] == LABEL_DOWN).sum())
        self.loss_class_counts[1] = int((loss_labels[active] == LABEL_FLAT).sum())
        self.loss_class_counts[2] = int((loss_labels[active] == LABEL_UP).sum())

        if self.max_samples > 0 and len(self.samples) > self.max_samples:
            rng = np.random.default_rng(self.random_seed)
            idx = np.sort(rng.choice(len(self.samples), size=self.max_samples, replace=False))
            self.samples = [self.samples[int(i)] for i in idx]
            self.class_counts[:] = 0
            self.loss_class_counts[:] = 0
            for _, label, loss_label, _, _ in self.samples:
                self.class_counts[label] += 1
                if loss_label != IGNORE_INDEX:
                    self.loss_class_counts[loss_label] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_idx, label, loss_label, real_return, epsilon = self.samples[idx]
        features = self.features[sample_idx, :-1].to(dtype=torch.float32)
        time_features = {
            key: value[sample_idx, :-1].to(dtype=torch.long)
            for key, value in self.cache["time_features"].items()
        }
        seq_stat = self.cache["seq_stats"][sample_idx]
        direction_sign = 0
        if label == LABEL_UP:
            direction_sign = 1
        elif label == LABEL_DOWN:
            direction_sign = -1
        result = {
            "features": features,
            "sector_id": int(self.cache["sector_ids"][sample_idx]),
            "time": time_features,
            "label": int(label),
            "loss_label": int(loss_label),
            "real_return": float(real_return),
            "epsilon": float(epsilon),
            "sample_id": f"cache:{int(sample_idx)}",
            "symbol": "cache",
            "prompt_mean": float(seq_stat["mean"][0]),
            "prompt_std": float(seq_stat["std"][0]),
            "direction_sign": direction_sign,
            "features_full": self.features[sample_idx].to(dtype=torch.float32),
        }
        if self.has_encoded:
            result["idx_coarse_full"] = self.cache["encoded_indices_coarse"][sample_idx].long()
            result["idx_fine_full"] = self.cache["encoded_indices_fine"][sample_idx].long()
        return result


def direction_collate(batch):
    has_encoded = all("idx_coarse_full" in item and "idx_fine_full" in item for item in batch)
    result = {
        "sector_ids": torch.as_tensor([item["sector_id"] for item in batch], dtype=torch.long),
        "time": {
            key: torch.stack([item["time"][key] for item in batch], dim=0).long()
            for key in ("minute", "day", "month", "year")
        },
        "labels": torch.as_tensor([item["label"] for item in batch], dtype=torch.long),
        "loss_labels": torch.as_tensor([item["loss_label"] for item in batch], dtype=torch.long),
        "real_returns": torch.as_tensor([item["real_return"] for item in batch], dtype=torch.float32),
        "epsilons": torch.as_tensor([item["epsilon"] for item in batch], dtype=torch.float32),
        "sample_ids": [item["sample_id"] for item in batch],
        "symbols": [item["symbol"] for item in batch],
        "prompt_means": torch.as_tensor([item["prompt_mean"] for item in batch], dtype=torch.float32),
        "prompt_stds": torch.as_tensor([item["prompt_std"] for item in batch], dtype=torch.float32),
        "direction_signs": torch.as_tensor([item["direction_sign"] for item in batch], dtype=torch.float32),
    }
    if has_encoded:
        result["idx_coarse_full"] = torch.stack([item["idx_coarse_full"] for item in batch], dim=0)
        result["idx_fine_full"] = torch.stack([item["idx_fine_full"] for item in batch], dim=0)
    else:
        result["features_full"] = torch.stack([item["features_full"] for item in batch], dim=0)
    return result


def _class_weights(counts, device):
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.zeros(3, dtype=np.float32)
    active = counts > 0
    if active.any():
        inv = counts[active].sum() / (counts[active] * active.sum())
        weights[active] = inv.astype(np.float32)
    weights[~active] = 0.0
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def last_step_token_ce(logits_c, logits_f, target_c, target_f):
    return F.cross_entropy(logits_c[:, -1, :].float(), target_c[:, -1]) + F.cross_entropy(
        logits_f[:, -1, :].float(),
        target_f[:, -1],
    )


def token_kl(current_c, current_f, ref_c, ref_f):
    kl_c = F.kl_div(
        F.log_softmax(current_c[:, -1, :].float(), dim=-1),
        F.softmax(ref_c[:, -1, :].float(), dim=-1),
        reduction="batchmean",
    )
    kl_f = F.kl_div(
        F.log_softmax(current_f[:, -1, :].float(), dim=-1),
        F.softmax(ref_f[:, -1, :].float(), dim=-1),
        reduction="batchmean",
    )
    return kl_c + kl_f


def latent_regularization_loss(latent_states):
    if latent_states is None or latent_states.shape[0] < 2:
        device = latent_states.device if latent_states is not None else "cpu"
        return torch.tensor(0.0, device=device)
    diff = latent_states[1:] - latent_states[:-1]
    diversity_loss = torch.exp(-diff.pow(2).sum(-1).sqrt().mean())
    latent_flat = latent_states.reshape(latent_states.shape[0], -1, latent_states.shape[-1])
    collapse_loss = torch.exp(-latent_flat.var(dim=1).mean())
    return 0.6 * diversity_loss + 0.0005 * collapse_loss


def _sample_next_tokens(step_logits, temperature, num_samples):
    temp = max(float(temperature), 1e-5)
    safe_logits = torch.nan_to_num(step_logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    probs = torch.softmax(safe_logits / temp, dim=-1)
    return torch.multinomial(probs, num_samples=max(1, int(num_samples)), replacement=True)


@torch.no_grad()
def _token_return_values(tokenizer, coarse_idx, fine_idx, means, stds):
    decoded = tokenizer.decode(coarse_idx, fine_idx)
    log_ret_norm = decoded[..., 0]
    dev = log_ret_norm.device
    means = means.to(dev)
    stds = stds.to(dev)
    if means.ndim == 1:
        mean_col = means
    else:
        mean_col = means[:, 0]
    if stds.ndim == 1:
        std_col = stds
    else:
        std_col = stds[:, 0]
    while mean_col.ndim < log_ret_norm.ndim:
        mean_col = mean_col.unsqueeze(-1)
        std_col = std_col.unsqueeze(-1)
    return log_ret_norm * std_col + mean_col


@torch.no_grad()
def _token_return_direction(tokenizer, coarse_idx, fine_idx, means, stds):
    log_ret = _token_return_values(tokenizer, coarse_idx, fine_idx, means, stds)
    return torch.sign(log_ret)


@torch.no_grad()
def _return_labels_from_values(log_ret, epsilons):
    eps = epsilons.to(log_ret.device).abs()
    while eps.ndim < log_ret.ndim:
        eps = eps.unsqueeze(-1)
    labels = torch.full_like(log_ret, LABEL_FLAT, dtype=torch.long)
    labels = torch.where(log_ret > eps, torch.full_like(labels, LABEL_UP), labels)
    labels = torch.where(log_ret < -eps, torch.full_like(labels, LABEL_DOWN), labels)
    return labels


@torch.no_grad()
def _token_return_labels(tokenizer, coarse_idx, fine_idx, means, stds, epsilons):
    log_ret = _token_return_values(tokenizer, coarse_idx, fine_idx, means, stds)
    return _return_labels_from_values(log_ret, epsilons)


def _candidate_logp(logits_coarse, logits_fine, candidate_coarse, candidate_fine):
    logp_c = F.log_softmax(logits_coarse.float(), dim=-1)
    logp_f = F.log_softmax(logits_fine.float(), dim=-1)
    return (
        logp_c.gather(1, candidate_coarse.long().unsqueeze(1)).squeeze(1)
        + logp_f.gather(1, candidate_fine.long().unsqueeze(1)).squeeze(1)
    )


def expo_regression_loss(
    logits_coarse,
    logits_fine,
    ref_logits_coarse,
    ref_logits_fine,
    winner_coarse,
    winner_fine,
    loser_coarse,
    loser_fine,
    valid_pair,
    reference_weight,
):
    valid_pair = valid_pair.to(dtype=torch.bool)
    theta_win = _candidate_logp(logits_coarse, logits_fine, winner_coarse, winner_fine)
    theta_lose = _candidate_logp(logits_coarse, logits_fine, loser_coarse, loser_fine)
    with torch.no_grad():
        ref_win = _candidate_logp(ref_logits_coarse, ref_logits_fine, winner_coarse, winner_fine)
        ref_lose = _candidate_logp(ref_logits_coarse, ref_logits_fine, loser_coarse, loser_fine)
        ref_pref = torch.sigmoid(ref_win - ref_lose)

    theta_pref = torch.sigmoid(theta_win - theta_lose)
    lam = max(0.0, min(1.0, float(reference_weight)))
    target_pref = (lam * ref_pref + (1.0 - lam)).clamp(0.0, 1.0)
    per_row = (theta_pref - target_pref).pow(2)
    valid_weight = valid_pair.to(dtype=per_row.dtype)
    loss = (per_row * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)

    prob_sums = torch.stack(
        [
            (target_pref.detach() * valid_weight).sum(),
            (theta_pref.detach() * valid_weight).sum(),
            (ref_pref.detach() * valid_weight).sum(),
        ]
    )
    return loss, prob_sums


@torch.no_grad()
def _sample_expo_pairs(
    tokenizer,
    ref_logits_coarse,
    ref_logits_fine,
    labels,
    loss_labels,
    real_returns,
    epsilons,
    prompt_means,
    prompt_stds,
    gold_coarse,
    gold_fine,
    temperature,
    num_candidates,
    direction_bonus,
    error_weight,
    min_score_margin,
    include_gold,
):
    """Sample reference-policy candidates and turn them into EXPO winner/loser pairs."""
    sampled_c = _sample_next_tokens(ref_logits_coarse, temperature, num_candidates)
    sampled_f = _sample_next_tokens(ref_logits_fine, temperature, num_candidates)
    if bool(include_gold):
        sampled_c = torch.cat([sampled_c, gold_coarse.long().unsqueeze(1)], dim=1)
        sampled_f = torch.cat([sampled_f, gold_fine.long().unsqueeze(1)], dim=1)

    candidate_returns = _token_return_values(
        tokenizer,
        sampled_c,
        sampled_f,
        prompt_means,
        prompt_stds,
    )
    candidate_labels = _return_labels_from_values(candidate_returns, epsilons).to(labels.device)

    valid_label = loss_labels != IGNORE_INDEX
    real = real_returns.to(candidate_returns.device, dtype=candidate_returns.dtype)
    eps = epsilons.to(candidate_returns.device, dtype=candidate_returns.dtype).abs()
    error_scale = torch.maximum(real.abs(), eps).clamp_min(1e-6)
    while real.ndim < candidate_returns.ndim:
        real = real.unsqueeze(-1)
        error_scale = error_scale.unsqueeze(-1)
    abs_error = (candidate_returns - real).abs()
    normalized_error = torch.nan_to_num(abs_error / error_scale, nan=1e6, posinf=1e6, neginf=1e6)
    direction_correct = candidate_labels.eq(labels.unsqueeze(1)) & valid_label.unsqueeze(1)
    scores = (
        direction_correct.to(dtype=candidate_returns.dtype) * float(direction_bonus)
        - normalized_error * float(error_weight)
    )
    scores = scores.masked_fill(~valid_label.unsqueeze(1), float("-inf"))

    winner_score, winner_idx = scores.max(dim=1)
    loser_score, loser_idx = scores.min(dim=1)
    score_margin = winner_score - loser_score
    valid_pair = (
        valid_label
        & torch.isfinite(winner_score)
        & torch.isfinite(loser_score)
        & (winner_idx != loser_idx)
        & (score_margin >= float(min_score_margin))
    )

    rows = torch.arange(sampled_c.size(0), device=sampled_c.device)
    winner_coarse = sampled_c[rows, winner_idx]
    winner_fine = sampled_f[rows, winner_idx]
    loser_coarse = sampled_c[rows, loser_idx]
    loser_fine = sampled_f[rows, loser_idx]
    winner_correct = direction_correct[rows, winner_idx]
    loser_correct = direction_correct[rows, loser_idx]
    winner_error = abs_error[rows, winner_idx]
    loser_error = abs_error[rows, loser_idx]
    winner_from_gold = (
        winner_idx == sampled_c.size(1) - 1
        if bool(include_gold)
        else torch.zeros_like(valid_pair, dtype=torch.bool)
    )

    valid_weight = valid_pair.to(dtype=candidate_returns.dtype)
    counts = torch.stack(
        [
            valid_label.sum(),
            valid_pair.sum(),
            (valid_label & ~valid_pair).sum(),
            (valid_pair & winner_correct).sum(),
            (valid_pair & loser_correct).sum(),
            (valid_pair & winner_from_gold).sum(),
        ]
    )
    sums = torch.stack(
        [
            (score_margin.clamp_min(0.0) * valid_weight).sum(),
            (winner_error * valid_weight).sum(),
            (loser_error * valid_weight).sum(),
        ]
    )
    return winner_coarse, winner_fine, loser_coarse, loser_fine, valid_pair, counts, sums


def _dataloader_worker_init(_worker_id):
    # Avoid CPU oversubscription when using multiple dataloader workers.
    torch.set_num_threads(1)


def _move_batch(batch, device, non_blocking=True):
    result = {
        "sector_ids": batch["sector_ids"].to(device=device, dtype=torch.long, non_blocking=non_blocking),
        "time": {
            key: value.to(device=device, dtype=torch.long, non_blocking=non_blocking)
            for key, value in batch["time"].items()
        },
        "labels": batch["labels"].to(device=device, dtype=torch.long, non_blocking=non_blocking),
        "loss_labels": batch["loss_labels"].to(device=device, dtype=torch.long, non_blocking=non_blocking),
        "real_returns": batch["real_returns"].to(device=device, dtype=torch.float32, non_blocking=non_blocking),
        "epsilons": batch["epsilons"].to(device=device, dtype=torch.float32, non_blocking=non_blocking),
        "sample_ids": batch["sample_ids"],
        "symbols": batch["symbols"],
        "prompt_means": batch["prompt_means"].to(device=device, dtype=torch.float32, non_blocking=non_blocking),
        "prompt_stds": batch["prompt_stds"].to(device=device, dtype=torch.float32, non_blocking=non_blocking),
        "direction_signs": batch["direction_signs"].to(device=device, dtype=torch.float32, non_blocking=non_blocking),
    }
    if "features_full" in batch:
        result["features_full"] = batch["features_full"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )
    if "idx_coarse_full" in batch and "idx_fine_full" in batch:
        result["idx_coarse_full"] = batch["idx_coarse_full"].to(
            device=device,
            dtype=torch.long,
            non_blocking=non_blocking,
        )
        result["idx_fine_full"] = batch["idx_fine_full"].to(
            device=device,
            dtype=torch.long,
            non_blocking=non_blocking,
        )
    return result


def _encode_prompt(tokenizer, features):
    with torch.no_grad():
        idx_coarse, idx_fine = tokenizer.encode(features)
    return idx_coarse.long(), idx_fine.long()


def compute_metrics(probs, labels, real_returns, confidence_threshold=0.55, margin_threshold=0.0):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    real_returns = np.asarray(real_returns, dtype=np.float64)
    if probs.size == 0:
        return {"num_samples": 0}

    p_down = probs[:, LABEL_DOWN]
    p_up = probs[:, LABEL_UP]
    side_conf = np.maximum(p_down, p_up)
    margin = np.abs(p_up - p_down)
    hard_side_pred = np.where(p_up >= p_down, LABEL_UP, LABEL_DOWN).astype(np.int64)
    pred = np.full(labels.shape, LABEL_FLAT, dtype=np.int64)
    decisive = (side_conf >= float(confidence_threshold)) & (margin > float(margin_threshold))
    pred[decisive & (p_up > p_down)] = LABEL_UP
    pred[decisive & (p_down >= p_up)] = LABEL_DOWN
    raw_pred = probs.argmax(axis=1)

    true_directional = labels != LABEL_FLAT
    pred_directional = pred != LABEL_FLAT
    covered_directional = true_directional & pred_directional
    decision_correct = pred == labels
    hard_correct = hard_side_pred == labels
    raw_correct = raw_pred == labels

    def _safe_mean(mask, values, default=0.0):
        if mask.sum() == 0:
            return float(default)
        return float(np.mean(values[mask]))

    recall_down = _safe_mean(labels == LABEL_DOWN, hard_side_pred == LABEL_DOWN)
    recall_up = _safe_mean(labels == LABEL_UP, hard_side_pred == LABEL_UP)
    balanced_accuracy = float(np.mean([recall_down, recall_up]))

    up_precision = _safe_mean(hard_side_pred == LABEL_UP, labels == LABEL_UP)
    down_precision = _safe_mean(hard_side_pred == LABEL_DOWN, labels == LABEL_DOWN)
    direction_accuracy = _safe_mean(true_directional, hard_correct)
    threshold_direction_accuracy = _safe_mean(covered_directional, decision_correct)
    threshold_direction_accuracy_all = _safe_mean(true_directional, decision_correct)

    confidences = probs.max(axis=1)
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bin_edges[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        ece += float(mask.mean()) * abs(float(raw_correct[mask].mean()) - float(confidences[mask].mean()))

    bucket_metrics = {}
    abs_returns = np.abs(real_returns)
    if len(abs_returns) >= 3:
        q1, q2 = np.quantile(abs_returns, [1.0 / 3.0, 2.0 / 3.0])
        buckets = {
            "small": abs_returns <= q1,
            "medium": (abs_returns > q1) & (abs_returns <= q2),
            "large": abs_returns > q2,
        }
        for name, mask in buckets.items():
            metric = _safe_mean(mask & true_directional, hard_correct)
            bucket_metrics[f"return_bucket_{name}_accuracy"] = metric

    return {
        "num_samples": int(len(labels)),
        "direction_accuracy": direction_accuracy,
        "raw_argmax_accuracy": float(np.mean(raw_correct)),
        "balanced_accuracy": balanced_accuracy,
        "threshold_direction_accuracy": threshold_direction_accuracy,
        "threshold_direction_accuracy_all": threshold_direction_accuracy_all,
        "coverage": float(np.mean(pred_directional)),
        "directional_coverage": float(np.mean(covered_directional[true_directional])) if true_directional.any() else 0.0,
        "up_precision": up_precision,
        "down_precision": down_precision,
        "recall_up": recall_up,
        "recall_down": recall_down,
        "flat_accuracy": _safe_mean(labels == LABEL_FLAT, pred == LABEL_FLAT),
        "ece": float(ece),
        "class_counts": {LABEL_NAMES[i]: int((labels == i).sum()) for i in range(3)},
        "hard_pred_counts": {LABEL_NAMES[i]: int((hard_side_pred == i).sum()) for i in range(3)},
        "pred_counts": {LABEL_NAMES[i]: int((pred == i).sum()) for i in range(3)},
        **bucket_metrics,
    }


@torch.no_grad()
def _token_logits_to_pseudo_probs(logits_c, logits_f, tokenizer, means, stds):
    pred_c = logits_c.float().argmax(dim=-1)
    pred_f = logits_f.float().argmax(dim=-1)
    direction = _token_return_direction(tokenizer, pred_c.unsqueeze(1), pred_f.unsqueeze(1), means, stds)
    if direction.ndim > 1:
        direction = direction.squeeze(-1)
    conf_c = F.softmax(logits_c.float(), dim=-1).max(dim=-1).values
    conf_f = F.softmax(logits_f.float(), dim=-1).max(dim=-1).values
    confidence = (conf_c * conf_f).sqrt()

    B = logits_c.size(0)
    conf = confidence.cpu().to(torch.float64)
    d = direction.clamp(-1, 1).long().cpu()
    pred_class = d + 1

    other = ((1.0 - conf) * 0.5).unsqueeze(-1)
    probs = other.expand(B, 3).clone()
    probs[torch.arange(B), pred_class] = conf
    return probs.numpy()


@torch.no_grad()
def evaluate_loader(model, tokenizer, loader, device, amp_enabled, amp_dtype, confidence_threshold, margin_threshold):
    model.eval()
    all_probs = []
    all_labels = []
    all_returns = []
    for raw_batch in tqdm(loader, desc="Validate DA_EXPO", leave=False):
        batch = _move_batch(raw_batch, device)
        t_minute = batch["time"]["minute"]
        t_day = batch["time"]["day"]
        t_month = batch["time"]["month"]
        t_year = batch["time"]["year"]
        if t_minute.size(1) == batch["idx_coarse_full"].size(1):
            t_minute = t_minute[:, :-1]
            t_day = t_day[:, :-1]
            t_month = t_month[:, :-1]
            t_year = t_year[:, :-1]
        with _autocast_context(device, amp_enabled, amp_dtype):
            logits_c, logits_f, _ = model(
                batch["idx_coarse_full"][:, :-1],
                batch["idx_fine_full"][:, :-1],
                batch["sector_ids"],
                t_minute,
                t_day,
                t_month,
                t_year,
                last_only=True,
            )
        pprobs = _token_logits_to_pseudo_probs(
            logits_c[:, -1, :], logits_f[:, -1, :],
            tokenizer,
            batch["prompt_means"], batch["prompt_stds"],
        )
        all_probs.append(pprobs)
        all_labels.append(batch["labels"].cpu().numpy())
        all_returns.append(batch["real_returns"].cpu().numpy())
    if not all_probs:
        return {"num_samples": 0}
    return compute_metrics(
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_returns, axis=0),
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )


def _select_last_n_days(eval_items, days):
    if not eval_items:
        return [], []
    dates = sorted({item["future_dates"][0] for item in eval_items})
    selected = set(dates[-max(1, int(days)):])
    return [item for item in eval_items if item["future_dates"][0] in selected], sorted(selected)


def _item_return_and_label(item, epsilon):
    last_close = max(float(item["hist_closes"][-1]), 1e-8)
    next_close = max(float(item["actual_future"][0]), 1e-8)
    real_return = math.log(next_close / last_close)
    return real_return, _direction_label(real_return, float(epsilon))


def _eval_item_epsilon(item, label_mode, global_epsilon, fixed_epsilon, z_threshold, min_epsilon, rolling_vol_window):
    label_mode = str(label_mode).strip().lower()
    if label_mode == "fixed":
        return max(float(min_epsilon), abs(float(fixed_epsilon)))
    if label_mode == "rolling_vol":
        prompt_norm = np.asarray(item["prompt_norm"], dtype=np.float64)
        prompt_mean = np.asarray(item["prompt_mean"], dtype=np.float64)
        prompt_std = np.asarray(item["prompt_std"], dtype=np.float64)
        if prompt_norm.ndim != 2 or prompt_norm.shape[0] == 0:
            return float(global_epsilon)
        prefix_returns = prompt_norm[:, 0] * float(prompt_std[0]) + float(prompt_mean[0])
        window = max(2, int(rolling_vol_window))
        vol = float(np.std(prefix_returns[-window:]))
        return max(float(min_epsilon), abs(vol) * abs(float(z_threshold)))
    return float(global_epsilon)


@torch.no_grad()
def _predict_direction_probs(model, tokenizer, items, device, batch_size, amp_enabled, amp_dtype):
    probs = []
    step = max(1, int(batch_size))
    for start in tqdm(range(0, len(items), step), desc="Validate demo rolling_1d", leave=False):
        batch_items = items[start : start + step]
        features = torch.from_numpy(
            np.stack([item["prompt_norm"] for item in batch_items], axis=0).astype(np.float32)
        ).to(device=device, dtype=torch.float32)
        sector_ids = torch.as_tensor(
            [item["sector_id"] for item in batch_items],
            dtype=torch.long,
            device=device,
        )
        time = {
            key: torch.from_numpy(
                np.stack([item["prompt_time"][key] for item in batch_items], axis=0).astype(np.int64)
            ).to(device=device, dtype=torch.long)
            for key in ("minute", "day", "month", "year")
        }
        idx_coarse, idx_fine = tokenizer.encode(features)
        with _autocast_context(device, amp_enabled, amp_dtype):
            logits_c, logits_f, _ = model(
                idx_coarse.long(),
                idx_fine.long(),
                sector_ids,
                time["minute"],
                time["day"],
                time["month"],
                time["year"],
                last_only=True,
            )
        means = torch.as_tensor(
            [item["prompt_mean"] for item in batch_items], dtype=torch.float32, device=device,
        )
        stds = torch.as_tensor(
            [item["prompt_std"] for item in batch_items], dtype=torch.float32, device=device,
        )
        pprobs = _token_logits_to_pseudo_probs(
            logits_c[:, -1, :], logits_f[:, -1, :], tokenizer, means, stds,
        )
        probs.append(pprobs)
    if not probs:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(probs, axis=0)


def _prepare_demo_eval_items(cfg):
    if int(cfg.eval_demo_days) <= 0:
        return [], {"enabled": False}

    demo_dataset = AShareDataset(mode="demo")
    symbols = sorted(demo_dataset.raw_data.keys())
    if int(cfg.max_stocks) > 0:
        symbols = symbols[: int(cfg.max_stocks)]
    eval_items, skip_reasons = build_rolling_1d_eval_items(demo_dataset=demo_dataset, symbols=symbols)
    eval_items, selected_dates = _select_last_n_days(eval_items, int(cfg.eval_demo_days))

    if int(cfg.max_eval_items) > 0 and len(eval_items) > int(cfg.max_eval_items):
        rng = np.random.default_rng(int(cfg.random_seed))
        chosen = np.sort(rng.choice(len(eval_items), size=int(cfg.max_eval_items), replace=False))
        eval_items = [eval_items[int(idx)] for idx in chosen]

    return eval_items, {
        "enabled": True,
        "num_items": int(len(eval_items)),
        "num_symbols": int(len(symbols)),
        "selected_dates": [item.date().isoformat() for item in selected_dates],
        "skip_reasons": skip_reasons,
    }


@torch.no_grad()
def evaluate_demo_items(model, tokenizer, eval_items, cfg, label_info, device, amp_enabled, amp_dtype):
    if not eval_items:
        return {"num_samples": 0}

    probs = _predict_direction_probs(
        model=model,
        tokenizer=tokenizer,
        items=eval_items,
        device=device,
        batch_size=int(cfg.batch_size),
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    labels = []
    returns = []
    for item in eval_items:
        epsilon = _eval_item_epsilon(
            item=item,
            label_mode=label_info.get("label_mode", "global_median"),
            global_epsilon=float(label_info.get("global_epsilon", cfg.min_epsilon)),
            fixed_epsilon=float(label_info.get("fixed_epsilon", cfg.fixed_epsilon)),
            z_threshold=float(label_info.get("z_threshold", cfg.z_threshold)),
            min_epsilon=float(label_info.get("min_epsilon", cfg.min_epsilon)),
            rolling_vol_window=int(label_info.get("rolling_vol_window", PostTrainDAConfig.rolling_vol_window)),
        )
        real_return, label = _item_return_and_label(item, epsilon)
        labels.append(label)
        returns.append(real_return)

    return compute_metrics(
        probs,
        np.asarray(labels, dtype=np.int64),
        np.asarray(returns, dtype=np.float64),
        confidence_threshold=float(cfg.eval_confidence_threshold),
        margin_threshold=float(cfg.eval_margin_threshold),
    )


def configure_trainable_params(model, cfg):
    if bool(cfg.freeze_backbone):
        for param in model.parameters():
            param.requires_grad = False

    if bool(cfg.train_lora):
        inject_lora(
            model,
            rank=int(cfg.lora_rank),
            alpha=float(cfg.lora_alpha),
            dropout=float(cfg.lora_dropout),
            target_keywords=tuple(cfg.lora_target_keywords),
            freeze_base=True,
        )

    if not bool(cfg.freeze_backbone) and not bool(cfg.train_lora):
        for param in model.parameters():
            param.requires_grad = True

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    param_groups = [{"params": trainable_params, "lr": float(cfg.learning_rate)}]
    return param_groups


def _build_adamw_optimizer(param_groups, cfg, device):
    base_kwargs = {"weight_decay": float(cfg.weight_decay)}
    candidates = []
    if device.type == "cuda":
        candidates.append({**base_kwargs, "fused": True})
        candidates.append({**base_kwargs, "foreach": True})
    candidates.append(base_kwargs)

    last_exc = None
    for kwargs in candidates:
        try:
            return torch.optim.AdamW(param_groups, **kwargs), kwargs
        except (TypeError, RuntimeError, ValueError) as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    return torch.optim.AdamW(param_groups, **base_kwargs), base_kwargs


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Post_Train_DA direction EXPO")
    parser.add_argument("--checkpoint-path", default=PostTrainDAConfig.checkpoint_path)
    parser.add_argument("--cache-path", default=PostTrainDAConfig.cache_path)
    parser.add_argument("--val-cache-path", default=getattr(PostTrainDAConfig, "val_cache_path", ""))
    parser.add_argument("--output-dir", default=PostTrainDAConfig.output_dir)
    parser.add_argument("--save-name", default=PostTrainDAConfig.save_name)
    parser.add_argument("--save-epoch-checkpoints", type=_as_bool, default=PostTrainDAConfig.save_epoch_checkpoints)
    parser.add_argument("--epochs", type=int, default=PostTrainDAConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=PostTrainDAConfig.batch_size)
    parser.add_argument("--accumulation-steps", type=int, default=PostTrainDAConfig.accumulation_steps)
    parser.add_argument("--num-workers", type=int, default=PostTrainDAConfig.num_workers)
    parser.add_argument("--cpu-threads", type=int, default=getattr(PostTrainDAConfig, "cpu_threads", 0))
    parser.add_argument("--lr", type=float, default=PostTrainDAConfig.learning_rate)
    parser.add_argument("--backbone-lr", type=float, default=PostTrainDAConfig.backbone_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=PostTrainDAConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=PostTrainDAConfig.grad_clip)
    parser.add_argument("--max-train-updates", type=int, default=PostTrainDAConfig.max_train_updates)
    parser.add_argument("--progress-interval", type=int, default=PostTrainDAConfig.progress_interval)
    parser.add_argument("--sample-stride", type=int, default=PostTrainDAConfig.sample_stride)
    parser.add_argument("--val-sample-stride", type=int, default=PostTrainDAConfig.val_sample_stride)
    parser.add_argument("--max-train-samples", type=int, default=PostTrainDAConfig.max_train_samples)
    parser.add_argument("--max-val-samples", type=int, default=PostTrainDAConfig.max_val_samples)
    parser.add_argument("--max-eval-items", type=int, default=PostTrainDAConfig.max_eval_items)
    parser.add_argument("--max-stocks", type=int, default=PostTrainDAConfig.max_stocks)
    parser.add_argument("--cache-val-ratio", type=float, default=PostTrainDAConfig.cache_val_ratio)
    parser.add_argument("--skip-cache-signature-check", action="store_true")
    parser.add_argument("--eval-demo-days", type=int, default=getattr(PostTrainDAConfig, "eval_demo_days", 10))
    parser.add_argument("--demo-score-weight", type=float, default=getattr(PostTrainDAConfig, "demo_score_weight", 0.5))
    parser.add_argument("--label-mode", choices=["global_median", "rolling_vol", "fixed"], default=PostTrainDAConfig.label_mode)
    parser.add_argument("--epsilon-scale", type=float, default=PostTrainDAConfig.epsilon_scale)
    parser.add_argument("--fixed-epsilon", type=float, default=PostTrainDAConfig.fixed_epsilon)
    parser.add_argument("--z-threshold", type=float, default=PostTrainDAConfig.z_threshold)
    parser.add_argument("--min-epsilon", type=float, default=PostTrainDAConfig.min_epsilon)
    parser.add_argument("--flat-policy", choices=["class", "ignore"], default=PostTrainDAConfig.flat_policy)
    parser.add_argument("--expo-temperature", type=float, default=PostTrainDAConfig.expo_temperature)
    parser.add_argument("--expo-num-candidates", type=int, default=PostTrainDAConfig.expo_num_candidates)
    parser.add_argument("--expo-reference-weight", type=float, default=PostTrainDAConfig.expo_reference_weight)
    parser.add_argument("--expo-score-margin", type=float, default=PostTrainDAConfig.expo_score_margin)
    parser.add_argument("--expo-direction-bonus", type=float, default=PostTrainDAConfig.expo_direction_bonus)
    parser.add_argument("--expo-error-weight", type=float, default=PostTrainDAConfig.expo_error_weight)
    parser.add_argument("--expo-include-gold", type=_as_bool, default=PostTrainDAConfig.expo_include_gold)
    parser.add_argument("--keep-auxiliary", type=_as_bool, default=PostTrainDAConfig.expo_keep_auxiliary)
    parser.add_argument("--token-ce-weight", type=float, default=PostTrainDAConfig.token_ce_weight)
    parser.add_argument("--kl-weight", type=float, default=PostTrainDAConfig.kl_weight)
    parser.add_argument("--latent-weight", type=float, default=PostTrainDAConfig.latent_weight)
    parser.add_argument("--confidence-threshold", type=float, default=PostTrainDAConfig.eval_confidence_threshold)
    parser.add_argument("--margin-threshold", type=float, default=PostTrainDAConfig.eval_margin_threshold)
    parser.add_argument("--freeze-backbone", type=_as_bool, default=PostTrainDAConfig.freeze_backbone)
    parser.add_argument("--train-lora", type=_as_bool, default=PostTrainDAConfig.train_lora)
    parser.add_argument("--use-amp", type=_as_bool, default=PostTrainDAConfig.use_amp)
    parser.add_argument("--amp-dtype", default=PostTrainDAConfig.amp_dtype)
    parser.add_argument("--use-tf32", type=_as_bool, default=PostTrainDAConfig.use_tf32)
    parser.add_argument("--deterministic", type=_as_bool, default=getattr(PostTrainDAConfig, "deterministic", False))
    parser.add_argument("--seed", type=int, default=PostTrainDAConfig.random_seed)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--torch-compile", action="store_true", help="Enable torch.compile for model acceleration")
    return parser


def _namespace_to_config(args):
    return argparse.Namespace(
        checkpoint_path=args.checkpoint_path,
        cache_path=args.cache_path,
        val_cache_path=args.val_cache_path,
        output_dir=args.output_dir,
        save_name=args.save_name,
        save_epoch_checkpoints=bool(args.save_epoch_checkpoints),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        accumulation_steps=max(1, int(args.accumulation_steps)),
        num_workers=max(0, int(args.num_workers)),
        cpu_threads=max(0, int(args.cpu_threads)),
        learning_rate=float(args.lr),
        backbone_learning_rate=float(args.backbone_lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        max_train_updates=max(0, int(args.max_train_updates)),
        progress_interval=max(1, int(args.progress_interval)),
        sample_stride=max(1, int(args.sample_stride)),
        val_sample_stride=max(1, int(args.val_sample_stride)),
        max_train_samples=max(0, int(args.max_train_samples)),
        max_val_samples=max(0, int(args.max_val_samples)),
        max_eval_items=max(0, int(args.max_eval_items)),
        max_stocks=max(0, int(args.max_stocks)),
        cache_val_ratio=max(0.01, min(0.5, float(args.cache_val_ratio))),
        eval_demo_days=max(0, int(args.eval_demo_days)),
        demo_score_weight=max(0.0, min(1.0, float(args.demo_score_weight))),
        label_mode=args.label_mode,
        epsilon_scale=float(args.epsilon_scale),
        fixed_epsilon=float(args.fixed_epsilon),
        z_threshold=float(args.z_threshold),
        min_epsilon=float(args.min_epsilon),
        rolling_vol_window=max(2, int(getattr(PostTrainDAConfig, "rolling_vol_window", 20))),
        flat_policy=args.flat_policy,
        expo_temperature=float(args.expo_temperature),
        expo_num_candidates=max(2, int(args.expo_num_candidates)),
        expo_reference_weight=max(0.0, min(1.0, float(args.expo_reference_weight))),
        expo_score_margin=max(0.0, float(args.expo_score_margin)),
        expo_direction_bonus=float(args.expo_direction_bonus),
        expo_error_weight=max(0.0, float(args.expo_error_weight)),
        expo_include_gold=bool(args.expo_include_gold),
        expo_keep_auxiliary=bool(args.keep_auxiliary),
        token_ce_weight=float(args.token_ce_weight),
        kl_weight=float(args.kl_weight),
        latent_weight=float(args.latent_weight),
        eval_confidence_threshold=float(args.confidence_threshold),
        eval_margin_threshold=float(args.margin_threshold),
        freeze_backbone=bool(args.freeze_backbone),
        train_lora=bool(args.train_lora),
        lora_rank=int(PostTrainDAConfig.lora_rank),
        lora_alpha=float(PostTrainDAConfig.lora_alpha),
        lora_dropout=float(PostTrainDAConfig.lora_dropout),
        lora_target_keywords=tuple(PostTrainDAConfig.lora_target_keywords),
        use_amp=bool(args.use_amp),
        amp_dtype=args.amp_dtype,
        use_tf32=bool(args.use_tf32),
        deterministic=bool(args.deterministic),
        random_seed=int(args.seed),
        eval_only=bool(args.eval_only),
        skip_cache_signature_check=bool(args.skip_cache_signature_check),
        torch_compile=bool(args.torch_compile),
    )


def _save_epoch_base_model(model, tokenizer, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"direction_expo_basemodel-{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "tokenizer_state_dict": tokenizer.state_dict(),
        },
        path,
    )
    print(f"Epoch base model saved: {path}")
    return path


def _save_checkpoint(path, model, tokenizer, cfg, label_info, metrics, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    adapter_path = ""
    if has_lora_layers(model):
        adapter_path = os.path.join(os.path.dirname(path), "direction_expo_lora_adapter.pt")
    payload = {
        "stage": "Post_Train_DA_B2_direction_expo",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_state_dict": model.state_dict() if not has_lora_layers(model) else None,
        "direction_head_state_dict": model.direction_head.state_dict(),
        "lora_state_dict": lora_state_dict(model) if has_lora_layers(model) else None,
        "lora_adapter_path": adapter_path,
        "tokenizer_state_dict": tokenizer.state_dict(),
        "model_config": {},
        "post_train_config": vars(cfg),
        "label_info": label_info,
        "metrics": metrics,
        "history": history,
    }
    base_checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    payload["model_config"] = base_checkpoint.get("model_config", {})
    torch.save(payload, path)

    if has_lora_layers(model):
        save_lora_adapter(
            model,
            adapter_path,
            rank=int(cfg.lora_rank),
            alpha=float(cfg.lora_alpha),
            dropout=float(cfg.lora_dropout),
            target_keywords=tuple(cfg.lora_target_keywords),
        )
    return path


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = _namespace_to_config(args)
    cfg.checkpoint_path = _resolve_project_path(cfg.checkpoint_path)
    cfg.cache_path = _resolve_project_path(cfg.cache_path)
    cfg.val_cache_path = _resolve_project_path(cfg.val_cache_path) if str(cfg.val_cache_path).strip() else ""
    cfg.output_dir = _resolve_project_path(cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    cpu_threads = int(cfg.cpu_threads)
    if cpu_threads <= 0:
        detected = os.cpu_count() or 4
        cpu_threads = max(1, detected // 2)
    torch.set_num_threads(cpu_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
        except RuntimeError:
            pass

    set_global_seed(int(cfg.random_seed), deterministic=bool(cfg.deterministic))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _amp_dtype(cfg.amp_dtype)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    if device.type == "cuda" and bool(cfg.use_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    print(f"Device: {device}")
    print(f"CPU threads: intra_op={torch.get_num_threads()}")
    print(f"Checkpoint: {cfg.checkpoint_path}")
    print(f"Train cache: {cfg.cache_path}")
    if cfg.val_cache_path:
        print(f"Val cache: {cfg.val_cache_path}")

    train_cache_payload = _load_cache_payload(cfg.cache_path, mode="train", skip_signature_check=cfg.skip_cache_signature_check)
    if "encoded_indices_coarse" not in train_cache_payload or "encoded_indices_fine" not in train_cache_payload:
        print("Warning: train cache has no valid precomputed tokenizer encodings; prompts will be encoded on the fly.")
    train_real_returns = _denormalized_last_returns(train_cache_payload)

    use_dedicated_val_cache = (
        bool(cfg.val_cache_path)
        and os.path.exists(cfg.val_cache_path)
        and os.path.abspath(cfg.val_cache_path) != os.path.abspath(cfg.cache_path)
    )
    split_strategy = "chronological_tail_holdout"
    if use_dedicated_val_cache:
        val_cache_payload = _load_cache_payload(cfg.val_cache_path, mode="val", skip_signature_check=cfg.skip_cache_signature_check)
        if "encoded_indices_coarse" not in val_cache_payload or "encoded_indices_fine" not in val_cache_payload:
            print("Warning: val cache has no valid precomputed tokenizer encodings; prompts will be encoded on the fly.")
        val_real_returns = _denormalized_last_returns(val_cache_payload)
        train_indices = np.arange(len(train_real_returns), dtype=np.int64)
        val_indices = np.arange(len(val_real_returns), dtype=np.int64)
        split_strategy = "dedicated_val_cache"
    else:
        if cfg.val_cache_path and os.path.abspath(cfg.val_cache_path) != os.path.abspath(cfg.cache_path):
            print(
                "Warning: val cache path is set but unavailable; fallback to in-cache chronological holdout: "
                f"{cfg.val_cache_path}"
            )
        val_cache_payload = train_cache_payload
        val_real_returns = train_real_returns
        train_indices, val_indices = _split_cache_indices(
            len(train_real_returns),
            val_ratio=float(cfg.cache_val_ratio),
        )

    if int(cfg.sample_stride) > 1:
        train_indices = train_indices[:: int(cfg.sample_stride)]
    if int(cfg.val_sample_stride) > 1:
        val_indices = val_indices[:: int(cfg.val_sample_stride)]
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise RuntimeError(
            "Empty train/val indices after stride sampling: "
            f"train={len(train_indices)}, val={len(val_indices)}"
        )

    train_abs_returns = np.abs(train_real_returns[train_indices])
    train_abs_returns = train_abs_returns[np.isfinite(train_abs_returns)]
    if cfg.label_mode == "fixed":
        global_epsilon = max(float(cfg.min_epsilon), abs(float(cfg.fixed_epsilon)))
    elif train_abs_returns.size > 0:
        global_epsilon = max(float(cfg.min_epsilon), float(np.median(train_abs_returns)) * float(cfg.epsilon_scale))
    else:
        global_epsilon = float(cfg.min_epsilon)

    train_dataset = CachedDirectionDataset(
        train_cache_payload,
        indices=train_indices,
        mode="train",
        real_returns=train_real_returns,
        global_epsilon=global_epsilon,
        max_samples=int(cfg.max_train_samples),
        label_mode=cfg.label_mode,
        fixed_epsilon=float(cfg.fixed_epsilon),
        z_threshold=float(cfg.z_threshold),
        min_epsilon=float(cfg.min_epsilon),
        rolling_vol_window=int(cfg.rolling_vol_window),
        flat_policy=cfg.flat_policy,
        random_seed=int(cfg.random_seed),
    )
    val_dataset = CachedDirectionDataset(
        val_cache_payload,
        indices=val_indices,
        mode="val",
        real_returns=val_real_returns,
        global_epsilon=global_epsilon,
        max_samples=int(cfg.max_val_samples),
        label_mode=cfg.label_mode,
        fixed_epsilon=float(cfg.fixed_epsilon),
        z_threshold=float(cfg.z_threshold),
        min_epsilon=float(cfg.min_epsilon),
        rolling_vol_window=int(cfg.rolling_vol_window),
        flat_policy="class",
        random_seed=int(cfg.random_seed),
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"Empty DA dataset: train={len(train_dataset)}, val={len(val_dataset)}")

    demo_eval_items, demo_eval_meta = _prepare_demo_eval_items(cfg)

    label_info = {
        "label_names": list(LABEL_NAMES),
        "label_mode": cfg.label_mode,
        "global_epsilon": global_epsilon,
        "fixed_epsilon": float(cfg.fixed_epsilon),
        "z_threshold": float(cfg.z_threshold),
        "min_epsilon": float(cfg.min_epsilon),
        "rolling_vol_window": int(cfg.rolling_vol_window),
        "flat_policy": cfg.flat_policy,
        "train_cache_path": os.path.abspath(cfg.cache_path),
        "val_cache_path": os.path.abspath(cfg.val_cache_path) if cfg.val_cache_path else "",
        "split_strategy": split_strategy,
        "cache_num_samples": int(len(train_real_returns)),
        "val_cache_num_samples": int(len(val_real_returns)),
        "cache_val_ratio": float(cfg.cache_val_ratio),
        "input_alignment": "cache_full_sequence[:, :-1] -> label sign(denorm(cache_full_sequence[:, -1, log_ret]))",
        "training_method": "expo_regression",
        "expo_temperature": float(cfg.expo_temperature),
        "expo_num_candidates": int(cfg.expo_num_candidates),
        "expo_reference_weight": float(cfg.expo_reference_weight),
        "expo_score_margin": float(cfg.expo_score_margin),
        "expo_direction_bonus": float(cfg.expo_direction_bonus),
        "expo_error_weight": float(cfg.expo_error_weight),
        "expo_include_gold": bool(cfg.expo_include_gold),
        "train_class_counts": {LABEL_NAMES[i]: int(train_dataset.class_counts[i]) for i in range(3)},
        "train_loss_class_counts": {LABEL_NAMES[i]: int(train_dataset.loss_class_counts[i]) for i in range(3)},
        "val_class_counts": {LABEL_NAMES[i]: int(val_dataset.class_counts[i]) for i in range(3)},
        "demo_eval": demo_eval_meta,
    }
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=not cfg.eval_only,
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(cfg.num_workers) > 0),
        prefetch_factor=2 if int(cfg.num_workers) > 0 else None,
        worker_init_fn=_dataloader_worker_init if int(cfg.num_workers) > 0 else None,
        collate_fn=direction_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(cfg.num_workers) > 0),
        prefetch_factor=2 if int(cfg.num_workers) > 0 else None,
        worker_init_fn=_dataloader_worker_init if int(cfg.num_workers) > 0 else None,
        collate_fn=direction_collate,
    )

    model, tokenizer = load_model(
        device,
        checkpoint_path=cfg.checkpoint_path,
        strict_checkpoint_compat=False,
    )
    tokenizer.eval()
    tokenizer.requires_grad_(False)

    if cfg.eval_only:
        metrics = {
            "val": evaluate_loader(
                model,
                tokenizer,
                val_loader,
                device,
                amp_enabled,
                amp_dtype,
                cfg.eval_confidence_threshold,
                cfg.eval_margin_threshold,
            )
        }
        if demo_eval_items:
            metrics["rolling_1d_demo"] = evaluate_demo_items(
                model=model,
                tokenizer=tokenizer,
                eval_items=demo_eval_items,
                cfg=cfg,
                label_info=label_info,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return metrics

    param_groups = configure_trainable_params(model, cfg)
    summary = trainable_parameter_summary(model)
    print(f"Trainable parameters: {summary['trainable']:,}/{summary['total']:,}")

    token_policy_trainable = any(
        param.requires_grad and not name.startswith("direction_head.")
        for name, param in model.named_parameters()
    )
    if not token_policy_trainable:
        raise RuntimeError(
            "EXPO optimizes next-token policy logits, but no token-policy parameters are trainable. "
            "Set freeze_backbone=False or train_lora=True."
        )

    optimizer, optimizer_kwargs = _build_adamw_optimizer(param_groups, cfg, device)
    print(f"Optimizer: AdamW {optimizer_kwargs}")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    total_steps = int(cfg.epochs) * len(train_loader) // int(cfg.accumulation_steps)
    warmup_steps = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=float(cfg.learning_rate) * 0.01
    )
    print(f"LR schedule: LinearWarmup({warmup_steps}) + CosineAnnealing({total_steps - warmup_steps}, eta_min={float(cfg.learning_rate) * 0.01:.2e})")

    keep_auxiliary = bool(cfg.expo_keep_auxiliary) and token_policy_trainable
    effective_token_ce_weight = float(cfg.token_ce_weight) if keep_auxiliary else 0.0
    effective_kl_weight = float(cfg.kl_weight) if keep_auxiliary else 0.0
    effective_latent_weight = float(cfg.latent_weight) if keep_auxiliary else 0.0
    reference_model = copy.deepcopy(model).to(device)
    reference_model.eval()
    reference_model.requires_grad_(False)

    use_compile = bool(getattr(cfg, "torch_compile", False)) and hasattr(torch, "compile") and device.type == "cuda"
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            print("Model compiled with torch.compile (reduce-overhead, dynamic=True)")
        except Exception as exc:
            print(f"torch.compile failed, continuing without compilation: {exc}")
            use_compile = False

    history = []
    best_score = -float("inf")
    best_path = os.path.join(cfg.output_dir, cfg.save_name)
    updates = 0

    for epoch in range(int(cfg.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {
            key: torch.zeros((), dtype=torch.float64, device=device)
            for key in ("loss", "expo", "token_ce", "kl", "latent")
        }
        expo_counts = torch.zeros(len(EXPO_COUNT_KEYS), dtype=torch.long, device=device)
        expo_sums = torch.zeros(len(EXPO_SUM_KEYS), dtype=torch.float64, device=device)
        batches = 0
        pbar = tqdm(train_loader, desc=f"Post_Train_DA_EXPO epoch {epoch + 1}/{cfg.epochs}")
        for step, raw_batch in enumerate(pbar, start=1):
            batch = _move_batch(raw_batch, device)

            if "idx_coarse_full" in batch and "idx_fine_full" in batch:
                idx_coarse_full = batch["idx_coarse_full"]
                idx_fine_full = batch["idx_fine_full"]
            else:
                idx_coarse_full, idx_fine_full = _encode_prompt(tokenizer, batch["features_full"])

            t_minute = batch["time"]["minute"]
            t_day = batch["time"]["day"]
            t_month = batch["time"]["month"]
            t_year = batch["time"]["year"]
            if t_minute.size(1) == idx_coarse_full.size(1):
                t_minute = t_minute[:, :-1]
                t_day = t_day[:, :-1]
                t_month = t_month[:, :-1]
                t_year = t_year[:, :-1]

            with _autocast_context(device, amp_enabled, amp_dtype):
                logits_c, logits_f, latent_states_forward = model(
                    idx_coarse_full[:, :-1],
                    idx_fine_full[:, :-1],
                    batch["sector_ids"],
                    t_minute,
                    t_day,
                    t_month,
                    t_year,
                    last_only=True,
                )

            target_c = idx_coarse_full[:, -1]
            target_f = idx_fine_full[:, -1]
            current_last_c = logits_c[:, -1, :]
            current_last_f = logits_f[:, -1, :]

            with torch.no_grad():
                with _autocast_context(device, amp_enabled, amp_dtype):
                    ref_logits_c, ref_logits_f, _ = reference_model(
                        idx_coarse_full[:, :-1],
                        idx_fine_full[:, :-1],
                        batch["sector_ids"],
                        t_minute,
                        t_day,
                        t_month,
                        t_year,
                        last_only=True,
                    )
                ref_last_c = ref_logits_c[:, -1, :]
                ref_last_f = ref_logits_f[:, -1, :]

            (
                winner_c,
                winner_f,
                loser_c,
                loser_f,
                valid_pair,
                expo_count_stats,
                expo_score_sums,
            ) = _sample_expo_pairs(
                tokenizer=tokenizer,
                ref_logits_coarse=ref_last_c,
                ref_logits_fine=ref_last_f,
                labels=batch["labels"],
                loss_labels=batch["loss_labels"],
                real_returns=batch["real_returns"],
                epsilons=batch["epsilons"],
                prompt_means=batch["prompt_means"],
                prompt_stds=batch["prompt_stds"],
                gold_coarse=target_c,
                gold_fine=target_f,
                temperature=float(cfg.expo_temperature),
                num_candidates=int(cfg.expo_num_candidates),
                direction_bonus=float(cfg.expo_direction_bonus),
                error_weight=float(cfg.expo_error_weight),
                min_score_margin=float(cfg.expo_score_margin),
                include_gold=bool(cfg.expo_include_gold),
            )

            with _autocast_context(device, amp_enabled, amp_dtype):
                expo_loss_value, expo_prob_sums = expo_regression_loss(
                    current_last_c,
                    current_last_f,
                    ref_last_c,
                    ref_last_f,
                    winner_c,
                    winner_f,
                    loser_c,
                    loser_f,
                    valid_pair,
                    reference_weight=float(cfg.expo_reference_weight),
                )

                token_ce_loss = expo_loss_value.new_zeros(())
                kl_loss = expo_loss_value.new_zeros(())
                latent_loss = expo_loss_value.new_zeros(())

                if keep_auxiliary:
                    token_ce_loss = last_step_token_ce(
                        logits_c, logits_f,
                        idx_coarse_full[:, 1:], idx_fine_full[:, 1:],
                    )
                    latent_loss = latent_regularization_loss(latent_states_forward)

                if effective_kl_weight > 0.0:
                    kl_loss = token_kl(logits_c, logits_f, ref_logits_c, ref_logits_f)

                loss = (
                    expo_loss_value
                    + effective_token_ce_weight * token_ce_loss
                    + effective_kl_weight * kl_loss
                    + effective_latent_weight * latent_loss
                )

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaled_loss = loss / int(cfg.accumulation_steps)
            scaler.scale(scaled_loss).backward()

            should_step = step % int(cfg.accumulation_steps) == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for group in param_groups for param in group["params"]],
                    float(cfg.grad_clip),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if updates <= warmup_steps:
                    scheduler.step()
                else:
                    cosine_scheduler.step()

            batches += 1
            totals["loss"] += loss.detach().to(dtype=torch.float64)
            totals["expo"] += expo_loss_value.detach().to(dtype=torch.float64)
            totals["token_ce"] += token_ce_loss.detach().to(dtype=torch.float64)
            totals["kl"] += kl_loss.detach().to(dtype=torch.float64)
            totals["latent"] += latent_loss.detach().to(dtype=torch.float64)
            expo_counts += expo_count_stats.to(device=device)
            expo_sums[:3] += expo_score_sums.to(device=device, dtype=torch.float64)
            expo_sums[3:] += expo_prob_sums.to(device=device, dtype=torch.float64)
            if (
                step % int(cfg.progress_interval) == 0
                or step == len(train_loader)
                or (int(cfg.max_train_updates) > 0 and updates >= int(cfg.max_train_updates))
            ):
                counts_cpu = {
                    key: int(value)
                    for key, value in zip(EXPO_COUNT_KEYS, expo_counts.detach().cpu().tolist())
                }
                valid_count = max(1, counts_cpu["valid"])
                pair_count = max(1, counts_cpu["pairs"])
                pbar.set_postfix(
                    {
                        "loss": float((totals["loss"] / max(1, batches)).detach().cpu()),
                        "pair": counts_cpu["pairs"] / valid_count,
                        "target": float((expo_sums[3] / pair_count).detach().cpu()),
                        "theta": float((expo_sums[4] / pair_count).detach().cpu()),
                        "updates": updates,
                    }
                )

            if int(cfg.max_train_updates) > 0 and updates >= int(cfg.max_train_updates):
                break

        val_metrics = evaluate_loader(
            model,
            tokenizer,
            val_loader,
            device,
            amp_enabled,
            amp_dtype,
            cfg.eval_confidence_threshold,
            cfg.eval_margin_threshold,
        )
        demo_metrics = None
        if demo_eval_items:
            demo_metrics = evaluate_demo_items(
                model=model,
                tokenizer=tokenizer,
                eval_items=demo_eval_items,
                cfg=cfg,
                label_info=label_info,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

        train_record = {
            key: float((value / max(1, batches)).detach().cpu())
            for key, value in totals.items()
        }
        expo_count_values = {
            key: int(value)
            for key, value in zip(EXPO_COUNT_KEYS, expo_counts.detach().cpu().tolist())
        }
        expo_sum_values = {
            key: float(value)
            for key, value in zip(EXPO_SUM_KEYS, expo_sums.detach().cpu().tolist())
        }
        valid_count = max(1, expo_count_values["valid"])
        pair_count = max(1, expo_count_values["pairs"])
        train_record.update(
            {
                "expo_valid_samples": int(expo_count_values["valid"]),
                "expo_pairs": int(expo_count_values["pairs"]),
                "expo_pair_rate": float(expo_count_values["pairs"] / valid_count),
                "expo_skipped_rate": float(expo_count_values["skipped"] / valid_count),
                "expo_winner_direction_correct_rate": float(
                    expo_count_values["winner_direction_correct"] / pair_count
                ),
                "expo_loser_direction_correct_rate": float(
                    expo_count_values["loser_direction_correct"] / pair_count
                ),
                "expo_winner_from_gold_rate": float(expo_count_values["winner_from_gold"] / pair_count),
                "expo_avg_preference_margin": float(expo_sum_values["preference_margin"] / pair_count),
                "expo_avg_winner_abs_error": float(expo_sum_values["winner_abs_error"] / pair_count),
                "expo_avg_loser_abs_error": float(expo_sum_values["loser_abs_error"] / pair_count),
                "expo_avg_target_prob": float(expo_sum_values["target_prob"] / pair_count),
                "expo_avg_theta_prob": float(expo_sum_values["theta_prob"] / pair_count),
                "expo_avg_ref_prob": float(expo_sum_values["ref_prob"] / pair_count),
            }
        )

        record = {
            "epoch": epoch + 1,
            "updates": updates,
            "train": train_record,
            "val": val_metrics,
        }
        if demo_metrics is not None:
            record["rolling_1d_demo"] = demo_metrics

        val_score = val_metrics.get("balanced_accuracy")
        if val_score is None:
            val_score = val_metrics.get("raw_argmax_accuracy", 0.0)
        score = float(val_score)
        if demo_metrics is not None and int(demo_metrics.get("num_samples", 0)) > 0:
            demo_score = demo_metrics.get("balanced_accuracy")
            if demo_score is None:
                demo_score = demo_metrics.get("raw_argmax_accuracy", 0.0)
            score = (1.0 - float(cfg.demo_score_weight)) * float(val_score) + float(cfg.demo_score_weight) * float(
                demo_score
            )
        record["selection_score"] = float(score)
        history.append(record)
        print(json.dumps(record, indent=2, ensure_ascii=False))

        history_path = os.path.join(cfg.output_dir, "direction_expo_history.json")
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)

        if bool(cfg.save_epoch_checkpoints):
            _save_epoch_base_model(
                model,
                tokenizer,
                epoch=epoch + 1,
                output_dir=cfg.output_dir,
            )

        if float(score) > best_score:
            best_score = float(score)
            _save_checkpoint(
                best_path,
                model,
                tokenizer,
                cfg,
                label_info=label_info,
                metrics={"val": val_metrics, **({"rolling_1d_demo": demo_metrics} if demo_metrics is not None else {})},
                history=history,
            )
            print(f"Saved best Post_Train_DA_EXPO checkpoint: {best_path}")

        if int(cfg.max_train_updates) > 0 and updates >= int(cfg.max_train_updates):
            break

    print(f"Best score: {best_score:.6f}")
    print(f"Best checkpoint: {best_path}")
    return {"best_score": best_score, "best_path": best_path, "history": history}


if __name__ == "__main__":
    main()
