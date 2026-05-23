# -*- coding: utf-8 -*-
"""Batch evaluate checkpoints on cached rolling 1-step samples.

The script predicts the trailing token(s) of each cached sequence from the
preceding ground-truth tokens, decodes the predicted returns, and compares them
with the denormalized target returns stored in the cache. By default it
evaluates a random subset; pass --full to evaluate every row in every matched
cache file.
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import TrainingConfig
from evaluate_predictions import load_model


TIME_KEYS = ("minute", "day", "month", "year")


def _as_abs(path):
    return os.path.abspath(os.path.expanduser(str(path)))


def _discover_checkpoints(checkpoint_glob):
    paths = sorted(glob.glob(checkpoint_glob, recursive=True))
    result = []
    for path in paths:
        name = os.path.basename(path).lower()
        if not path.lower().endswith(".pt"):
            continue
        if "tokenizer" in name or "adapter" in name:
            continue
        result.append(_as_abs(path))
    return result


def _discover_caches(cache_globs):
    result = []
    for pattern in cache_globs:
        result.extend(glob.glob(pattern, recursive=True))
    return sorted({_as_abs(path) for path in result if path.lower().endswith(".pt")})


def _torch_load_cache(path):
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        return torch.load(path, **kwargs)


def _load_cache(path):
    payload = _torch_load_cache(path)
    required = {"features", "sector_ids", "time_features", "seq_stats"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise KeyError(f"Cache {path} is missing keys: {missing}")
    return payload


def _cache_num_samples(path):
    payload = _load_cache(path)
    try:
        return int(payload["features"].shape[0])
    finally:
        del payload


def _sample_cache_rows(cache_paths, sample_size, seed, full=False):
    counts = []
    total = 0
    for path in cache_paths:
        count = _cache_num_samples(path)
        counts.append((path, count))
        total += count
    if total <= 0:
        raise RuntimeError("No samples found in cache files.")

    if full or int(sample_size) <= 0:
        selected = {
            path: np.arange(count, dtype=np.int64)
            for path, count in counts
            if count > 0
        }
        return counts, selected

    size = min(int(sample_size), total)
    rng = np.random.default_rng(int(seed))
    global_indices = np.sort(rng.choice(total, size=size, replace=False))

    selected = defaultdict(list)
    cursor = 0
    pos = 0
    for path, count in counts:
        end = cursor + count
        while pos < len(global_indices) and cursor <= int(global_indices[pos]) < end:
            selected[path].append(int(global_indices[pos] - cursor))
            pos += 1
        cursor = end

    return counts, {path: np.asarray(indices, dtype=np.int64) for path, indices in selected.items()}


def _seq_stats_arrays(seq_stats, indices):
    means = []
    stds = []
    for idx in indices.tolist():
        item = seq_stats[int(idx)]
        means.append(np.asarray(item["mean"], dtype=np.float32))
        stds.append(np.asarray(item["std"], dtype=np.float32))
    return np.stack(means, axis=0), np.stack(stds, axis=0)


def _validate_last_steps(last_steps, seq_len):
    steps = int(last_steps)
    if steps < 1:
        raise ValueError(f"--last-steps must be >= 1, got {last_steps}")
    max_steps = int(seq_len) - 1
    if steps > max_steps:
        raise ValueError(
            f"--last-steps must be <= seq_len - 1 ({max_steps}), got {steps}"
        )
    return steps


def _denormalized_returns(features, means, stds, last_steps):
    target_norm = features[:, -int(last_steps):, 0].to(dtype=torch.float32)
    return target_norm * stds[:, 0].unsqueeze(1) + means[:, 0].unsqueeze(1)


def _has_valid_encoded_indices(payload, features):
    coarse = payload.get("encoded_indices_coarse")
    fine = payload.get("encoded_indices_fine")
    if not isinstance(coarse, torch.Tensor) or not isinstance(fine, torch.Tensor):
        return False
    if coarse.ndim != 2 or fine.ndim != 2:
        return False
    expected = tuple(features.shape[:2])
    return tuple(coarse.shape) == expected and tuple(fine.shape) == expected


def _extract_sample_batch(cache_paths, selected_by_cache, last_steps, use_cached_encodings):
    feature_parts = []
    sector_parts = []
    time_parts = {key: [] for key in TIME_KEYS}
    mean_parts = []
    std_parts = []
    coarse_parts = []
    fine_parts = []
    can_use_cached_encodings = bool(use_cached_encodings)
    source_rows = []

    for path in cache_paths:
        indices = selected_by_cache.get(path)
        if indices is None or len(indices) == 0:
            continue
        payload = _load_cache(path)
        features = payload["features"]
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features, dtype=torch.float32)
        sectors = payload["sector_ids"]
        if not isinstance(sectors, torch.Tensor):
            sectors = torch.as_tensor(sectors, dtype=torch.long)

        feature_parts.append(features[indices].to(dtype=torch.float32).contiguous())
        sector_parts.append(sectors[indices].to(dtype=torch.long).contiguous())
        for key in TIME_KEYS:
            values = payload["time_features"][key]
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values, dtype=torch.long)
            time_parts[key].append(values[indices].to(dtype=torch.long).contiguous())

        means, stds = _seq_stats_arrays(payload["seq_stats"], indices)
        mean_parts.append(torch.from_numpy(means))
        std_parts.append(torch.from_numpy(stds))

        if can_use_cached_encodings and _has_valid_encoded_indices(payload, features):
            coarse_parts.append(
                payload["encoded_indices_coarse"][indices].to(dtype=torch.long).contiguous()
            )
            fine_parts.append(
                payload["encoded_indices_fine"][indices].to(dtype=torch.long).contiguous()
            )
        else:
            can_use_cached_encodings = False
            coarse_parts.clear()
            fine_parts.clear()

        source_name = os.path.basename(path)
        source_rows.extend({"cache": source_name, "cache_index": int(idx)} for idx in indices.tolist())
        del payload

    if not feature_parts:
        raise RuntimeError("Sample selection produced no rows.")

    features = torch.cat(feature_parts, dim=0)
    sector_ids = torch.cat(sector_parts, dim=0)
    time_features = {key: torch.cat(parts, dim=0) for key, parts in time_parts.items()}
    means = torch.cat(mean_parts, dim=0).to(dtype=torch.float32)
    stds = torch.cat(std_parts, dim=0).to(dtype=torch.float32)
    last_steps = _validate_last_steps(last_steps, features.size(1))
    actual_returns = _denormalized_returns(features, means, stds, last_steps)
    result = {
        "features": features,
        "sector_ids": sector_ids,
        "time_features": time_features,
        "means": means,
        "stds": stds,
        "actual_returns": actual_returns,
        "last_steps": int(last_steps),
        "source_rows": source_rows,
    }
    if can_use_cached_encodings and len(coarse_parts) == len(feature_parts):
        result["idx_coarse"] = torch.cat(coarse_parts, dim=0)
        result["idx_fine"] = torch.cat(fine_parts, dim=0)
        result["uses_cached_encodings"] = True
    else:
        result["uses_cached_encodings"] = False
    return result


def _model_label(path):
    rel = os.path.relpath(path, os.getcwd()).replace("\\", "/")
    if rel.startswith("../"):
        rel = os.path.basename(path)
    return rel[:-3] if rel.lower().endswith(".pt") else rel


def _autocast_context(device, enabled, dtype_name):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    name = str(dtype_name).strip().lower()
    dtype = torch.bfloat16 if name in {"bf16", "bfloat16"} else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def _cached_encodings_match_tokenizer(tokenizer, sample, device, check_rows):
    if not sample.get("uses_cached_encodings"):
        return False
    rows = min(int(check_rows), int(sample["features"].size(0)))
    if rows <= 0:
        return True
    features = sample["features"][:rows].to(device=device, dtype=torch.float32, non_blocking=True)
    expected_coarse = sample["idx_coarse"][:rows]
    expected_fine = sample["idx_fine"][:rows]
    actual_coarse, actual_fine = tokenizer.encode(features)
    return (
        torch.equal(actual_coarse.cpu(), expected_coarse.cpu())
        and torch.equal(actual_fine.cpu(), expected_fine.cpu())
    )


@torch.inference_mode()
def _predict_1step_returns(
    model,
    tokenizer,
    sample,
    device,
    batch_size,
    use_amp,
    amp_dtype,
    zero_sector_ids,
    use_cached_encodings,
):
    model.eval()
    tokenizer.eval()
    features = sample["features"]
    sector_ids = sample["sector_ids"]
    time_features = sample["time_features"]
    means = sample["means"]
    stds = sample["stds"]
    last_steps = int(sample["last_steps"])
    use_cached_encodings = bool(use_cached_encodings and sample.get("uses_cached_encodings"))

    preds = []
    desc = "Predict 1-step" if last_steps == 1 else f"Predict rolling last-{last_steps}"
    for start in tqdm(range(0, features.size(0), int(batch_size)), desc=desc, leave=False):
        end = min(start + int(batch_size), features.size(0))
        batch_features = None
        if use_cached_encodings:
            idx_coarse = sample["idx_coarse"][start:end, :-1].to(
                device=device, dtype=torch.long, non_blocking=True
            )
            idx_fine = sample["idx_fine"][start:end, :-1].to(
                device=device, dtype=torch.long, non_blocking=True
            )
        else:
            batch_features = features[start:end, :-1].to(device=device, dtype=torch.float32, non_blocking=True)
            idx_coarse, idx_fine = tokenizer.encode(batch_features)

        batch_sector = sector_ids[start:end].to(device=device, dtype=torch.long, non_blocking=True)
        if zero_sector_ids:
            batch_sector = torch.zeros_like(batch_sector)
        batch_time = {
            key: values[start:end, :-1].to(device=device, dtype=torch.long, non_blocking=True)
            for key, values in time_features.items()
        }
        batch_means = means[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        batch_stds = stds[start:end].to(device=device, dtype=torch.float32, non_blocking=True)

        with _autocast_context(device, use_amp, amp_dtype):
            logits_c, logits_f, _ = model(
                idx_coarse.long(),
                idx_fine.long(),
                batch_time["minute"],
                batch_time["day"],
                batch_time["month"],
                batch_time["year"],
                last_only=(last_steps == 1),
            )

        pred_c = logits_c[:, -last_steps:, :].float().argmax(dim=-1)
        pred_f = logits_f[:, -last_steps:, :].float().argmax(dim=-1)
        decoded = tokenizer.decode(pred_c, pred_f)
        pred_norm = decoded[:, :, 0].float()
        pred_return = pred_norm * batch_stds[:, 0].unsqueeze(1) + batch_means[:, 0].unsqueeze(1)
        preds.append(pred_return.detach().cpu())

        del batch_features, batch_time, batch_means, batch_stds
        del idx_coarse, idx_fine, logits_c, logits_f, pred_c, pred_f, decoded, pred_return

    return torch.cat(preds, dim=0).numpy()


def _compute_metrics(pred_returns, actual_returns, mape_eps):
    pred = np.asarray(pred_returns, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual_returns, dtype=np.float64).reshape(-1)
    finite = np.isfinite(pred) & np.isfinite(actual)
    if finite.sum() == 0:
        return {
            "num_samples": 0,
            "mape": math.nan,
            "return_mape": math.nan,
            "da": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "pred_up_ratio": math.nan,
            "actual_up_ratio": math.nan,
        }

    pred = pred[finite]
    actual = actual[finite]
    pred_ratio = np.exp(np.clip(pred, -50.0, 50.0))
    actual_ratio = np.exp(np.clip(actual, -50.0, 50.0))
    ratio_denom = np.maximum(np.abs(actual_ratio), float(mape_eps))
    return_denom = np.maximum(np.abs(actual), float(mape_eps))
    mape = float(np.mean(np.abs((pred_ratio - actual_ratio) / ratio_denom)) * 100.0)
    return_mape = float(np.mean(np.abs((pred - actual) / return_denom)) * 100.0)
    pred_sign = np.where(pred >= 0.0, 1, -1)
    actual_sign = np.where(actual >= 0.0, 1, -1)
    err = pred - actual
    return {
        "num_samples": int(len(actual)),
        "mape": mape,
        "return_mape": return_mape,
        "da": float(np.mean(pred_sign == actual_sign) * 100.0),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pred_up_ratio": float(np.mean(pred_sign > 0) * 100.0),
        "actual_up_ratio": float(np.mean(actual_sign > 0) * 100.0),
    }


def _plot_results(rows, output_path, last_steps):
    labels = [row["model"] for row in rows]
    x = np.arange(len(labels))
    mape = np.asarray([row["mape"] for row in rows], dtype=np.float64)
    da = np.asarray([row["da"] for row in rows], dtype=np.float64)

    width = max(10.0, 0.72 * len(labels) + 4.0)
    fig, axes = plt.subplots(2, 1, figsize=(width, 8.5), sharex=True)

    axes[0].bar(x, mape, color="#4C78A8")
    axes[0].set_ylabel("MAPE (%)")
    title_prefix = "1-Step" if int(last_steps) == 1 else f"Rolling 1-Step Last {int(last_steps)}"
    axes[0].set_title(f"{title_prefix} Close-Ratio MAPE by Checkpoint")
    axes[0].grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(mape):
        axes[0].text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, da, color="#59A14F")
    axes[1].set_ylabel("DA (%)")
    axes[1].set_title(f"{title_prefix} Direction Accuracy by Checkpoint")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(da):
        axes[1].text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "model",
        "checkpoint_path",
        "last_steps",
        "num_sequences",
        "used_cached_encodings",
        "num_samples",
        "mape",
        "return_mape",
        "da",
        "mae",
        "rmse",
        "pred_up_ratio",
        "actual_up_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints on cached 1-step samples.")
    parser.add_argument("--checkpoint-glob", default=os.path.join("checkpoints", "**", "*.pt"))
    parser.add_argument(
        "--cache-glob",
        action="append",
        default=None,
        help="Cache glob. Can be passed multiple times.",
    )
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument(
        "--full",
        "--full-mode",
        action="store_true",
        help="Evaluate all rows from all matched cache files. Also enabled by --sample-size 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--last-steps",
        "--forecast-steps",
        "--prediction-days",
        dest="last_steps",
        type=int,
        default=1,
        help=(
            "Number of trailing tokens/days to score per cached sequence. "
            "Each target is a rolling one-step prediction that uses the true previous token."
        ),
    )
    parser.add_argument("--mape-eps", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=os.path.join("outputs", "checkpoint_1step_eval"))
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--amp-dtype", default="bfloat16")
    parser.add_argument("--zero-sector-ids", action="store_true")
    parser.add_argument(
        "--no-cached-encodings",
        action="store_true",
        help="Ignore encoded_indices_* tensors in cache and re-run tokenizer.encode for every checkpoint.",
    )
    parser.add_argument(
        "--encoding-check-rows",
        type=int,
        default=8,
        help="Rows used to verify cached token ids against each checkpoint tokenizer before reuse.",
    )
    args = parser.parse_args(argv)
    last_steps = int(args.last_steps)
    if last_steps < 1:
        raise ValueError(f"--last-steps must be >= 1, got {last_steps}")

    cache_globs = args.cache_glob or ["dataset_*.pt"]
    checkpoint_paths = _discover_checkpoints(args.checkpoint_glob)
    cache_paths = _discover_caches(cache_globs)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found by glob: {args.checkpoint_glob}")
    if not cache_paths:
        raise FileNotFoundError(f"No cache files found by glob(s): {cache_globs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(TrainingConfig, "use_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(getattr(TrainingConfig, "use_tf32", True))
        torch.set_float32_matmul_precision("high")

    full_mode = bool(args.full or int(args.sample_size) <= 0)
    counts, selected_by_cache = _sample_cache_rows(
        cache_paths,
        args.sample_size,
        args.seed,
        full=full_mode,
    )
    sample = _extract_sample_batch(
        cache_paths,
        selected_by_cache,
        last_steps=last_steps,
        use_cached_encodings=not bool(args.no_cached_encodings),
    )
    actual_returns = sample["actual_returns"].numpy()

    output_dir = _as_abs(args.output_dir)
    rows = []
    for checkpoint_path in tqdm(checkpoint_paths, desc="Evaluate checkpoints"):
        model, tokenizer = load_model(
            device=device,
            checkpoint_path=checkpoint_path,
            strict_checkpoint_compat=False,
        )
        use_cached_encodings = _cached_encodings_match_tokenizer(
            tokenizer=tokenizer,
            sample=sample,
            device=device,
            check_rows=args.encoding_check_rows,
        )
        pred_returns = _predict_1step_returns(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            device=device,
            batch_size=max(1, int(args.batch_size)),
            use_amp=bool(args.use_amp),
            amp_dtype=args.amp_dtype,
            zero_sector_ids=bool(args.zero_sector_ids),
            use_cached_encodings=use_cached_encodings,
        )
        metrics = _compute_metrics(pred_returns, actual_returns, args.mape_eps)
        rows.append({
            "model": _model_label(checkpoint_path),
            "checkpoint_path": checkpoint_path,
            "last_steps": int(last_steps),
            "num_sequences": int(sample["features"].size(0)),
            "used_cached_encodings": bool(use_cached_encodings),
            **metrics,
        })
        del model, tokenizer, pred_returns
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_path = os.path.join(output_dir, "checkpoint_1step_mape_da.png")
    csv_path = os.path.join(output_dir, "checkpoint_1step_metrics.csv")
    json_path = os.path.join(output_dir, "checkpoint_1step_metrics.json")
    _plot_results(rows, plot_path, last_steps=last_steps)
    _write_csv(rows, csv_path)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "sample_mode": "full" if full_mode else "random",
        "full_mode": bool(full_mode),
        "sample_size_arg": int(args.sample_size),
        "sample_size_requested": int(sample["features"].size(0)) if full_mode else int(args.sample_size),
        "sample_size_used": int(sample["features"].size(0)),
        "prediction_mode": "rolling_one_step_teacher_forced",
        "last_steps": int(last_steps),
        "num_predictions_used": int(sample["features"].size(0)) * int(last_steps),
        "seed": int(args.seed),
        "mape_eps": float(args.mape_eps),
        "metric_notes": {
            "mape": "MAPE on rolling one-step close ratio, computed from exp(pred_log_return) vs exp(actual_log_return).",
            "return_mape": "MAPE directly on denormalized log returns, included only as a diagnostic because near-zero returns can dominate it.",
            "da": "Up/down direction accuracy on denormalized predicted and actual log returns.",
            "rolling_one_step_teacher_forced": "For last_steps > 1, each trailing target is predicted from ground-truth prior tokens in the same cache sequence; predicted tokens are not fed back.",
        },
        "zero_sector_ids": bool(args.zero_sector_ids),
        "cached_encodings_available": bool(sample.get("uses_cached_encodings")),
        "cached_encodings_disabled": bool(args.no_cached_encodings),
        "encoding_check_rows": int(args.encoding_check_rows),
        "cache_counts": [{"cache_path": path, "num_samples": count} for path, count in counts],
        "selected_counts": {
            os.path.basename(path): int(len(indices)) for path, indices in selected_by_cache.items()
        },
        "checkpoints": checkpoint_paths,
        "metrics": rows,
        "plot_path": plot_path,
        "csv_path": csv_path,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
