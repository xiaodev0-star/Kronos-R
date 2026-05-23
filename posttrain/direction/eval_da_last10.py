# -*- coding: utf-8 -*-
"""Evaluate a Post_Train_DA EXPO checkpoint on the last 10 rolling demo days."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import PostTrainDAConfig, TrainingConfig
from data_processor import AShareDataset
from evaluate_predictions import build_rolling_1d_eval_items, load_model
from model.lora import inject_lora, load_lora_state_dict
from posttrain.direction.train_da import (
    LABEL_DOWN,
    LABEL_FLAT,
    LABEL_NAMES,
    LABEL_UP,
    _amp_dtype,
    _autocast_context,
    _direction_label,
    _token_return_direction,
    compute_metrics,
)


def _load_direction_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    post_cfg = checkpoint.get("post_train_config", {})
    base_checkpoint = post_cfg.get("checkpoint_path") or TrainingConfig.base_model_path
    stage = checkpoint.get("stage", "")

    if checkpoint.get("model_state_dict") is not None:
        model, tokenizer = load_model(
            device,
            checkpoint_path=path,
            strict_checkpoint_compat=False,
        )
    elif "lora" in str(stage).lower() or checkpoint.get("lora_state_dict"):
        model, tokenizer = load_model(
            device,
            checkpoint_path=base_checkpoint,
            strict_checkpoint_compat=False,
        )
        if checkpoint.get("lora_state_dict"):
            inject_lora(
                model,
                rank=int(post_cfg.get("lora_rank", 8)),
                alpha=float(post_cfg.get("lora_alpha", 16.0)),
                dropout=float(post_cfg.get("lora_dropout", 0.05)),
                target_keywords=tuple(post_cfg.get("lora_target_keywords", ())),
                freeze_base=False,
            )
            load_lora_state_dict(model, checkpoint["lora_state_dict"], strict=False)
        direction_state = checkpoint.get("direction_head_state_dict")
        if direction_state:
            model.direction_head.load_state_dict(direction_state, strict=False)
    else:
        model, tokenizer = load_model(
            device,
            checkpoint_path=base_checkpoint,
            strict_checkpoint_compat=False,
        )

    model.eval()
    tokenizer.eval()
    return model, tokenizer, checkpoint


def _item_return_and_label(item, epsilon):
    last_close = max(float(item["hist_closes"][-1]), 1e-8)
    next_close = max(float(item["actual_future"][0]), 1e-8)
    real_return = math.log(next_close / last_close)
    return real_return, _direction_label(real_return, float(epsilon))


def _select_last_n_days(eval_items, days):
    dates = sorted({item["future_dates"][0] for item in eval_items})
    selected = set(dates[-max(1, int(days)):])
    return [item for item in eval_items if item["future_dates"][0] in selected], sorted(selected)


@torch.no_grad()
def _predict_direction_batches(model, tokenizer, items, device, batch_size, amp_enabled, amp_dtype):
    probs = []
    for start in tqdm(range(0, len(items), max(1, int(batch_size))), desc="Eval Post_Train_DA_EXPO"):
        batch_items = items[start : start + max(1, int(batch_size))]
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
        pred_c = logits_c[:, -1, :].float().argmax(dim=-1)
        pred_f = logits_f[:, -1, :].float().argmax(dim=-1)
        direction = _token_return_direction(
            tokenizer, pred_c.unsqueeze(1), pred_f.unsqueeze(1), means, stds,
        ).squeeze(1)
        confidence = F.softmax(logits_c[:, -1, :].float(), dim=-1).max(dim=-1).values

        B = logits_c.size(0)
        conf = confidence.cpu().to(torch.float64)
        d = direction.clamp(-1, 1).long().cpu()
        pred_class = d + 1
        other = ((1.0 - conf) * 0.5).unsqueeze(-1)
        batch_probs = other.expand(B, 3).clone()
        batch_probs[torch.arange(B), pred_class] = conf
        probs.append(batch_probs.numpy())
    return np.concatenate(probs, axis=0) if probs else np.empty((0, 3), dtype=np.float32)


def _plot_daily_da(daily_rows, output_path):
    dates = [row["date"] for row in daily_rows]
    x = np.arange(len(dates))
    da = np.asarray([row["direction_accuracy"] * 100.0 for row in daily_rows], dtype=np.float64)
    bacc = np.asarray([row["balanced_accuracy"] * 100.0 for row in daily_rows], dtype=np.float64)
    coverage = np.asarray([row["coverage"] * 100.0 for row in daily_rows], dtype=np.float64)
    threshold_da = np.asarray(
        [row["threshold_direction_accuracy"] * 100.0 for row in daily_rows],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, da, marker="o", linewidth=2.2, label="DA hard up/down")
    ax.plot(x, bacc, marker="s", linewidth=1.8, label="Balanced accuracy")
    ax.plot(x, threshold_da, marker="^", linewidth=1.6, label="Threshold DA")
    ax.bar(x, coverage, alpha=0.16, label="Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=35, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    ax.set_title("Post_Train_DA Last 10 Rolling Days")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Post_Train_DA on last rolling days")
    parser.add_argument("--checkpoint-path", default=os.path.join(PostTrainDAConfig.output_dir, PostTrainDAConfig.save_name))
    parser.add_argument("--output-dir", default=os.path.join("outputs", "post_train_da"))
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confidence-threshold", type=float, default=PostTrainDAConfig.eval_confidence_threshold)
    parser.add_argument("--margin-threshold", type=float, default=PostTrainDAConfig.eval_margin_threshold)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--amp-dtype", default=PostTrainDAConfig.amp_dtype)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _amp_dtype(args.amp_dtype)
    amp_enabled = bool(args.use_amp and device.type == "cuda")

    model, tokenizer, checkpoint = _load_direction_checkpoint(args.checkpoint_path, device)
    label_info = checkpoint.get("label_info", {})
    epsilon = float(label_info.get("global_epsilon", PostTrainDAConfig.min_epsilon))

    demo_dataset = AShareDataset(mode="demo")
    eval_items, skip_reasons = build_rolling_1d_eval_items(demo_dataset)
    eval_items, selected_dates = _select_last_n_days(eval_items, args.days)
    if not eval_items:
        raise RuntimeError(f"No rolling_1d eval items. skip_reasons={skip_reasons}")

    probs = _predict_direction_batches(
        model,
        tokenizer,
        eval_items,
        device,
        batch_size=args.batch_size,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    labels = []
    returns = []
    by_date = defaultdict(lambda: {"idx": [], "labels": [], "returns": []})
    for idx, item in enumerate(eval_items):
        real_return, label = _item_return_and_label(item, epsilon)
        date_key = item["future_dates"][0].date().isoformat()
        labels.append(label)
        returns.append(real_return)
        by_date[date_key]["idx"].append(idx)
        by_date[date_key]["labels"].append(label)
        by_date[date_key]["returns"].append(real_return)

    labels = np.asarray(labels, dtype=np.int64)
    returns = np.asarray(returns, dtype=np.float64)
    overall = compute_metrics(
        probs,
        labels,
        returns,
        confidence_threshold=float(args.confidence_threshold),
        margin_threshold=float(args.margin_threshold),
    )

    daily_rows = []
    for date_key in sorted(by_date.keys()):
        idx = np.asarray(by_date[date_key]["idx"], dtype=np.int64)
        row_metrics = compute_metrics(
            probs[idx],
            np.asarray(by_date[date_key]["labels"], dtype=np.int64),
            np.asarray(by_date[date_key]["returns"], dtype=np.float64),
            confidence_threshold=float(args.confidence_threshold),
            margin_threshold=float(args.margin_threshold),
        )
        daily_rows.append({"date": date_key, **row_metrics})

    plot_path = os.path.join(args.output_dir, "post_train_da_last10_da.png")
    csv_path = os.path.join(args.output_dir, "post_train_da_last10_daily_metrics.csv")
    json_path = os.path.join(args.output_dir, "post_train_da_last10_metrics.json")
    _plot_daily_da(daily_rows, plot_path)

    os.makedirs(args.output_dir, exist_ok=True)
    fieldnames = ["date"] + [key for key in daily_rows[0].keys() if key != "date" and not isinstance(daily_rows[0][key], dict)]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(daily_rows)

    payload = {
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "epsilon": epsilon,
        "selected_dates": [date.date().isoformat() for date in selected_dates],
        "num_items": int(len(eval_items)),
        "skip_reasons": skip_reasons,
        "overall": overall,
        "daily": daily_rows,
        "plot_path": os.path.abspath(plot_path),
        "csv_path": os.path.abspath(csv_path),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
