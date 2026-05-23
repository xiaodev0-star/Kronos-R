# -*- coding: utf-8 -*-
"""Evaluate checkpoints on strict train/val 10-step autoregressive rollout."""

import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PostTrainRolloutConfig, TrainingConfig
from evaluate_predictions import load_model
from posttrain.rollout.data import RolloutWindowDataset, resolve_project_path, rollout_cache_path, rollout_collate
from posttrain.rollout.train_rollout import (
    _amp_dtype,
    _as_bool,
    _namespace_from_args,
    _cfg_to_dict,
    compute_rollout_metrics,
    predict_autoregressive_returns,
)


def _model_label(path):
    rel = os.path.relpath(path, os.getcwd()).replace("\\", "/")
    if rel.startswith("../"):
        rel = os.path.basename(path)
    return rel[:-3] if rel.lower().endswith(".pt") else rel


def _discover_checkpoints(paths):
    result = []
    for item in paths:
        text = str(item).strip()
        if not text:
            continue
        if any(ch in text for ch in "*?[]"):
            import glob

            result.extend(glob.glob(text, recursive=True))
        else:
            result.append(text)
    return [resolve_project_path(path) for path in result if str(path).lower().endswith(".pt")]


def _write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "model",
        "checkpoint_path",
        "mode",
        "num_sequences",
        "horizon",
        "num_samples",
        "mape",
        "path_mape",
        "return_mape",
        "path_return_mape",
        "da",
        "mae",
        "path_mae",
        "rmse",
        "path_rmse",
        "pred_up_ratio",
        "actual_up_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(text):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))[:120]


def _write_prediction_diff_csv(pred, actual, dataset, model_label, output_dir, mape_eps, mode):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"prediction_diff_{_safe_name(model_label)}.csv")
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    fields = [
        "model",
        "sequence_index",
        "sample_id",
        "symbol",
        "target_date",
        "step",
        "pred_return",
        "actual_return",
        "abs_error",
        "squared_error",
        "close_ratio_mape",
        "cumulative_pred_close_ratio",
        "cumulative_actual_close_ratio",
        "path_mape",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seq_pos, source_idx in enumerate(dataset.indices.tolist()):
            symbol = dataset.symbols[source_idx] if source_idx < len(dataset.symbols) else ""
            dates = dataset.target_dates[source_idx] if source_idx < len(dataset.target_dates) else []
            pred_cum = 0.0
            actual_cum = 0.0
            for step in range(pred.shape[1]):
                p = float(pred[seq_pos, step])
                a = float(actual[seq_pos, step])
                err = p - a
                pred_cum += p
                actual_cum += a
                p_ratio = float(np.exp(np.clip(p, -50.0, 50.0)))
                a_ratio = float(np.exp(np.clip(a, -50.0, 50.0)))
                ratio_mape = abs(p_ratio - a_ratio) / max(abs(a_ratio), float(mape_eps)) * 100.0
                pred_path_ratio = float(np.exp(np.clip(pred_cum, -50.0, 50.0)))
                actual_path_ratio = float(np.exp(np.clip(actual_cum, -50.0, 50.0)))
                path_mape = (
                    abs(pred_path_ratio - actual_path_ratio)
                    / max(abs(actual_path_ratio), float(mape_eps))
                    * 100.0
                )
                writer.writerow({
                    "model": model_label,
                    "sequence_index": int(seq_pos),
                    "sample_id": f"{mode}:{int(source_idx)}",
                    "symbol": symbol,
                    "target_date": dates[step] if step < len(dates) else "",
                    "step": int(step + 1),
                    "pred_return": p,
                    "actual_return": a,
                    "abs_error": abs(err),
                    "squared_error": err * err,
                    "close_ratio_mape": ratio_mape,
                    "cumulative_pred_close_ratio": pred_path_ratio,
                    "cumulative_actual_close_ratio": actual_path_ratio,
                    "path_mape": path_mape,
                })
    return path


def build_parser():
    parser = argparse.ArgumentParser(description="Strict rollout autoregressive evaluation")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint path or glob. Can be passed multiple times.",
    )
    parser.add_argument("--include-base", type=_as_bool, default=True)
    parser.add_argument("--mode", choices=["train", "val", "demo"], default="val")
    parser.add_argument("--checkpoint-path", default=PostTrainRolloutConfig.checkpoint_path)
    parser.add_argument("--output-dir", default=os.path.join("outputs", "post_train_rollout"))
    parser.add_argument("--prefix-len", type=int, default=PostTrainRolloutConfig.prefix_len)
    parser.add_argument("--horizon", type=int, default=PostTrainRolloutConfig.horizon)
    parser.add_argument("--stride-ratio", type=float, default=PostTrainRolloutConfig.stride_ratio)
    parser.add_argument("--cache-dir", default=PostTrainRolloutConfig.cache_dir)
    parser.add_argument("--cache-rebuild", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=PostTrainRolloutConfig.max_stocks)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=PostTrainRolloutConfig.max_val_samples)
    parser.add_argument("--batch-size", type=int, default=PostTrainRolloutConfig.eval_batch_size)
    parser.add_argument("--use-amp", type=_as_bool, default=PostTrainRolloutConfig.use_amp)
    parser.add_argument("--amp-dtype", default=PostTrainRolloutConfig.amp_dtype)
    parser.add_argument("--use-tf32", type=_as_bool, default=PostTrainRolloutConfig.use_tf32)
    parser.add_argument("--zero-sector-ids", type=_as_bool, default=PostTrainRolloutConfig.zero_sector_ids)
    parser.add_argument("--mape-eps", type=float, default=PostTrainRolloutConfig.mape_eps)
    parser.add_argument("--seed", type=int, default=PostTrainRolloutConfig.random_seed)
    # ── Experiment S: Temperature Annealing ──
    parser.add_argument("--sample-temp-start", type=float, default=0.0)
    parser.add_argument("--sample-temp-end", type=float, default=0.0)
    parser.add_argument("--sample-temp-steps", type=int, default=0)
    # ── Output CSV ──
    parser.add_argument("--output-csv", default="")
    return parser


def _eval_cfg_from_args(args):
    ns = argparse.Namespace(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        save_name=PostTrainRolloutConfig.save_name,
        save_epoch_checkpoints=False,
        prefix_len=args.prefix_len,
        horizon=args.horizon,
        stride_ratio=args.stride_ratio,
        cache_dir=args.cache_dir,
        cache_rebuild=args.cache_rebuild,
        max_stocks=args.max_stocks,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        epochs=1,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        accumulation_steps=1,
        num_workers=0,
        lr=PostTrainRolloutConfig.learning_rate,
        weight_decay=PostTrainRolloutConfig.weight_decay,
        grad_clip=PostTrainRolloutConfig.grad_clip,
        max_train_updates=0,
        progress_interval=20,
        rollout_ratio_start=PostTrainRolloutConfig.rollout_ratio_start,
        rollout_ratio_end=PostTrainRolloutConfig.rollout_ratio_end,
        anchor_weight=PostTrainRolloutConfig.anchor_weight,
        kl_weight=PostTrainRolloutConfig.kl_weight,
        numeric_mape_weight=PostTrainRolloutConfig.numeric_mape_weight,
        numeric_top_k=PostTrainRolloutConfig.numeric_top_k,
        numeric_soft_ce_weight=PostTrainRolloutConfig.numeric_soft_ce_weight,
        numeric_soft_ce_top_k=PostTrainRolloutConfig.numeric_soft_ce_top_k,
        numeric_soft_ce_temp=PostTrainRolloutConfig.numeric_soft_ce_temp,
        step_weight_gamma=PostTrainRolloutConfig.step_weight_gamma,
        use_sampling=False,
        sampling_temperature=1.0,
        freeze_backbone=False,
        trainable_scope=PostTrainRolloutConfig.trainable_scope,
        use_gradient_checkpointing=False,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        use_tf32=args.use_tf32,
        zero_sector_ids=args.zero_sector_ids,
        mape_eps=args.mape_eps,
        deterministic=False,
        seed=args.seed,
        eval_only=True,
    )
    return _namespace_from_args(ns)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _eval_cfg_from_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.use_tf32)
        torch.set_float32_matmul_precision("high")
    amp_dtype = _amp_dtype(cfg.amp_dtype)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda" and amp_dtype is not None)

    max_samples = int(cfg.max_train_samples if args.mode == "train" else cfg.max_val_samples)
    if args.mode == "demo":
        max_samples = 0  # demo: use all windows
    dataset = RolloutWindowDataset(
        args.mode,
        cfg=cfg,
        max_samples=max_samples,
        seed=int(args.seed) + (0 if args.mode == "train" else 17),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=rollout_collate,
    )

    checkpoint_paths = []
    if bool(args.include_base):
        checkpoint_paths.append(resolve_project_path(args.checkpoint_path or TrainingConfig.base_model_path))
    if args.checkpoint:
        checkpoint_paths.extend(_discover_checkpoints(args.checkpoint))
    seen = set()
    checkpoint_paths = [path for path in checkpoint_paths if not (path in seen or seen.add(path))]
    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoint paths supplied for rollout evaluation.")

    output_dir = resolve_project_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    details = []
    for path in tqdm(checkpoint_paths, desc="Eval rollout checkpoints"):
        model, tokenizer = load_model(
            device=device,
            checkpoint_path=path,
            strict_checkpoint_compat=False,
        )
        pred, actual = predict_autoregressive_returns(
            model=model,
            tokenizer=tokenizer,
            loader=loader,
            cfg=cfg,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            sample_temp_start=float(args.sample_temp_start),
            sample_temp_end=float(args.sample_temp_end),
            sample_temp_steps=int(args.sample_temp_steps),
        )
        metrics = compute_rollout_metrics(pred, actual, mape_eps=float(cfg.mape_eps))
        label = _model_label(path)
        diff_csv = _write_prediction_diff_csv(
            pred=pred,
            actual=actual,
            dataset=dataset,
            model_label=label,
            output_dir=output_dir,
            mape_eps=float(cfg.mape_eps),
            mode=args.mode,
        )
        row = {
            "model": label,
            "checkpoint_path": path,
            "mode": args.mode,
            **{key: value for key, value in metrics.items() if key != "per_step"},
        }
        rows.append(row)
        details.append({**row, "per_step": metrics.get("per_step", []), "prediction_diff_csv": diff_csv})
        del model, tokenizer, pred, actual
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json_path = os.path.join(output_dir, f"rollout_eval_{args.mode}.json")
    csv_path = os.path.join(output_dir, f"rollout_eval_{args.mode}.csv")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "mode": args.mode,
        "config": _cfg_to_dict(cfg),
        "cache_path": rollout_cache_path(args.mode, cfg),
        "num_sequences": int(len(dataset)),
        "data_policy": {
            "demo_usage": "not used; evaluator choices are train/val only",
            "rollout": "pure autoregressive; only the 1023-token prefix is real",
        },
        "metrics": details,
        "csv_path": csv_path,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    _write_csv(rows, csv_path)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
