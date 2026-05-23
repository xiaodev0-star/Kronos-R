# -*- coding: utf-8 -*-
"""Train-only numeric calibration for strict rollout predictions.

The calibrator is fitted on train-window autoregressive predictions only and
then applied to val predictions.  It does not touch demo data and it does not
feed future true tokens into the rollout context.
"""

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
from posttrain.rollout.data import RolloutWindowDataset, resolve_project_path, rollout_collate
from posttrain.rollout.eval_rollout import _discover_checkpoints, _model_label
from posttrain.rollout.train_rollout import (
    _amp_dtype,
    _as_bool,
    _cfg_to_dict,
    _namespace_from_args,
    compute_rollout_metrics,
    predict_autoregressive_returns,
)
from reproducibility import set_global_seed


def _build_parser():
    parser = argparse.ArgumentParser(description="Fit train-only rollout path-MAPE calibrators")
    parser.add_argument("--checkpoint", action="append", default=None)
    parser.add_argument("--include-base", type=_as_bool, default=True)
    parser.add_argument("--checkpoint-path", default=PostTrainRolloutConfig.checkpoint_path)
    parser.add_argument("--output-dir", default=os.path.join("outputs", "post_train_rollout_calibrated"))
    parser.add_argument("--prefix-len", type=int, default=PostTrainRolloutConfig.prefix_len)
    parser.add_argument("--horizon", type=int, default=PostTrainRolloutConfig.horizon)
    parser.add_argument("--stride-ratio", type=float, default=PostTrainRolloutConfig.stride_ratio)
    parser.add_argument("--cache-dir", default=PostTrainRolloutConfig.cache_dir)
    parser.add_argument("--cache-rebuild", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=PostTrainRolloutConfig.max_stocks)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=PostTrainRolloutConfig.eval_batch_size)
    parser.add_argument("--use-amp", type=_as_bool, default=PostTrainRolloutConfig.use_amp)
    parser.add_argument("--amp-dtype", default=PostTrainRolloutConfig.amp_dtype)
    parser.add_argument("--use-tf32", type=_as_bool, default=PostTrainRolloutConfig.use_tf32)
    parser.add_argument("--zero-sector-ids", type=_as_bool, default=PostTrainRolloutConfig.zero_sector_ids)
    parser.add_argument("--mape-eps", type=float, default=PostTrainRolloutConfig.mape_eps)
    parser.add_argument("--seed", type=int, default=PostTrainRolloutConfig.random_seed)
    return parser


def _cfg_from_args(args):
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


def _apply_calibrator(pred, calibrator):
    pred = np.asarray(pred, dtype=np.float64)
    kind = calibrator["kind"]
    if kind == "identity":
        return pred.copy()
    if kind in {"mean_bias", "median_bias"}:
        bias = np.asarray(calibrator["bias"], dtype=np.float64).reshape(1, -1)
        return pred + bias
    if kind == "affine":
        slope = np.asarray(calibrator["slope"], dtype=np.float64).reshape(1, -1)
        intercept = np.asarray(calibrator["intercept"], dtype=np.float64).reshape(1, -1)
        return pred * slope + intercept
    raise ValueError(f"Unknown calibrator kind: {kind}")


def _fit_affine(pred, actual):
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    slopes = []
    intercepts = []
    for step in range(pred.shape[1]):
        x = pred[:, step]
        y = actual[:, step]
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 4 or float(np.var(x[finite])) < 1e-10:
            slopes.append(1.0)
            intercepts.append(float(np.mean(y[finite] - x[finite])) if finite.any() else 0.0)
            continue
        design = np.stack([x[finite], np.ones(int(finite.sum()))], axis=1)
        slope, intercept = np.linalg.lstsq(design, y[finite], rcond=None)[0]
        slopes.append(float(np.clip(slope, -3.0, 3.0)))
        intercepts.append(float(np.clip(intercept, -0.05, 0.05)))
    return {"kind": "affine", "slope": slopes, "intercept": intercepts}


def _fit_calibrators(train_pred, train_actual, mape_eps):
    residual = np.asarray(train_actual, dtype=np.float64) - np.asarray(train_pred, dtype=np.float64)
    candidates = [
        {"kind": "identity"},
        {"kind": "mean_bias", "bias": np.mean(residual, axis=0).astype(float).tolist()},
        {"kind": "median_bias", "bias": np.median(residual, axis=0).astype(float).tolist()},
        _fit_affine(train_pred, train_actual),
    ]
    scored = []
    for cal in candidates:
        calibrated = _apply_calibrator(train_pred, cal)
        metrics = compute_rollout_metrics(calibrated, train_actual, mape_eps=mape_eps)
        row = dict(cal)
        row["train_mape"] = float(metrics["mape"])
        row["train_path_mape"] = float(metrics["path_mape"])
        row["train_mae"] = float(metrics["mae"])
        row["train_rmse"] = float(metrics["rmse"])
        scored.append(row)
    scored.sort(key=lambda item: (item["train_path_mape"], item["train_mape"], item["train_mae"]))
    return scored[0], scored


def _write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "model",
        "checkpoint_path",
        "variant",
        "selected_calibrator",
        "train_mape",
        "train_path_mape",
        "num_sequences",
        "num_samples",
        "mape",
        "path_mape",
        "return_mape",
        "path_return_mape",
        "mae",
        "path_mae",
        "rmse",
        "path_rmse",
        "da",
        "pred_up_ratio",
        "actual_up_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = _cfg_from_args(args)
    set_global_seed(int(args.seed), deterministic=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.use_tf32)
        torch.set_float32_matmul_precision("high")
    amp_dtype = _amp_dtype(cfg.amp_dtype)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda" and amp_dtype is not None)

    train_dataset = RolloutWindowDataset(
        "train",
        cfg=cfg,
        max_samples=int(cfg.max_train_samples),
        seed=int(args.seed),
    )
    val_dataset = RolloutWindowDataset(
        "val",
        cfg=cfg,
        max_samples=int(cfg.max_val_samples),
        seed=int(args.seed) + 17,
    )
    loader_kwargs = {
        "batch_size": max(1, int(cfg.eval_batch_size)),
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": rollout_collate,
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    checkpoints = []
    if bool(args.include_base):
        checkpoints.append(resolve_project_path(TrainingConfig.base_model_path))
    checkpoints.extend(_discover_checkpoints(args.checkpoint or []))
    if not checkpoints:
        raise RuntimeError("No checkpoints to calibrate.")

    rows = []
    details = []
    for checkpoint_path in tqdm(checkpoints, desc="Calibrate rollout checkpoints"):
        model, tokenizer = load_model(
            device=device,
            checkpoint_path=checkpoint_path,
            strict_checkpoint_compat=False,
        )
        label = _model_label(checkpoint_path)
        train_pred, train_actual = predict_autoregressive_returns(
            model=model,
            tokenizer=tokenizer,
            loader=train_loader,
            cfg=cfg,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        val_pred, val_actual = predict_autoregressive_returns(
            model=model,
            tokenizer=tokenizer,
            loader=val_loader,
            cfg=cfg,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        raw_metrics = compute_rollout_metrics(val_pred, val_actual, mape_eps=float(cfg.mape_eps))
        selected, candidates = _fit_calibrators(train_pred, train_actual, mape_eps=float(cfg.mape_eps))
        calibrated_pred = _apply_calibrator(val_pred, selected)
        calibrated_metrics = compute_rollout_metrics(calibrated_pred, val_actual, mape_eps=float(cfg.mape_eps))

        for variant, metrics, calibrator in (
            ("raw", raw_metrics, {"kind": "identity", "train_mape": None}),
            ("train_calibrated", calibrated_metrics, selected),
        ):
            row = {
                "model": label,
                "checkpoint_path": checkpoint_path,
                "variant": variant,
                "selected_calibrator": calibrator["kind"],
                "train_mape": calibrator.get("train_mape"),
                "train_path_mape": calibrator.get("train_path_mape"),
                **metrics,
            }
            rows.append(row)
        details.append({
            "model": label,
            "checkpoint_path": checkpoint_path,
            "selected_calibrator": selected,
            "candidate_calibrators": candidates,
        })

    os.makedirs(cfg.output_dir, exist_ok=True)
    csv_path = os.path.join(cfg.output_dir, "rollout_calibrated_eval_val.csv")
    _write_csv(rows, csv_path)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "config": _cfg_to_dict(cfg),
        "num_train_sequences": len(train_dataset),
        "num_val_sequences": len(val_dataset),
        "data_policy": {
            "calibration_fit": "train rollout predictions only; calibrator selected by cumulative path MAPE",
            "validation": "val rollout predictions only",
            "demo_usage": "not used",
            "rollout": "pure autoregressive prediction; no future true token is fed back",
        },
        "rows": rows,
        "details": details,
        "csv_path": csv_path,
    }
    json_path = os.path.join(cfg.output_dir, "rollout_calibrated_eval_val.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
