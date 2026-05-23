# -*- coding: utf-8 -*-
"""Conformal calibration for prediction intervals.

Implements split-conformal quantile correction:
  1. Train quantile head on training set only.
  2. On calibration set, compute nonconformity scores:
     s_i = max(q_low_i - y_i, y_i - q_high_i, 0)
  3. Conformal offset: δ_C = quantile(s, ceil((n+1)(1-α))/n)
  4. Evaluate widened intervals [q_low - δ_C, q_high + δ_C] on eval set.
"""

import numpy as np
from typing import Dict, List, Tuple


def nonconformity_scores(pred_lower, pred_upper, actual):
    """Compute nonconformity scores. Handles both torch.Tensor and np.ndarray."""
    import torch as _torch
    if isinstance(pred_lower, _torch.Tensor):
        lower = pred_lower.detach().cpu()
        upper = pred_upper.detach().cpu()
        y = actual.detach().cpu()
        scores = _torch.maximum(lower - y, _torch.maximum(y - upper, _torch.tensor(0.0)))
        return scores.numpy()
    lower = np.asarray(pred_lower, dtype=np.float64).copy()
    upper = np.asarray(pred_upper, dtype=np.float64).copy()
    y = np.asarray(actual, dtype=np.float64).copy()
    return np.maximum(np.maximum(lower - y, 0.0), np.maximum(y - upper, 0.0))


def conformal_offset(
    scores: np.ndarray, alpha: float, per_step: bool = True,
) -> np.ndarray:
    """Compute conformal correction offset.

    δ = quantile(scores, ceil((n+1)(1-α))/n)

    Parameters
    ----------
    scores : ndarray [N, H] or [N]
    alpha : float  1 - confidence_level
    per_step : bool  If True, compute per-step offsets [H]; else global scalar.

    Returns
    -------
    offset : ndarray [H] or scalar
    """
    scores = np.asarray(scores, dtype=np.float64)
    if per_step and scores.ndim == 2:
        N, H = scores.shape
        offsets = np.zeros(H, dtype=np.float64)
        for h in range(H):
            offsets[h] = _compute_single_offset(scores[:, h], alpha)
        return offsets
    else:
        return _compute_single_offset(scores.ravel(), alpha)


def _compute_single_offset(scores_1d: np.ndarray, alpha: float) -> float:
    """Compute single offset value."""
    n = len(scores_1d)
    if n == 0:
        return 0.0
    # Finite scores only
    finite = np.isfinite(scores_1d)
    if not finite.any():
        return 0.0
    vals = np.sort(scores_1d[finite])
    # q = ceil((n+1)(1-α)) / n, clamped to [0, 1]
    n_valid = len(vals)
    q_idx = min(int(np.ceil((n_valid + 1) * (1.0 - alpha))), n_valid)
    q_idx = max(q_idx, 0)
    if q_idx >= n_valid:
        return float(vals[-1])
    if q_idx == 0:
        return 0.0
    return float(vals[q_idx - 1])


def apply_conformal_correction(
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
    offset: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Widen intervals by conformal offset.

    L' = L - δ,  U' = U + δ
    """
    lower = np.asarray(pred_lower, dtype=np.float64) - np.asarray(offset, dtype=np.float64)
    upper = np.asarray(pred_upper, dtype=np.float64) + np.asarray(offset, dtype=np.float64)
    return lower, upper


def calibrate_and_evaluate(
    pred_lower_calib: np.ndarray,
    pred_upper_calib: np.ndarray,
    actual_calib: np.ndarray,
    pred_lower_eval: np.ndarray,
    pred_upper_eval: np.ndarray,
    actual_eval: np.ndarray,
    confidence_level: float = 0.80,
    per_step: bool = True,
) -> Dict:
    """Full conformal calibration pipeline.

    1. Compute nonconformity scores on calib set.
    2. Compute conformal offsets.
    3. Apply correction to eval predictions.
    4. Return corrected predictions + offset info.

    Returns
    -------
    dict with keys:
        pred_lower_corrected, pred_upper_corrected : ndarray
        offset : ndarray or scalar
        calib_coverage : float (coverage on calib set before correction)
    """
    alpha = 1.0 - float(confidence_level)

    # Scores on calibration set
    scores_calib = nonconformity_scores(pred_lower_calib, pred_upper_calib, actual_calib)
    calib_coverage = float(np.mean(scores_calib == 0.0))

    # Offset
    offset = conformal_offset(scores_calib, alpha, per_step=per_step)

    # Apply to eval
    lower_corr, upper_corr = apply_conformal_correction(
        pred_lower_eval, pred_upper_eval, offset,
    )

    return {
        "pred_lower_corrected": lower_corr,
        "pred_upper_corrected": upper_corr,
        "offset": offset.tolist() if isinstance(offset, np.ndarray) else float(offset),
        "calib_coverage": calib_coverage,
        "calib_num_samples": int(len(actual_calib)),
        "calib_mean_score": float(np.mean(scores_calib)),
        "calib_max_score": float(np.max(scores_calib)),
    }


def split_indices_time_ordered(
    n_total: int, calib_ratio: float = 0.4, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices in time order (first calib_ratio for calibration).

    Returns (calib_indices, eval_indices).
    """
    n_calib = max(1, int(n_total * calib_ratio))
    indices = np.arange(n_total, dtype=np.int64)
    return indices[:n_calib], indices[n_calib:]


def split_indices_random(
    n_total: int, calib_ratio: float = 0.4, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices randomly."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_calib = max(1, int(n_total * calib_ratio))
    return indices[:n_calib], indices[n_calib:]
