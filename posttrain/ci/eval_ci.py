# -*- coding: utf-8 -*-
"""Confidence Interval evaluation metrics.

Metrics follow the interval-score framework (Gneiting & Raftery, 2007):
  IS_alpha = (u - l) + (2/alpha)·(l - y)_+ + (2/alpha)·(y - u)_+

This is a *proper scoring rule* — the expected score is minimised when
the interval matches the true conditional distribution.  Lower is better.
"""

import math
from typing import Dict, List

import numpy as np


def _safe_mean(values, default=0.0):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return float(default)
    return float(arr[finite].mean())


def interval_score(lower, upper, actual, alpha):
    """Proper scoring rule for a single prediction interval.

    Parameters
    ----------
    lower, upper : ndarray
        Predicted interval bounds (same shape as actual).
    actual : ndarray
        Ground-truth values.
    alpha : float
        1 - confidence_level (e.g. 0.2 for an 80 % interval).

    Returns
    -------
    score : ndarray (same shape)
        IS = width + (2/alpha)·penalty_low + (2/alpha)·penalty_high
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    width = upper - lower
    penalty_low = np.maximum(lower - actual, 0.0)
    penalty_high = np.maximum(actual - upper, 0.0)
    return width + (2.0 / max(float(alpha), 1e-8)) * (penalty_low + penalty_high)


def coverage_rate(lower, upper, actual):
    """Fraction of actual values that fall inside [lower, upper]."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    covered = (actual >= lower) & (actual <= upper)
    return float(covered.mean())


def average_width(lower, upper):
    """Mean interval width."""
    return _safe_mean(np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64))


def compute_ci_metrics(
    pred_lower,
    pred_upper,
    actual_returns,
    confidence_level=0.80,
    mape_eps=1e-4,
):
    """Compute per-step and aggregate CI quality metrics.

    Parameters
    ----------
    pred_lower, pred_upper : ndarray  [N, H]
        Predicted lower / upper bounds for each of H future steps.
    actual_returns : ndarray  [N, H]
        Ground-truth log-returns.
    confidence_level : float
        Nominal confidence level (e.g. 0.80).
    mape_eps : float
        Epsilon for MAPE denominator clamping.

    Returns
    -------
    dict with keys:
        num_sequences, horizon, confidence_level, coverage, avg_width,
        avg_interval_score, per_step (list of per-step dicts).
    """
    pred_lower = np.asarray(pred_lower, dtype=np.float64)
    pred_upper = np.asarray(pred_upper, dtype=np.float64)
    actual = np.asarray(actual_returns, dtype=np.float64)

    if pred_lower.size == 0:
        return {"num_sequences": 0}

    if pred_lower.shape != actual.shape or pred_upper.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: lower={pred_lower.shape}, upper={pred_upper.shape}, "
            f"actual={actual.shape}"
        )

    alpha = 1.0 - float(confidence_level)
    N, H = actual.shape

    # --- aggregate ---
    iscore = interval_score(pred_lower, pred_upper, actual, alpha)
    cov = coverage_rate(pred_lower, pred_upper, actual)
    avg_w = average_width(pred_lower, pred_upper)

    # MAPE-style return errors
    actual_ratio = np.exp(np.clip(actual, -50.0, 50.0))
    denom = np.maximum(np.abs(actual_ratio), float(mape_eps))
    pred_mid = (pred_lower + pred_upper) * 0.5
    pred_mid_ratio = np.exp(np.clip(pred_mid, -50.0, 50.0))
    mape_val = _safe_mean(np.abs(pred_mid_ratio - actual_ratio) / denom) * 100.0

    # direction accuracy of interval midpoint
    pred_mid_sign = np.where(pred_mid >= 0.0, 1, -1)
    actual_sign = np.where(actual >= 0.0, 1, -1)
    da_val = _safe_mean(pred_mid_sign == actual_sign) * 100.0

    # --- cumulative (path) metrics ---
    pred_lower_cum = np.cumsum(pred_lower, axis=1)
    pred_upper_cum = np.cumsum(pred_upper, axis=1)
    actual_cum = np.cumsum(actual, axis=1)
    path_iscore = interval_score(pred_lower_cum, pred_upper_cum, actual_cum, alpha)
    path_cov = coverage_rate(pred_lower_cum, pred_upper_cum, actual_cum)
    path_width = average_width(pred_lower_cum, pred_upper_cum)

    # --- per-step ---
    per_step: List[Dict] = []
    for step in range(H):
        p_iscore = interval_score(pred_lower[:, step], pred_upper[:, step], actual[:, step], alpha)
        p_cov = coverage_rate(pred_lower[:, step], pred_upper[:, step], actual[:, step])
        p_width = average_width(pred_lower[:, step], pred_upper[:, step])

        pp_iscore = interval_score(
            pred_lower_cum[:, step], pred_upper_cum[:, step], actual_cum[:, step], alpha
        )
        pp_cov = coverage_rate(pred_lower_cum[:, step], pred_upper_cum[:, step], actual_cum[:, step])
        pp_width = average_width(pred_lower_cum[:, step], pred_upper_cum[:, step])

        per_step.append({
            "step": int(step + 1),
            "coverage": float(p_cov),
            "avg_width": float(p_width),
            "avg_interval_score": float(_safe_mean(p_iscore)),
            "path_coverage": float(pp_cov),
            "path_avg_width": float(pp_width),
            "path_avg_interval_score": float(_safe_mean(pp_iscore)),
        })

    return {
        "num_sequences": int(N),
        "horizon": int(H),
        "confidence_level": float(confidence_level),
        "alpha": float(alpha),
        "coverage": float(cov),
        "avg_width": float(avg_w),
        "avg_interval_score": float(_safe_mean(iscore)),
        "path_coverage": float(path_cov),
        "path_avg_width": float(path_width),
        "path_avg_interval_score": float(_safe_mean(path_iscore)),
        "mape_midpoint": float(mape_val),
        "da_midpoint": float(da_val),
        "per_step": per_step,
    }
