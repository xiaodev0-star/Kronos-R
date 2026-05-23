# -*- coding: utf-8 -*-
"""Training logic for CIQuantileHead.

Losses:
  - Pinball (quantile) loss — proper scoring rule for quantiles.
  - Interval score — fully differentiable with explicit boundaries.
  - Teacher distillation — SmoothL1 to high-N sampling quantiles.
  - Monotonicity penalties — step-width and quantile-crossing.
  - Coverage-constrained dual optimisation (Method C).

All losses operate on denormalised log-return space.
"""

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Pinball / Quantile Loss
# ═══════════════════════════════════════════════════════════════

def pinball_loss(q: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """Pinball (quantile) loss.

    PB_τ(q, y) = max(τ·(y-q), (τ-1)·(y-q))
                = τ·max(y-q, 0) + (1-τ)·max(q-y, 0)

    Parameters
    ----------
    q : [B, H]  Predicted τ-quantile.
    y : [B, H]  True value.
    tau : float  Quantile level (0 < τ < 1).

    Returns
    -------
    scalar loss
    """
    diff = y - q
    return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()


def dual_pinball_loss(
    q_low: torch.Tensor, q_high: torch.Tensor, y: torch.Tensor, confidence: float
) -> torch.Tensor:
    """Pinball loss for both tails of a prediction interval.

    Parameters
    ----------
    q_low : [B, H]   Predicted α/2 quantile.
    q_high : [B, H]  Predicted 1-α/2 quantile.
    y : [B, H]       True value.
    confidence : float  Confidence level (1-α).

    Returns
    -------
    scalar loss
    """
    alpha = 1.0 - float(confidence)
    tau_low = alpha / 2.0
    tau_high = 1.0 - alpha / 2.0
    return pinball_loss(q_low, y, tau_low) + pinball_loss(q_high, y, tau_high)


# ═══════════════════════════════════════════════════════════════
# Interval Score (fully differentiable)
# ═══════════════════════════════════════════════════════════════

def interval_score_loss(
    q_low: torch.Tensor, q_high: torch.Tensor, y: torch.Tensor, confidence: float
) -> torch.Tensor:
    """Gneiting-Raftery interval score.

    IS_α(L, U; y) = (U-L) + (2/α)·max(L-y, 0) + (2/α)·max(y-U, 0)

    Fully differentiable w.r.t. L and U.
    """
    alpha = 1.0 - float(confidence)
    alpha = max(alpha, 1e-8)
    width = q_high - q_low
    penalty_low = torch.clamp(q_low - y, min=0.0)
    penalty_high = torch.clamp(y - q_high, min=0.0)
    return (width + (2.0 / alpha) * (penalty_low + penalty_high)).mean()


# ═══════════════════════════════════════════════════════════════
# Teacher Distillation
# ═══════════════════════════════════════════════════════════════

def teacher_distillation_loss(
    q_low: torch.Tensor,
    q_high: torch.Tensor,
    teacher_low: torch.Tensor,
    teacher_high: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Distill high-N sampling quantiles into the quantile head.

    Parameters
    ----------
    q_low, q_high : [B, H]  Student quantiles.
    teacher_low, teacher_high : [B, H]  Teacher quantiles (detached).
    loss_type : "smooth_l1" | "l1" | "mse"
    """
    if loss_type == "smooth_l1":
        loss_low = F.smooth_l1_loss(q_low, teacher_low)
        loss_high = F.smooth_l1_loss(q_high, teacher_high)
    elif loss_type == "l1":
        loss_low = F.l1_loss(q_low, teacher_low)
        loss_high = F.l1_loss(q_high, teacher_high)
    else:  # mse
        loss_low = F.mse_loss(q_low, teacher_low)
        loss_high = F.mse_loss(q_high, teacher_high)
    return loss_low + loss_high


# ═══════════════════════════════════════════════════════════════
# Monotonicity penalties
# ═══════════════════════════════════════════════════════════════

def step_width_monotonicity_penalty(sigma_t: torch.Tensor) -> torch.Tensor:
    """Penalise decreasing width across steps.

    Encourages σ_{t+1} >= σ_t — uncertainty should grow over time.

    Parameters
    ----------
    sigma_t : [B, H]  Per-step width parameter.

    Returns
    -------
    scalar penalty
    """
    if sigma_t.size(1) < 2:
        return sigma_t.new_zeros(())
    # Penalise when later step has smaller sigma than previous
    diffs = sigma_t[:, :-1] - sigma_t[:, 1:]  # [B, H-1], positive = violation
    return torch.clamp(diffs, min=0.0).mean()


def quantile_crossing_penalty(q_low: torch.Tensor, q_high: torch.Tensor) -> torch.Tensor:
    """Penalise quantile crossing: q_low should be < q_high."""
    crossing = torch.clamp(q_low - q_high, min=0.0)  # positive = crossing
    return crossing.mean()


def multi_confidence_monotonicity_penalty(
    quantiles: dict, sigma_t: torch.Tensor, confidence_levels: tuple
) -> torch.Tensor:
    """Penalise if higher confidence doesn't produce wider intervals.

    For C1 < C2, we expect width(C2) > width(C1).
    """
    sorted_c = sorted(confidence_levels)
    if len(sorted_c) < 2:
        return sigma_t.new_zeros(())
    penalty = sigma_t.new_zeros(())
    for i in range(len(sorted_c) - 1):
        c_lo, c_hi = sorted_c[i], sorted_c[i + 1]
        _, upper_lo = quantiles[c_lo]
        _, upper_hi = quantiles[c_hi]
        width_lo = upper_lo - quantiles[c_lo][0]  # upper - lower
        width_hi = upper_hi - quantiles[c_hi][0]
        penalty = penalty + torch.clamp(width_lo - width_hi, min=0.0).mean()
    return penalty


# ═══════════════════════════════════════════════════════════════
# Coverage-Constrained Dual (Method C)
# ═══════════════════════════════════════════════════════════════

class CoverageDualController:
    """Lagrangian dual controller for coverage-constrained optimisation.

    Maintains a dual variable λ per confidence level.
    Updates λ online: λ ← clip(λ + η * (coverage_gap), 0, λ_max).
    """

    def __init__(self, confidence_levels=(0.68, 0.80, 0.90), eta_dual=0.1, lambda_max=10.0):
        self.confidence_levels = tuple(confidence_levels)
        self.eta_dual = float(eta_dual)
        self.lambda_max = float(lambda_max)
        self.lambdas = {c: 0.0 for c in confidence_levels}
        self._coverage_history = {c: [] for c in confidence_levels}

    def soft_coverage(self, q_low, q_high, y, tau=0.01):
        """Differentiable soft coverage via sigmoid relaxation.

        soft_cov = σ((y-L)/τ) · σ((U-y)/τ)
        """
        return (
            torch.sigmoid((y - q_low) / tau) * torch.sigmoid((q_high - y) / tau)
        ).mean()

    def update_and_get_penalty(self, q_low, q_high, y, confidence):
        """Compute dual penalty term and update λ.

        Returns λ * relu(coverage_gap) as the penalty term.
        """
        c = float(confidence)
        if c not in self.lambdas:
            return q_low.new_zeros(())

        alpha = 1.0 - c
        soft_cov = self.soft_coverage(q_low, q_high, y)
        coverage_gap = c - soft_cov  # positive = under-coverage

        # Update dual variable
        self.lambdas[c] = max(0.0, min(self.lambda_max,
            self.lambdas[c] + self.eta_dual * float(coverage_gap.detach().item())))
        self._coverage_history[c].append(float(soft_cov.detach().item()))

        # Penalty: λ * relu(coverage_gap)
        return self.lambdas[c] * torch.clamp(coverage_gap, min=0.0)

    def get_lambda(self, confidence):
        return self.lambdas.get(float(confidence), 0.0)

    def get_coverage_history(self, confidence):
        return self._coverage_history.get(float(confidence), [])


# ═══════════════════════════════════════════════════════════════
# Composite training loss
# ═══════════════════════════════════════════════════════════════

def compute_quantile_training_loss(
    quantile_head,
    hidden_states,
    actual_returns,
    teacher_quantiles=None,
    confidence_levels=(0.68, 0.80, 0.90),
    lambda_pinball=1.0,
    lambda_teacher=0.3,
    lambda_is_score=0.3,
    lambda_mono_step=0.05,
    lambda_mono_quantile=0.01,
    lambda_mono_multi_c=0.01,
    dual_controller=None,
):
    """Compute composite quantile training loss.

    Parameters
    ----------
    quantile_head : CIQuantileHead
    hidden_states : [B, H, dim]
    actual_returns : [B, H]  denormalised log-returns.
    teacher_quantiles : dict or None
        {c: (teacher_low [B,H], teacher_high [B,H])} for distillation.
    confidence_levels : tuple of float
    lambda_* : float  Loss weights.
    dual_controller : CoverageDualController or None

    Returns
    -------
    total_loss : scalar
    stats : dict  Per-component loss values for monitoring.
    """
    m_t, sigma_t, quantiles = quantile_head(hidden_states)

    total = m_t.new_zeros(())
    stats = {}

    # ── Pinball loss (all confidence levels) ──
    pinball_val = m_t.new_zeros(())
    for c in confidence_levels:
        q_low, q_high = quantiles[c]
        pb = dual_pinball_loss(q_low, q_high, actual_returns, c)
        pinball_val = pinball_val + pb
    pinball_val = pinball_val / max(1, len(confidence_levels))
    total = total + lambda_pinball * pinball_val
    stats["pinball"] = float(pinball_val.detach().item())

    # ── Interval score (primary confidence level) ──
    primary_c = confidence_levels[0]
    q_low_p, q_high_p = quantiles[primary_c]
    is_val = interval_score_loss(q_low_p, q_high_p, actual_returns, primary_c)
    total = total + lambda_is_score * is_val
    stats["interval_score"] = float(is_val.detach().item())

    # ── Teacher distillation ──
    teacher_val = m_t.new_zeros(())
    if teacher_quantiles is not None and lambda_teacher > 0:
        for c in confidence_levels:
            if c in teacher_quantiles:
                t_low, t_high = teacher_quantiles[c]
                q_low, q_high = quantiles[c]
                teacher_val = teacher_val + teacher_distillation_loss(q_low, q_high, t_low, t_high)
        if len(confidence_levels) > 0:
            teacher_val = teacher_val / max(1, len(
                [c for c in confidence_levels if c in teacher_quantiles]))
        total = total + lambda_teacher * teacher_val
    stats["teacher_distill"] = float(teacher_val.detach().item())

    # ── Monotonicity penalties ──
    mono_step = step_width_monotonicity_penalty(sigma_t)
    total = total + lambda_mono_step * mono_step
    stats["mono_step"] = float(mono_step.detach().item())

    mono_q = quantile_crossing_penalty(q_low_p, q_high_p)
    total = total + lambda_mono_quantile * mono_q
    stats["mono_quantile"] = float(mono_q.detach().item())

    mono_mc = multi_confidence_monotonicity_penalty(quantiles, sigma_t, confidence_levels)
    total = total + lambda_mono_multi_c * mono_mc
    stats["mono_multi_c"] = float(mono_mc.detach().item())

    # ── Dual coverage constraint (Method C) ──
    dual_val = m_t.new_zeros(())
    if dual_controller is not None:
        for c in confidence_levels:
            q_low_c, q_high_c = quantiles[c]
            dual_val = dual_val + dual_controller.update_and_get_penalty(
                q_low_c, q_high_c, actual_returns, c)
        dual_val = dual_val / max(1, len(confidence_levels))
        total = total + dual_val
    stats["dual_coverage"] = float(dual_val.detach().item())

    # ── Coverage metrics (monitoring, no gradient) ──
    with torch.no_grad():
        covered = ((actual_returns >= q_low_p) & (actual_returns <= q_high_p)).float().mean()
        width = (q_high_p - q_low_p).mean()
    stats["coverage"] = float(covered.item())
    stats["avg_width"] = float(width.item())

    return total, stats
