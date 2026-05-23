# -*- coding: utf-8 -*-
"""Scale-family CI Quantile Head for Kronos-R.

Design (confirmed):
  - Shared centre m_t and base width σ_t per step.
  - Per-confidence-level scaling via normal quantile constants k(C).
  - Learnable global scale a_C (init 0) for mild heavy-tail correction.
  - Step embedding for step-aware width prediction.

  q_low(C, t) = m_t - σ_t · exp(a_C) · k(C)
  q_high(C, t) = m_t + σ_t · exp(a_C) · k(C)

  where k(C) = Φ⁻¹((1+C)/2), σ_t = softplus(raw_σ_t) + ε.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Normal quantile constants ──
# k(C) = Φ⁻¹((1+C)/2)  — the (1+C)/2 quantile of N(0,1)
# We pre-compute these as fixed float constants to avoid import-time torch dependency.
# Values verified against scipy.stats.norm.ppf((1+C)/2).
CONFIDENCE_LEVELS = (0.50, 0.68, 0.80, 0.90, 0.95)
K_CONSTANTS = {
    0.50: 0.6744897501960817,   # Φ⁻¹(0.75)
    0.68: 0.9944578832097532,   # Φ⁻¹(0.84)
    0.80: 1.2815515655446004,   # Φ⁻¹(0.90)
    0.90: 1.6448536269514722,   # Φ⁻¹(0.95)
    0.95: 1.959963984540054,    # Φ⁻¹(0.975)
}
# C=0.50 → k≈0.674, C=0.68 → k≈0.994, C=0.80 → k≈1.282, C=0.90 → k≈1.645, C=0.95 → k≈1.960


class CIQuantileHead(nn.Module):
    """Lightweight head that maps hidden states → quantile parameters.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of backbone hidden states.
    num_steps : int
        Max horizon (default 10).
    step_embedding_dim : int
        Dimension of learned step embedding.
    head_hidden_dim : int
        Hidden dimension of the small MLP inside the head.
    sigma_eps : float
        Small constant added to softplus for numerical stability.
    share_aC : bool
        If True, a_C is a single learnable scalar (tied across C).
        If False, one a_C per confidence level.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        num_steps: int = 10,
        step_embedding_dim: int = 16,
        head_hidden_dim: int = 128,
        sigma_eps: float = 1e-3,
        share_aC: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.sigma_eps = sigma_eps
        self.share_aC = share_aC
        self.num_confidence_levels = len(CONFIDENCE_LEVELS)

        # Step embedding
        self.step_embed = nn.Embedding(num_steps, step_embedding_dim)

        # Small MLP: hidden + step_embed → m, raw_σ
        input_dim = hidden_dim + step_embedding_dim
        self.fc1 = nn.Linear(input_dim, head_hidden_dim)
        self.fc2 = nn.Linear(head_hidden_dim, head_hidden_dim // 2)
        self.head_mu = nn.Linear(head_hidden_dim // 2, 1)      # centre m_t
        self.head_sigma = nn.Linear(head_hidden_dim // 2, 1)   # raw width

        # Learnable global scale per C (or shared)
        if share_aC:
            self.a_C_raw = nn.Parameter(torch.zeros(1))
        else:
            self.a_C_raw = nn.Parameter(torch.zeros(self.num_confidence_levels))

        # K constants buffer (non-trainable)
        k_vals = torch.tensor([K_CONSTANTS[c] for c in CONFIDENCE_LEVELS], dtype=torch.float32)
        self.register_buffer("k_values", k_vals)  # [num_C]

        self._init_weights()

    def _init_weights(self):
        for module in [self.fc1, self.fc2, self.head_mu, self.head_sigma]:
            if hasattr(module, "weight") and module.weight.ndim >= 2:
                nn.init.xavier_uniform_(module.weight, gain=0.5)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        # head_sigma bias → small positive so σ starts near 0.01
        nn.init.constant_(self.head_sigma.bias, -2.0)  # softplus(-2) ≈ 0.127 → raw starts small
        nn.init.zeros_(self.head_mu.bias)

    def forward(self, hidden_states, step_indices=None):
        """Predict quantile parameters for each step.

        Parameters
        ----------
        hidden_states : [B, H, hidden_dim]
            Hidden states at each horizon position.
        step_indices : [H] or None
            Step indices (0..H-1). If None, uses arange(H).

        Returns
        -------
        m_t : [B, H]           Predicted centre (log-return).
        sigma_t : [B, H]       Predicted base width (log-return, > 0).
        quantiles : dict[C → (lower [B,H], upper [B,H])]
            Prediction intervals for each confidence level.
        """
        B, H, _ = hidden_states.shape
        if step_indices is None:
            step_indices = torch.arange(H, device=hidden_states.device)

        # Step embedding
        step_emb = self.step_embed(step_indices)  # [H, step_emb_dim]
        step_emb = step_emb.unsqueeze(0).expand(B, H, -1)  # [B, H, step_emb_dim]

        # Concatenate
        x = torch.cat([hidden_states, step_emb], dim=-1)  # [B, H, input_dim]

        # MLP
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))

        m_t = self.head_mu(x).squeeze(-1)          # [B, H]
        raw_sigma = self.head_sigma(x).squeeze(-1)  # [B, H]
        sigma_t = F.softplus(raw_sigma) + self.sigma_eps  # [B, H] > 0

        # Quantiles for each confidence level
        a_C = self.a_C_raw  # [1] or [num_C]
        quantiles = {}
        for i, c in enumerate(CONFIDENCE_LEVELS):
            k = self.k_values[i]  # scalar
            a = a_C if self.share_aC else a_C[i]
            half_width = sigma_t * k * torch.exp(a)  # [B, H]
            quantiles[c] = (m_t - half_width, m_t + half_width)

        return m_t, sigma_t, quantiles

    def get_interval(self, hidden_states, confidence_level, step_indices=None):
        """Convenience: return (lower, upper) for a single confidence level."""
        _, _, quantiles = self.forward(hidden_states, step_indices)
        c = float(confidence_level)
        # find nearest supported C
        nearest = min(CONFIDENCE_LEVELS, key=lambda x: abs(x - c))
        return quantiles[nearest]
