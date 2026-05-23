"""Kronos-R inference-optimized model with KV-cache, torch.compile, and CUDA graph support.

Enhanced from model/kronos_reasoning.py with:
  - SparseAttention: forward_with_cache / forward_incremental (KV-cache)
  - RingAttentionBlock: corresponding cache-aware methods
  - KronosReasoningGPT: optimized AR predict with KV-cache
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig


def _is_torch_compiling():
    compiler_api = getattr(torch, "compiler", None)
    if compiler_api is None or not hasattr(compiler_api, "is_compiling"):
        return False
    try:
        return bool(compiler_api.is_compiling())
    except Exception:
        return False


@dataclass
class KVCache:
    """KV cache for SparseAttention blocks during autoregressive inference.

    Each layer stores (K, V) tensors of shape [B, num_kv_heads, seq_len, head_dim].
    """
    k_cache: List[torch.Tensor] = field(default_factory=list)
    v_cache: List[torch.Tensor] = field(default_factory=list)
    prefix_hidden: Optional[torch.Tensor] = None
    prefix_len: int = 0

    def clear(self):
        self.k_cache.clear()
        self.v_cache.clear()
        self.prefix_hidden = None
        self.prefix_len = 0


class RevIN(nn.Module):
    """Reversible Instance Normalization for time-series forecasting."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    def forward(self, x, mode="norm"):
        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        return x

    def _normalize(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm, mean, std

    def _denormalize(self, x, mean=None, std=None):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        if mean is not None and std is not None:
            x = x * std + mean
        return x


class SparseAttention(nn.Module):
    """DSA (Differential Sparse Attention) + GQA with KV-cache support for fast AR inference."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        window_size: int | None = None,
        position_encoding: str = "rope",
        rope_base: float = 10000.0,
        alibi_decay_base: float = 0.02,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.position_encoding = str(position_encoding).lower()
        self.rope_base = float(rope_base)
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._rope_cache: dict = {}

        head_idx = torch.arange(self.num_heads, dtype=torch.float32)
        alibi_decays = float(alibi_decay_base) * (head_idx + 1.0) / max(1, self.num_heads)
        self.register_buffer("alibi_slopes", alibi_decays.view(1, self.num_heads, 1, 1), persistent=False)

    def _get_rope_trig(self, seq_len: int, device, dtype):
        rot_dim = (self.head_dim // 2) * 2
        key = (seq_len, device.index if device.index is not None else -1, str(dtype), rot_dim)
        if key in self._rope_cache:
            return self._rope_cache[key]

        trig_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        pos = torch.arange(seq_len, device=device, dtype=trig_dtype)
        freq = torch.arange(0, rot_dim, 2, device=device, dtype=trig_dtype)
        inv_freq = 1.0 / (self.rope_base ** (freq / rot_dim))
        angles = torch.einsum("n,d->nd", pos, inv_freq)
        sin = angles.sin().to(dtype=dtype)
        cos = angles.cos().to(dtype=dtype)
        self._rope_cache[key] = (sin, cos)
        return sin, cos

    def _apply_rope(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to x. x: [B, H, S, D] or [B, H, 1, D]."""
        rot_dim = sin.shape[-1] * 2
        x_main = x[..., :rot_dim]
        x_even = x_main[..., ::2]
        x_odd = x_main[..., 1::2]
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)
        x_rot = torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
        if rot_dim == x.shape[-1]:
            return x_rot
        return torch.cat([x_rot, x[..., rot_dim:]], dim=-1)

    def _apply_alibi(self, scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, device=scores.device, dtype=torch.float32)
        distances = (positions.unsqueeze(1) - positions.unsqueeze(0)).clamp_min(0)
        biases = -self.alibi_slopes * distances.unsqueeze(0)
        return scores + biases

    def _gqa_expand_kv(self, k, v):
        if self.num_kv_heads < self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)
        return k, v

    def _build_attn_mask(self, S, device, dtype):
        """Build combined causal + window attention mask for SDPA.

        Returns float additive mask: 0.0 = attend, -inf = mask.
        Supports ALiBi bias injection.
        """
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1
        )

        has_window = self.window_size is not None and self.window_size < S

        if self.position_encoding == "alibi":
            # Build float mask with ALiBi bias
            positions = torch.arange(S, device=device, dtype=torch.float32)
            distances = (positions.unsqueeze(1) - positions.unsqueeze(0)).clamp_min(0)
            biases = -self.alibi_slopes.to(device=device) * distances.unsqueeze(0)
            # Add causal masking
            biases = biases.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            if has_window:
                idx = torch.arange(S, device=device)
                dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp_min(0)
                window_mask = dist >= self.window_size
                biases = biases.masked_fill(window_mask.unsqueeze(0), float("-inf"))
            return biases.to(dtype=dtype)

        if has_window:
            # Build boolean mask combining causal + window
            idx = torch.arange(S, device=device)
            dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp_min(0)
            window_mask = dist >= self.window_size
            combined = causal_mask | window_mask
            # SDPA attn_mask: True = keep, False = mask (when bool)
            # OR float: 0 = keep, -inf = mask
            return combined.unsqueeze(0).unsqueeze(0)

        # Full attention: just causal
        return "causal"  # SDPA recognizes this string literal

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, S, D = x.shape

        if return_attention:
            return self._forward_manual(x, return_attention=True)

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.position_encoding == "rope":
            sin, cos = self._get_rope_trig(S, q.device, q.dtype)
            q = self._apply_rope(q, sin, cos)
            k = self._apply_rope(k, sin, cos)

        # Use SDPA (Flash Attention on supported GPUs) for fast attention
        # SDPA natively supports GQA — pass Q, K, V with different head counts
        attn_mask = self._build_attn_mask(S, q.device, q.dtype)

        if attn_mask == "causal":
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout.p if isinstance(self.dropout, nn.Dropout) and self.dropout.p > 0 else 0.0,
                scale=self.scale,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout.p if isinstance(self.dropout, nn.Dropout) and self.dropout.p > 0 else 0.0,
                scale=self.scale,
            )

        out = out.transpose(1, 2).reshape(B, S, D)
        out = self.out_proj(out)
        return out

    def _forward_manual(self, x: torch.Tensor, return_attention: bool = False):
        """Fallback manual attention for when return_attention=True is needed."""
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.position_encoding == "rope":
            sin, cos = self._get_rope_trig(S, q.device, q.dtype)
            q = self._apply_rope(q, sin, cos)
            k = self._apply_rope(k, sin, cos)

        k_exp, v_exp = self._gqa_expand_kv(k, v)
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale

        if self.position_encoding == "alibi":
            scores = self._apply_alibi(scores, S)

        causal_mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if self.window_size is not None and self.window_size < S:
            idx = torch.arange(S, device=scores.device)
            dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp_min(0)
            window_mask = dist >= self.window_size
            scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).reshape(B, S, D)
        out = self.out_proj(out)

        if return_attention:
            return out, attn
        return out

    @torch.no_grad()
    def forward_with_cache(self, x: torch.Tensor):
        """Process full prefix sequence and return KV cache for incremental decoding.

        Returns:
            output: [B, S, D]
            k_cache: [B, num_kv_heads, S, head_dim]
            v_cache: [B, num_kv_heads, S, head_dim]
        """
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.position_encoding == "rope":
            sin, cos = self._get_rope_trig(S, q.device, q.dtype)
            q = self._apply_rope(q, sin, cos)
            k = self._apply_rope(k, sin, cos)

        k_expanded, v_expanded = self._gqa_expand_kv(k, v)

        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale

        if self.position_encoding == "alibi":
            scores = self._apply_alibi(scores, S)

        causal_mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if self.window_size is not None and self.window_size < S:
            idx = torch.arange(S, device=scores.device)
            dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp_min(0)
            window_mask = dist >= self.window_size
            scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_expanded)
        out = out.transpose(1, 2).reshape(B, S, D)
        out = self.out_proj(out)

        # Trim KV cache to window_size if needed
        if self.window_size is not None and self.window_size < S:
            k = k[:, :, -self.window_size:, :].contiguous()
            v = v[:, :, -self.window_size:, :].contiguous()

        return out, k, v

    @torch.no_grad()
    def forward_incremental(
        self,
        x_new: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        start_pos: int,
    ):
        """Process a single new token using cached KV from previous steps.

        Args:
            x_new: [B, 1, D] single new token
            k_cache: [B, num_kv_heads, cache_len, head_dim]
            v_cache: [B, num_kv_heads, cache_len, head_dim]
            start_pos: position index of this new token

        Returns:
            output: [B, 1, D]
            k_cache_new: updated K cache
            v_cache_new: updated V cache
        """
        B, S, D = x_new.shape  # S == 1

        q = self.q_proj(x_new).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x_new).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x_new).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.position_encoding == "rope":
            sin, cos = self._get_rope_trig(start_pos + 1, q.device, q.dtype)
            # _apply_rope unsqueezes internally: [S, D/2] → [1, 1, S, D/2]
            sin_pos = sin[start_pos:start_pos + 1]
            cos_pos = cos[start_pos:start_pos + 1]
            q = self._apply_rope(q, sin_pos, cos_pos)
            k_new = self._apply_rope(k_new, sin_pos, cos_pos)

        # Concatenate with cache
        k_full = torch.cat([k_cache, k_new], dim=2)
        v_full = torch.cat([v_cache, v_new], dim=2)

        k_expanded, v_expanded = self._gqa_expand_kv(k_full, v_full)

        total_len = k_full.shape[2]
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale

        if self.position_encoding == "alibi":
            # ALiBi bias for the new query position
            positions = torch.arange(total_len, device=scores.device, dtype=torch.float32)
            query_pos = float(start_pos)
            distances = (query_pos - positions).clamp_min(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            biases = -self.alibi_slopes[:, :, :, :1] * distances
            scores = scores + biases

        # Causal mask is automatically satisfied (new token is last)
        # Only need window mask if applicable
        if self.window_size is not None and total_len > self.window_size:
            idx = torch.arange(total_len, device=scores.device)
            dist = (total_len - 1 - idx).clamp_min(0)
            window_mask = dist >= self.window_size
            scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_expanded)
        out = out.transpose(1, 2).reshape(B, S, D)
        out = self.out_proj(out)

        # Update and trim caches
        if self.window_size is not None and total_len > self.window_size:
            k_cache_new = k_full[:, :, -self.window_size:, :].contiguous()
            v_cache_new = v_full[:, :, -self.window_size:, :].contiguous()
        else:
            k_cache_new = k_full.contiguous()
            v_cache_new = v_full.contiguous()

        return out, k_cache_new, v_cache_new

    def clear_runtime_caches(self):
        self._rope_cache.clear()


class LinearAttention(nn.Module):
    """Causal linear attention with chunked long-sequence support."""

    def __init__(
        self,
        dim=256,
        num_heads=4,
        chunk_size=1024,
        position_encoding="rope",
        rope_base=10000.0,
        alibi_decay_base=0.02,
        local_window_size=512,
        local_windows=None,
        local_attention_mix_init=0.5,
        multi_scale_fusion="gated",
        scale_gate_init=0.3,
        local_attention_max_seq=2048,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size
        self.position_encoding = str(position_encoding).lower()
        self.rope_base = float(rope_base)
        if local_windows is None:
            windows = [int(local_window_size)]
        else:
            windows = [int(window) for window in local_windows]
        self.local_windows = [window for window in windows if window > 0]
        self.local_window_size = max(self.local_windows) if self.local_windows else int(local_window_size)
        self.local_attention_max_seq = int(local_attention_max_seq)
        fusion = str(multi_scale_fusion).lower()
        self.multi_scale_fusion = fusion if fusion in {"gated", "weighted", "concat"} else "gated"
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        self._rope_cache = {}
        self._pos_cache = {}

        head_idx = torch.arange(self.num_heads, dtype=torch.float32)
        alibi_decays = float(alibi_decay_base) * (head_idx + 1.0) / max(1, self.num_heads)
        self.register_buffer("alibi_decays", alibi_decays.view(1, self.num_heads, 1, 1), persistent=False)

        mix_prob = float(local_attention_mix_init)
        mix_prob = min(max(mix_prob, 1e-4), 1 - 1e-4)
        mix_logit = math.log(mix_prob / (1.0 - mix_prob))
        self.local_mix_logit = nn.Parameter(torch.tensor(mix_logit, dtype=torch.float32))

        num_scales = 1 + len(self.local_windows)
        self.scale_gates = nn.Parameter(
            torch.full((num_scales,), float(scale_gate_init), dtype=torch.float32)
        )
        self.multi_scale_out = None
        if self.multi_scale_fusion == "concat" and len(self.local_windows) > 0:
            self.multi_scale_out = nn.Linear(self.head_dim * num_scales, self.head_dim)

    def _cache_key(self, seq_len, device, dtype, extra=None):
        dev_key = f"{device.type}:{device.index if device.index is not None else -1}"
        key = (seq_len, dev_key, str(dtype))
        if extra is not None:
            key = key + (extra,)
        return key

    def _can_use_runtime_cache(self):
        return (not self.training) and (not torch.is_grad_enabled()) and (not _is_torch_compiling())

    def clear_runtime_caches(self):
        self._rope_cache.clear()
        self._pos_cache.clear()

    def _get_rope_trig(self, seq_len, device, dtype, rot_dim):
        use_cache = self._can_use_runtime_cache()
        key = self._cache_key(seq_len, device, dtype, extra=f"rope:{rot_dim}")
        if use_cache and key in self._rope_cache:
            return self._rope_cache[key]

        trig_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        pos = torch.arange(seq_len, device=device, dtype=trig_dtype)
        freq = torch.arange(0, rot_dim, 2, device=device, dtype=trig_dtype)
        inv_freq = 1.0 / (self.rope_base ** (freq / rot_dim))
        angles = torch.einsum("n,d->nd", pos, inv_freq)
        sin = angles.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        cos = angles.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        if use_cache:
            self._rope_cache[key] = (sin, cos)
        return sin, cos

    def _get_pos_distance(self, seq_len, device):
        use_cache = self._can_use_runtime_cache()
        key = self._cache_key(seq_len, device, torch.int64, extra="pos")
        cached = self._pos_cache.get(key) if use_cache else None
        if cached is not None:
            return cached

        q_pos = torch.arange(seq_len, device=device)
        k_pos = torch.arange(seq_len, device=device)
        distance = (q_pos[:, None] - k_pos[None, :]).clamp_min(0)
        if use_cache:
            self._pos_cache[key] = (q_pos, k_pos, distance)
        return q_pos, k_pos, distance

    def _apply_rope(self, q, k):
        rot_dim = (self.head_dim // 2) * 2
        if rot_dim < 2:
            return q, k
        sin, cos = self._get_rope_trig(q.size(2), q.device, q.dtype, rot_dim)

        def _rotate_half(x):
            x_main = x[..., :rot_dim]
            x_even = x_main[..., ::2]
            x_odd = x_main[..., 1::2]
            x_rot = torch.stack(
                [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos],
                dim=-1,
            ).flatten(-2)
            if rot_dim == x.shape[-1]:
                return x_rot
            return torch.cat([x_rot, x[..., rot_dim:]], dim=-1)

        return _rotate_half(q), _rotate_half(k)

    def _apply_alibi_linear_scaling(self, q, k, start_idx=0, anchor_idx=0):
        if self.position_encoding != "alibi":
            return q, k
        pos = torch.arange(start_idx, start_idx + q.size(2), device=q.device, dtype=torch.float32)
        rel_pos = (pos - float(anchor_idx)).view(1, 1, -1, 1)
        decays = self.alibi_decays.to(device=q.device, dtype=torch.float32)
        exp_arg_q = torch.clamp(-decays * rel_pos, min=-60.0, max=60.0)
        exp_arg_k = torch.clamp(decays * rel_pos, min=-60.0, max=60.0)
        q_scale = torch.exp(exp_arg_q).to(dtype=q.dtype)
        k_scale = torch.exp(exp_arg_k).to(dtype=k.dtype)
        return q * q_scale, k * k_scale

    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape
        qkv = (
            self.to_qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.position_encoding == "rope":
            q, k = self._apply_rope(q, k)

        if return_attention:
            return self._causal_softmax_attention(q, k, v)

        if seq_len <= self.chunk_size:
            global_out = self._linear_attention_chunk(q, k, v)
        else:
            global_out = self._linear_attention_chunked_long(q, k, v)

        if (not self.local_windows) or seq_len > self.local_attention_max_seq:
            return self.to_out(global_out.transpose(1, 2).reshape(batch_size, seq_len, -1))

        local_outputs = self._local_multi_scale_attention(q, k, v)
        if not local_outputs:
            return self.to_out(global_out.transpose(1, 2).reshape(batch_size, seq_len, -1))

        if self.multi_scale_fusion == "concat" and self.multi_scale_out is not None:
            mixed = self.multi_scale_out(torch.cat([global_out] + local_outputs, dim=-1))
        else:
            gates = F.softmax(self.scale_gates[: 1 + len(local_outputs)], dim=0).to(dtype=global_out.dtype)
            mixed = gates[0] * global_out
            for idx, local_out in enumerate(local_outputs):
                mixed = mixed + gates[idx + 1] * local_out
        return self.to_out(mixed.transpose(1, 2).reshape(batch_size, seq_len, -1))

    def _causal_softmax_attention(self, q, k, v):
        _, _, seq_len, _ = q.shape
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if self.position_encoding == "alibi":
            _, _, distance = self._get_pos_distance(seq_len, q.device)
            distance = distance.to(dtype=attn.dtype)
            decays = self.alibi_decays.to(dtype=attn.dtype, device=q.device)
            attn = attn - decays * distance.unsqueeze(0).unsqueeze(0)

        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
        attn = attn.masked_fill(mask, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], seq_len, -1)
        return self.to_out(out), attn

    def _linear_attention_chunk(self, q, k, v):
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        q, k = self._apply_alibi_linear_scaling(q, k, start_idx=0, anchor_idx=0)

        kv = torch.einsum("bhnd,bhne->bhnde", k, v)
        kv_cum = torch.cumsum(kv, dim=2)
        k_cum = torch.cumsum(k, dim=2)

        out = torch.einsum("bhnd,bhnde->bhne", q, kv_cum)
        denom = torch.einsum("bhnd,bhnd->bhn", q, k_cum).unsqueeze(-1) + 1e-6
        return out / denom

    def _linear_attention_chunked_long(self, q, k, v):
        batch_size, heads, seq_len, head_dim = q.shape
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        out = torch.zeros_like(v)
        kv_state = torch.zeros(batch_size, heads, head_dim, head_dim, device=q.device, dtype=q.dtype)
        k_state = torch.zeros(batch_size, heads, head_dim, device=q.device, dtype=q.dtype)
        state_anchor = 0

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            q_chunk = q[:, :, start:end]
            k_chunk = k[:, :, start:end]
            v_chunk = v[:, :, start:end]

            if self.position_encoding == "alibi" and start != state_anchor:
                decays = self.alibi_decays.to(device=q.device, dtype=torch.float32)
                delta = float(start - state_anchor)
                rescale = torch.exp(torch.clamp(-decays * delta, min=-60.0, max=60.0)).to(dtype=q.dtype)
                kv_state = kv_state * rescale
                k_state = k_state * rescale.squeeze(-1)
                state_anchor = start

            q_chunk, k_chunk = self._apply_alibi_linear_scaling(q_chunk, k_chunk, start_idx=start, anchor_idx=state_anchor)

            kv_chunk = torch.einsum("bhnd,bhne->bhnde", k_chunk, v_chunk)
            kv_cum = torch.cumsum(kv_chunk, dim=2) + kv_state.unsqueeze(2)
            k_cum = torch.cumsum(k_chunk, dim=2) + k_state.unsqueeze(2)

            out_chunk = torch.einsum("bhnd,bhnde->bhne", q_chunk, kv_cum)
            denom = torch.einsum("bhnd,bhnd->bhn", q_chunk, k_cum).unsqueeze(-1) + 1e-6
            out[:, :, start:end] = out_chunk / denom

            kv_state = kv_cum[:, :, -1]
            k_state = k_cum[:, :, -1]

        return out

    def _local_multi_scale_attention(self, q, k, v):
        if not self.local_windows:
            return []
        _, _, seq_len, _ = q.shape
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        q_pos, k_pos, distance = self._get_pos_distance(seq_len, q.device)

        if self.position_encoding == "alibi":
            decays = self.alibi_decays.to(dtype=scores.dtype, device=q.device)
            scores = scores - decays * distance.to(dtype=scores.dtype).unsqueeze(0).unsqueeze(0)

        causal_mask = (k_pos[None, :] > q_pos[:, None]).unsqueeze(0).unsqueeze(0)
        min_value = torch.finfo(scores.dtype).min
        local_outputs = []

        for window_size in self.local_windows:
            if window_size > 0:
                window_mask = (distance >= window_size).unsqueeze(0).unsqueeze(0)
                full_mask = causal_mask | window_mask
            else:
                full_mask = causal_mask

            curr_scores = scores.masked_fill(full_mask, min_value)
            attn = F.softmax(curr_scores, dim=-1)
            local_outputs.append(torch.matmul(attn, v))

        return local_outputs


class RingAttentionBlock(nn.Module):
    """Residual attention block with KV-cache support for incremental inference."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        window_size: int | None = None,
        position_encoding: str = "rope",
        rope_base: float = 10000.0,
        alibi_decay_base: float = 0.02,
        dropout: float = 0.08,
    ):
        super().__init__()
        self.attn = SparseAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            position_encoding=position_encoding,
            rope_base=rope_base,
            alibi_decay_base=alibi_decay_base,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self._use_checkpoint = False

    def enable_gradient_checkpointing(self, enable: bool = True):
        self._use_checkpoint = enable

    def _forward_impl(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            return x + self.ffn(self.norm2(x)), attn_weights

        if self._use_checkpoint and self.training and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)

        return self._forward_impl(x)

    @torch.no_grad()
    def forward_with_cache(self, x: torch.Tensor):
        """Process full prefix and return output + KV cache."""
        attn_out, k_cache, v_cache = self.attn.forward_with_cache(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, k_cache, v_cache

    @torch.no_grad()
    def forward_incremental(
        self,
        x_new: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        start_pos: int,
    ):
        """Process single new token using KV cache."""
        attn_out, k_cache_new, v_cache_new = self.attn.forward_incremental(
            self.norm1(x_new), k_cache, v_cache, start_pos
        )
        x_new = x_new + attn_out
        x_new = x_new + self.ffn(self.norm2(x_new))
        return x_new, k_cache_new, v_cache_new


class LatentReasoner(nn.Module):
    """Parallel Latent Reasoner using learnable latent tokens with cross+self attention."""

    def __init__(
        self,
        dim=ModelConfig.dim,
        num_latent_tokens=ModelConfig.num_latent_tokens,
        depth=ModelConfig.latent_reasoner_depth,
        cross_heads=ModelConfig.latent_cross_heads,
        dropout=ModelConfig.dropout,
    ):
        super().__init__()
        self.num_latent_tokens = max(1, int(num_latent_tokens))
        self.dim = dim

        self.latent_tokens = nn.Parameter(torch.randn(1, self.num_latent_tokens, dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(max(1, int(depth))):
            self.layers.append(nn.ModuleDict({
                "latent_norm": nn.LayerNorm(dim),
                "cross_attn_norm": nn.LayerNorm(dim),
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=dim, num_heads=max(1, int(cross_heads)),
                    batch_first=True, dropout=dropout,
                ),
                "self_attn_norm": nn.LayerNorm(dim),
                "self_attn": nn.MultiheadAttention(
                    embed_dim=dim, num_heads=max(1, int(cross_heads)),
                    batch_first=True, dropout=dropout,
                ),
                "ffn_norm": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout),
                ),
            }))

        self.gate = nn.Linear(dim * 2, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        latent = self.latent_tokens.expand(batch_size, -1, -1)
        latent_states = []

        for layer in self.layers:
            latent_normed = layer["latent_norm"](latent)
            history_normed = layer["cross_attn_norm"](x)
            cross_out, _ = layer["cross_attn"](latent_normed, history_normed, history_normed, need_weights=False)
            latent = latent + cross_out
            self_normed = layer["self_attn_norm"](latent)
            self_out, _ = layer["self_attn"](self_normed, self_normed, self_normed, need_weights=False)
            latent = latent + self_out
            latent = latent + layer["ffn"](layer["ffn_norm"](latent))
            latent_states.append(latent)

        latent_for_history = latent
        gate_input = torch.cat([x, latent_for_history.mean(dim=1, keepdim=True).expand_as(x)], dim=-1)
        gate_val = torch.sigmoid(self.gate(gate_input))
        latent_broadcast = latent_for_history.mean(dim=1, keepdim=True).expand_as(x)
        output = self.out_norm(x + gate_val * (latent_broadcast - x))
        stacked_latents = torch.stack(latent_states, dim=0)
        return output, stacked_latents


class HorizonDecoder(nn.Module):
    """Horizon Decoder: 30 future query tokens for parallel future prediction."""

    def __init__(
        self,
        dim=ModelConfig.dim,
        horizon_tokens=30,
        depth=2,
        num_heads=4,
        dropout=0.1,
        vocab_size_coarse=1024,
        vocab_size_fine=1024,
    ):
        super().__init__()
        self.horizon_tokens = max(1, int(horizon_tokens))
        self.dim = dim

        self.horizon_emb = nn.Parameter(torch.randn(1, self.horizon_tokens, dim) * 0.02)
        self.horizon_day_emb = nn.Embedding(31, dim)
        self.horizon_month_emb = nn.Embedding(12, dim)

        self.layers = nn.ModuleList()
        for _ in range(max(1, int(depth))):
            self.layers.append(nn.ModuleDict({
                "cross_norm_q": nn.LayerNorm(dim),
                "cross_norm_kv": nn.LayerNorm(dim),
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=dim, num_heads=max(1, int(num_heads)),
                    batch_first=True, dropout=dropout,
                ),
                "self_norm": nn.LayerNorm(dim),
                "self_attn": nn.MultiheadAttention(
                    embed_dim=dim, num_heads=max(1, int(num_heads)),
                    batch_first=True, dropout=dropout,
                ),
                "ffn_norm": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim), nn.Dropout(dropout),
                ),
            }))

        self.norm = nn.LayerNorm(dim)
        self.horizon_head_coarse = nn.Linear(dim, vocab_size_coarse)
        self.coarse_to_fine = nn.Linear(dim, dim)
        self.fine_gate = nn.Linear(dim * 2, dim)
        self.fine_norm = nn.LayerNorm(dim)
        self.horizon_head_fine = nn.Linear(dim, vocab_size_fine)

        causal_mask = torch.triu(
            torch.ones(self.horizon_tokens, self.horizon_tokens, dtype=torch.bool), diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, history_hidden, future_day=None, future_month=None):
        batch_size = history_hidden.size(0)
        queries = self.horizon_emb.expand(batch_size, -1, -1).clone()

        if future_day is not None:
            day_clamped = torch.clamp(future_day[:, :self.horizon_tokens], 0, 30)
            queries[:, :day_clamped.size(1), :] = (
                queries[:, :day_clamped.size(1), :] + self.horizon_day_emb(day_clamped)
            )
        if future_month is not None:
            month_clamped = torch.clamp(future_month[:, :self.horizon_tokens], 0, 11)
            queries[:, :month_clamped.size(1), :] = (
                queries[:, :month_clamped.size(1), :] + self.horizon_month_emb(month_clamped)
            )

        for layer in self.layers:
            q_normed = layer["cross_norm_q"](queries)
            kv_normed = layer["cross_norm_kv"](history_hidden)
            cross_out, _ = layer["cross_attn"](q_normed, kv_normed, kv_normed, need_weights=False)
            queries = queries + cross_out
            self_normed = layer["self_norm"](queries)
            self_out, _ = layer["self_attn"](self_normed, self_normed, self_normed, attn_mask=self.causal_mask, need_weights=False)
            queries = queries + self_out
            queries = queries + layer["ffn"](layer["ffn_norm"](queries))

        queries = self.norm(queries)
        logits_coarse = self.horizon_head_coarse(queries)

        coarse_probs = F.softmax(logits_coarse.float(), dim=-1).to(dtype=queries.dtype)
        coarse_ctx = torch.matmul(coarse_probs, self.horizon_head_coarse.weight)
        coarse_ctx = self.coarse_to_fine(coarse_ctx)
        fine_gate_val = torch.sigmoid(self.fine_gate(torch.cat([queries, coarse_ctx], dim=-1)))
        fine_hidden = self.fine_norm(queries + fine_gate_val * coarse_ctx)
        logits_fine = self.horizon_head_fine(fine_hidden)

        return logits_coarse, logits_fine


class KronosReasoningGPT(nn.Module):
    """Kronos reasoning model with DSA+GQA and optimized KV-cache AR inference."""

    def __init__(
        self,
        dim=ModelConfig.dim,
        depth=ModelConfig.depth,
        heads=ModelConfig.heads,
        num_kv_heads=2,
        dsa_windows=None,
        dropout=ModelConfig.dropout,
        vocab_size_coarse=ModelConfig.vocab_size_coarse,
        vocab_size_fine=ModelConfig.vocab_size_fine,
        num_latent_tokens=ModelConfig.num_latent_tokens,
        latent_reasoner_depth=ModelConfig.latent_reasoner_depth,
        latent_cross_heads=ModelConfig.latent_cross_heads,
        position_encoding=ModelConfig.position_encoding,
        rope_base=ModelConfig.rope_base,
        alibi_decay_base=ModelConfig.alibi_decay_base,
        max_len=ModelConfig.max_len,
        use_revin=ModelConfig.use_revin,
        num_factor_tokens=ModelConfig.num_factor_tokens,
        **legacy_kwargs,
    ):
        super().__init__()
        self.position_encoding = str(position_encoding).lower()
        self.max_len = int(max_len)
        self.use_revin = use_revin
        self.vocab_size_coarse = int(vocab_size_coarse)
        self.vocab_size_fine = int(vocab_size_fine)
        self.depth = depth

        if dsa_windows is None:
            base = [None, 512, 512, None]
            dsa_windows = [base[i % len(base)] for i in range(depth)]
        self.dsa_windows = list(dsa_windows)

        self.token_emb_coarse = nn.Embedding(self.vocab_size_coarse, dim)
        self.token_emb_fine = nn.Embedding(self.vocab_size_fine, dim)

        self.time_emb_min = nn.Embedding(240, dim)
        self.time_emb_day = nn.Embedding(31, dim)
        self.time_emb_month = nn.Embedding(12, dim)
        self.time_emb_year = nn.Embedding(100, dim)

        self.blocks = nn.ModuleList()
        for layer_idx in range(depth):
            self.blocks.append(
                RingAttentionBlock(
                    dim=dim,
                    num_heads=heads,
                    num_kv_heads=num_kv_heads,
                    window_size=self.dsa_windows[layer_idx],
                    position_encoding=self.position_encoding,
                    rope_base=rope_base,
                    alibi_decay_base=alibi_decay_base,
                    dropout=dropout,
                )
            )

        self.latent_reasoner = LatentReasoner(
            dim=dim, num_latent_tokens=num_latent_tokens,
            depth=latent_reasoner_depth, cross_heads=latent_cross_heads, dropout=dropout,
        )

        self.horizon_decoder = None

        self.norm = nn.LayerNorm(dim)
        self.head_coarse = nn.Linear(dim, self.vocab_size_coarse)
        self.coarse_to_fine = nn.Linear(dim, dim)
        self.fine_gate = nn.Linear(dim * 2, dim)
        self.fine_norm = nn.LayerNorm(dim)
        self.head_fine = nn.Linear(dim, self.vocab_size_fine)

        self.pos_emb = None
        if self.position_encoding == "learned":
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
            nn.init.normal_(self.pos_emb, std=0.02)

        if use_revin:
            self.revin = RevIN(dim, eps=1e-5, affine=True)
        else:
            self.revin = None

        if num_factor_tokens > 0:
            self.factor_tokens = nn.Parameter(torch.randn(1, num_factor_tokens, dim) * 0.02)
            self.factor_cross_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=max(1, heads // 2), batch_first=True,
            )
            self.factor_norm = nn.LayerNorm(dim)
        else:
            self.factor_tokens = None

        direction_hidden = max(1, dim // 2)
        self.direction_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, direction_hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(direction_hidden, 3),
        )

        value_hidden = max(1, dim // 4)
        self.value_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, value_hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(value_hidden, 1),
        )

        plan_hidden = max(1, dim // 2)
        self.plan_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, plan_hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(plan_hidden, 3),
        )

        error_hidden = max(1, dim // 4)
        self.error_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, error_hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(error_hidden, 1),
        )

    def _compute_embedding(self, idx_coarse, idx_fine, t_min, t_day, t_month, t_year):
        seq_len = idx_coarse.shape[1]
        idx_coarse = torch.clamp(idx_coarse, 0, self.vocab_size_coarse - 1)
        idx_fine = torch.clamp(idx_fine, 0, self.vocab_size_fine - 1)
        t_min = torch.clamp(t_min, 0, 239)
        t_day = torch.clamp(t_day, 0, 30)
        t_month = torch.clamp(t_month, 0, 11)
        t_year = torch.clamp(t_year, 0, 99)

        x = self.token_emb_coarse(idx_coarse) + self.token_emb_fine(idx_fine)
        x = (
            x
            + self.time_emb_min(t_min)
            + self.time_emb_day(t_day)
            + self.time_emb_month(t_month)
            + self.time_emb_year(t_year)
        )
        if self.pos_emb is not None:
            x = x + self.pos_emb[:, :seq_len, :]
        return x

    def _compute_embedding_single(self, idx_coarse, idx_fine, t_min, t_day, t_month, t_year, pos):
        """Compute embedding for a single token at a given position."""
        idx_coarse = torch.clamp(idx_coarse, 0, self.vocab_size_coarse - 1)
        idx_fine = torch.clamp(idx_fine, 0, self.vocab_size_fine - 1)
        t_min = torch.clamp(t_min, 0, 239)
        t_day = torch.clamp(t_day, 0, 30)
        t_month = torch.clamp(t_month, 0, 11)
        t_year = torch.clamp(t_year, 0, 99)

        x = self.token_emb_coarse(idx_coarse) + self.token_emb_fine(idx_fine)
        x = (
            x
            + self.time_emb_min(t_min)
            + self.time_emb_day(t_day)
            + self.time_emb_month(t_month)
            + self.time_emb_year(t_year)
        )
        if self.pos_emb is not None:
            x = x + self.pos_emb[:, pos:pos + 1, :]
        return x

    def _compute_output_logits(self, x, last_only=False):
        if last_only:
            x = x[:, -1:, :]
        x = self.norm(x)
        logits_coarse = self.head_coarse(x)

        coarse_probs = F.softmax(logits_coarse.float(), dim=-1).to(dtype=x.dtype)
        coarse_ctx = torch.matmul(coarse_probs, self.token_emb_coarse.weight)
        coarse_ctx = self.coarse_to_fine(coarse_ctx)
        fine_gate = torch.sigmoid(self.fine_gate(torch.cat([x, coarse_ctx], dim=-1)))
        fine_hidden = self.fine_norm(x + fine_gate * coarse_ctx)
        logits_fine = self.head_fine(fine_hidden)

        return logits_coarse, logits_fine

    def _apply_revin(self, x):
        if self.revin is None:
            return x
        x_norm, _, _ = self.revin(x, mode="norm")
        return x_norm

    def _apply_factor_tokens(self, x):
        if self.factor_tokens is None:
            return x
        batch_size = x.size(0)
        factor = self.factor_tokens.expand(batch_size, -1, -1)
        normed_x = self.factor_norm(x)
        attn_out, _ = self.factor_cross_attn(factor, normed_x, normed_x, need_weights=False)
        factor_updated = factor + attn_out
        factor_mean = factor_updated.mean(dim=1, keepdim=True)
        return x + factor_mean

    def _run_post_backbone(self, x, last_only=False):
        x = self._apply_revin(x)
        x = self._apply_factor_tokens(x)
        x, latent_states = self.latent_reasoner(x)
        logits_coarse, logits_fine = self._compute_output_logits(x, last_only=last_only)
        return x, latent_states, logits_coarse, logits_fine

    def _compute_direction_logits(self, x, latent_states):
        last_hidden = x[:, -1, :]
        if latent_states is not None and latent_states.ndim == 4:
            latent_mean = latent_states[-1].mean(dim=1)
            direction_state = last_hidden + latent_mean
        else:
            direction_state = last_hidden
        return self.direction_head(direction_state)

    # ─── Standard forward (unchanged from original) ───

    def forward(
        self, idx_coarse, idx_fine, t_min, t_day, t_month, t_year,
        return_attention=False, last_only=False, return_hidden=False,
    ):
        _, seq_len = idx_coarse.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max length {self.max_len}")

        x = self._compute_embedding(idx_coarse, idx_fine, t_min, t_day, t_month, t_year)

        attention_weights = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_weights.append(attn)
            else:
                x = block(x)

        x, latent_states, logits_coarse, logits_fine = self._run_post_backbone(x, last_only=last_only)

        if return_attention:
            if return_hidden:
                return logits_coarse, logits_fine, latent_states, attention_weights, x
            return logits_coarse, logits_fine, latent_states, attention_weights
        if return_hidden:
            return logits_coarse, logits_fine, latent_states, x
        return logits_coarse, logits_fine, latent_states

    # ─── KV-Cache Autoregressive Inference ───

    @torch.no_grad()
    def predict_ar_kv_cache(
        self,
        idx_coarse: torch.Tensor,
        idx_fine: torch.Tensor,
        t_min: torch.Tensor,
        t_day: torch.Tensor,
        t_month: torch.Tensor,
        t_year: torch.Tensor,
        horizon: int,
        temperature: float = 1.0,
        use_sampling: bool = False,
    ):
        """Autoregressive prediction using KV-cache for fast multi-step forecasting.

        This is the core optimized inference method:
          1. Process the prefix through all blocks WITH KV caching (once)
          2. Run LatentReasoner + output heads on prefix hidden states
          3. For each step: incremental block forward (fast) → LatentReasoner → decode
          4. Accumulate predictions

        Args:
            idx_coarse/fine: [B, prefix_len] tokenized prefix
            t_*: time features for prefix tokens
            horizon: number of future steps to predict
            temperature: sampling temperature (1.0 = argmax)
            use_sampling: if True, sample from logits; else argmax

        Returns:
            pred_indices_coarse: [B, horizon] predicted coarse token indices
            pred_indices_fine: [B, horizon] predicted fine token indices
        """
        B, prefix_len = idx_coarse.shape
        device = idx_coarse.device
        dtype = next(self.parameters()).dtype

        # Step 1: Embed prefix
        x = self._compute_embedding(idx_coarse, idx_fine, t_min, t_day, t_month, t_year)

        # Step 2: Forward through blocks with KV cache collection
        k_caches = []
        v_caches = []
        for block in self.blocks:
            x, k_cache, v_cache = block.forward_with_cache(x)
            k_caches.append(k_cache)
            v_caches.append(v_cache)

        # Step 3: Run post-backbone on prefix
        prefix_hidden = x  # save for incremental steps
        _, _, logits_c, logits_f = self._run_post_backbone(x, last_only=True)

        # Step 4: Get first prediction
        pred_c_list = []
        pred_f_list = []

        if use_sampling and temperature > 0:
            probs_c = F.softmax(logits_c[:, -1, :].float() / max(temperature, 1e-6), dim=-1)
            pc = torch.multinomial(probs_c, 1).squeeze(-1)
            probs_f = F.softmax(logits_f[:, -1, :].float() / max(temperature, 1e-6), dim=-1)
            pf = torch.multinomial(probs_f, 1).squeeze(-1)
        else:
            pc = logits_c[:, -1, :].argmax(dim=-1)
            pf = logits_f[:, -1, :].argmax(dim=-1)

        pred_c_list.append(pc.unsqueeze(1))
        pred_f_list.append(pf.unsqueeze(1))

        # Step 5: Autoregressive loop with incremental forward
        cur_c = pc.clone()
        cur_f = pf.clone()
        cur_t_min = t_min[:, -1:]  # reuse last prefix time (approximation)
        cur_t_day = t_day[:, -1:]
        cur_t_month = t_month[:, -1:]
        cur_t_year = t_year[:, -1:]

        for step in range(1, horizon):
            # Embed single new token
            x_new = self._compute_embedding_single(
                cur_c.unsqueeze(1), cur_f.unsqueeze(1),
                cur_t_min, cur_t_day, cur_t_month, cur_t_year,
                pos=prefix_len + step - 1,
            )

            # Incremental forward through blocks
            new_k_caches = []
            new_v_caches = []
            for block, kc, vc in zip(self.blocks, k_caches, v_caches):
                x_new, nkc, nvc = block.forward_incremental(
                    x_new, kc, vc, start_pos=prefix_len + step - 1
                )
                new_k_caches.append(nkc)
                new_v_caches.append(nvc)

            # Update caches
            k_caches = new_k_caches
            v_caches = new_v_caches

            # Concatenate hidden states and run post-backbone
            prefix_hidden = torch.cat([prefix_hidden, x_new], dim=1)
            _, _, logits_c, logits_f = self._run_post_backbone(prefix_hidden, last_only=True)

            # Decode prediction
            if use_sampling and temperature > 0:
                probs_c = F.softmax(logits_c[:, -1, :].float() / max(temperature, 1e-6), dim=-1)
                pc = torch.multinomial(probs_c, 1).squeeze(-1)
                probs_f = F.softmax(logits_f[:, -1, :].float() / max(temperature, 1e-6), dim=-1)
                pf = torch.multinomial(probs_f, 1).squeeze(-1)
            else:
                pc = logits_c[:, -1, :].argmax(dim=-1)
                pf = logits_f[:, -1, :].argmax(dim=-1)

            pred_c_list.append(pc.unsqueeze(1))
            pred_f_list.append(pf.unsqueeze(1))
            cur_c = pc.clone()
            cur_f = pf.clone()

        pred_coarse = torch.cat(pred_c_list, dim=1)
        pred_fine = torch.cat(pred_f_list, dim=1)
        return pred_coarse, pred_fine

    # ─── Auxiliary heads (unchanged) ───

    def predict_value(self, hidden_state):
        return self.value_head(hidden_state).squeeze(-1)

    def predict_plan(self, hidden_state):
        return self.plan_head(hidden_state)

    def predict_error(self, hidden_state):
        return self.error_head(hidden_state).squeeze(-1)

    def forward_direction(
        self, idx_coarse, idx_fine, t_min, t_day, t_month, t_year, return_token_logits=False,
    ):
        _, seq_len = idx_coarse.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max length {self.max_len}")

        x = self._compute_embedding(idx_coarse, idx_fine, t_min, t_day, t_month, t_year)
        for block in self.blocks:
            x = block(x)
        x, latent_states, logits_coarse, logits_fine = self._run_post_backbone(x)
        direction_logits = self._compute_direction_logits(x, latent_states)

        if return_token_logits:
            return direction_logits, logits_coarse, logits_fine, latent_states
        return direction_logits, latent_states

    def forward_horizon(
        self, idx_coarse, idx_fine, t_min, t_day, t_month, t_year,
        future_day=None, future_month=None,
    ):
        _, seq_len = idx_coarse.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max length {self.max_len}")

        x = self._compute_embedding(idx_coarse, idx_fine, t_min, t_day, t_month, t_year)
        for block in self.blocks:
            x = block(x)
        x, latent_states, logits_coarse, logits_fine = self._run_post_backbone(x)

        horizon_logits_coarse, horizon_logits_fine = self.horizon_decoder(
            x, future_day=future_day, future_month=future_month
        )
        return logits_coarse, logits_fine, latent_states, horizon_logits_coarse, horizon_logits_fine

    def enable_gradient_checkpointing(self, enable: bool = True):
        for block in self.blocks:
            block.enable_gradient_checkpointing(enable)

    def clear_runtime_caches(self):
        for block in self.blocks:
            if hasattr(block.attn, 'clear_runtime_caches'):
                block.attn.clear_runtime_caches()
