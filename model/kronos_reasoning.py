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
    linear_attn_states: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]] = field(default_factory=dict)
    prefix_hidden: Optional[torch.Tensor] = None
    prefix_len: int = 0
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

    def clear(self):
        self.linear_attn_states.clear()
        self.prefix_hidden = None
        self.prefix_len = 0
        self.device = None
        self.dtype = None

    def is_valid(self, device: torch.device, dtype: torch.dtype) -> bool:
        if self.prefix_hidden is None:
            return False
        return self.device == device and self.dtype == dtype


class RevIN(nn.Module):
    """Reversible Instance Normalization for time-series forecasting.

    Normalizes input along the temporal dimension, then denormalizes output.
    Preserves non-stationary information by learning affine parameters.
    """

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
    """DSA (Differential Sparse Attention) + GQA for Kronos-R.

    Standard causal softmax attention with:
      - Per-layer sliding-window control (None = full attention)
      - Grouped Query Attention (GQA): num_kv_heads <= num_heads
      - RoPE position encoding
      - Optional ALiBi bias

    DSA pattern across layers (4 layers):
      Layer 0: full attention  (capture global dependencies)
      Layer 1: window=512       (local patterns)
      Layer 2: window=512       (local patterns)
      Layer 3: full attention  (global integration)
    """

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

        # QKV projections — separate for GQA
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # RoPE cache
        self._rope_cache: dict = {}

        # ALiBi slopes (per-head, for num_heads)
        head_idx = torch.arange(self.num_heads, dtype=torch.float32)
        alibi_decays = float(alibi_decay_base) * (head_idx + 1.0) / max(1, self.num_heads)
        self.register_buffer("alibi_slopes", alibi_decays.view(1, self.num_heads, 1, 1), persistent=False)

    def _get_rope_trig(self, seq_len: int, device, dtype):
        """Cached RoPE sin/cos tables."""
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
        # sin, cos: [S, rot_dim/2] → [1, 1, S, rot_dim/2]
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)
        x_rot = torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
        if rot_dim == x.shape[-1]:
            return x_rot
        return torch.cat([x_rot, x[..., rot_dim:]], dim=-1)

    def _apply_alibi(self, scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Add ALiBi biases to attention scores."""
        positions = torch.arange(seq_len, device=scores.device, dtype=torch.float32)
        distances = (positions.unsqueeze(1) - positions.unsqueeze(0)).clamp_min(0)  # [S, S]
        biases = -self.alibi_slopes * distances.unsqueeze(0)  # [1, H, S, S]
        return scores + biases

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: [batch, seq_len, dim]
            return_attention: if True, return (output, attention_weights)

        Returns:
            output: [batch, seq_len, dim]
        """
        B, S, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if self.position_encoding == "rope":
            sin, cos = self._get_rope_trig(S, q.device, q.dtype)
            q = self._apply_rope(q, sin, cos)
            k = self._apply_rope(k, sin, cos)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_heads < self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]

        # ALiBi
        if self.position_encoding == "alibi":
            scores = self._apply_alibi(scores, S)

        # Causal mask
        causal_mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Sliding window mask
        if self.window_size is not None and self.window_size < S:
            idx = torch.arange(S, device=scores.device)
            dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp_min(0)  # [S, S], distance when j<=i
            window_mask = dist >= self.window_size
            scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, S, D]
        out = out.transpose(1, 2).reshape(B, S, D)
        out = self.out_proj(out)

        if return_attention:
            return out, attn
        return out

    def clear_runtime_caches(self):
        self._rope_cache.clear()


class LinearAttention(nn.Module):
    """支持分块长序列的因果线性注意力。"""

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
        pos = torch.arange(
            start_idx,
            start_idx + q.size(2),
            device=q.device,
            dtype=torch.float32,
        )
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

        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1
        )
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
        kv_state = torch.zeros(
            batch_size, heads, head_dim, head_dim, device=q.device, dtype=q.dtype
        )
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
                rescale = torch.exp(torch.clamp(-decays * delta, min=-60.0, max=60.0)).to(
                    dtype=q.dtype
                )
                kv_state = kv_state * rescale
                k_state = k_state * rescale.squeeze(-1)
                state_anchor = start

            q_chunk, k_chunk = self._apply_alibi_linear_scaling(
                q_chunk,
                k_chunk,
                start_idx=start,
                anchor_idx=state_anchor,
            )

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

    def forward_incremental(
        self,
        x_new: torch.Tensor,
        kv_state: torch.Tensor,
        k_state: torch.Tensor,
        state_anchor: int,
        start_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch_size = x_new.size(0)
        qkv = (
            self.to_qkv(x_new)
            .reshape(batch_size, 1, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.position_encoding == "rope":
            rot_dim = (self.head_dim // 2) * 2
            if rot_dim >= 2:
                sin, cos = self._get_rope_trig(start_idx + 1, q.device, q.dtype, rot_dim)
                sin = sin[:, :, start_idx:start_idx+1, :]
                cos = cos[:, :, start_idx:start_idx+1, :]

                def _rotate_half_single(x):
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

                q, k = _rotate_half_single(q), _rotate_half_single(k)

        q = F.elu(q) + 1
        k = F.elu(k) + 1

        new_anchor = state_anchor
        if self.position_encoding == "alibi" and start_idx != state_anchor:
            decays = self.alibi_decays.to(device=q.device, dtype=torch.float32)
            delta = float(start_idx - state_anchor)
            rescale = torch.exp(torch.clamp(-decays * delta, min=-60.0, max=60.0)).to(
                dtype=q.dtype
            )
            kv_state = kv_state * rescale
            k_state = k_state * rescale.squeeze(-1)
            new_anchor = start_idx

        if self.position_encoding == "alibi":
            q, k = self._apply_alibi_linear_scaling(q, k, start_idx=start_idx, anchor_idx=new_anchor)

        kv_new = torch.einsum("bhnd,bhne->bhnde", k, v).squeeze(2)
        new_kv_state = kv_state + kv_new
        new_k_state = k_state + k.squeeze(2)

        out = torch.einsum("bhnd,bhde->bhne", q, new_kv_state)
        denom = torch.einsum("bhnd,bhd->bhn", q, new_k_state).unsqueeze(-1) + 1e-6
        global_out = out / denom

        out_final = self.to_out(global_out.transpose(1, 2).reshape(batch_size, 1, -1))

        return out_final, new_kv_state, new_k_state, new_anchor

    def forward_with_cache(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch_size, seq_len, _ = x.shape
        qkv = (
            self.to_qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.position_encoding == "rope":
            q, k = self._apply_rope(q, k)

        q_elu = F.elu(q) + 1
        k_elu = F.elu(k) + 1

        out = torch.zeros_like(v)
        kv_state = torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            device=q.device, dtype=q.dtype
        )
        k_state = torch.zeros(batch_size, self.num_heads, self.head_dim, device=q.device, dtype=q.dtype)
        state_anchor = 0

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            q_chunk = q_elu[:, :, start:end]
            k_chunk = k_elu[:, :, start:end]
            v_chunk = v[:, :, start:end]

            if self.position_encoding == "alibi" and start != state_anchor:
                decays = self.alibi_decays.to(device=q.device, dtype=torch.float32)
                delta = float(start - state_anchor)
                rescale = torch.exp(torch.clamp(-decays * delta, min=-60.0, max=60.0)).to(
                    dtype=q.dtype
                )
                kv_state = kv_state * rescale
                k_state = k_state * rescale.squeeze(-1)
                state_anchor = start

            q_chunk_scaled, k_chunk_scaled = self._apply_alibi_linear_scaling(
                q_chunk, k_chunk, start_idx=start, anchor_idx=state_anchor
            )

            kv_chunk = torch.einsum("bhnd,bhne->bhnde", k_chunk_scaled, v_chunk)
            kv_cum = torch.cumsum(kv_chunk, dim=2) + kv_state.unsqueeze(2)
            k_cum = torch.cumsum(k_chunk_scaled, dim=2) + k_state.unsqueeze(2)

            out_chunk = torch.einsum("bhnd,bhnde->bhne", q_chunk_scaled, kv_cum)
            denom = torch.einsum("bhnd,bhnd->bhn", q_chunk_scaled, k_cum).unsqueeze(-1) + 1e-6
            out[:, :, start:end] = out_chunk / denom

            kv_state = kv_cum[:, :, -1]
            k_state = k_cum[:, :, -1]

        global_out = out

        if self.local_windows and seq_len <= self.local_attention_max_seq:
            local_outputs = self._local_multi_scale_attention(q, k, v)
            if local_outputs:
                if self.multi_scale_fusion == "concat" and self.multi_scale_out is not None:
                    global_out = self.multi_scale_out(torch.cat([global_out] + local_outputs, dim=-1))
                else:
                    gates = F.softmax(self.scale_gates[: 1 + len(local_outputs)], dim=0).to(dtype=global_out.dtype)
                    mixed = gates[0] * global_out
                    for idx, local_out in enumerate(local_outputs):
                        mixed = mixed + gates[idx + 1] * local_out
                    global_out = mixed

        final_out = self.to_out(global_out.transpose(1, 2).reshape(batch_size, seq_len, -1))

        return final_out, kv_state, k_state, seq_len


class RingAttentionBlock(nn.Module):
    """Residual attention block — uses DSA (SparseAttention) with GQA.

    Pre-LN: LayerNorm → Attention → residual → LayerNorm → FFN → residual
    """

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


class LatentReasoner(nn.Module):
    """并行 Latent Reasoner，替代 GRU ThinkingLayer。

    使用 learnable latent tokens 做 cross-attention + self-attention，
    将推理从时间递推改为 token 空间并行计算。
    """

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
                    embed_dim=dim,
                    num_heads=max(1, int(cross_heads)),
                    batch_first=True,
                    dropout=dropout,
                ),
                "self_attn_norm": nn.LayerNorm(dim),
                "self_attn": nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=max(1, int(cross_heads)),
                    batch_first=True,
                    dropout=dropout,
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
        """前向传播。

        Args:
            x: 输入 [batch, seq, dim]

        Returns:
            output: [batch, seq, dim]
            latent_states: [num_layers, batch, num_latent_tokens, dim] 用于 regularization
        """
        batch_size, seq_len, _ = x.shape
        latent = self.latent_tokens.expand(batch_size, -1, -1)

        latent_states = []

        for layer in self.layers:
            latent_normed = layer["latent_norm"](latent)
            history_normed = layer["cross_attn_norm"](x)

            cross_out, _ = layer["cross_attn"](
                latent_normed, history_normed, history_normed,
                need_weights=False,
            )
            latent = latent + cross_out

            self_normed = layer["self_attn_norm"](latent)
            self_out, _ = layer["self_attn"](
                self_normed, self_normed, self_normed,
                need_weights=False,
            )
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

    def forward_incremental(self, x_new):
        """Incremental forward for single new token (inference only).

        For incremental inference, we run the full latent reasoner on the
        single-token input. Since latent tokens are learned (not sequential),
        this is fully parallel and efficient.
        """
        return self.forward(x_new)


class HorizonDecoder(nn.Module):
    """Horizon Decoder: 30 个 future query tokens 并行预测未来 30 天。

    每个 query token 带 horizon embedding + future calendar embedding，
    通过 cross-attend history memory 和 causal self-attend 交互，
    一次 forward 输出整个 30 天 future block。
    """

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
                    embed_dim=dim,
                    num_heads=max(1, int(num_heads)),
                    batch_first=True,
                    dropout=dropout,
                ),
                "self_norm": nn.LayerNorm(dim),
                "self_attn": nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=max(1, int(num_heads)),
                    batch_first=True,
                    dropout=dropout,
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

        self.norm = nn.LayerNorm(dim)
        self.horizon_head_coarse = nn.Linear(dim, vocab_size_coarse)
        self.coarse_to_fine = nn.Linear(dim, dim)
        self.fine_gate = nn.Linear(dim * 2, dim)
        self.fine_norm = nn.LayerNorm(dim)
        self.horizon_head_fine = nn.Linear(dim, vocab_size_fine)

        causal_mask = torch.triu(
            torch.ones(self.horizon_tokens, self.horizon_tokens, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, history_hidden, future_day=None, future_month=None):
        """前向传播。

        Args:
            history_hidden: [batch, seq_len, dim] history encoder output
            future_day: [batch, horizon_tokens] optional future day features
            future_month: [batch, horizon_tokens] optional future month features

        Returns:
            logits_coarse: [batch, horizon_tokens, vocab_coarse]
            logits_fine: [batch, horizon_tokens, vocab_fine]
        """
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
            cross_out, _ = layer["cross_attn"](
                q_normed, kv_normed, kv_normed,
                need_weights=False,
            )
            queries = queries + cross_out

            self_normed = layer["self_norm"](queries)
            self_out, _ = layer["self_attn"](
                self_normed, self_normed, self_normed,
                attn_mask=self.causal_mask,
                need_weights=False,
            )
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
    """Kronos reasoning model with DSA (Differential Sparse Attention) + GQA.

    Architecture:
    A. History Encoder: SparseAttention blocks with per-layer window config
    B. Latent Reasoner: learnable latent tokens
    C. Horizon Decoder: 30 future query tokens (parallel prediction)
    D. RevIN: reversible instance normalization

    DSA layer pattern (4 layers):
      Layer 0 → full attention    (global dependencies)
      Layer 1 → window=512        (local patterns)
      Layer 2 → window=512        (local patterns)
      Layer 3 → full attention    (global integration)
    """

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
        **legacy_kwargs,  # silently accept removed params (horizon_*, revin_*, etc.)
    ):
        super().__init__()
        self.position_encoding = str(position_encoding).lower()
        self.max_len = int(max_len)
        self.use_revin = use_revin
        self.vocab_size_coarse = int(vocab_size_coarse)
        self.vocab_size_fine = int(vocab_size_fine)
        self.depth = depth

        # DSA per-layer window config
        if dsa_windows is None:
            # Default: [full, 512, 512, full] repeated/downsampled to match depth
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
            dim=dim,
            num_latent_tokens=num_latent_tokens,
            depth=latent_reasoner_depth,
            cross_heads=latent_cross_heads,
            dropout=dropout,
        )

        self.horizon_decoder = None  # removed — LLM-style next-token only

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
                embed_dim=dim,
                num_heads=max(1, heads // 2),
                batch_first=True,
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

        # ── Value Head: predicts expected path_mape from hidden state ──
        value_hidden = max(1, dim // 4)
        self.value_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, value_hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(value_hidden, 1),
        )

        # ── Plan Head: predicts cumulative returns at sub-goal positions (3 steps) ──
        plan_hidden = max(1, dim // 2)
        self.plan_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, plan_hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(plan_hidden, 3),  # cum return at step 3, 6, 10
        )

        # ── Error Head: predicts per-step error contribution ──
        error_hidden = max(1, dim // 4)
        self.error_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, error_hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(error_hidden, 1),  # predicted path_mape contribution
        )

    def _compute_embedding(
        self,
        idx_coarse,
        idx_fine,
        t_min,
        t_day,
        t_month,
        t_year,
    ):
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
        attn_out, _ = self.factor_cross_attn(
            factor, normed_x, normed_x,
            need_weights=False,
        )
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

    def compute_direction_logits_at_positions(self, x, latent_states, start=0, end=-1):
        """Compute 3-class direction logits at arbitrary position range.

        Phase 8-2: Enables per-step direction classification during STAR-CAST
        training. Unlike _compute_direction_logits (last-only), this handles
        arbitrary [start:end) slices for multi-step simultaneous training.

        Args:
            x: [B, S, dim] hidden states from _run_post_backbone
            latent_states: [L, B, N, dim] latent reasoner states, or None
            start: first position index (default 0)
            end: last position index (default -1 means seq_len)

        Returns:
            dir_logits: [B, H, 3] where H = end - start
        """
        if end < 0:
            end = x.size(1) + end + 1  # -1 -> seq_len
        h = x[:, start:end, :]  # [B, H, dim]
        B, H, D = h.shape

        if latent_states is not None and latent_states.ndim == 4:
            latent_mean = latent_states[-1].mean(dim=1)  # [B, D]
            direction_state = h + latent_mean.unsqueeze(1)  # [B, H, D]
        else:
            direction_state = h

        flat = direction_state.reshape(B * H, D)
        logits = self.direction_head(flat)  # [B*H, 3]
        return logits.view(B, H, 3)

    def forward(
        self,
        idx_coarse,
        idx_fine,
        t_min,
        t_day,
        t_month,
        t_year,
        return_attention=False,
        last_only=False,
        return_hidden=False,
        neftune_alpha=0.0,
    ):
        _, seq_len = idx_coarse.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max length {self.max_len}"
            )

        x = self._compute_embedding(
            idx_coarse, idx_fine, t_min, t_day, t_month, t_year
        )

        # NEFTune: inject scaled uniform noise to embedding output.
        # Applied whenever neftune_alpha > 0, regardless of grad mode —
        # during STAR-CAST exploration (no_grad) we still want noisy embeddings.
        if neftune_alpha > 0:
            B, L, D = x.shape
            noise = (torch.rand_like(x) * 2 - 1) * (float(neftune_alpha) / math.sqrt(L * D))
            x = x + noise

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

    def predict_value(self, hidden_state):
        """Predict expected path_mape from hidden state (single vector per sample)."""
        return self.value_head(hidden_state).squeeze(-1)

    def predict_plan(self, hidden_state):
        """Predict cumulative log-returns at 3 sub-goal positions from hidden state."""
        return self.plan_head(hidden_state)

    def predict_error(self, hidden_state):
        """Predict per-step error contribution from hidden state."""
        return self.error_head(hidden_state).squeeze(-1)

    def forward_direction(
        self,
        idx_coarse,
        idx_fine,
        t_min,
        t_day,
        t_month,
        t_year,
        return_token_logits=False,
    ):
        _, seq_len = idx_coarse.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max length {self.max_len}"
            )

        x = self._compute_embedding(
            idx_coarse, idx_fine, t_min, t_day, t_month, t_year
        )

        for block in self.blocks:
            x = block(x)

        x, latent_states, logits_coarse, logits_fine = self._run_post_backbone(x)
        direction_logits = self._compute_direction_logits(x, latent_states)

        if return_token_logits:
            return direction_logits, logits_coarse, logits_fine, latent_states
        return direction_logits, latent_states

    def forward_horizon(
        self,
        idx_coarse,
        idx_fine,
        t_min,
        t_day,
        t_month,
        t_year,
        future_day=None,
        future_month=None,
    ):
        """Forward pass that also produces horizon predictions.

        Returns:
            logits_coarse: causal logits [batch, seq, vocab_coarse]
            logits_fine: causal logits [batch, seq, vocab_fine]
            latent_states: latent reasoner states
            horizon_logits_coarse: [batch, horizon_tokens, vocab_coarse]
            horizon_logits_fine: [batch, horizon_tokens, vocab_fine]
        """
        _, seq_len = idx_coarse.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max length {self.max_len}"
            )

        x = self._compute_embedding(
            idx_coarse, idx_fine, t_min, t_day, t_month, t_year
        )

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

    def forward_with_cache(
        self,
        idx_coarse,
        idx_fine,
        t_min,
        t_day,
        t_month,
        t_year,
    ):
        batch_size, seq_len = idx_coarse.shape

        x = self._compute_embedding(
            idx_coarse, idx_fine, t_min, t_day, t_month, t_year
        )

        cache = KVCache()
        cache.device = x.device
        cache.dtype = x.dtype
        cache.prefix_len = seq_len

        for layer_idx, block in enumerate(self.blocks):
            x, kv_state, k_state, state_anchor = block.forward_with_cache(x)
            cache.linear_attn_states[layer_idx] = (kv_state, k_state, state_anchor)

        cache.prefix_hidden = x.clone()

        _, latent_states, logits_coarse, logits_fine = self._run_post_backbone(cache.prefix_hidden)

        return logits_coarse, logits_fine, latent_states, cache

    def forward_incremental(
        self,
        idx_coarse_new,
        idx_fine_new,
        t_min_new,
        t_day_new,
        t_month_new,
        t_year_new,
        cache,
    ):
        start_pos = cache.prefix_len

        idx_coarse_new = torch.clamp(idx_coarse_new, 0, self.vocab_size_coarse - 1)
        idx_fine_new = torch.clamp(idx_fine_new, 0, self.vocab_size_fine - 1)
        t_min_new = torch.clamp(t_min_new, 0, 239)
        t_day_new = torch.clamp(t_day_new, 0, 30)
        t_month_new = torch.clamp(t_month_new, 0, 11)
        t_year_new = torch.clamp(t_year_new, 0, 99)

        x_new = self.token_emb_coarse(idx_coarse_new) + self.token_emb_fine(idx_fine_new)
        x_new = (
            x_new
            + self.time_emb_min(t_min_new)
            + self.time_emb_day(t_day_new)
            + self.time_emb_month(t_month_new)
            + self.time_emb_year(t_year_new)
        )
        if self.pos_emb is not None:
            seq_len_new = idx_coarse_new.shape[1]
            x_new = x_new + self.pos_emb[:, start_pos:start_pos + seq_len_new, :]

        for layer_idx, block in enumerate(self.blocks):
            kv_state, k_state, state_anchor = cache.linear_attn_states[layer_idx]
            x_new, new_kv_state, new_k_state, new_anchor = block.forward_incremental(
                x_new, kv_state, k_state, state_anchor, start_pos
            )
            cache.linear_attn_states[layer_idx] = (new_kv_state, new_k_state, new_anchor)

        prefix_hidden = x_new
        if cache.prefix_hidden is not None:
            prefix_hidden = torch.cat([cache.prefix_hidden, x_new], dim=1)

        cache.prefix_hidden = prefix_hidden
        cache.prefix_len = start_pos + idx_coarse_new.shape[1]

        _, latent_states, logits_coarse, logits_fine = self._run_post_backbone(prefix_hidden)

        return logits_coarse[:, -1:, :], logits_fine[:, -1:, :], latent_states, cache
