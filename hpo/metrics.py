"""BSQ tokenizer evaluation metrics.

BSQ (Binary Spherical Quantization) differs fundamentally from VQ-VAE:
- No explicit codebook → no codebook collapse
- Every 2^k binary code is always reachable by construction
- "Dead codes" = encoder diversity issue, not codebook update failure

Evaluation dimensions:
  1. Reconstruction quality  – MSE / MAE of encode→decode round-trip
  2. Bit-level health         – per-hyperplane activation probability & entropy
  3. Token-level diversity    – distribution of the 2^k integer codes
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_per_bit_stats(bits_tensor: torch.Tensor) -> dict:
    """Compute per-bit activation probability and entropy.

    Args:
        bits_tensor: [N, k] tensor of {0,1} binary codes.

    Returns:
        dict with per-bit p1, entropy, and aggregate stats.
    """
    k = bits_tensor.shape[1]
    per_bit = {}
    bit_entropies = []

    for i in range(k):
        p1 = bits_tensor[:, i].float().mean().item()
        if p1 <= 0.0 or p1 >= 1.0:
            bit_ent = 0.0
        else:
            bit_ent = -(p1 * math.log(p1) + (1.0 - p1) * math.log(1.0 - p1))

        per_bit[f"bit_{i}_p1"] = round(p1, 6)
        per_bit[f"bit_{i}_entropy"] = round(bit_ent, 6)
        bit_entropies.append(bit_ent)

    max_ent = math.log(2)  # ≈ 0.693

    return {
        "per_bit": per_bit,
        "bit_entropies": bit_entropies,
        "mean_bit_entropy": round(float(sum(bit_entropies)) / max(k, 1), 6),
        "effective_bits": sum(1 for e in bit_entropies if e > 0.3),
        "dead_bits": sum(1 for e in bit_entropies if e < 0.01),
        "max_bit_entropy": max_ent,
    }


def compute_token_distribution(token_ids: torch.Tensor, vocab_size: int) -> dict:
    """Compute token-level distribution statistics.

    Args:
        token_ids: [N] integer token IDs in [0, vocab_size).
        vocab_size: total number of possible tokens (2^k).

    Returns:
        dict with utilization, entropy, dead token count, top-K concentration.
    """
    counts = torch.bincount(token_ids, minlength=vocab_size).float()
    total = counts.sum()
    if total == 0:
        return {
            "vocab_size": vocab_size,
            "utilization": 0.0,
            "norm_entropy": 0.0,
            "raw_entropy": 0.0,
            "dead_tokens": vocab_size,
            "top20_concentration": 0.0,
        }

    probs = counts / total
    raw_entropy = -(probs * (probs + 1e-12).log()).sum().item()
    max_entropy = math.log(vocab_size)
    norm_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

    used = (counts > 0).sum().item()
    dead = vocab_size - used

    # Top 20% token concentration
    top_k = max(1, vocab_size // 5)
    top20_conc = probs.topk(top_k).values.sum().item()

    return {
        "vocab_size": vocab_size,
        "utilization": round(used / vocab_size, 6),
        "norm_entropy": round(norm_entropy, 6),
        "raw_entropy": round(raw_entropy, 6),
        "dead_tokens": dead,
        "top20_concentration": round(top20_conc, 6),
    }


def extract_bits_from_quantizer(quantizer, z_norm: torch.Tensor) -> torch.Tensor:
    """Extract {0,1} bits from a BSQ quantizer given normalized latent.

    Args:
        quantizer: BSQQuantizer module.
        z_norm: [N, embedding_dim] normalized latent vectors.

    Returns:
        [N, k] tensor of {0,1} bits.
    """
    logits = quantizer.project(z_norm)
    return ((torch.sign(logits) + 1.0) * 0.5).long()


def evaluate_tokenizer_bsq(tokenizer, dataloader, device, raw_data_dir: str | None = None) -> dict:
    """Full BSQ tokenizer evaluation.

    Runs one pass over the dataloader and computes:
      - Reconstruction MSE / MAE
      - Per-bit activation & entropy (coarse + fine)
      - Token distribution statistics (coarse + fine)

    If raw_data_dir is provided, saves paper-plot-ready raw arrays:
      - coarse_bits.npy:        [N, k_c] raw {0,1} bits
      - fine_bits.npy:          [N, k_f] raw {0,1} bits
      - coarse_token_ids.npy:   [N] integer token IDs
      - fine_token_ids.npy:     [N] integer token IDs
      - coarse_token_counts.npy:[vocab_size] frequency per token
      - fine_token_counts.npy:  [vocab_size] frequency per token
      - coarse_bit_p1.npy:      [k_c] P(bit_i=1)
      - fine_bit_p1.npy:        [k_f] P(bit_i=1)

    Args:
        tokenizer: HierarchicalQuantizer in eval mode.
        dataloader: DataLoader yielding (features, sector_ids, time_features, encodings).
        device: torch device.
        raw_data_dir: if set, save raw .npy arrays here.

    Returns:
        dict of metrics.
    """
    tokenizer.eval()

    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0

    all_bits_coarse = []
    all_bits_fine = []

    for batch_data in tqdm(dataloader, desc="BSQ eval"):
        data = batch_data[0].to(device)

        z = tokenizer.encoder(data)
        z_norm = F.normalize(z, dim=-1)

        # --- coarse bits ---
        bits_c = extract_bits_from_quantizer(tokenizer.bsq_coarse, z_norm)
        all_bits_coarse.append(bits_c.reshape(-1, bits_c.shape[-1]).cpu())

        # --- fine bits (if present) ---
        if tokenizer.num_quantizers > 1:
            b_c_hard = torch.sign(tokenizer.bsq_coarse.project(z_norm))
            z_q_c = tokenizer.bsq_coarse.decode_proj(b_c_hard)
            residual = z - z_q_c
            residual_norm = F.normalize(residual, dim=-1)
            bits_f = extract_bits_from_quantizer(tokenizer.bsq_fine, residual_norm)
            all_bits_fine.append(bits_f.reshape(-1, bits_f.shape[-1]).cpu())

        # --- reconstruction ---
        idx_c, idx_f = tokenizer.encode(data)
        x_recon = tokenizer.decode(idx_c, idx_f)
        total_mse += F.mse_loss(x_recon, data).item()
        total_mae += F.l1_loss(x_recon, data).item()
        num_batches += 1

    metrics = {
        "val_recon_mse": round(total_mse / max(num_batches, 1), 8),
        "val_recon_mae": round(total_mae / max(num_batches, 1), 8),
        "num_samples_evaluated": num_batches,
    }

    # --- bit-level stats & raw data ---
    if all_bits_coarse:
        bits_coarse = torch.cat(all_bits_coarse, dim=0)
        coarse_bit_stats = compute_per_bit_stats(bits_coarse)
        metrics["coarse"] = coarse_bit_stats

        # token-level from coarse
        k_c = bits_coarse.shape[1]
        powers_c = 2 ** torch.arange(k_c)
        token_ids_c = (bits_coarse * powers_c).sum(dim=-1)
        token_counts_c = torch.bincount(token_ids_c, minlength=1 << k_c)
        metrics["coarse"]["token_dist"] = compute_token_distribution(token_ids_c, 1 << k_c)

        if raw_data_dir is not None:
            os.makedirs(raw_data_dir, exist_ok=True)
            import numpy as np
            np.save(os.path.join(raw_data_dir, "coarse_bits.npy"), bits_coarse.numpy())
            np.save(os.path.join(raw_data_dir, "coarse_token_ids.npy"), token_ids_c.numpy())
            np.save(os.path.join(raw_data_dir, "coarse_token_counts.npy"), token_counts_c.numpy())
            np.save(os.path.join(raw_data_dir, "coarse_bit_p1.npy"),
                    np.array([coarse_bit_stats["per_bit"][f"bit_{i}_p1"]
                              for i in range(k_c)], dtype=np.float32))

    if all_bits_fine:
        bits_fine = torch.cat(all_bits_fine, dim=0)
        fine_bit_stats = compute_per_bit_stats(bits_fine)
        metrics["fine"] = fine_bit_stats

        k_f = bits_fine.shape[1]
        powers_f = 2 ** torch.arange(k_f)
        token_ids_f = (bits_fine * powers_f).sum(dim=-1)
        token_counts_f = torch.bincount(token_ids_f, minlength=1 << k_f)
        metrics["fine"]["token_dist"] = compute_token_distribution(token_ids_f, 1 << k_f)

        if raw_data_dir is not None:
            os.makedirs(raw_data_dir, exist_ok=True)
            import numpy as np
            np.save(os.path.join(raw_data_dir, "fine_bits.npy"), bits_fine.numpy())
            np.save(os.path.join(raw_data_dir, "fine_token_ids.npy"), token_ids_f.numpy())
            np.save(os.path.join(raw_data_dir, "fine_token_counts.npy"), token_counts_f.numpy())
            np.save(os.path.join(raw_data_dir, "fine_bit_p1.npy"),
                    np.array([fine_bit_stats["per_bit"][f"bit_{i}_p1"]
                              for i in range(k_f)], dtype=np.float32))

    return metrics


def metrics_health_check(metrics: dict) -> tuple[bool, list[str]]:
    """Check if BSQ tokenizer metrics pass basic health thresholds.

    Returns:
        (passed, warnings) – True if healthy, plus list of warning messages.
    """
    warnings = []

    # Coarse effective bits
    coarse_eff = metrics.get("coarse", {}).get("effective_bits")
    if coarse_eff is not None:
        dead = metrics["coarse"].get("dead_bits", 0)
        if dead > 0:
            warnings.append(f"coarse dead_bits={dead} (some hyperplanes not working)")
        total_bits = len(metrics["coarse"].get("bit_entropies", []))
        if total_bits > 0 and coarse_eff < total_bits * 0.5:
            warnings.append(
                f"coarse effective_bits={coarse_eff}/{total_bits} (<50% bits active)"
            )

    # Fine effective bits
    fine_eff = metrics.get("fine", {}).get("effective_bits")
    if fine_eff is not None:
        dead = metrics["fine"].get("dead_bits", 0)
        if dead > 0:
            warnings.append(f"fine dead_bits={dead}")
        total_bits = len(metrics["fine"].get("bit_entropies", []))
        if total_bits > 0 and fine_eff < total_bits * 0.5:
            warnings.append(
                f"fine effective_bits={fine_eff}/{total_bits} (<50% bits active)"
            )

    # Token-level: >30% dead tokens is suspicious
    for level in ["coarse", "fine"]:
        tok = metrics.get(level, {}).get("token_dist", {})
        dead_t = tok.get("dead_tokens", 0)
        vocab = tok.get("vocab_size", 1)
        if vocab > 0 and dead_t / vocab > 0.5:
            warnings.append(f"{level} dead_tokens={dead_t}/{vocab} (>50% unused)")

    passed = len(warnings) == 0
    return passed, warnings


def format_metrics_summary(metrics: dict) -> str:
    """Format metrics as a human-readable string for logging."""
    lines = []
    lines.append(f"  recon_mse={metrics['val_recon_mse']:.6f}  recon_mae={metrics['val_recon_mae']:.6f}")

    for level in ["coarse", "fine"]:
        if level not in metrics:
            continue
        s = metrics[level]
        lines.append(
            f"  [{level}] effective_bits={s['effective_bits']}/{len(s.get('bit_entropies',[]))}  "
            f"dead_bits={s['dead_bits']}  mean_bit_ent={s['mean_bit_entropy']:.4f}"
        )
        tok = s.get("token_dist", {})
        if tok:
            lines.append(
                f"  [{level}] utilization={tok['utilization']:.4f}  "
                f"norm_ent={tok['norm_entropy']:.4f}  "
                f"dead_tokens={tok['dead_tokens']}/{tok['vocab_size']}  "
                f"top20_conc={tok['top20_concentration']:.4f}"
            )

    return "\n".join(lines)


def save_metrics(metrics: dict, path: str):
    """Save metrics dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
