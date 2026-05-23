import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TokenizerConfig


class BSQQuantizer(nn.Module):
    """Binary Spherical Quantization — implicit-codebook quantizer.

    Projects a latent vector onto the unit sphere, then through learnable
    hyperplanes to produce a k-bit binary code b ∈ {−1, 1}^k.  The
    vocabulary of 2^k codes is implicit — every code is reachable.
    """

    def __init__(self, embedding_dim, bits, commitment_cost=0.05, entropy_weight=0.05):
        super().__init__()
        self.bits = int(bits)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.entropy_weight = float(entropy_weight)

        self.project = nn.Linear(embedding_dim, self.bits, bias=False)
        self.decode_proj = nn.Linear(self.bits, embedding_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.project.weight)
        nn.init.orthogonal_(self.decode_proj.weight)

    @staticmethod
    def _bits_to_int(bits_01):
        k = bits_01.shape[-1]
        powers = 2 ** torch.arange(k, device=bits_01.device, dtype=bits_01.dtype)
        return (bits_01 * powers).sum(dim=-1)

    @staticmethod
    def _int_to_bits(indices, bits):
        mask = 2 ** torch.arange(bits, device=indices.device)
        return ((indices.unsqueeze(-1) & mask) != 0).to(dtype=torch.float32) * 2.0 - 1.0

    def forward(self, z):
        z_norm = F.normalize(z, dim=-1)
        logits = self.project(z_norm)
        b_hard = torch.sign(logits)
        b_soft = torch.tanh(logits)
        b = b_hard + b_soft - b_soft.detach()

        bits_01 = ((b_hard + 1.0) * 0.5).long().clamp(0, 1)
        indices = self._bits_to_int(bits_01)

        commit_loss = F.mse_loss(logits, b_hard.detach()) * self.commitment_cost
        codebook_loss = F.mse_loss(logits.detach(), b_soft) * self.commitment_cost

        ent_loss = torch.tensor(0.0, device=z.device)
        if self.entropy_weight > 0 and self.training:
            prob = torch.sigmoid(logits)
            ent = -(prob * torch.log(prob + 1e-10) + (1.0 - prob) * torch.log(1.0 - prob + 1e-10))
            ent_loss = -ent.mean() * self.entropy_weight

        quant_loss = commit_loss + codebook_loss + ent_loss
        return b, indices, quant_loss

    def quantize(self, z):
        z_norm = F.normalize(z, dim=-1)
        logits = self.project(z_norm)
        b_hard = torch.sign(logits)
        bits_01 = ((b_hard + 1.0) * 0.5).long().clamp(0, 1)
        indices = self._bits_to_int(bits_01)
        return indices

    def decode_ids(self, indices):
        b = self._int_to_bits(indices, self.bits).to(dtype=self.decode_proj.weight.dtype)
        return self.decode_proj(b)

    def vocab_size(self):
        return 1 << self.bits


class HierarchicalQuantizer(nn.Module):
    """BSQ-based hierarchical tokenizer for financial K-line sequences.

    2-level coarse→fine quantization:
      - encoder: MLP (input_dim → hidden_dim → embedding_dim)
      - BSQ coarse: k₁ bits → captures principal structure
      - BSQ fine:   k₂ bits → encodes residual detail
      - decoder: MLP (embedding_dim → hidden_dim → input_dim)

    Loss = L_coarse(recon from coarse-only) + L_fine(recon from full) + quant_loss
    """

    def __init__(
        self,
        input_dim=TokenizerConfig.input_dim,
        hidden_dim=TokenizerConfig.hidden_dim,
        embedding_dim=TokenizerConfig.embedding_dim,
        num_quantizers=TokenizerConfig.num_quantizers,
        bits_per_quantizer=None,
        commitment_cost=None,
        entropy_weight=None,
    ):
        super().__init__()
        self.num_quantizers = max(1, int(num_quantizers))
        self._embedding_dim = int(embedding_dim)

        _bits = (
            int(bits_per_quantizer)
            if bits_per_quantizer is not None
            else getattr(TokenizerConfig, "bits_per_quantizer", 10)
        )
        _commit = (
            float(commitment_cost)
            if commitment_cost is not None
            else getattr(TokenizerConfig, "bsq_commitment_cost", 0.05)
        )
        _ent = (
            float(entropy_weight)
            if entropy_weight is not None
            else getattr(TokenizerConfig, "bsq_entropy_weight", 0.05)
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self._embedding_dim),
            nn.LayerNorm(self._embedding_dim),
        )

        self.bsq_quantizers = nn.ModuleList([
            BSQQuantizer(self._embedding_dim, _bits, _commit, _ent)
            for _ in range(self.num_quantizers)
        ])
        self.bsq_coarse = self.bsq_quantizers[0]
        self.bsq_fine = self.bsq_quantizers[1] if self.num_quantizers > 1 else self.bsq_quantizers[0]

        self.decoder = nn.Sequential(
            nn.Linear(self._embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def _bsq_quantize_latent(self, z):
        residual = z
        z_q_total = torch.zeros_like(z)
        total_loss = z.new_zeros(())
        indices = []
        z_q_coarse = None

        for i, bsq in enumerate(self.bsq_quantizers):
            b, idx, q_loss = bsq(residual)
            z_q = bsq.decode_proj(b)
            total_loss = total_loss + q_loss
            z_q_total = z_q_total + z_q
            residual = residual - z_q
            indices.append(idx)
            if i == 0:
                z_q_coarse = z_q

        return total_loss, z_q_total, indices, z_q_coarse

    def forward(self, x, return_all=False):
        z = self.encoder(x)
        quant_loss, z_q, indices, z_q_coarse = self._bsq_quantize_latent(z)
        x_recon_full = self.decoder(z_q)

        if return_all:
            return quant_loss, x_recon_full, [], indices

        recon_loss = F.mse_loss(x_recon_full, x)
        if z_q_coarse is not None and self.num_quantizers > 1:
            x_recon_coarse = self.decoder(z_q_coarse)
            recon_loss = recon_loss + F.mse_loss(x_recon_coarse, x)

        total_loss = quant_loss + recon_loss
        return total_loss, x_recon_full, indices

    def encode_all(self, x):
        z = self.encoder(x)
        residual = z
        indices = []
        for bsq in self.bsq_quantizers:
            idx = bsq.quantize(residual)
            z_q = bsq.decode_proj(bsq._int_to_bits(idx, bsq.bits).to(dtype=residual.dtype))
            residual = residual - z_q
            indices.append(idx.long())
        return torch.stack(indices, dim=-1)

    def encode(self, x):
        all_indices = self.encode_all(x)
        idx_c = all_indices[:, :, 0]
        idx_f = all_indices[:, :, 1] if self.num_quantizers > 1 else idx_c
        return idx_c, idx_f

    def decode_all(self, all_indices):
        if isinstance(all_indices, torch.Tensor):
            if all_indices.dim() != 3 or all_indices.size(-1) != self.num_quantizers:
                raise ValueError(
                    f"Expected indices tensor of shape [B, N, {self.num_quantizers}], "
                    f"got {tuple(all_indices.shape)}"
                )
            indices_per_level = [all_indices[:, :, i] for i in range(self.num_quantizers)]
        else:
            indices_per_level = list(all_indices)
            if len(indices_per_level) != self.num_quantizers:
                raise ValueError(
                    f"Expected {self.num_quantizers} levels, got {len(indices_per_level)}"
                )

        z_q = torch.zeros(
            indices_per_level[0].shape[0],
            indices_per_level[0].shape[1],
            self._embedding_dim,
            device=indices_per_level[0].device,
        )
        for indices, bsq in zip(indices_per_level, self.bsq_quantizers):
            z_q = z_q + bsq.decode_ids(indices)

        return self.decoder(z_q)

    def decode(self, idx_coarse, idx_fine):
        if self.num_quantizers == 1:
            all_indices = torch.stack([idx_coarse], dim=-1)
            return self.decode_all(all_indices)

        levels = [idx_coarse, idx_fine]
        for _ in range(max(0, self.num_quantizers - 2)):
            levels.append(torch.zeros_like(idx_coarse))
        all_indices = torch.stack(levels, dim=-1)
        return self.decode_all(all_indices)

    def codebook_stats(self):
        stats = {}
        for i, bsq in enumerate(self.bsq_quantizers):
            stats[f"level_{i}"] = {"type": "BSQ", "bits": bsq.bits, "vocab_size": 1 << bsq.bits}
        if "level_0" in stats:
            stats["coarse"] = stats["level_0"]
        if "level_1" in stats:
            stats["fine"] = stats["level_1"]
        return stats
