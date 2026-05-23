from config import TokenizerConfig


def build_tokenizer_kwargs(config_dict=None):
    cfg = config_dict or {}
    return {
        "input_dim": cfg.get("input_dim", TokenizerConfig.input_dim),
        "hidden_dim": cfg.get("hidden_dim", TokenizerConfig.hidden_dim),
        "embedding_dim": cfg.get("embedding_dim", TokenizerConfig.embedding_dim),
        "num_quantizers": cfg.get("num_quantizers", TokenizerConfig.num_quantizers),
        "bits_per_quantizer": cfg.get(
            "bits_per_quantizer", getattr(TokenizerConfig, "bits_per_quantizer", 10)
        ),
        "commitment_cost": cfg.get(
            "bsq_commitment_cost", getattr(TokenizerConfig, "bsq_commitment_cost", 0.05)
        ),
        "entropy_weight": cfg.get(
            "bsq_entropy_weight", getattr(TokenizerConfig, "bsq_entropy_weight", 0.05)
        ),
    }


def export_tokenizer_config():
    return {
        "input_dim": TokenizerConfig.input_dim,
        "hidden_dim": TokenizerConfig.hidden_dim,
        "embedding_dim": TokenizerConfig.embedding_dim,
        "num_quantizers": TokenizerConfig.num_quantizers,
        "bits_per_quantizer": getattr(TokenizerConfig, "bits_per_quantizer", 10),
        "bsq_commitment_cost": getattr(TokenizerConfig, "bsq_commitment_cost", 0.05),
        "bsq_entropy_weight": getattr(TokenizerConfig, "bsq_entropy_weight", 0.05),
    }
