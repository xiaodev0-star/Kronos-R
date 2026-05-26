# -*- coding: utf-8 -*-
"""项目配置。"""

import json
import os


class DataConfig:
    """数据集处理配置。"""

    data_dir = "dataset/"
    seq_len = 1024
    demo_ratio = 0.05
    demo_days = 250
    max_stocks = None
    train_val_split = 0.9
    outlier_sigma = 5
    stride_ratio = 0.5
    feature_cols = ["log_ret", "log_high", "log_low", "log_open", "log_vol", "log_amt"]
    base_year = 2010
    random_seed = 42


class TokenizerConfig:
    """BSQ Tokenizer config — Phase 2 trial 015 optimized."""

    input_dim = 6
    hidden_dim = 192              # Phase 2
    embedding_dim = 48            # Phase 2
    num_quantizers = 2

    bits_per_quantizer = 10       # Phase 1
    bsq_commitment_cost = 0.194   # Phase 2
    bsq_entropy_weight = 0.01     # Phase 2 (fixed, r=0.001)

    epochs = 200
    random_seed = 42
    learning_rate = 1e-4
    batch_size = 512
    grad_clip = 1.0
    save_path = "checkpoints/tokenizer.pt"


class ModelConfig:
    """Kronos reasoning model config — Phase 1-4 optimized."""

    # ── Backbone (Phase 3 trial 047) ──
    dim = 384
    depth = 3
    heads = 4
    num_kv_heads = 1           # GQA 4:1
    dsa_windows = [None, 512, 512]  # DSA: full, window, window

    # ── Latent Reasoner (Phase 4: defaults confirmed) ──
    num_latent_tokens = 16
    latent_reasoner_depth = 4
    latent_cross_heads = 2     # Phase 4: 2 > 4

    # ── Position encoding ──
    position_encoding = "rope"
    rope_base = 10000.0
    alibi_decay_base = 0.02

    # ── Regularization ──
    max_len = 10000
    dropout = 0.1323

    # ── Vocabulary (Phase 1: bits=10) ──
    vocab_size_coarse = 1 << getattr(TokenizerConfig, "bits_per_quantizer", 10)
    vocab_size_fine   = 1 << getattr(TokenizerConfig, "bits_per_quantizer", 10)

    # ── Horizon Decoder (not used in current config) ──
    horizon_tokens = 0
    horizon_decoder_depth = 0
    horizon_decoder_heads = 0

    # ── RevIN ──
    revin_affine = False
    revin_eps = 1e-5

    # ── Ablated / removed ──
    use_revin = False          # Phase 4 ablation: delta=0.004pp
    num_factor_tokens = 0      # Phase 4: factor=0 better, delete


class TrainingConfig:
    """Kronos 推理训练配置。"""

    epochs = 30
    early_stop_patience = 5
    max_train_updates = 0
    random_seed = 42
    deterministic = False

    batch_size = 32
    accumulation_steps = 1
    num_workers = 0
    pin_memory = True
    persistent_workers = False
    prefetch_factor = 4
    use_cuda_prefetch = True

    learning_rate = 1.08e-3     # Phase 3 trial 047
    weight_decay = 1.4e-5       # Phase 3 trial 047
    grad_clip = 0.3
    use_tf32 = True
    optimizer_use_fused = True
    optimizer_use_foreach = True

    # 基础训练参数。
    train_local_attention_max_seq = 2048

    # Latent 正则参数。
    diversity_weight = 0.6
    collapse_weight = 0.0005

    # 训练优化参数。
    use_gradient_checkpointing = True

    scheduler_T_max = epochs
    scheduler_by_updates = False
    scheduler_eta_min = 3.797544798678408e-08

    save_dir = "checkpoints"
    tokenizer_path = "checkpoints/tokenizer.pt"
    base_model_path = "checkpoints/base_model.pt"
    use_memmap_cache = False
    memmap_cache_dir = "dataset_cache"


class EvaluationConfig:
    """评估与预测配置。"""

    num_stocks = 100
    pred_steps = 30
    temperature = 0.5
    use_sampling = True
    eval_batch_size = 0
    eval_use_amp = True
    eval_amp_dtype = "bfloat16"
    eval_use_tf32 = True
    eval_non_blocking_transfer = True
    enable_stock_sft = False
    output_dir = "outputs"


class PathConfig:
    """路径配置。"""

    checkpoint_dir = "checkpoints"
    output_dir = "outputs"
    cache_file = "dataset_train.pt"


class LoRAConfig:
    """LoRA 微调配置。"""

    enabled = False
    random_seed = 42
    rank = 8
    alpha = 16
    dropout = 0.05
    target_keywords = ("to_qkv", "to_out", "head_coarse", "head_fine", "horizon_head_coarse", "horizon_head_fine")
    save_dir = "checkpoints/lora"


class PostTrainDAConfig:
    """Direction-accuracy EXPO post-train defaults (Stage B2 - EXPO)."""

    random_seed = 42
    deterministic = False
    output_dir = "checkpoints/post_train_da"
    checkpoint_path = TrainingConfig.base_model_path
    cache_path = "dataset_train.pt"
    val_cache_path = "dataset_val.pt"
    save_name = "direction_expo.pt"
    save_epoch_checkpoints = True

    epochs = 10
    batch_size = 4
    accumulation_steps = 1
    num_workers = 0
    learning_rate = 1e-4
    backbone_learning_rate = 1e-4
    weight_decay = 1e-4
    grad_clip = 1.0
    max_train_updates = 0
    progress_interval = 10
    use_amp = True
    amp_dtype = "bfloat16"
    use_tf32 = True

    freeze_backbone = False
    trainable_scope = "all"
    train_lora = False
    lora_rank = LoRAConfig.rank
    lora_alpha = LoRAConfig.alpha
    lora_dropout = LoRAConfig.dropout
    lora_target_keywords = LoRAConfig.target_keywords

    sample_stride = 1
    val_sample_stride = 1
    max_train_samples = 0
    max_val_samples = 0
    max_eval_items = 0
    max_stocks = 0
    cache_val_ratio = 0.1
    eval_demo_days = 10
    demo_score_weight = 0.5

    label_mode = "rolling_vol"
    epsilon_scale = 0.5
    fixed_epsilon = 0.0
    rolling_vol_window = 20
    z_threshold = 0.1
    min_epsilon = 1e-5
    flat_policy = "ignore"

    expo_temperature = 1.0
    expo_num_candidates = 192
    expo_reference_weight = 0.6
    expo_score_margin = 0.05
    expo_direction_bonus = 1.0
    expo_error_weight = 0.25
    expo_include_gold = True
    expo_keep_auxiliary = True

    token_ce_weight = 0.0
    kl_weight = 0.05
    latent_weight = 0.0
    eval_confidence_threshold = 0.55
    eval_margin_threshold = 0.0


class PostTrainRolloutConfig:
    """Rollout post-train defaults for strict multi-step autoregressive use."""

    random_seed = 42
    deterministic = False
    output_dir = "checkpoints/post_train_rollout"
    checkpoint_path = TrainingConfig.base_model_path
    save_name = "rollout_scheduled.pt"
    save_epoch_checkpoints = True

    prefix_len = 1023
    horizon = 10
    stride_ratio = DataConfig.stride_ratio
    cache_dir = "posttrain/rollout/cache"
    cache_rebuild = False
    max_stocks = 0
    max_train_samples = 0
    max_val_samples = 0

    epochs = 3
    batch_size = 2
    eval_batch_size = 8
    accumulation_steps = 1
    num_workers = 0
    learning_rate = 2e-5
    weight_decay = 1e-4
    grad_clip = 0.5
    max_train_updates = 0
    progress_interval = 20

    rollout_ratio_start = 0.50
    rollout_ratio_end = 0.90
    anchor_weight = 0.20
    kl_weight = 0.02
    numeric_mape_weight = 25.0
    numeric_top_k = 16
    numeric_soft_ce_weight = 0.0
    numeric_soft_ce_top_k = 8
    numeric_soft_ce_temp = 0.005
    step_weight_gamma = 0.50
    use_sampling = False
    sampling_temperature = 1.0

    freeze_backbone = False
    trainable_scope = "all"
    train_lora = False
    use_gradient_checkpointing = True
    use_amp = True
    amp_dtype = "bfloat16"
    use_tf32 = True
    zero_sector_ids = not bool(getattr(TrainingConfig, "use_sector_ids", False))

    mape_eps = 1e-4


class PostTrainCIConfig:
    """Confidence-Interval post-training defaults.

    Trains the model to produce well-calibrated, sharp prediction
    distributions that yield narrow confidence intervals with correct
    coverage for multi-step autoregressive return predictions.

    Two complementary loss terms:

      Concentration loss  (differentiable)
          L_conc = E_{p}[ |r - y| ] — expected absolute error under the
          model's predicted distribution.  Sharpens the distribution
          around the true value → narrower intervals.

      Interval-score surrogate
          Smooth approximation of the Gneiting-Raftery interval score.
          Penalises wide intervals and missed coverage at the nominal
          confidence level.
    """

    random_seed = 42
    deterministic = False
    output_dir = "checkpoints/post_train_ci"
    checkpoint_path = TrainingConfig.base_model_path
    save_name = "ci_model.pt"
    save_epoch_checkpoints = True

    # ── Data (reuses rollout cache format) ──
    prefix_len = 1023
    horizon = 10
    stride_ratio = DataConfig.stride_ratio
    cache_dir = "posttrain/rollout/cache"
    cache_rebuild = False
    max_stocks = 0
    max_train_samples = 0
    max_val_samples = 0

    # ── CI parameters ──
    ci_confidence_level = 0.80       # nominal confidence (80 %)
    ci_top_k = 32                    # top-K tokens for distribution estimation
    ci_num_samples = 64              # for sampling-based CI (Idea 1)

    # ── Loss weights ──
    concentration_weight = 1.0       # L_conc multiplier
    interval_score_weight = 0.3      # interval-score guidance multiplier
    kl_weight = 0.02                 # KL to reference model

    # ── Training schedule ──
    epochs = 3
    batch_size = 2
    eval_batch_size = 8
    accumulation_steps = 1
    num_workers = 0
    learning_rate = 2e-5
    weight_decay = 1e-4
    grad_clip = 0.5
    max_train_updates = 0
    progress_interval = 20

    # ── Self-rollout curriculum ──
    rollout_ratio_start = 0.70
    rollout_ratio_end = 0.95
    step_weight_gamma = 0.50

    # ── Optimisation ──
    freeze_backbone = False
    trainable_scope = "all"
    use_gradient_checkpointing = True
    use_amp = True
    amp_dtype = "bfloat16"
    use_tf32 = True

    mape_eps = 1e-4


class PostTrainStarCastConfig:
    """Phase 8: STAR-CAST engine — online self-training with noisy exploration,
    oracle filtering, and dual-engine (continuous + discrete) fine-tuning.

    Core idea: at each training step, the model explores N parallel trajectories
    with NEFTune noise + temperature sampling, the Oracle picks the best one
    (correct direction + minimal path error), and the model is updated with:
      1. Asymmetric direction-aware loss on continuous expected returns
      2. STaR-style CE loss on discrete golden trajectories
    """

    random_seed = 42
    deterministic = False
    output_dir = "checkpoints/post_train_star_cast"
    checkpoint_path = TrainingConfig.base_model_path
    save_name = "star_cast.pt"
    save_epoch_checkpoints = True

    # ── Data (reuses Phase 6 rollout cache) ──
    prefix_len = 1023
    horizon = 10
    stride_ratio = DataConfig.stride_ratio
    cache_dir = "posttrain/rollout/cache"
    cache_rebuild = False
    max_stocks = 0
    max_train_samples = 0
    max_val_samples = 0

    # ── Training schedule ──
    epochs = 3
    batch_size = 2
    eval_batch_size = 8
    accumulation_steps = 1
    num_workers = 0
    learning_rate = 1.98e-5        # HPO best
    weight_decay = 1e-4
    grad_clip = 0.5
    max_train_updates = 0
    progress_interval = 20
    checkpoint_interval = 120       # save every N updates (0 = epoch only)

    # ── STAR-CAST hyperparameters (Phase 8 HPO best: trial 13, path_mape=4.86%) ──
    neftune_alpha = 5.78             # NEFTune noise strength
    num_trajectories = 4             # N parallel exploration trajectories
    exploration_temperature = 0.517  # temperature for exploration sampling
    top_k_expected_return = 16       # Top-K for soft expected return computation
    asymmetric_alpha = 3.0           # base penalty multiplier for wrong direction
    asymmetric_beta = 10.0           # scale penalty for wrong direction magnitude
    path_asymmetric_alpha = 4.0      # path-level base penalty (harsher) — not searched
    path_asymmetric_beta = 15.0      # path-level scale penalty (harsher) — not searched
    step_asym_weight = 1.0           # loss weight: step-level asymmetric
    path_asym_weight = 1.5           # loss weight: path-level asymmetric
    star_ce_weight = 0.174           # loss weight: STaR cross-entropy

    # ── Phase 9 improvements: break the zero-collapse trap ──
    timidity_penalty_weight = 2.0    # penalize correct-direction-but-too-conservative preds (push-forward)
    timidity_ratio_threshold = 0.5   # threshold: |pred| < |actual| * this -> timid
    oracle_magnitude_penalty = 2.0   # penalize low-volatility trajectories in Oracle filtering
    prob_sharpening_temp = 0.5       # temperature for probability sharpening (<1.0 = sharper)
    actionable_da_threshold = 0.005  # threshold for "actionable" DA (only count |pred| > threshold)

    # ── Optimisation ──
    freeze_backbone = False
    trainable_scope = "all"
    use_gradient_checkpointing = True
    use_amp = True
    amp_dtype = "bfloat16"
    use_tf32 = True

    mape_eps = 1e-4


def _apply_runtime_overrides():
    override_path = os.environ.get("KRONOS_OVERRIDE_JSON", "").strip()
    if not override_path:
        return
    if not os.path.exists(override_path):
        raise FileNotFoundError(f"KRONOS_OVERRIDE_JSON not found: {override_path}")

    with open(override_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError("Override payload must be a JSON object.")

    config_map = {
        "DataConfig": DataConfig,
        "TokenizerConfig": TokenizerConfig,
        "ModelConfig": ModelConfig,
        "TrainingConfig": TrainingConfig,
        "PostTrainDAConfig": PostTrainDAConfig,
        "PostTrainRolloutConfig": PostTrainRolloutConfig,
        "PostTrainCIConfig": PostTrainCIConfig,
        "PostTrainStarCastConfig": PostTrainStarCastConfig,
        "EvaluationConfig": EvaluationConfig,
        "PathConfig": PathConfig,
        "LoRAConfig": LoRAConfig,
    }

    for config_name, overrides in payload.items():
        if overrides is None:
            continue
        target = config_map.get(config_name)
        if target is None:
            raise KeyError(f"Unknown config section in override payload: {config_name}")
        if not isinstance(overrides, dict):
            raise TypeError(f"{config_name} override must be a JSON object.")
        for key, value in overrides.items():
            setattr(target, key, value)


_apply_runtime_overrides()
