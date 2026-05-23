# -*- coding: utf-8 -*-
"""Data loading for Confidence Interval post-training.

Reuses the Rollout cache format (prefix_len=1023, horizon=10) because
CI prediction needs the same multi-step autoregressive setup.  The
existing RolloutWindowDataset and rollout_cache_path are re-exported
here so the CI module stays self-contained.
"""

from posttrain.rollout.data import (
    FEATURE_COLS,
    TIME_KEYS,
    RolloutSplitInfo,
    RolloutWindowDataset,
    build_rollout_cache,
    load_rollout_cache,
    resolve_project_path,
    rollout_cache_path,
    rollout_collate,
)
