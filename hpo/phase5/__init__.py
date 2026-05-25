# -*- coding: utf-8 -*-
"""Phase 5 DA: Integrated post-training comparison package.

Methods:
  Old — CE, ExPO, DPO, RSFT (token-space preference optimization)
  New — GRPO (Group Relative Policy Optimization for financial alignment)

Usage::

    python -m hpo.phase5.run          # run all methods
    python -m hpo.phase5.run --method grpo  # single method
    python -m hpo.phase5.plot          # generate comparison plots
"""

from hpo.phase5.core import (
    LABEL_DOWN, LABEL_FLAT, LABEL_UP, IGNORE_INDEX,
    SEED, P3_CKPT, TOK_PATH, OUT_DIR, VOCAB, P3,
    DEFAULT_CFG,
    load_tokenizer, build_model, load_base_weights,
    build_trainable_model, build_ref_model,
    get_dataloaders, move_batch, prepare_inputs,
    sample_tokens, token_returns, token_direction,
    candidate_logp, build_winner_loser,
    evaluate, compute_metrics,
)
from hpo.phase5.methods import (
    METHOD_REGISTRY,
    loss_ce, loss_expo, loss_dpo, loss_rsft,
    loss_grpo,
)
from hpo.phase5.train import run_method
