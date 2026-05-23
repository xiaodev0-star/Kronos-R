# -*- coding: utf-8 -*-
"""Confidence Interval post-training for Kronos-R.

Provides two complementary approaches:
  - ci_sampling : Temperature-sampling-based CI at inference time (no training).
  - train_ci    : CI-aware post-training that optimises interval sharpness
                  and coverage via proper scoring rules.
"""
