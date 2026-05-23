# -*- coding: utf-8 -*-
"""Convenience entry-point for Confidence-Interval post-training.

Usage
-----
  python Post_Train_CI.py                          # train with defaults
  python Post_Train_CI.py --epochs 5 --lr 1e-5     # custom hyperparams
  python Post_Train_CI.py --eval-only              # evaluate only
"""

from posttrain.ci.train_ci import main

if __name__ == "__main__":
    main()
