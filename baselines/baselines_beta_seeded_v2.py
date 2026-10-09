"""Step 5+6 baselines (PCA/OCSVM/IsolationForest) on Beta features, run with
multiple seeds to match the AE models' seed protocol.

Seed affects the OCSVM/IsolationForest train subsample and IsolationForest's
own randomness; PCA is deterministic so its seed has no effect.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from common.baselines import run
from common.paths import p

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run(feat_dir=p("beta_features_v2"), n_features=52, seed=args.seed,
        results_path=p("krok7_results_baselines_beta_v2_seed{}.json".format(args.seed)),
        include_seed_in_results=True)
