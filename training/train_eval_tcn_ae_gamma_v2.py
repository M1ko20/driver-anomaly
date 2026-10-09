"""Step 4+5+6+7 (v2 data, real per-driver split with anomalies) on Gamma features.

Windowing: Dense (SEQ_LEN=60, FRAME_STRIDE=1, WINDOW_STEP=1). Model: TCN
autoencoder (common/tcn_ae.py), for a direct comparison against Alpha/Beta.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from common import tcn_ae_eval
from common.paths import p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tcn_ae_eval.run(
        feat_dir=p("gamma_features_v2"),
        n_features=104,
        seed=args.seed,
        ckpt_path=p("tcn_ae_gamma_v2_seed{}_best.pt".format(args.seed)),
        results_path=p("krok7_results_gamma_v2_seed{}.json".format(args.seed)),
        title="Gamma, YOLO26-pose, TCN-AE, Dense window",
    )


if __name__ == "__main__":
    main()
