"""Step 4+5+6+7 (v2 data, real per-driver split with anomalies) on Alpha features.

Windowing: Dense (SEQ_LEN=60, FRAME_STRIDE=1, WINDOW_STEP=1). Model: TCN
autoencoder (common/tcn_ae.py). Trains on safe-drive only; early stopping
uses val MSE computed on safe-drive clips only.

Evaluation: each clip is a single labeled interval, so no window-to-event
aggregation is needed. Threshold is chosen on val (Youden's J), then applied
once to test for ROC-AUC, PR-AUC, and frozen-threshold precision/recall/F1.
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
        feat_dir=p("alpha_features_v2"),
        n_features=39,
        seed=args.seed,
        ckpt_path=p("tcn_ae_alpha_v2_seed{}_best.pt".format(args.seed)),
        results_path=p("krok7_results_alpha_v2_seed{}.json".format(args.seed)),
        title="Alpha, YOLO26-pose, TCN-AE, Dense window",
    )


if __name__ == "__main__":
    main()
