"""Step 4+5+6+7 (v2 data) on Beta features with a long-term window
(SEQ_LEN=30, FRAME_STRIDE=3, i.e. ~3x the temporal span of the Dense
window at the same sequence length). Model: TCN autoencoder (common/tcn_ae.py).
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
        feat_dir=p("beta_features_v2"),
        n_features=52,
        seq_len=30,
        frame_stride=3,
        seed=args.seed,
        ckpt_path=p("tcn_ae_beta_longterm_v2_seed{}_best.pt".format(args.seed)),
        results_path=p("krok7_results_beta_longterm_v2_seed{}.json".format(args.seed)),
        title="Beta, YOLO26-pose, TCN-AE, Long-term(30/3) window",
    )


if __name__ == "__main__":
    main()
