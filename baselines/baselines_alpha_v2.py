"""Step 5 baselines: PCA reconstruction, One-Class SVM, and Isolation Forest on
the same Alpha features and Dense windows as the TCN autoencoder
(train_eval_tcn_ae_v2.py), for a direct comparison against the AE.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.baselines import run
from common.paths import p

if __name__ == "__main__":
    run(feat_dir=p("alpha_features_v2"), n_features=39,
        results_path=p("krok7_results_baselines_alpha_v2.json"))
