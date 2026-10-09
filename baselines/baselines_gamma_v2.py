"""Step 5 baselines (PCA/OCSVM/IsolationForest) on the same Gamma features and
Dense windows as the TCN autoencoder, for a direct comparison against the AE.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.baselines import run
from common.paths import p

if __name__ == "__main__":
    run(feat_dir=p("gamma_features_v2"), n_features=104,
        results_path=p("krok7_results_baselines_gamma_v2.json"))
