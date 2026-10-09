"""Aggregate the per-seed baseline (PCA/OCSVM/IsolationForest) results on Beta (v2)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.aggregate import aggregate_baselines
from common.paths import p

SEEDS = [0, 1, 2]
MODELS = ["PCA_reconstruction", "OneClassSVM", "IsolationForest"]

if __name__ == "__main__":
    aggregate_baselines(p("krok7_results_baselines_beta_v2_seed{}.json"), SEEDS, MODELS,
                         p("krok7_results_baselines_beta_v2_seed_summary.json"))
