"""Aggregate the per-seed Krok 7 results for the predictive model on Beta (v2).

No per-seed table, just the mean/std summary.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.aggregate import aggregate_seeds
from common.paths import p

SEEDS = [0, 1, 2]

if __name__ == "__main__":
    aggregate_seeds(p("krok7_results_predictive_beta_v2_seed{}.json"), SEEDS,
                     p("krok7_results_predictive_beta_v2_seed_summary.json"),
                     show_table=False)
