"""Aggregate the per-seed Krok 7 results for Beta long-term (v2)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.aggregate import aggregate_seeds
from common.paths import p

SEEDS = [0, 1, 2]

if __name__ == "__main__":
    aggregate_seeds(p("krok7_results_beta_longterm_v2_seed{}.json"), SEEDS,
                     p("krok7_results_beta_longterm_v2_seed_summary.json"))
