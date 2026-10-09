"""Aggregate the per-seed Krok 7 results for ST-GCN gamma-per-node (v2).

Terse variant: no per-seed table and no "Saved summary." message.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.aggregate import aggregate_seeds
from common.paths import p

SEEDS = [0, 1, 2]

if __name__ == "__main__":
    aggregate_seeds(p("krok7_results_stgcn_gammapernode_v2_seed{}.json"), SEEDS,
                     p("krok7_results_stgcn_gammapernode_v2_seed_summary.json"),
                     show_table=False, show_saved_message=False)
