"""Step 2: skeleton normalization (v1 data). See common/normalize_skeleton_core.py."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.normalize_skeleton_core import run
from common.paths import p

IN_DIR = p("pose_json")
OUT_DIR = p("pose_norm")

if __name__ == "__main__":
    run(IN_DIR, OUT_DIR)
