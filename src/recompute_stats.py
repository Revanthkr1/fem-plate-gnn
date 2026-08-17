"""One-off: recompute normalization stats + cache tensors over the full case set.

Run whenever the number of generated cases changes (e.g. 10 -> 200 here) --
stats computed over a small subset don't reflect the full distribution once
more cases with different hole sizes/positions/loads are added.
"""
import glob
import os

import numpy as np

from src.normalization import compute_stats
from src.preprocess import preprocess_split

RAW_DIR = "data/raw"
CACHE_DIR = "data/cache"
STATS_PATH = "data/norm_stats.npz"


def main():
    n_cases = len(glob.glob(os.path.join(RAW_DIR, "case_*.json")))
    case_ids = list(range(n_cases))
    print("found %d cases" % n_cases)

    stats = compute_stats(RAW_DIR, case_ids)
    np.savez(STATS_PATH, **stats)
    print("wrote %s (n_cases=%d, n_nodes=%d)" % (
        STATS_PATH, stats["n_cases"], stats["n_nodes"]))

    preprocess_split(RAW_DIR, case_ids, CACHE_DIR)
    print("cached %d/%d cases to %s" % (
        len(glob.glob(os.path.join(CACHE_DIR, "case_*.pt"))), n_cases, CACHE_DIR))


if __name__ == "__main__":
    main()
