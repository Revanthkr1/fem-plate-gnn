"""Phase 11 splits: separates cases by hole COUNT, not just a random holdout.

The actual test of non-parametric geometric generalization is whether the
model works on a hole count it never saw during training -- interpolating
within a count it has seen (even a "new" random hole position/radius) is a
much easier problem. `hole_count_splits()` holds out one count bucket
entirely (default: 3) from training, in addition to a normal random
in-distribution validation holdout from the remaining buckets.
"""
import glob
import os
import random

from src.data import load_case


def hole_counts(raw_dir):
    """{case_id: number of holes in that case}, read from each case's own
    params -- hole count isn't derivable from case_id alone."""
    counts = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "case_*.json"))):
        case_id = int(os.path.basename(path)[5:8])
        case = load_case(raw_dir, case_id)
        counts[case_id] = len(case["params"]["holes"]) if "holes" in case["params"] else 1
    return counts


def hole_count_splits(raw_dir, held_out_count=3, n_val=20, seed=0):
    """Returns {'train': [...], 'val': [...], 'test_ood': [...]}.

    'test_ood' -- every case with exactly `held_out_count` holes -- is held
    out of training entirely; this is the actual generalization test. 'val'
    is a normal random in-distribution holdout drawn from the remaining
    (in-training-distribution) cases, same role as the phase 8-10b holdout.
    """
    counts = hole_counts(raw_dir)
    test_ood = sorted(cid for cid, n in counts.items() if n == held_out_count)
    pool = sorted(cid for cid, n in counts.items() if n != held_out_count)

    rng = random.Random(seed)
    pool_shuffled = pool[:]
    rng.shuffle(pool_shuffled)
    val = sorted(pool_shuffled[:n_val])
    train = sorted(pool_shuffled[n_val:])

    return {"train": train, "val": val, "test_ood": test_ood}
