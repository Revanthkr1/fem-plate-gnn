"""One-time conversion: raw case JSON -> cached graph tensors (real mesh connectivity).

Training re-reads the same cases every epoch; without caching that means
re-parsing JSON and rebuilding the PyVista UnstructuredGrid (for edge
extraction) every epoch. Cases here are far smaller than AirfRANS's (~5-6k
nodes vs ~180k), so this matters less for runtime, but the pattern -- and the
edge_attr-is-recomputable-so-don't-cache-it trick -- carries over unchanged.
"""
import os

import torch

from src.data import load_case
from src.graph import build_graph


def cache_path(cache_dir, case_id):
    return os.path.join(cache_dir, "case_%03d.pt" % case_id)


def preprocess_case(raw_dir, case_id, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    out_path = cache_path(cache_dir, case_id)
    if not os.path.exists(out_path):
        case = load_case(raw_dir, case_id)
        # edge_attr is dropped -- fully determined by position + edge_index
        # (dst - src), so CachedPlateHoleDataset recomputes it instead of storing it.
        node_features, edge_index, _, targets = build_graph(case)
        torch.save(
            {
                "node_features": torch.tensor(node_features, dtype=torch.float32),
                "edge_index": torch.tensor(edge_index, dtype=torch.int32),
                "targets": torch.tensor(targets, dtype=torch.float32),
                "case_id": case_id,
            },
            out_path,
        )
    return out_path


def preprocess_split(raw_dir, case_ids, cache_dir, log_every=50):
    for i, case_id in enumerate(case_ids):
        preprocess_case(raw_dir, case_id, cache_dir)
        if (i + 1) % log_every == 0:
            print("cached %d/%d" % (i + 1, len(case_ids)), flush=True)
