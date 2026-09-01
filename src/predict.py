"""Prediction backend for the demo: geometry in, predicted fields out.

Ties together src/mesh_gen.py (Abaqus-independent meshing), src/graph.py's
_mesh_edges() (reused unchanged), and the trained checkpoint.
"""
import numpy as np
import torch

from src.evaluate import load_trained_model
from src.graph import _mesh_edges
from src.mesh_gen import generate_mesh

_model_cache = {}


def _get_model(checkpoint_path):
    if checkpoint_path not in _model_cache:
        _model_cache[checkpoint_path] = load_trained_model(checkpoint_path)
    return _model_cache[checkpoint_path]


@torch.no_grad()
def predict(holes, load, checkpoint_path="data/model_release.ckpt",
            stats_path="data/norm_stats.npz"):
    """holes: list of {"hole_r", "hole_x", "hole_y"} dicts (possibly empty).
    Returns (positions, elements, u_x, u_y, von_mises) -- positions (N, 2),
    elements {label: [node labels]} (for plotting the mesh), and three
    (N,) arrays of predicted, denormalized field values.
    """
    positions, elements = generate_mesh(holes)
    label_to_idx = {str(i): i for i in range(len(positions))}
    edge_index, edge_attr = _mesh_edges(elements, label_to_idx, positions)

    load_col = np.full(len(positions), load)
    node_features = np.concatenate([positions, load_col[:, None]], axis=1)

    stats = dict(np.load(stats_path))
    x = torch.tensor(
        (node_features - stats["node_mean"]) / stats["node_std"], dtype=torch.float32
    )
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)

    model = _get_model(checkpoint_path)
    pred_norm = model(x, edge_index_t, edge_attr_t).numpy()
    pred = pred_norm * stats["target_std"] + stats["target_mean"]

    return positions, elements, pred[:, 0], pred[:, 1], pred[:, 2]
