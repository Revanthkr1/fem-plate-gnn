"""Evaluate a trained checkpoint on a held-out split of plate-with-hole cases.

Ported from the AirfRANS project's evaluate.py: same load-checkpoint / loop-
over-cases / summarize shape. Lift/drag has no FEM equivalent here, so the
per-case physical sanity check is different in kind: rather than a single
derived scalar computed the same way for every case (Cd/Cl), it checks
whether the model finds the right *stress concentration* -- these cases are
mostly elastoplastic (not purely elastic), so a literal Kirsch SCF doesn't
apply case-by-case the way it did in the phase-1 setup check. Peak von Mises
magnitude and location are the closest per-case analogue: does the surrogate
correctly find both how big and where the worst stress is, not just get the
average field right.
"""
import numpy as np
import torch

from src.data import load_case
from src.graph import build_graph
from src.metrics import FIELD_NAMES, relative_l2_per_field
from src.train import DEFAULT_MODEL_KWARGS, TrainModule


def load_trained_model(checkpoint_path, **model_kwargs):
    # target_mean/target_std are only placeholders here (need the right shape,
    # (3,) -- u_x, u_y, von_mises) -- load_state_dict overwrites these
    # registered buffers with the real values saved in the checkpoint right
    # after construction.
    module = TrainModule.load_from_checkpoint(
        checkpoint_path,
        target_mean=np.zeros(3),
        target_std=np.ones(3),
        map_location="cpu",
        **(model_kwargs or DEFAULT_MODEL_KWARGS),
    )
    module.eval()
    return module.model


@torch.no_grad()
def evaluate_case(model, raw_dir, case_id, stats):
    case = load_case(raw_dir, case_id)
    node_features, edge_index, edge_attr, targets = build_graph(case)

    x = torch.tensor(
        (node_features - stats["node_mean"]) / stats["node_std"], dtype=torch.float32
    )
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)

    pred_norm = model(x, edge_index_t, edge_attr_t).numpy()
    pred = pred_norm * stats["target_std"] + stats["target_mean"]

    field_errors = relative_l2_per_field(torch.tensor(pred), torch.tensor(targets))

    # von_mises is column 2 (see src/metrics.py FIELD_NAMES) -- peak location
    # is the node index of the max, not a physical (x, y) yet.
    true_peak_idx = int(np.argmax(targets[:, 2]))
    pred_peak_idx = int(np.argmax(pred[:, 2]))
    position = node_features[:, :2]
    peak_location_error = float(
        np.linalg.norm(position[pred_peak_idx] - position[true_peak_idx])
    )

    return {
        "case_id": case_id,
        "field_errors": field_errors,
        "true_peak_von_mises": float(targets[true_peak_idx, 2]),
        "pred_peak_von_mises": float(pred[pred_peak_idx, 2]),
        "peak_location_error_mm": peak_location_error,
        # With 0 holes there's no real stress concentration -- the field is
        # smooth/near-uniform, so "true peak location" is essentially
        # arbitrary mesh noise, not a physically meaningful target. Peak-
        # location error is only meaningful when there's an actual hole to
        # localize (see PROJECT_FLOW.md phase 11 -- this was found by 0-hole
        # cases showing huge "errors" that were noise-vs-noise, not real
        # model failure).
        "n_holes": len(case["params"]["holes"]) if "holes" in case["params"] else 1,
    }


def evaluate_split(checkpoint_path, raw_dir, stats_path, case_ids, log_every=20, **model_kwargs):
    stats = dict(np.load(stats_path))
    model = load_trained_model(checkpoint_path, **model_kwargs)

    results = []
    for i, case_id in enumerate(case_ids):
        results.append(evaluate_case(model, raw_dir, case_id, stats))
        if (i + 1) % log_every == 0:
            print(f"evaluated {i + 1}/{len(case_ids)}", flush=True)
    return results


def summarize(results):
    """Aggregate a list of evaluate_case() results into headline numbers --
    mean relative L2 per field (never blended), plus peak-von-Mises magnitude
    relative error and mean peak-location error.

    Peak-location error is computed only over cases with >=1 hole -- a
    0-hole case has no real stress concentration (the field is smooth/near-
    uniform), so its "true peak" location is essentially arbitrary mesh
    noise, and including it just adds noise-vs-noise error that looks like
    model failure but isn't (see PROJECT_FLOW.md phase 11). Peak magnitude
    and the per-field relative L2 are still meaningful for 0-hole cases
    (there IS a real, if unremarkable, max value and field to compare), so
    those stay computed over the full set.
    """
    summary = {}
    for field in FIELD_NAMES:
        errors = [r["field_errors"][field] for r in results]
        summary[f"{field}_rel_l2_mean"] = float(np.mean(errors))

    true_peaks = np.array([r["true_peak_von_mises"] for r in results])
    pred_peaks = np.array([r["pred_peak_von_mises"] for r in results])
    summary["peak_von_mises_rel_l2"] = float(
        np.linalg.norm(pred_peaks - true_peaks) / np.linalg.norm(true_peaks)
    )

    with_holes = [r for r in results if r.get("n_holes", 1) > 0]
    summary["mean_peak_location_error_mm"] = float(
        np.mean([r["peak_location_error_mm"] for r in with_holes])
    )
    summary["n_cases_with_holes"] = len(with_holes)
    summary["n_cases_excluded_zero_holes"] = len(results) - len(with_holes)
    return summary
