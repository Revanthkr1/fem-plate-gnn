"""Framework-agnostic per-case graph dataset, plus a thin PyTorch Geometric wrapper.

Same three-tier pattern as the AirfRANS project (raw dataset -> PyG Data wrapper
-> cached-tensor dataset), minus the subsampling tier -- these meshes are already
small (~5-6k nodes vs AirfRANS's ~180k), so there's no need for a cheap local
sanity-check subsample; the real graph is already cheap enough to overfit on.
"""
import torch
from torch_geometric.data import Data, Dataset

from src.data import load_case
from src.graph import build_graph
from src.preprocess import cache_path


class PlateHoleGraphDataset:
    def __init__(self, raw_dir, case_ids, stats=None):
        self.raw_dir = raw_dir
        self.case_ids = case_ids
        self.stats = stats

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case = load_case(self.raw_dir, self.case_ids[idx])
        node_features, edge_index, edge_attr, targets = build_graph(case)

        if self.stats is not None:
            node_features = (node_features - self.stats["node_mean"]) / self.stats["node_std"]
            targets = (targets - self.stats["target_mean"]) / self.stats["target_std"]

        return {
            "case_id": self.case_ids[idx],
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "targets": targets,
        }


def to_pyg_data(item):
    return Data(
        x=torch.tensor(item["node_features"], dtype=torch.float32),
        edge_index=torch.tensor(item["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(item["edge_attr"], dtype=torch.float32),
        y=torch.tensor(item["targets"], dtype=torch.float32),
        case_id=item["case_id"],
    )


class PyGPlateHoleDataset(Dataset):
    """Wraps PlateHoleGraphDataset, converting each item to a torch_geometric Data object."""

    def __init__(self, raw_dir, case_ids, stats=None):
        super().__init__()
        self.inner = PlateHoleGraphDataset(raw_dir, case_ids, stats=stats)

    def len(self):
        return len(self.inner)

    def get(self, idx):
        return to_pyg_data(self.inner[idx])


class CachedPyGPlateHoleDataset(Dataset):
    """Reads pre-built graph tensors from src.preprocess's cache -- no JSON
    parsing or PyVista edge extraction at train time. Run
    src.preprocess.preprocess_split over `case_ids` first."""

    def __init__(self, cache_dir, case_ids, stats=None):
        super().__init__()
        self.cache_dir = cache_dir
        self.case_ids = case_ids
        self.stats = stats

    def len(self):
        return len(self.case_ids)

    def get(self, idx):
        case_id = self.case_ids[idx]
        item = torch.load(cache_path(self.cache_dir, case_id), weights_only=True)
        x, targets = item["node_features"], item["targets"]
        edge_index = item["edge_index"].long()  # PyG requires int64 edge_index

        # edge_attr isn't cached (see src/preprocess.py) -- recompute from RAW
        # (pre-normalization) position, matching what build_graph() would return.
        position = x[:, :2]
        edge_attr = position[edge_index[1]] - position[edge_index[0]]

        if self.stats is not None:
            node_mean = torch.as_tensor(self.stats["node_mean"], dtype=torch.float32)
            node_std = torch.as_tensor(self.stats["node_std"], dtype=torch.float32)
            target_mean = torch.as_tensor(self.stats["target_mean"], dtype=torch.float32)
            target_std = torch.as_tensor(self.stats["target_std"], dtype=torch.float32)
            x = (x - node_mean) / node_std
            targets = (targets - target_mean) / target_std

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=targets,
            case_id=item["case_id"],
        )
