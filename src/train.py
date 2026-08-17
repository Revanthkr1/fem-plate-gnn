"""Training loop for the plate-with-hole GNN surrogate.

Ported from the AirfRANS project's train.py: the Lightning module shell
(optimizer/scheduler, resume/checkpoint recovery, generic per-field relative-L2
validation logging) carries over unchanged. Two CFD-specific pieces do NOT
carry over and have no FEM equivalent, so they're simply gone rather than
replaced: the wall-distance-weighted loss (distance_weighted_mse /
WALL_WEIGHT_PEAK/LENGTH_SCALE) and the surface-node MAE metric (no-slip-wall
specific). Training loss here is plain MSE.

These meshes are also far smaller than AirfRANS's (~5-6k nodes vs ~180k), so
there's no batch_size=1/accumulate_grad_batches memory pressure driving those
choices the way it did there -- kept the same defaults anyway since they're
harmless at this scale and this hasn't been benchmarked against a larger
batch size yet.
"""
import glob
import os
import shutil

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader

from src.dataset import CachedPyGPlateHoleDataset
from src.metrics import relative_l2_per_field
from src.model import MeshGraphNet

# MeshGraphNet's own defaults -- NOT scaled up yet. Per project convention
# (CLAUDE.md): don't tune architecture/hyperparameters before the model has
# been shown to overfit deliberately on a handful of cases.
DEFAULT_MODEL_KWARGS = {"latent_dim": 32, "hidden_dim": 64, "n_message_passing": 4}


class TrainModule(L.LightningModule):
    def __init__(
        self,
        target_mean,
        target_std,
        lr=1e-3,
        max_epochs=100,
        **model_kwargs,
    ):
        super().__init__()
        self.model = MeshGraphNet(**model_kwargs)
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))
        self.lr = lr
        self.max_epochs = max_epochs

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        self.log("train_loss", loss, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=True)

        pred_phys = pred * self.target_std + self.target_mean
        target_phys = batch.y * self.target_std + self.target_mean
        for field, err in relative_l2_per_field(pred_phys, target_phys).items():
            self.log(f"val_rel_l2_{field}", err, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main(
    cache_dir,
    stats_path,
    checkpoint_path,
    case_ids,
    max_epochs=100,
    batch_size=1,
    accumulate_grad_batches=4,
    n_val=2,
    lr=1e-3,
    model_kwargs=None,
    checkpoint_every_n_epochs=5,
    num_workers=2,
    precision="32-true",
    resume_from_checkpoint=None,
):
    """checkpoint_every_n_epochs / resume_from_checkpoint: same crash-recovery
    behavior as the AirfRANS project -- periodic checkpoints next to
    checkpoint_path so progress survives a disconnect, auto-resuming from the
    most recent one (by epoch number) unless a path is given explicitly.
    """
    stats = dict(np.load(stats_path))
    train_ids = case_ids[:-n_val] if n_val > 0 else case_ids
    val_ids = case_ids[-n_val:] if n_val > 0 else case_ids

    train_ds = CachedPyGPlateHoleDataset(cache_dir, train_ids, stats=stats)
    val_ds = CachedPyGPlateHoleDataset(cache_dir, val_ids, stats=stats)

    loader_kwargs = dict(
        num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    module = TrainModule(
        stats["target_mean"],
        stats["target_std"],
        lr=lr,
        max_epochs=max_epochs,
        **(model_kwargs or DEFAULT_MODEL_KWARGS),
    )
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    periodic_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="mgn-{epoch:03d}",
        every_n_epochs=checkpoint_every_n_epochs,
        save_top_k=-1,
    )

    if resume_from_checkpoint is None:
        existing = sorted(glob.glob(os.path.join(checkpoint_dir, "mgn-epoch=*.ckpt")))
        resume_from_checkpoint = existing[-1] if existing else None

    if resume_from_checkpoint and os.path.dirname(
        os.path.abspath(resume_from_checkpoint)
    ) != os.path.abspath(checkpoint_dir):
        local_resume_path = os.path.join(
            checkpoint_dir, os.path.basename(resume_from_checkpoint)
        )
        shutil.copy2(resume_from_checkpoint, local_resume_path)
        resume_from_checkpoint = local_resume_path

    if resume_from_checkpoint:
        print(f"Resuming from {resume_from_checkpoint}")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision=precision,
        log_every_n_steps=10,
        logger=False,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[periodic_ckpt],
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_from_checkpoint)

    trainer.save_checkpoint(checkpoint_path)
    print(f"Saved final checkpoint to {checkpoint_path}")
    print(f"Periodic checkpoints (every {checkpoint_every_n_epochs} epochs) in {checkpoint_dir}")


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config(os.path.join("configs", "base.yaml"))
    paths = config["paths"]
    training = config["training"]

    n_cases = len(glob.glob(os.path.join(paths["raw_dir"], "case_*.json")))
    main(
        cache_dir=paths["cache_dir"],
        stats_path=paths["stats_path"],
        checkpoint_path=paths["checkpoint_path"],
        case_ids=list(range(n_cases)),
        model_kwargs=config["model"],
        **training,
    )
