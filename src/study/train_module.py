from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from .models_autoenc import FragmentAutoencoder
from .fragmentation import FragmentBatch


class FragmentAE(pl.LightningModule):
    """LightningModule that trains a CNN autoencoder on unordered fragments.

    Optimization objective: MSE reconstruction.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = FragmentAutoencoder(embedding_dim=embedding_dim)
        self.criterion = nn.MSELoss()

    def forward(self, fragments: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns reconstruction and embeddings
        x_hat, z = self.model(fragments)
        return x_hat, z

    def training_step(self, batch: FragmentBatch, batch_idx: int) -> torch.Tensor:
        x = batch.fragments
        x_hat, _ = self(x)
        loss = self.criterion(x_hat, x)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: FragmentBatch, batch_idx: int) -> None:
        x = batch.fragments
        x_hat, _ = self(x)
        loss = self.criterion(x_hat, x)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
