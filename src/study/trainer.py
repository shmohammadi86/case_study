"""Fragment Autoencoder Trainer with clustering metrics."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn_extra.cluster import KMedoids
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric

from .datasets import FragmentBatchDataset
from .models import (
    count_parameters,
    create_autoencoder,
    create_linear_autoencoder,
    create_pca_autoencoder,
    create_supervised_classifier,
)


class ModelProtocol(Protocol):
    """Protocol for models that return (output, embeddings) tuple."""

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def parameters(self) -> Iterator[torch.nn.Parameter]: ...
    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]: ...


logger = logging.getLogger(__name__)


class FragmentAutoencoderTrainer(pl.LightningModule):
    def __init__(
        self,
        data_path: str,
        model_type: str = "conv",
        images_per_batch: int = 10,
        batch_size: int = 8,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        encoder_channels: list[int] | None = None,
        latent_dim: int = 128,
        early_stopping_patience: int = 15,
        num_workers: int = 4,
        steps_per_epoch: int = 1000,
        output_dir: str = "outputs",
        num_classes: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model hyperparameters
        self.model_type = model_type
        self.encoder_channels = encoder_channels or [32, 64]
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # Training hyperparameters
        self.images_per_batch = images_per_batch
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.num_workers = num_workers
        self.steps_per_epoch = steps_per_epoch

        # Paths
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create model based on type
        self.model: ModelProtocol
        if model_type == "conv":
            self.model = create_autoencoder(
                input_channels=3, latent_dim=latent_dim, encoder_channels=encoder_channels, input_size=16
            )
        elif model_type == "linear":
            self.model = create_linear_autoencoder(input_channels=3, latent_dim=latent_dim, input_size=16)
        elif model_type == "pca":
            self.model = create_pca_autoencoder(input_channels=3, latent_dim=latent_dim, input_size=16)
        elif model_type == "supervised":
            self.model = create_supervised_classifier(
                input_channels=3, latent_dim=latent_dim, num_classes=num_classes, input_size=16
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Initialize loss function
        self.criterion: torch.nn.CrossEntropyLoss | torch.nn.MSELoss = (
            torch.nn.CrossEntropyLoss() if model_type == "supervised" else torch.nn.MSELoss()
        )

        # Initialize accuracy metrics
        self.train_accuracy: Accuracy | None = None
        self.val_accuracy: Accuracy | None = None

        # Initialize torchmetrics
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Initialize torchmetrics for proper distributed training support."""
        # Initialize accuracy metrics if they haven't been set yet
        if self.model_type == "supervised" and self.train_accuracy is None:
            self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        # Clustering metrics (using MeanMetric for proper aggregation)
        self.train_kmeans_ari = MeanMetric()
        self.train_kmeans_nmi = MeanMetric()
        self.train_kmedoids_ari = MeanMetric()
        self.train_kmedoids_nmi = MeanMetric()
        self.train_mean_ari = MeanMetric()
        self.train_mean_nmi = MeanMetric()
        self.train_mean_clustering_score = MeanMetric()

        self.val_kmeans_ari = MeanMetric()
        self.val_kmeans_nmi = MeanMetric()
        self.val_kmedoids_ari = MeanMetric()
        self.val_kmedoids_nmi = MeanMetric()
        self.val_mean_ari = MeanMetric()
        self.val_mean_nmi = MeanMetric()
        self.val_mean_clustering_score = MeanMetric()

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = FragmentBatchDataset(
                images_dir=self.data_path,
                images_per_sample=self.images_per_batch,
                steps_per_epoch=self.steps_per_epoch,
                seed=42,
                augment=True,
            )

            self.val_dataset = FragmentBatchDataset(
                images_dir=self.data_path,
                images_per_sample=self.images_per_batch,
                steps_per_epoch=self.steps_per_epoch // 4,
                seed=123,
                augment=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Returns:
            For autoencoders: (reconstructed, embeddings)
            For supervised: (logits, embeddings)
        """
        return self.model(x)

    def _compute_clustering_metrics(self, embeddings: torch.Tensor, source_ids: torch.Tensor) -> dict[str, float]:
        embeddings_np = embeddings.detach().cpu().numpy()
        source_ids_np = source_ids.cpu().numpy()

        n_clusters = len(torch.unique(source_ids))
        if n_clusters < 2:
            return {
                "kmeans_ari": 0.0,
                "kmeans_nmi": 0.0,
                "kmedoids_ari": 0.0,
                "kmedoids_nmi": 0.0,
                "mean_ari": 0.0,
                "mean_nmi": 0.0,
                "mean_clustering_score": 0.0,
            }

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings_np)
        kmeans_ari = adjusted_rand_score(source_ids_np, kmeans_labels)
        kmeans_nmi = normalized_mutual_info_score(source_ids_np, kmeans_labels)

        # K-medoids clustering
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, init="k-medoids++")
        kmedoids_labels = kmedoids.fit_predict(embeddings_np)
        kmedoids_ari = adjusted_rand_score(source_ids_np, kmedoids_labels)
        kmedoids_nmi = normalized_mutual_info_score(source_ids_np, kmedoids_labels)

        # Compute means
        mean_ari = (kmeans_ari + kmedoids_ari) / 2
        mean_nmi = (kmeans_nmi + kmedoids_nmi) / 2
        mean_clustering_score = (mean_ari + mean_nmi) / 2

        return {
            "kmeans_ari": float(kmeans_ari),
            "kmeans_nmi": float(kmeans_nmi),
            "kmedoids_ari": float(kmedoids_ari),
            "kmedoids_nmi": float(kmedoids_nmi),
            "mean_ari": float(mean_ari),
            "mean_nmi": float(mean_nmi),
            "mean_clustering_score": float(mean_clustering_score),
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        fragments: torch.Tensor = batch.fragments
        source_ids: torch.Tensor = batch.source_ids
        loss: torch.Tensor

        if self.model_type == "supervised":
            # Supervised learning: predict source image ID
            logits, embeddings = self(fragments)
            loss = self.criterion(logits, source_ids)

            # Update accuracy using torchmetrics
            if self.train_accuracy is not None:
                self.train_accuracy(logits, source_ids)

            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            if self.train_accuracy is not None:
                self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        else:
            # Autoencoder: reconstruction loss
            reconstructed, embeddings = self(fragments)
            loss = self.criterion(reconstructed, fragments)
            self.log("train_recon_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Clustering metrics for all models
        clustering_metrics = self._compute_clustering_metrics(embeddings, source_ids)

        # Update torchmetrics
        self.train_kmeans_ari(clustering_metrics["kmeans_ari"])
        self.train_kmeans_nmi(clustering_metrics["kmeans_nmi"])
        self.train_kmedoids_ari(clustering_metrics["kmedoids_ari"])
        self.train_kmedoids_nmi(clustering_metrics["kmedoids_nmi"])
        self.train_mean_ari(clustering_metrics["mean_ari"])
        self.train_mean_nmi(clustering_metrics["mean_nmi"])
        self.train_mean_clustering_score(clustering_metrics["mean_clustering_score"])

        # Log metrics using torchmetrics
        self.log("train_kmeans_ari", self.train_kmeans_ari, on_step=False, on_epoch=True)
        self.log("train_kmeans_nmi", self.train_kmeans_nmi, on_step=False, on_epoch=True)
        self.log("train_kmedoids_ari", self.train_kmedoids_ari, on_step=False, on_epoch=True)
        self.log("train_kmedoids_nmi", self.train_kmedoids_nmi, on_step=False, on_epoch=True)
        self.log("train_mean_ari", self.train_mean_ari, on_step=False, on_epoch=True)
        self.log("train_mean_nmi", self.train_mean_nmi, on_step=False, on_epoch=True)
        self.log("train_mean_clustering_score", self.train_mean_clustering_score, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        fragments: torch.Tensor = batch.fragments
        source_ids: torch.Tensor = batch.source_ids
        loss: torch.Tensor

        if self.model_type == "supervised":
            # Supervised learning: predict source image ID
            logits, embeddings = self(fragments)
            loss = self.criterion(logits, source_ids)

            # Update accuracy using torchmetrics
            if self.val_accuracy is not None:
                self.val_accuracy(logits, source_ids)

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            if self.val_accuracy is not None:
                self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        else:
            # Autoencoder: reconstruction loss
            reconstructed, embeddings = self(fragments)
            loss = self.criterion(reconstructed, fragments)
            self.log("val_recon_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Clustering metrics for all models
        clustering_metrics = self._compute_clustering_metrics(embeddings, source_ids)

        # Update torchmetrics
        self.val_kmeans_ari(clustering_metrics["kmeans_ari"])
        self.val_kmeans_nmi(clustering_metrics["kmeans_nmi"])
        self.val_kmedoids_ari(clustering_metrics["kmedoids_ari"])
        self.val_kmedoids_nmi(clustering_metrics["kmedoids_nmi"])
        self.val_mean_ari(clustering_metrics["mean_ari"])
        self.val_mean_nmi(clustering_metrics["mean_nmi"])
        self.val_mean_clustering_score(clustering_metrics["mean_clustering_score"])

        # Log metrics using torchmetrics
        self.log("val_kmeans_ari", self.val_kmeans_ari, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_kmeans_nmi", self.val_kmeans_nmi, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_kmedoids_ari", self.val_kmedoids_ari, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_kmedoids_nmi", self.val_kmedoids_nmi, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mean_ari", self.val_mean_ari, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mean_nmi", self.val_mean_nmi, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_mean_clustering_score", self.val_mean_clustering_score, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_mean_clustering_score", "frequency": 1},
        }

    def configure_callbacks(self) -> list[pl.callbacks.Callback]:
        callbacks: list[pl.callbacks.Callback] = []

        checkpoint_path = self.output_dir / "checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_path),
                filename="best_model-{epoch:02d}-{val_mean_clustering_score:.3f}",
                monitor="val_mean_clustering_score",
                mode="max",
                save_top_k=1,
                save_last=True,
                verbose=True,
            )
        )

        callbacks.append(
            EarlyStopping(
                monitor="val_mean_clustering_score",
                mode="max",
                patience=self.early_stopping_patience,
                verbose=True,
                min_delta=0.001,
            )
        )

        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        return callbacks

    def on_train_end(self) -> None:
        # Save final model state dict
        state_dict_path = self.output_dir / "model_state_dict.pt"
        try:
            cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(cpu_state, state_dict_path)
        except Exception as e:
            logger.warning(f"Failed to save CPU state_dict: {e}")

        # Save training info
        training_info = {
            "model_config": {
                "model_type": self.model_type,
                "encoder_channels": self.encoder_channels,
                "latent_dim": self.latent_dim,
                "input_size": 16,
                "input_channels": 3,
                "num_classes": self.num_classes if self.model_type == "supervised" else None,
            },
            "training_config": {
                "images_per_batch": self.images_per_batch,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "early_stopping_patience": self.early_stopping_patience,
                "steps_per_epoch": self.steps_per_epoch,
            },
            "model_parameters": count_parameters(self.model),
        }

        with open(self.output_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Any,  # Compatible with supertype: Any | IO[Any]
        map_location: torch.device | str | dict[str, str] | None = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = True,
        **kwargs: Any,
    ) -> FragmentAutoencoderTrainer:
        instance: FragmentAutoencoderTrainer = super().load_from_checkpoint(
            str(checkpoint_path),
            map_location=map_location,
            hparams_file=str(hparams_file) if hparams_file is not None else None,
            strict=strict,
            **kwargs,
        )
        return instance


if __name__ == "__main__":
    # Example usage for all 4 model types
    model_configs = [
        {"model_type": "conv", "output_dir": "outputs/conv_autoencoder", "description": "Convolutional Autoencoder"},
        {
            "model_type": "linear",
            "output_dir": "outputs/linear_autoencoder",
            "description": "Multi-layer Linear Autoencoder",
        },
        {
            "model_type": "pca",
            "output_dir": "outputs/pca_autoencoder",
            "description": "PCA-like Single Layer Autoencoder",
        },
        {
            "model_type": "supervised",
            "output_dir": "outputs/supervised_classifier",
            "description": "Supervised Fragment Classifier",
        },
    ]

    # Train the first model (conv) as example
    config = model_configs[0]
    trainer = FragmentAutoencoderTrainer(
        data_path="data/train_data",
        model_type=config["model_type"],
        images_per_batch=10,
        batch_size=8,
        epochs=100,
        learning_rate=1e-3,
        encoder_channels=[32, 64],
        latent_dim=128,
        output_dir=config["output_dir"],
        num_classes=1000,  # For supervised model
    )

    print(f"Training {config['description']} with {count_parameters(trainer.model):,} parameters")

    # Train using PyTorch Lightning Trainer
    pl_trainer = pl.Trainer(
        max_epochs=trainer.epochs,
        callbacks=trainer.configure_callbacks(),
        logger=CSVLogger(save_dir=str(trainer.output_dir), name="training_logs"),
        accelerator="auto",
        devices="auto",
        precision=32,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
    )

    pl_trainer.fit(trainer)
    print(f"Training completed! Check {trainer.output_dir} for results.")

    print("\nTo train other models, change model_type to:")
    for i, config in enumerate(model_configs[1:], 1):
        print(f"  {i + 1}. '{config['model_type']}' - {config['description']}")
