"""
Training Pipeline for ImageNet-64 Classification using PyTorch Lightning
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from .data import ImageNet64Dataset, get_augmentation_transforms
from .models import count_parameters, create_cnn_model, create_transfer_learning_model

logger = logging.getLogger(__name__)


class ImageNet64LightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for ImageNet-64 classification.
    """

    def __init__(self, model_config: dict[str, Any], training_config: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config

        # Create model
        if self.model_config.get("use_transfer_learning", False):
            self.model = create_transfer_learning_model(
                base_model_name=self.model_config.get("base_model", "resnet50"),
                num_classes=self.model_config.get("num_classes", 1000),
                trainable_layers=self.model_config.get("trainable_layers", 0),
            )
        else:
            self.model = create_cnn_model(
                num_classes=self.model_config.get("num_classes", 1000),
                architecture=self.model_config.get("architecture", "simple"),
            )

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1000)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1000)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1000)

        self.train_top5_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5)
        self.val_top5_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5)
        self.test_top5_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5)

        logger.info(f"Created model with {count_parameters(self.model):,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate metrics
        self.train_accuracy(logits, y)
        self.train_top5_accuracy(logits, y)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_top5_acc", self.train_top5_accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate metrics
        self.val_accuracy(logits, y)
        self.val_top5_accuracy(logits, y)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_top5_acc", self.val_top5_accuracy, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate metrics
        self.test_accuracy(logits, y)
        self.test_top5_accuracy(logits, y)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test_top5_acc", self.test_top5_accuracy, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.get("learning_rate", 0.001),
            weight_decay=self.training_config.get("weight_decay", 1e-4),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "frequency": 1},
        }


class ImageNet64Trainer:
    """
    Training pipeline for ImageNet-64 classification models using PyTorch Lightning.
    """

    def __init__(
        self,
        data_path: str,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        output_dir: str = "outputs",
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        data_path : str
            Path to ImageNet-64 dataset
        model_config : Dict[str, Any]
            Model configuration parameters
        training_config : Dict[str, Any]
            Training configuration parameters
        output_dir : str
            Directory to save outputs
        """
        self.data_path = Path(data_path)
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Lightning module
        self.lightning_module = ImageNet64LightningModule(model_config, training_config)

        # Initialize data loaders
        self._setup_data()

        # Training results
        self.trainer_results = None

    def _setup_data(self) -> None:
        """Setup data loaders."""
        batch_size = self.training_config.get("batch_size", 32)
        num_workers = self.training_config.get("num_workers", 4)

        # Training dataset with augmentation
        train_transform = (
            get_augmentation_transforms(training=True) if self.training_config.get("augmentation", True) else None
        )
        self.train_dataset = ImageNet64Dataset(self.data_path, split="train", transform=train_transform)

        # Validation dataset without augmentation
        val_transform = get_augmentation_transforms(training=False)
        self.val_dataset = ImageNet64Dataset(self.data_path, split="test", transform=val_transform)

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Training batches per epoch: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")

    def _create_callbacks(self) -> list[pl.callbacks.Callback]:
        """Create PyTorch Lightning callbacks."""
        callbacks: list[pl.callbacks.Callback] = []

        # Model checkpoint
        checkpoint_path = self.output_dir / "checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_path),
                filename="best_model-{epoch:02d}-{val_acc:.3f}",
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                save_last=True,
                verbose=True,
            )
        )

        # Early stopping
        if self.training_config.get("early_stopping", True):
            callbacks.append(
                EarlyStopping(
                    monitor="val_acc", mode="max", patience=self.training_config.get("patience", 10), verbose=True
                )
            )

        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        return callbacks

    def train(self) -> dict[str, Any]:
        """
        Train the model using PyTorch Lightning.

        Returns
        -------
        Dict[str, Any]
            Training results and metrics
        """
        logger.info("Starting training...")

        # Create logger
        csv_logger = CSVLogger(save_dir=str(self.output_dir), name="training_logs")

        # Create PyTorch Lightning trainer
        trainer = pl.Trainer(
            max_epochs=self.training_config.get("epochs", 100),
            callbacks=self._create_callbacks(),
            logger=csv_logger,
            accelerator="auto",
            devices="auto",
            precision=self.training_config.get("precision", 32),
            gradient_clip_val=self.training_config.get("gradient_clip_val", 0.0),
            accumulate_grad_batches=self.training_config.get("accumulate_grad_batches", 1),
            log_every_n_steps=50,
            enable_progress_bar=True,
            enable_model_summary=True,
        )

        # Determine checkpoint to resume from if available
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_path = None
        try:
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)
                logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        except Exception:
            ckpt_path = None

        # Train (with resume if checkpoint exists)
        trainer.fit(
            self.lightning_module,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=ckpt_path,
        )

        # Save final model (Lightning checkpoint)
        model_path = self.output_dir / "final_model.ckpt"
        trainer.save_checkpoint(str(model_path))
        logger.info(f"Saved final model to {model_path}")

        # Additionally save a CPU-friendly state dict for local inference
        state_dict_path = self.output_dir / "model_state_dict.pt"
        try:
            cpu_state = {k: v.cpu() for k, v in self.lightning_module.model.state_dict().items()}
            torch.save(cpu_state, state_dict_path)
            logger.info(f"Saved model state_dict to {state_dict_path}")
        except Exception as e:
            logger.warning(f"Failed to save CPU state_dict: {e}")

        # Generate training plots
        self._plot_training_history(csv_logger.log_dir)

        # Evaluate model
        results = self.evaluate(trainer)

        return results

    def evaluate(self, trainer: pl.Trainer | None = None, ckpt_path: str | None = None) -> dict[str, Any]:
        """
        Evaluate the trained model.

        Parameters
        ----------
        trainer : pl.Trainer, optional
            PyTorch Lightning trainer instance

        Returns
        -------
        Dict[str, Any]
            Evaluation metrics
        """
        logger.info("Evaluating model...")

        if trainer is None:
            # Create a new trainer for evaluation
            trainer = pl.Trainer(accelerator="auto", devices="auto", logger=False, enable_progress_bar=True)

        # Optionally load checkpoint weights for evaluation
        if ckpt_path:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if "state_dict" in ckpt:
                    self.lightning_module.load_state_dict(ckpt["state_dict"], strict=False)
                    logger.info(f"Loaded Lightning state_dict from {ckpt_path}")
                else:
                    # if a plain model state dict
                    self.lightning_module.model.load_state_dict(ckpt, strict=False)
                    logger.info(f"Loaded model state_dict from {ckpt_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint for evaluation: {e}")

        # Test the model
        test_results = trainer.test(self.lightning_module, dataloaders=self.val_loader, verbose=True)

        # Extract results
        eval_results = test_results[0] if test_results else {}

        logger.info("Evaluation Results:")
        for metric, value in eval_results.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Save results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        # Ensure a concrete dict type for callers / typing
        return dict(eval_results)

    def _plot_training_history(self, log_dir: str) -> None:
        """Plot training history from CSV logs."""
        import pandas as pd

        try:
            # Read metrics from CSV log
            csv_path = Path(log_dir) / "metrics.csv"
            if not csv_path.exists():
                logger.warning(f"No metrics file found at {csv_path}")
                return

            df = pd.read_csv(csv_path)

            # Filter out NaN values and group by epoch
            df = df.dropna()

            _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Accuracy
            if "train_acc_epoch" in df.columns and "val_acc" in df.columns:
                train_acc = df.dropna(subset=["train_acc_epoch"])
                val_acc = df.dropna(subset=["val_acc"])

                axes[0, 0].plot(train_acc["epoch"], train_acc["train_acc_epoch"], label="Training")
                axes[0, 0].plot(val_acc["epoch"], val_acc["val_acc"], label="Validation")
                axes[0, 0].set_title("Model Accuracy")
                axes[0, 0].set_xlabel("Epoch")
                axes[0, 0].set_ylabel("Accuracy")
                axes[0, 0].legend()

            # Loss
            if "train_loss_epoch" in df.columns and "val_loss" in df.columns:
                train_loss = df.dropna(subset=["train_loss_epoch"])
                val_loss = df.dropna(subset=["val_loss"])

                axes[0, 1].plot(train_loss["epoch"], train_loss["train_loss_epoch"], label="Training")
                axes[0, 1].plot(val_loss["epoch"], val_loss["val_loss"], label="Validation")
                axes[0, 1].set_title("Model Loss")
                axes[0, 1].set_xlabel("Epoch")
                axes[0, 1].set_ylabel("Loss")
                axes[0, 1].legend()

            # Top-5 Accuracy
            if "train_top5_acc" in df.columns and "val_top5_acc" in df.columns:
                train_top5 = df.dropna(subset=["train_top5_acc"])
                val_top5 = df.dropna(subset=["val_top5_acc"])

                axes[1, 0].plot(train_top5["epoch"], train_top5["train_top5_acc"], label="Training")
                axes[1, 0].plot(val_top5["epoch"], val_top5["val_top5_acc"], label="Validation")
                axes[1, 0].set_title("Top-5 Accuracy")
                axes[1, 0].set_xlabel("Epoch")
                axes[1, 0].set_ylabel("Top-5 Accuracy")
                axes[1, 0].legend()

            # Learning Rate
            if "lr-Adam" in df.columns:
                lr_data = df.dropna(subset=["lr-Adam"])
                axes[1, 1].plot(lr_data["epoch"], lr_data["lr-Adam"])
                axes[1, 1].set_title("Learning Rate")
                axes[1, 1].set_xlabel("Epoch")
                axes[1, 1].set_ylabel("Learning Rate")
                axes[1, 1].set_yscale("log")

            plt.tight_layout()
            plot_path = self.output_dir / "training_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved training plots to {plot_path}")

        except Exception as e:
            logger.warning(f"Could not create training plots: {e}")

    def predict_sample(self, n_samples: int = 10) -> None:
        """
        Make predictions on sample images and visualize results.

        Parameters
        ----------
        n_samples : int
            Number of samples to predict
        """
        # Get sample batch from validation loader
        self.lightning_module.eval()

        with torch.no_grad():
            # Get a batch from validation loader
            data_iter = iter(self.val_loader)
            x_batch, y_batch = next(data_iter)

            # Limit to n_samples
            x_batch = x_batch[:n_samples]
            y_batch = y_batch[:n_samples]

            # Make predictions
            logits = self.lightning_module(x_batch)
            predictions = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(predictions, dim=1)

        # Convert to numpy for plotting
        x_batch_np = x_batch.cpu().numpy()
        y_batch_np = y_batch.cpu().numpy()
        predicted_classes_np = predicted_classes.cpu().numpy()

        # Plot results
        _fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i in range(min(n_samples, 10)):
            # Display image (convert from CHW to HWC)
            img = x_batch_np[i].transpose(1, 2, 0)

            # Denormalize if needed
            if img.max() <= 1.0:
                img = np.clip(img, 0, 1)

            axes[i].imshow(img)
            axes[i].set_title(f"True: {y_batch_np[i]}\nPred: {predicted_classes_np[i]}")
            axes[i].axis("off")

        plt.tight_layout()
        plot_path = self.output_dir / "sample_predictions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved sample predictions to {plot_path}")


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from JSON file."""
    from typing import cast

    with open(config_path) as f:
        data = json.load(f)
    return cast(dict[str, Any], data)


if __name__ == "__main__":
    # Example usage
    model_config = {"architecture": "simple", "num_classes": 1000}

    training_config = {
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "augmentation": True,
        "early_stopping": True,
        "patience": 10,
        "num_workers": 4,
    }

    trainer = ImageNet64Trainer(
        data_path="data/imagenet64", model_config=model_config, training_config=training_config, output_dir="outputs"
    )

    results = trainer.train()
    print("Training completed!")
    print(f"Final test accuracy: {results.get('test_acc', 0):.4f}")
