#!/usr/bin/env python3
"""
CLI entrypoint for full training using FragmentAutoencoderTrainer.

Example:
    python train.py \
      --data-path experiments/data/train_data \
      --output-dir experiments/checkpoints/conv_autoencoder \
      --model-type conv \
      --images-per-batch 10 --batch-size 64 --epochs 50 \
      --steps-per-epoch 1000 --num-workers 8

This will automatically use GPU if available; otherwise it runs on CPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from src.study.trainer import FragmentAutoencoderTrainer
from src.study.models import count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fragment autoencoder/classifier")
    # data and io
    p.add_argument("--data-path", type=str, required=True, help="Path to train/val data directory")
    p.add_argument("--output-dir", type=str, default="experiments/checkpoints/conv_autoencoder", help="Output directory")

    # model config
    p.add_argument("--model-type", type=str, default="conv", choices=["conv", "linear", "pca", "supervised"],
                   help="Model type to train")
    p.add_argument("--encoder-channels", type=int, nargs="*", default=[32, 64],
                   help="Encoder channels progression, e.g. --encoder-channels 32 64")
    p.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    p.add_argument("--num-classes", type=int, default=1000, help="Classes for supervised model")

    # training config
    p.add_argument("--images-per-batch", type=int, default=10, help="Images per synthetic batch for fragmentation")
    p.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    p.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--early-stopping-patience", type=int, default=15, help="Early stopping patience")
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--steps-per-epoch", type=int, default=1000, help="Synthetic steps per epoch in dataset")
    p.add_argument("--precision", type=str, default="32", help="Lightning precision (e.g., 16-mixed, bf16-mixed, 32)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FragmentAutoencoderTrainer(
        data_path=args.data_path,
        model_type=args.model_type,
        images_per_batch=args.images_per_batch,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        encoder_channels=list(args.encoder_channels) if args.encoder_channels else [32, 64],
        latent_dim=args.latent_dim,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        steps_per_epoch=args.steps_per_epoch,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
    )

    print(f"Training {args.model_type} with {count_parameters(model.model):,} parameters")

    logger = CSVLogger(save_dir=str(out_dir), name="training_logs")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=model.configure_callbacks(),
        logger=logger,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
    )

    model.setup(stage="fit")
    trainer.fit(model)
    print(f"Training complete. Outputs are saved under: {out_dir}")


if __name__ == "__main__":
    main()
