#!/usr/bin/env python3
"""
Generate evaluation figures from a trained checkpoint on CPU.

Outputs:
- t-SNE of embeddings colored by TRUE source ids
- t-SNE of embeddings colored by PREDICTED (KMeans) cluster ids

Example:
  python experiments/scripts/evaluate_figures.py \
    --data-path experiments/data/dev_data \
    --checkpoint-dir experiments/checkpoints/conv_autoencoder \
    --output-dir experiments/figures \
    --images-per-batch 10 --tsne-perplexity 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import pytorch_lightning as pl

from src.study.trainer import FragmentAutoencoderTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate evaluation figures from a trained checkpoint (CPU)")
    p.add_argument("--data-path", type=str, required=True, help="Path to evaluation data directory (dev/test)")
    p.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing best_model-*.ckpt")
    p.add_argument("--output-dir", type=str, default="experiments/figures", help="Where to save figures")

    p.add_argument("--images-per-batch", type=int, default=10, help="Number of images per synthetic batch")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader (not critical here)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0=compat)")
    p.add_argument("--steps-per-epoch", type=int, default=1, help="How many samples to synthesize for a quick plot")

    p.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--tsne-random-state", type=int, default=42, help="t-SNE random state")

    return p.parse_args()


def pick_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("best_model-*.ckpt"))
    if ckpts:
        return ckpts[0]
    # Fallback to last.ckpt if best isn't present
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = pick_checkpoint(Path(args.checkpoint_dir))
    print(f"Using checkpoint: {ckpt_path}")

    # Load model strictly on CPU
    model = FragmentAutoencoderTrainer.load_from_checkpoint(
        ckpt_path, map_location=torch.device("cpu")
    )

    # Override runtime params for evaluation
    model.data_path = Path(args.data_path)
    model.images_per_batch = args.images_per_batch
    model.batch_size = args.batch_size
    model.num_workers = args.num_workers
    model.steps_per_epoch = args.steps_per_epoch

    # Prepare datasets and a CPU trainer
    model.setup(stage="fit")
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False)

    # Pull a single batch from val_dataloader
    val_loader = model.val_dataloader()
    batch = next(iter(val_loader))
    fragments: torch.Tensor = batch.fragments
    source_ids: torch.Tensor = batch.source_ids

    # Forward pass to get embeddings
    with torch.no_grad():
        _, embeddings = model(fragments)

    # Compute KMeans clustering for predicted labels
    n_clusters = len(torch.unique(source_ids))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = km.fit_predict(embeddings.numpy())

    # t-SNE on embeddings
    tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, random_state=args.tsne_random_state, init="pca")
    xy = tsne.fit_transform(embeddings.numpy())

    # Plot TRUE labels
    plt.figure(figsize=(6, 5), dpi=150)
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=source_ids.numpy(), s=12, cmap="tab20")
    plt.title("t-SNE of embeddings (colored by TRUE source id)")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    true_path = out_dir / "tsne_true.png"
    plt.savefig(true_path)
    plt.close()

    # Plot PREDICTED clusters
    plt.figure(figsize=(6, 5), dpi=150)
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=pred_labels, s=12, cmap="tab20")
    plt.title("t-SNE of embeddings (colored by PREDICTED cluster)")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    pred_path = out_dir / "tsne_pred.png"
    plt.savefig(pred_path)
    plt.close()

    print(f"Saved figures to:\n - {true_path}\n - {pred_path}")


if __name__ == "__main__":
    main()
