# Case Study: Image Fragment Reconstruction (Self-Supervised)

Author: Shahin Mohammadi  
Date: 2025-09-21

## Objective
Group unordered 16×16 fragments back to their source 64×64 images using a self-supervised approach that runs on a local machine (CPU for inference/evaluation).

## Methodology (Minimal and Practical)
- Data: ImageNet-64 batches under `/mnt/localssd/datasets/case/{train_data,dev_data}` (pickle format). Loader auto-detected.
- Fragmentation: For each sample, pick N=10 images, split each 64×64 into 4×4 non-overlapping fragments (16×16). Shuffle all fragments.
- Model: Small CNN Autoencoder (`src/study/models_autoenc.py`) trained with MSE recon on fragments.
- Embeddings: Use encoder outputs; cluster fragments with KMeans (k=N) and compare to true source IDs.
- Training: PyTorch Lightning, 8×H100 (DDP, BF16). Checkpoints to `outputs/fragment_clustering_baseline/checkpoints/`.
- Evaluation: ARI, NMI, Purity over many samples; CPU-centric notebook shows usage and visuals.

## Results
- Best checkpoint: `outputs/fragment_clustering_baseline/checkpoints/fragment-ae-epoch=16-val_loss=0.0173.ckpt`
- Visualization (single sample):
  - ![Visualization (real data)](../outputs/fragment_clustering_baseline/results/visualization_real.png)
- Metrics over 1000 validation samples (N=10 per sample):
  - File: [`outputs/fragment_clustering_baseline/results/metrics.json`](../outputs/fragment_clustering_baseline/results/metrics.json)
  - Summary:
    - Model: ARI=0.1657, NMI=0.3771, Purity=0.4031
    - Baseline (Random): ARI≈0.0002, NMI≈0.1328, Purity≈0.2245
    - Baseline (Raw-Pixel KMeans): ARI≈0.1516, NMI≈0.3546, Purity≈0.3863

Interpretation:
- The model’s embedding-based clustering outperforms both random and raw-pixel KMeans baselines across ARI/NMI/Purity.

## CPU-Centric Inference & Evaluation
- Notebook: [`notebooks/fragment_inference.ipynb`](../notebooks/fragment_inference.ipynb)
  - Forces CPU (`CUDA_VISIBLE_DEVICES=''`).
  - Loads best checkpoint, evaluates ARI/NMI/Purity over a subset, and visualizes clusters.
  - Executed copy: [`outputs/fragment_clustering_baseline/results/fragment_inference_exec.ipynb`](../outputs/fragment_clustering_baseline/results/fragment_inference_exec.ipynb)

## Commands
- Train (8×H100):
```
uv run python scripts/train_model.py \
  --train-dir /mnt/localssd/datasets/case/train_data \
  --val-dir /mnt/localssd/datasets/case/dev_data \
  --output-dir outputs/fragment_clustering_baseline \
  --images-per-sample 10 \
  --steps-per-epoch 100 --val-steps 50 --max-epochs 50 \
  --num-workers 16 --gpus 8 --strategy ddp --precision bf16-mixed
```
- Evaluate (metrics):
```
uv run python scripts/evaluate_performance.py \
  --val-dir /mnt/localssd/datasets/case/dev_data \
  --ckpt outputs/fragment_clustering_baseline/checkpoints/fragment-ae-epoch=16-val_loss=0.0173.ckpt \
  --num-samples 1000 --images-per-sample 10 --with-baselines \
  > outputs/fragment_clustering_baseline/results/metrics.json
```
- Visualize (one sample):
```
uv run python scripts/visualize_sample.py \
  --val-dir /mnt/localssd/datasets/case/dev_data \
  --ckpt outputs/fragment_clustering_baseline/checkpoints/fragment-ae-epoch=16-val_loss=0.0173.ckpt \
  --images-per-sample 10 \
  --save-path outputs/fragment_clustering_baseline/results/visualization_real.png
```

## Notes & Next Steps
- Simple autoencoder baseline works; further improvements could use contrastive learning or clustering-aware objectives (e.g., DeepCluster).
- Try varying `images_per_sample` and adding noise/rotations for robustness analysis (optional challenge).
