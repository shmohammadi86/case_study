# Case Study Report: Image Fragment Reconstruction via Self-Supervised Learning

Author: Shahin Mohammadi
Date: <YYYY-MM-DD>

## 1. Objective

Briefly restate the task: group unordered 16x16 fragments back to their 64x64 source images using a self-supervised approach.

## 2. Methodology

### 2.1 Data Preparation
- 64x64 RGB images fragmented into 4x4 grid (16x16 patches)
- Each training sample: N=10 images -> 160 unordered fragments
- Augmentations: horizontal flip

### 2.2 Model Architecture
- CNN Autoencoder (encoder -> embedding -> decoder)
- Embedding dimension: 64 (configurable)
- Loss: MSE reconstruction

### 2.3 Training Strategy
- Optimizer: Adam (lr=1e-3, wd=0.0)
- Framework: PyTorch Lightning
- Checkpoint: best by val_loss
- Hardware: CPU for smoke tests; H100 multi-GPU for final run

## 3. Evaluation Metrics
- Clustering with KMeans (k=N=10) on embeddings
- Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Purity
- Report mean metrics over multiple samples

## 4. Results
- Provide tables/plots summarizing ARI/NMI/Purity over validation samples
- Include representative visualization grids of predicted vs. true groups

## 5. Discussion
- Strengths/limitations of autoencoder-based embeddings
- Effect of noise/rotation (if attempted)
- Sensitivity to fragment size/number (if attempted)

## 6. Next Steps
- Try contrastive/self-supervised objectives (SimCLR-style) on fragments
- Explore clustering-aware losses (e.g., DeepCluster)
- Tune embedding dimension and encoder depth
- Consider positional priors for layout-aware grouping

## 7. Reproducibility
- Python: >=3.9
- Environment managed by uv; see `pyproject.toml`
- Commands used:
  - Training: `uv run python scripts/train_model.py --gpus 8 --strategy ddp --precision bf16-mixed --max-epochs 50`
  - Evaluation: `uv run python scripts/evaluate_performance.py --ckpt <path> --num-samples 1000`
  - Visualization: `uv run python scripts/visualize_sample.py --ckpt <path>`

## 8. References
- PyTorch Lightning docs
- scikit-learn clustering and metrics
