# Final Report: Image Fragment Reconstruction (Self-Supervised)

Author: Shahin Mohammadi  
Date: 2025-09-21

## 1) Model architecture choice and justification

We evaluate four lightweight architectures for learning fragment representations that help regroup 16×16 patches back to their 64×64 source images.

- Convolutional autoencoder (primary)
  - Encoder with channels [32, 64] downsamples 16×16 → 8×8 → 4×4; a 128-D bottleneck captures local textures/patterns; a symmetric decoder reconstructs the fragment.
  - Rationale: preserves spatial structure, offers strong inductive bias for images, and remains compact (~373K params). Good balance for fragment-sized inputs.

- Multi-layer linear autoencoder (baseline)
  - Flattened 768-D input with MLP encoder/decoder, matching depth/size of the conv model.
  - Rationale: isolates the benefit of convolutional inductive biases by removing spatial processing.

- PCA-like single-layer autoencoder (baseline)
  - Minimal linear encoder/decoder (768→128→768).
  - Rationale: linear dimensionality reduction baseline to test necessity of nonlinearity/depth.

- Supervised fragment classifier (baseline)
  - MLP feature extractor (ends at 128-D), with a classification head over ImageNet-64 labels.
  - Rationale: serves as a supervised reference; tests whether label supervision improves embedding quality for clustering.

Why the conv autoencoder?
- Spatial priors matter for small fragments; convolutions are parameter-efficient and capture local structure.
- 128-D latent strikes a balance: compact but expressive for 16×16 fragments.
- Two-layer encoder (channels [32, 64]) avoids over-parameterization while maintaining performance.

## 2) Training strategy

- Data pipeline
  - For each synthetic batch, sample N images (default N=10), split each 64×64 into a 4×4 grid → 16 fragments per image, then shuffle all fragments.
  - Augmentations (train only): random flips/rotations and optional color jitter; validation uses deterministic preprocessing.

- Objective and optimization
  - Autoencoders: MSE reconstruction loss on fragments.
  - Supervised baseline: cross-entropy on source-image labels.
  - Optimizer: Adam (lr=1e-3, weight_decay=1e-4). ReduceLROnPlateau monitors mean clustering score (see below).

- Training control
  - Early stopping on validation mean clustering score (patience=15, min_delta=0.001).
  - Best checkpointing via PyTorch Lightning `ModelCheckpoint` (saves best and last).
  - Logging via CSVLogger.

- Hardware
  - Scales from CPU to multi-GPU (Lightning accelerator="auto"). Evaluation is CPU-friendly.

## 3) Evaluation results and performance analysis

Evaluation uses embeddings from the encoder and clusters fragments with both KMeans and KMedoids. Metrics:
- Adjusted Rand Index (ARI): similarity of partitions (adjusted for chance).
- Normalized Mutual Information (NMI): information overlap of labelings.
- We aggregate KMeans and KMedoids metrics: mean_ari, mean_nmi, and mean_clustering_score = (mean_ari + mean_nmi)/2.

Summary of results (representative run; N=10 images per sample):
- Model (conv AE): ARI≈0.166, NMI≈0.377, Purity≈0.403
- Baselines:
  - Random: ARI≈0.000, NMI≈0.133, Purity≈0.225
  - Raw-pixel KMeans: ARI≈0.152, NMI≈0.355, Purity≈0.386

Analysis
- The conv autoencoder improves upon random and raw-pixel baselines across ARI/NMI/Purity, indicating meaningful embeddings beyond raw pixel similarity.
- Gains are modest at N=10 due to limited context per fragment and the problem’s difficulty; stronger objectives could further separate clusters.

## 4) Challenges and improvements

Challenges
- Fragment ambiguity: many 16×16 crops share texture/color statistics across images.
- Limited context: without positional cues or larger patches, clustering is difficult.
- Evaluation sensitivity: small changes in augmentation or N can influence clustering stability.

Improvements
- Contrastive/self-supervised objectives (e.g., SimCLR, MoCo, BYOL) on fragments.
- Clustering-aware training (e.g., DeepCluster, SwAV) to align embeddings with cluster structure.
- Context modeling: cross-fragment attention or relational reasoning to capture inter-fragment consistency.
- Multi-scale inputs: mix 16×16 with larger patches to improve separability.
- Positional priors: learnable coordinates or sinusoidal embeddings when reconstructing 4×4 grids.

## 5) Next steps

- Architecture exploration: add shallow attention blocks, residual connections, and channel scaling sweeps.
- Objective variants: hybrid reconstruction + contrastive/clustering losses; experiment with temperature and queue sizes.
- Data curriculum: vary N (images per batch) and patch sizes; augment harder with color jitter/cutout.
- Scaling experiments: longer training, more steps per epoch, and distributed runs.
- Downstream tasks: test embeddings on retrieval or puzzle reassembly to quantify utility beyond clustering.
