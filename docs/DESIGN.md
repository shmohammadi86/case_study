# ImageNet-64 Fragment Autoencoder Design Document

## Overview

This document describes the design and implementation of our ImageNet-64 fragment-based autoencoder system. The system is designed to process 16x16 image fragments extracted from 64x64 ImageNet images using a configurable convolutional autoencoder architecture.

## 1. Dataset Implementation

### 1.1 ImageNet64Dataset

The `ImageNet64Dataset` class in `src/study/datasets.py` provides a PyTorch-compatible interface for loading ImageNet-64 data:

**Key Features:**
- Loads data from pickle files containing 64x64 RGB images
- Supports both training and test splits
- Handles label conversion from 1-based to 0-based indexing
- Provides optional transform pipeline integration
- Validates data integrity (1000 classes maximum)

**Data Loading Process:**
1. **Training Data**: Iterates through multiple pickle files in `train_data/` directory
2. **Test Data**: Loads from single file `dev_data/dev_data_batch_1`
3. **Format Conversion**: Reshapes from flat arrays to (H, W, C) format
4. **Normalization**: Converts to [0, 1] range and CHW tensor format

### 1.2 FragmentBatchDataset

The `FragmentBatchDataset` class generates batches of 16x16 image fragments:

**Architecture:**
- Samples N images per batch (default: 10)
- Splits each 64x64 image into 4x4 grid = 16 fragments of 16x16 each
- Shuffles fragments to break spatial locality
- Returns `FragmentBatch` dataclass with fragments and source IDs

**Fragment Processing Pipeline:**
1. **Image Sampling**: Randomly selects images from directory
2. **Resizing**: Ensures all images are exactly 64x64
3. **Augmentation**: Optional random flips and rotations
4. **Fragmentation**: Splits into 16x16 patches using `split_into_patches()`
5. **Shuffling**: Randomizes fragment order to prevent spatial bias

### 1.3 Transform Functions

Located in `src/study/transforms.py`:

**Core Functions:**
- `normalize_img()`: Converts [0, 255] → [0, 1] range
- `get_augmentation_transforms()`: Provides training/validation transforms
- `basic_augment()`: Simple PIL-based augmentations (flip, rotate)
- `split_into_patches()`: Converts 64x64 image → 16 fragments of 16x16

**Augmentation Strategy:**
- Random horizontal flips (50% probability)
- Random rotation (-10° to +10°)
- Color jitter for training robustness
- Maintains fragment spatial integrity

## 2. Model Architecture Comparison

### 2.1 Four Model Variants for Comprehensive Evaluation

We implement four distinct model architectures to provide comprehensive baseline comparisons for fragment reconstruction and representation learning:

#### 2.1.1 Convolutional Autoencoder (Primary Model)
The `ConvAutoencoder` class implements a symmetric encoder-decoder architecture optimized for spatial feature learning:

**Core Architecture:**
```
Input: [B, 3, 16, 16]
├── Encoder: Convolutional downsampling (3→32→64 channels)
├── Bottleneck: Fully connected latent space (128D)
├── Decoder: Transposed convolutional upsampling (64→32→3 channels)
Output: [B, 3, 16, 16]
```

**Key Components:**
- **Encoder**: Progressive downsampling with stride-2 convolutions
- **Latent Space**: Fully connected bottleneck (default: 128 dimensions)
- **Decoder**: Symmetric upsampling using transposed convolutions
- **Building Blocks**: Modular `ConvBlock` and `DeconvBlock` from `modules.py`

#### 2.1.2 Multi-Layer Linear Autoencoder (Baseline 1)
The `LinearAutoencoder` provides a non-convolutional baseline with equivalent depth:

**Architecture:**
```
Input: [B, 3, 16, 16] → Flatten → [B, 768]
├── Encoder: 768 → 512 → 256 → 128 (with ReLU + Dropout)
├── Decoder: 128 → 256 → 512 → 768 (with ReLU + Dropout)
Output: [B, 768] → Reshape → [B, 3, 16, 16]
```

**Design Rationale:**
- **Depth Matching**: Same number of layers as convolutional model
- **Flattened Input**: Tests importance of spatial structure preservation
- **Regularization**: Dropout (0.1) prevents overfitting in linear layers
- **Capacity Control**: Hidden dimensions chosen to match conv model complexity

#### 2.1.3 PCA-Like Single Layer Autoencoder (Baseline 2)
The `PCAAutoencoder` provides minimal architecture for linear dimensionality reduction:

**Architecture:**
```
Input: [B, 3, 16, 16] → Flatten → [B, 768]
├── Encoder: Single linear layer 768 → 128
├── Decoder: Single linear layer 128 → 768 + Sigmoid
Output: [B, 768] → Reshape → [B, 3, 16, 16]
```

**Design Rationale:**
- **Minimal Complexity**: Single linear transformation (similar to PCA)
- **Linear Baseline**: Tests if non-linearity is necessary
- **Xavier Initialization**: Optimized for linear learning
- **Direct Comparison**: Isolates effect of architectural depth

#### 2.1.4 Supervised Fragment Classifier (Baseline 3)
The `SupervisedClassifier` uses classification loss instead of reconstruction:

**Architecture:**
```
Input: [B, 3, 16, 16] → Flatten → [B, 768]
├── Feature Extractor: 768 → 512 → 256 → 128 (with ReLU + Dropout 0.2)
├── Classifier Head: 128 → num_classes (1000 ImageNet classes)
Output: [B, num_classes] logits
```

**Design Rationale:**
- **Supervised Learning**: Uses fragment source image labels
- **Classification Loss**: Cross-entropy instead of reconstruction MSE
- **Feature Learning**: Tests if supervised signals improve clustering
- **Higher Dropout**: 0.2 for classification robustness

### 2.2 Configurable Architecture

**Parameters:**
- `input_channels`: Input channels (default: 3 for RGB)
- `latent_dim`: Latent space dimensionality (default: 128)
- `encoder_channels`: List defining layer progression (default: [32, 64])
- `input_size`: Input image size (default: 16 for fragments)

**Architecture Flexibility:**
- **Layer Count**: Configurable via `encoder_channels` list length
- **Channel Progression**: Customizable feature map sizes
- **Input Size**: Supports different image dimensions
- **Symmetric Design**: Decoder automatically mirrors encoder

### 2.3 Default Architecture Choices

**For 16x16 Fragments:**
- **Default Channels**: `[32, 64]` (2 layers)
- **Spatial Progression**: 16×16 → 8×8 → 4×4
- **Parameter Count**: ~373K parameters
- **Latent Dimension**: 128 (compact representation)

**Design Rationale:**

1. **Two-Layer Architecture**: 
   - Balances feature extraction with parameter efficiency
   - Prevents over-parameterization for small 16x16 inputs
   - Maintains meaningful spatial information at 4x4 bottleneck

2. **Channel Progression [32, 64]**:
   - Gradual feature complexity increase
   - Sufficient capacity for fragment-level features
   - Avoids excessive parameters for small input size

3. **16x16 Input Size**:
   - Matches fragment size from 4x4 grid decomposition
   - Optimal for local texture and pattern learning
   - Enables efficient batch processing of fragments

4. **128-Dimensional Latent Space**:
   - Provides good compression ratio (768 → 128)
   - Sufficient capacity for fragment reconstruction
   - Enables meaningful latent space interpolation

### 2.4 Architecture Constraints

**Spatial Limitations:**
- Maximum layers for 16x16 input: 4 layers (16→8→4→2→1)
- Current default uses 2 layers for optimal balance
- Validation prevents impossible architectures

**Parameter Scaling:**
- **1 Layer**: ~1.1M parameters (over-parameterized)
- **2 Layers**: ~373K parameters (optimal)
- **3 Layers**: ~631K parameters (acceptable)

## 3. Evaluation Metrics and Model Selection

### 3.1 Clustering-Based Evaluation Strategy

We evaluate all models using clustering metrics on learned embeddings to assess representation quality:

#### 3.1.1 Dual Clustering Algorithm Approach
**K-means Clustering:**
- Standard centroid-based clustering algorithm
- Fast and widely used baseline
- Assumes spherical cluster shapes
- Sensitive to initialization (mitigated with random_state=42)

**K-medoids Clustering:**
- Uses actual data points as cluster centers
- More robust to outliers than K-means
- Better handles non-spherical cluster shapes
- Provides complementary clustering perspective

#### 3.1.2 Clustering Metrics
**Adjusted Rand Index (ARI):**
- Measures similarity between predicted and true clusters
- Range: [-1, 1], where 1 = perfect clustering
- Adjusted for chance (0 = random clustering)
- Robust to cluster size imbalances

**Normalized Mutual Information (NMI):**
- Measures information shared between clusterings
- Range: [0, 1], where 1 = perfect clustering
- Normalized to account for cluster count differences
- Complementary to ARI for comprehensive evaluation

#### 3.1.3 Aggregated Scoring System
```python
# Compute clustering metrics for both algorithms
kmeans_ari, kmeans_nmi = cluster_with_kmeans(embeddings, labels)
kmedoids_ari, kmedoids_nmi = cluster_with_kmedoids(embeddings, labels)

# Aggregate metrics for robust evaluation
mean_ari = (kmeans_ari + kmedoids_ari) / 2
mean_nmi = (kmeans_nmi + kmedoids_nmi) / 2
mean_clustering_score = (mean_ari + mean_nmi) / 2
```

**Rationale for Aggregation:**
- **Robustness**: Reduces bias from single clustering algorithm
- **Fairness**: No model advantages from algorithm-specific biases
- **Comprehensive**: Captures both partition similarity (ARI) and information content (NMI)
- **Model Selection**: Single metric for early stopping and checkpointing

### 3.2 TorchMetrics Integration for Production-Ready Training

#### 3.2.1 Professional Metric Computation
We use `torchmetrics` instead of sklearn for several critical advantages:

**Distributed Training Support:**
```python
# Automatic synchronization across GPUs/nodes
self.val_mean_clustering_score = MeanMetric()
self.log("val_mean_clustering_score", self.val_mean_clustering_score)
```

**Memory Efficiency:**
- Incremental metric computation (no storing all predictions)
- Automatic state management across epochs
- Proper tensor device handling

**PyTorch Lightning Integration:**
- Seamless logging with `self.log()`
- Automatic metric reset between epochs
- Progress bar integration for real-time monitoring

#### 3.2.2 Metric Implementation Details
```python
# Training step metric updates
self.train_kmeans_ari(clustering_metrics["kmeans_ari"])
self.train_kmeans_nmi(clustering_metrics["kmeans_nmi"])
self.train_mean_clustering_score(clustering_metrics["mean_clustering_score"])

# Validation step with automatic aggregation
self.log("val_mean_clustering_score", self.val_mean_clustering_score, 
         on_step=False, on_epoch=True, prog_bar=True)
```

### 3.3 Model Selection and Early Stopping Strategy

#### 3.3.1 Primary Metric: Mean Clustering Score
**Monitoring Target:** `val_mean_clustering_score`
- Combines ARI and NMI from both clustering algorithms
- Single metric for consistent model comparison
- Higher values indicate better representation learning

#### 3.3.2 Early Stopping Configuration
```python
early_stop_callback = EarlyStopping(
    monitor="val_mean_clustering_score",
    mode="max",                    # Higher clustering score is better
    patience=15,                   # Wait 15 epochs for improvement
    min_delta=0.001,              # Minimum improvement threshold
    verbose=True
)
```

#### 3.3.3 Model Checkpointing
```python
checkpoint_callback = ModelCheckpoint(
    monitor="val_mean_clustering_score",
    mode="max",
    save_top_k=1,                 # Keep only best model
    filename="best-{epoch}-{val_mean_clustering_score:.4f}"
)
```

**Benefits of This Approach:**
- **Unsupervised Evaluation**: No reliance on classification accuracy
- **Representation Quality**: Directly measures embedding usefulness
- **Fair Comparison**: Same evaluation criteria across all 4 model types
- **Robust Selection**: Aggregated metrics reduce noise in model selection

## 4. Integration and Usage

### 4.1 Factory Functions for All Model Types

```python
# Convolutional autoencoder (primary model)
model = create_conv_autoencoder(latent_dim=128)

# Linear autoencoder baseline
model = create_linear_autoencoder(latent_dim=128)

# PCA-like autoencoder baseline  
model = create_pca_autoencoder(latent_dim=128)

# Supervised classifier baseline
model = create_supervised_classifier(num_classes=1000, latent_dim=128)
```

### 4.2 Unified Training Pipeline

```python
trainer = FragmentAutoencoderTrainer(
    model_type="conv",            # "conv", "linear", "pca", "supervised"
    data_path="data",
    images_per_batch=10,
    batch_size=8,
    epochs=100,
    latent_dim=128,
    early_stopping_patience=15
)
```

**Training Features:**
- Automatic model type selection and loss function
- TorchMetrics integration for robust metric computation
- Early stopping based on clustering performance
- Comprehensive logging of all metrics

## 4. Design Benefits

### 4.1 Modularity
- Separate concerns: datasets, transforms, models, modules
- Reusable components across different experiments
- Clean interfaces and type hints

### 4.2 Configurability
- Flexible architecture without code changes
- Easy experimentation with different layer counts
- Adaptable to various input sizes and tasks

### 4.3 Efficiency
- Optimized defaults for 16x16 fragments
- Balanced parameter count vs. capacity
- Efficient fragment processing pipeline

### 4.4 Extensibility
- Easy to add new augmentations
- Modular building blocks for new architectures
- Compatible with transfer learning approaches

## 5. Future Considerations

### 5.1 Potential Enhancements
- Skip connections for better gradient flow
- Attention mechanisms for fragment relationships
- Multi-scale processing capabilities
- Advanced augmentation strategies

### 5.2 Scalability
- Support for larger fragment sizes
- Batch processing optimizations
- Memory-efficient data loading
- Distributed training compatibility

This design provides a solid foundation for fragment-based image reconstruction and representation learning tasks while maintaining flexibility for future research directions.
