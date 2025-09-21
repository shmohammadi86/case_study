# ImageNet-64 Case Study

A complete implementation for ImageNet-64 image classification using TensorFlow/Keras. This project provides a comprehensive pipeline for training CNN models on the ImageNet-64 dataset with 64x64 pixel images across 1000 classes.

## Features

- **Data Management**: Automated dataset download and preprocessing
- **Multiple Architectures**: Simple CNN, ResNet-style, and Efficient CNN models
- **Transfer Learning**: Support for pre-trained models (ResNet50, EfficientNet)
- **Training Pipeline**: Complete training with callbacks, checkpointing, and monitoring
- **Evaluation Tools**: Comprehensive metrics and visualization
- **Configuration**: JSON-based configuration system
- **Examples**: Basic and advanced usage examples

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/shmohammadi86/case_study.git
cd case_study

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from study import create_sample_data, ImageNet64Trainer

# Create sample data for testing
create_sample_data("data/imagenet64", n_samples=1000)

# Configure model and training
model_config = {
    "architecture": "simple",
    "input_shape": (64, 64, 3),
    "num_classes": 1000
}

training_config = {
    "batch_size": 32,
    "epochs": 50,
    "augmentation": True
}

# Train model
trainer = ImageNet64Trainer(
    data_path="data/imagenet64",
    model_config=model_config,
    training_config=training_config,
    output_dir="outputs"
)

results = trainer.train()
```

### 3. Command Line Training

```bash
# Train with default configuration
python train.py --create-sample-data

# Train with custom configuration
python train.py --config config/training_config.json --output-dir outputs/experiment1

# Create sample data only
python train.py --create-sample-data --data-dir data/test
```

## Project Structure

```
case_study/
├── src/study/           # Main package
│   ├── data.py         # Dataset loading and preprocessing
│   ├── models.py       # CNN model architectures
│   ├── trainer.py      # Training pipeline
│   └── download.py     # Dataset download utilities
├── examples/           # Usage examples
│   ├── basic_usage.py
│   └── advanced_usage.py
├── config/            # Configuration files
│   └── training_config.json
├── train.py          # Main training script
└── README.md
```

## Model Architectures

### Simple CNN
- 4 convolutional layers with batch normalization
- MaxPooling and dropout for regularization
- ~2M parameters

### ResNet-style CNN
- Residual blocks with skip connections
- Deeper architecture with better gradient flow
- ~8M parameters

### Transfer Learning
- Pre-trained ResNet50 or EfficientNet backbone
- Custom classification head
- Fine-tuning support

## Configuration

The training pipeline uses JSON configuration files:

```json
{
  "model": {
    "architecture": "simple",
    "input_shape": [64, 64, 3],
    "num_classes": 1000
  },
  "training": {
    "batch_size": 32,
    "epochs": 50,
    "augmentation": true,
    "early_stopping": true
  }
}
```

## Examples

### Basic Training
```bash
python examples/basic_usage.py
```

### Architecture Comparison
```bash
python examples/advanced_usage.py
```

### Transfer Learning
```python
model_config = {
    "use_transfer_learning": True,
    "base_model": "ResNet50",
    "trainable_layers": 10
}
```

## Dataset

This implementation expects ImageNet-64 data in the following structure:
```
data/imagenet64/
├── train_data/
│   ├── train_data_batch_1
│   ├── train_data_batch_2
│   └── ...
└── dev_data/
    └── dev_data_batch_1
```

For testing without the full dataset, use `create_sample_data()` to generate synthetic data.

## Results

The training pipeline provides:
- Training/validation accuracy and loss curves
- Top-1 and Top-5 accuracy metrics
- Model checkpoints and final weights
- Sample prediction visualizations
- Comprehensive logging

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- tqdm

## License

MIT License - see LICENSE file for details.

## Author

Shahin Mohammadi (shahin.mohammadi@gmail.com)Machine Learning