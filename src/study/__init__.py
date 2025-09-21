"""
ImageNet-64 Case Study Package

A focused implementation for image processing with convolutional autoencoders including:
- Dataset loading and preprocessing
- Convolutional autoencoder architecture
- Training pipeline
- Evaluation tools
"""
# isort: skip_file

from .datasets import FragmentBatch, FragmentBatchDataset, ImageNet64Dataset, Imagenet64
from .models import ConvAutoencoder, count_parameters, create_autoencoder
from .modules import ConvBlock, DeconvBlock
from .trainer import FragmentAutoencoderTrainer
from .transforms import get_augmentation_transforms, normalize_img

__version__ = "1.0.0"
__author__ = "Shahin Mohammadi"

__all__ = [
    # Datasets
    "ImageNet64Dataset",
    "Imagenet64",
    "FragmentBatchDataset",
    "FragmentBatch",
    # Models
    "ConvAutoencoder",
    "create_autoencoder",
    "count_parameters",
    # Modules
    "ConvBlock",
    "DeconvBlock",
    # Transforms
    "normalize_img",
    "get_augmentation_transforms",
    # Training
    "FragmentAutoencoderTrainer",
]
