"""
ImageNet-64 Case Study Package

A complete implementation for ImageNet-64 image classification including:
- Dataset loading and preprocessing
- CNN model architectures
- Training pipeline
- Evaluation tools
"""

from .data import Imagenet64, get_augmentation_transforms, normalize_img
from .download import create_sample_data, download_imagenet64, setup_data_structure
from .models import create_cnn_model, create_transfer_learning_model
from .trainer import ImageNet64Trainer

__version__ = "1.0.0"
__author__ = "Shahin Mohammadi"

__all__ = [
    "Imagenet64",
    "normalize_img",
    "get_augmentation_transforms",
    "create_cnn_model",
    "create_transfer_learning_model",
    "ImageNet64Trainer",
    "download_imagenet64",
    "setup_data_structure",
    "create_sample_data"
]
