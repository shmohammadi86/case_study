"""
Data preprocessing and augmentation transforms.
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def normalize_img(img_batch: np.ndarray) -> torch.Tensor:
    """
    Normalize image batch from [0, 255] to [0, 1] range.

    Parameters
    ----------
    img_batch : np.ndarray
        Image batch with values in [0, 255] range

    Returns
    -------
    torch.Tensor
        Normalized tensor with values in [0, 1] range
    """
    img_tensor = torch.from_numpy(img_batch).float()
    normalized_tensor = img_tensor / 255.0
    return normalized_tensor


def get_augmentation_transforms(training: bool = True) -> transforms.Compose:
    """
    Get data augmentation transforms for training or validation.

    Parameters
    ----------
    training : bool
        Whether to apply training augmentations

    Returns
    -------
    transforms.Compose
        Composed transforms
    """
    if training:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )


def basic_augment(rng: np.random.RandomState, img: Image.Image) -> Image.Image:
    """Apply basic augmentations to a PIL image."""
    # Random horizontal flip
    if rng.rand() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random rotation (-10 to 10 degrees)
    angle = rng.uniform(-10, 10)
    img = img.rotate(angle, fillcolor=(128, 128, 128))
    return img


def split_into_patches(img: Image.Image) -> list[torch.Tensor]:
    """Split a 64x64 PIL image into 16 patches of 16x16 each."""
    patches = []
    img_array = np.array(img)  # Shape: (64, 64, 3)
    for i in range(4):
        for j in range(4):
            patch = img_array[i*16:(i+1)*16, j*16:(j+1)*16, :]
            # Convert to tensor and normalize to [0, 1]
            patch_tensor = torch.from_numpy(patch).float() / 255.0
            # Convert from HWC to CHW
            patch_tensor = patch_tensor.permute(2, 0, 1)
            patches.append(patch_tensor)
    return patches
