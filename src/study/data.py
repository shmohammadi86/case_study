import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

N_CLASSES = 1000
IMAGE_SIZE = (64, 64)


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


class ImageNet64Dataset(Dataset):
    """
    PyTorch Dataset for ImageNet-64 data.

    Parameters
    ----------
    data_path : str or Path
        Path to the ImageNet-64 data directory
    split : str
        Dataset split ('train' or 'test')
    transform : transforms.Compose, optional
        Transforms to apply to images
    """

    def __init__(self, data_path: str | Path, split: str = "train", transform: transforms.Compose | None = None):
        self.data_path = Path(str(data_path))
        self.split = split
        self.transform = transform

        if split == "train":
            self.images, self.labels = self._load_train_data()
        elif split == "test":
            self.images, self.labels = self._load_test_data()
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")

        # Validate data
        assert len(self.images) == len(self.labels)
        assert len(np.unique(self.labels)) <= N_CLASSES

    def _load_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load training data from pickle files."""
        train_files = os.listdir(self.data_path / "train_data")
        x_train: list[np.ndarray] = []
        y_train: list[np.ndarray] = []

        for train_file in tqdm(train_files, desc="Loading training data"):
            with open(self.data_path / "train_data" / train_file, "rb") as fo:
                data = pickle.load(fo)
                x = data["data"].reshape((data["data"].shape[0], 3, 64, 64)).transpose((0, 2, 3, 1))
                y = np.array(data["labels"]) - 1  # Convert to 0-based indexing

                x_train.append(x)
                y_train.append(y)

        x_train_arr: np.ndarray = np.concatenate(x_train, axis=0)
        y_train_arr: np.ndarray = np.concatenate(y_train, axis=0)

        return x_train_arr, y_train_arr

    def _load_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load test data from pickle file."""
        with open(self.data_path / "dev_data/dev_data_batch_1", "rb") as fo:
            data = pickle.load(fo)
            x_test = data["data"].reshape((data["data"].shape[0], 3, 64, 64)).transpose((0, 2, 3, 1))
            y_test = np.array(data["labels"]) - 1  # Convert to 0-based indexing

        return x_test, y_test

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to uint8 for PIL compatibility
        image = image.astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)  # HWC to CHW

        label = torch.tensor(label, dtype=torch.long)

        return image, label


class Imagenet64:
    """
    Legacy wrapper class for backward compatibility.
    Use ImageNet64Dataset for new implementations.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(str(data_path))

        # Load all data into memory for legacy compatibility
        train_dataset = ImageNet64Dataset(data_path, split="train")
        test_dataset = ImageNet64Dataset(data_path, split="test")

        self.data = {
            "x_train": train_dataset.images,
            "y_train": train_dataset.labels,
            "x_test": test_dataset.images,
            "y_test": test_dataset.labels,
        }

        # Validate data (basic sanity checks)
        n_classes = N_CLASSES
        assert len(np.unique(self.data["y_train"])) <= n_classes
        assert len(np.unique(self.data["y_train"])) >= len(np.unique(self.data["y_test"]))
