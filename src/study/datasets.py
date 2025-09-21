"""Dataset classes for ImageNet-64 and fragment-based learning."""

from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

N_CLASSES = 1000
IMAGE_SIZE = (64, 64)


@dataclass
class FragmentBatch:
    fragments: torch.Tensor
    source_ids: torch.Tensor


class ImageNet64Dataset(Dataset):

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

        assert len(self.images) == len(self.labels)
        assert len(np.unique(self.labels)) <= N_CLASSES

    def _load_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        if (self.data_path / "train_data").exists():
            train_dir = self.data_path / "train_data"
        else:
            train_dir = self.data_path / "imagenet64" / "train_data"
        train_files = os.listdir(train_dir)
        x_train: list[np.ndarray] = []
        y_train: list[np.ndarray] = []

        for train_file in tqdm(train_files, desc="Loading training data"):
            with open(train_dir / train_file, "rb") as fo:
                data = pickle.load(fo)
                x = data["data"].reshape((data["data"].shape[0], 3, 64, 64)).transpose((0, 2, 3, 1))
                y = np.array(data["labels"]) - 1  # Convert to 0-based indexing

                x_train.append(x)
                y_train.append(y)

        x_train_arr: np.ndarray = np.concatenate(x_train, axis=0)
        y_train_arr: np.ndarray = np.concatenate(y_train, axis=0)

        return x_train_arr, y_train_arr

    def _load_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        if (self.data_path / "dev_data").exists():
            dev_path = self.data_path / "dev_data" / "dev_data_batch_1"
        else:
            dev_path = self.data_path / "dev_data" / "dev_data_batch_1"
        with open(dev_path, "rb") as fo:
            data = pickle.load(fo)
            x_test = data["data"].reshape((data["data"].shape[0], 3, 64, 64)).transpose((0, 2, 3, 1))
            y_test = np.array(data["labels"]) - 1  # Convert to 0-based indexing

        return x_test, y_test

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]

        image = image.astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


class FragmentBatchDataset(Dataset):

    def __init__(
        self,
        images_dir: str | Path,
        images_per_sample: int = 10,
        steps_per_epoch: int = 100,
        seed: int | None = 42,
        augment: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.images_per_sample = images_per_sample
        self.steps_per_epoch = steps_per_epoch
        self.augment = augment

        rng = np.random.RandomState(seed if seed is not None else 0)
        self._rng = rng

        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.image_paths: list[Path] = [p for p in self.images_dir.rglob("*") if p.suffix.lower() in exts]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found under {self.images_dir}. Supported: {exts}")

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> FragmentBatch:
        from .transforms import basic_augment, split_into_patches

        # Sample N distinct images
        idxs = self._rng.choice(len(self.image_paths), size=self.images_per_sample, replace=False)
        chosen = [self.image_paths[i] for i in idxs]

        fragments: list[torch.Tensor] = []
        labels: list[int] = []
        for src_id, path in enumerate(chosen):
            img = Image.open(path).convert("RGB")
            # Ensure 64x64
            img = img.resize((64, 64), Image.Resampling.BILINEAR)
            if self.augment:
                img = basic_augment(self._rng, img)
            patches = split_into_patches(img)
            fragments.extend(patches)
            labels.extend([src_id] * len(patches))

        combined = list(zip(fragments, labels, strict=True))
        random.shuffle(combined)
        fragments_shuffled, labels_shuffled = zip(*combined, strict=True)
        fragments = list(fragments_shuffled)
        labels = list(labels_shuffled)

        fragments_tensor = torch.stack(fragments)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return FragmentBatch(fragments=fragments_tensor, source_ids=labels_tensor)


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

        n_classes = N_CLASSES
        assert len(np.unique(self.data["y_train"])) <= n_classes
        assert len(np.unique(self.data["y_train"])) >= len(np.unique(self.data["y_test"]))
