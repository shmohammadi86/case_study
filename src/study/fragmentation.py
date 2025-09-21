from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .data import ImageNet64Dataset


@dataclass
class FragmentBatch:
    """A batch containing unordered fragments from N images and their source ids.

    Attributes
    ----------
    fragments: torch.Tensor
        Tensor of shape [num_frags, 3, 16, 16] with values in [0, 1].
    source_ids: torch.Tensor
        Tensor of shape [num_frags] containing integer ids 0..N-1
        indicating the original image each fragment came from.
    """

    fragments: torch.Tensor
    source_ids: torch.Tensor


class FragmentBatchDataset(Dataset):
    """Generates batches of image fragments from a directory of 64x64 RGB images.

    Each item (index ignored) yields a `FragmentBatch` formed by:
    - Sampling N images
    - Splitting each image into a 4x4 grid of 16x16 patches
    - Collecting all patches, shuffling them, and returning fragments and labels
    """

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
        self.image_paths: List[Path] = [
            p for p in self.images_dir.rglob("*") if p.suffix.lower() in exts
        ]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found under {self.images_dir}. Supported: {exts}"
            )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> FragmentBatch:
        # Sample N distinct images
        idxs = self._rng.choice(len(self.image_paths), size=self.images_per_sample, replace=False)
        chosen = [self.image_paths[i] for i in idxs]

        fragments: List[torch.Tensor] = []
        labels: List[int] = []
        for src_id, path in enumerate(chosen):
            img = Image.open(path).convert("RGB")
            # Ensure 64x64
            img = img.resize((64, 64), Image.BILINEAR)
            if self.augment:
                img = basic_augment(self._rng, img)
            patches = split_into_patches(img)
            fragments.extend(patches)
            labels.extend([src_id] * len(patches))

        # Shuffle fragments
        order = list(range(len(fragments)))
        random.shuffle(order)
        fragments = [fragments[i] for i in order]
        labels = [labels[i] for i in order]

        frags_tensor = torch.stack(fragments, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return FragmentBatch(fragments=frags_tensor, source_ids=labels_tensor)


# --------------------
# Module-level helpers
# --------------------

def split_into_patches(img: Image.Image) -> List[torch.Tensor]:
    """Split a 64x64 PIL image into a 4x4 grid of 16x16 patches as tensors [3,16,16]."""
    arr = np.asarray(img, dtype=np.uint8)
    patches: List[torch.Tensor] = []
    for ry in range(4):
        for rx in range(4):
            patch = arr[ry * 16 : (ry + 1) * 16, rx * 16 : (rx + 1) * 16, :]
            # Make a writable copy to avoid PyTorch warning about non-writable arrays
            patch = patch.copy()
            t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patches.append(t)
    return patches


def basic_augment(rng: np.random.RandomState, img: Image.Image) -> Image.Image:
    """Lightweight augmentation: horizontal flip p=0.5."""
    if rng.rand() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

    def _basic_augment(self, img: Image.Image) -> Image.Image:
        return basic_augment(self._rng, img)

def collate_fragment_batch(batch: List[FragmentBatch]) -> FragmentBatch:
    # We expect DataLoader(batch_size=1). If larger, we merge.
    frags = torch.cat([b.fragments for b in batch], dim=0)
    labels = torch.cat([b.source_ids for b in batch], dim=0)
    return FragmentBatch(fragments=frags, source_ids=labels)


def _has_pickle_batches(dir_path: Path, split: str) -> bool:
    """Detect ImageNet-64 pickle batches inside the given split directory."""
    if split == "train":
        pattern = "train_data_batch_*"
    else:
        pattern = "dev_data_batch_*"
    return any(dir_path.glob(pattern))


class FragmentBatchDatasetFromPickle(Dataset):
    """Fragmentation dataset backed by ImageNet-64 pickle batches.

    Uses `ImageNet64Dataset` to load arrays, then samples N images per sample and
    returns their fragmented patches as an unordered collection.
    """

    def __init__(
        self,
        base_dir: str | Path,
        split: str = "train",
        images_per_sample: int = 10,
        steps_per_epoch: int = 100,
        seed: int | None = 42,
        augment: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.images_per_sample = images_per_sample
        self.steps_per_epoch = steps_per_epoch
        self.augment = augment

        rng = np.random.RandomState(seed if seed is not None else 0)
        self._rng = rng

        # Load underlying dataset arrays
        split_name = "train" if split == "train" else "test"
        self.ds = ImageNet64Dataset(self.base_dir, split=split_name, transform=None)
        self.num_images = len(self.ds)
        if self.num_images <= 0:
            raise RuntimeError(f"No images found in batches under {self.base_dir} for split={split}")

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> FragmentBatch:
        # sample N distinct indices
        idxs = self._rng.choice(self.num_images, size=self.images_per_sample, replace=False)
        fragments: List[torch.Tensor] = []
        labels: List[int] = []
        for sid, i in enumerate(idxs):
            img_np, _ = self.ds.images[i], self.ds.labels[i]
            # ensure PIL image for augmentation & resizing safety
            img = Image.fromarray(img_np.astype(np.uint8), mode="RGB")
            if self.augment:
                img = basic_augment(self._rng, img)
            patches = split_into_patches(img)
            fragments.extend(patches)
            labels.extend([sid] * len(patches))

        order = list(range(len(fragments)))
        random.shuffle(order)
        fragments = [fragments[i] for i in order]
        labels = [labels[i] for i in order]
        frags_tensor = torch.stack(fragments, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return FragmentBatch(fragments=frags_tensor, source_ids=labels_tensor)
