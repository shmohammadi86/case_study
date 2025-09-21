"""
ImageNet-64 Dataset Download and Setup Utilities
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGENET64_URLS = {
    "train": "https://image-net.org/data/downsample/Imagenet64_train.zip",
    "val": "https://image-net.org/data/downsample/Imagenet64_val.zip",
}


def download_imagenet64(data_dir: str = "data/imagenet64", subset: str = "both") -> Path:
    """
    Download ImageNet-64 dataset.

    Parameters
    ----------
    data_dir : str
        Directory to save the dataset
    subset : str
        Which subset to download: 'train', 'val', or 'both'

    Returns
    -------
    Path
        Path to the downloaded dataset directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if subset in ["train", "both"]:
        train_path = data_path / "train_data"
        if not train_path.exists():
            logger.info("Downloading ImageNet-64 training data...")
            # Note: Actual download would require authentication
            # This is a placeholder for the download logic
            logger.warning("ImageNet-64 requires manual download from image-net.org")
            logger.info("Please download Imagenet64_train.zip and extract to train_data/")

    if subset in ["val", "both"]:
        val_path = data_path / "dev_data"
        if not val_path.exists():
            logger.info("Downloading ImageNet-64 validation data...")
            logger.warning("ImageNet-64 requires manual download from image-net.org")
            logger.info("Please download Imagenet64_val.zip and extract to dev_data/")

    return data_path


def setup_data_structure(data_dir: str) -> bool:
    """
    Verify and setup the expected data structure.

    Parameters
    ----------
    data_dir : str
        Path to the dataset directory

    Returns
    -------
    bool
        True if data structure is correct
    """
    data_path = Path(data_dir)

    # Check for required directories
    train_dir = data_path / "train_data"
    dev_dir = data_path / "dev_data"

    if not train_dir.exists():
        logger.error(f"Training data directory not found: {train_dir}")
        return False

    if not dev_dir.exists():
        logger.error(f"Development data directory not found: {dev_dir}")
        return False

    # Check for data files
    train_files = list(train_dir.glob("*.pkl")) + list(train_dir.glob("*batch*"))
    dev_files = list(dev_dir.glob("*.pkl")) + list(dev_dir.glob("*batch*"))

    if not train_files:
        logger.error("No training data files found")
        return False

    if not dev_files:
        logger.error("No development data files found")
        return False

    logger.info(f"Found {len(train_files)} training files and {len(dev_files)} dev files")
    return True


def create_sample_data(data_dir: str, n_samples: int = 1000) -> None:
    """
    Create sample data for testing (when real ImageNet-64 is not available).

    Parameters
    ----------
    data_dir : str
        Directory to create sample data
    n_samples : int
        Number of sample images per batch
    """
    import pickle

    import numpy as np

    data_path = Path(data_dir)
    train_dir = data_path / "train_data"
    dev_dir = data_path / "dev_data"

    train_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)

    # Create sample training data
    for i in range(3):  # 3 training batches
        sample_data = {
            "data": np.random.randint(0, 256, (n_samples, 3 * 64 * 64), dtype=np.uint8),
            "labels": np.random.randint(1, 1001, n_samples).tolist(),
        }

        with open(train_dir / f"train_data_batch_{i + 1}", "wb") as f:
            pickle.dump(sample_data, f)

    # Create sample dev data
    sample_dev_data = {
        "data": np.random.randint(0, 256, (n_samples // 10, 3 * 64 * 64), dtype=np.uint8),
        "labels": np.random.randint(1, 1001, n_samples // 10).tolist(),
    }

    with open(dev_dir / "dev_data_batch_1", "wb") as f:
        pickle.dump(sample_dev_data, f)

    logger.info(f"Created sample data in {data_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample data for testing
    data_dir = "data/imagenet64"
    create_sample_data(data_dir)

    # Verify setup
    if setup_data_structure(data_dir):
        print("Data structure is ready!")
    else:
        print("Data structure setup failed!")
