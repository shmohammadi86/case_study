# Case Study: Image Fragment Reconstruction (Self-Supervised)

This repository implements a minimal, self-supervised pipeline that groups unordered 16×16 fragments back to their source 64×64 images. The goal is to learn fragment embeddings with a lightweight CNN autoencoder and recover the original grouping via clustering.

## What’s included

- Minimal Python modules under `src/study/` for data loading, fragment generation, and the model.
- A single public notebook with the full workflow (training, CPU-centric evaluation, baselines, visualization):
  - `experiments/notebooks/fragment_reconstruction_pipeline.ipynb`
- Final artifacts tracked with Git LFS:
  - `outputs/fragment_clustering_baseline/results/metrics.json`
  - `outputs/fragment_clustering_baseline/results/visualization_real.png`

## Git LFS setup (required for data and checkpoints)

This repository stores large files (datasets in `data/`, outputs/artifacts in `outputs/`, and `*.ckpt` checkpoints) using Git LFS. You must install and enable Git LFS to fetch these files correctly.

1) Install Git LFS (one-time):

```bash
# Linux
sudo apt-get update && sudo apt-get install -y git-lfs

# Or via package manager of your distribution
# See: https://git-lfs.com/
```

2) Initialize LFS in your Git configuration (one-time):

```bash
git lfs install
```

3) Clone and fetch LFS files:

```bash
git clone https://github.com/shmohammadi86/case_study.git
cd case_study
git lfs pull
```

If you cloned before LFS migration (history was rewritten), reset your local main to the new history:

```bash
git fetch origin
git checkout main
git reset --hard origin/main
git lfs pull
```

Verify LFS objects are present:

```bash
git lfs ls-files | head
git lfs fsck
```

## Dataset

The pipeline expects the legacy ImageNet-64 pickle batches at:

```text
/mnt/localssd/datasets/case/
├── train_data/
└── dev_data/
```
The code auto-detects this format and loads it without modification.

Local layout tip (optional): you may symlink or copy your data into the repository under `data/` so the examples work out-of-the-box:

```text
data/
├── train_data/
└── dev_data/
```

## Results & Report

- Final report: `experiments/docs/final_report.md`
- Notebook: `experiments/notebooks/fragment_reconstruction_pipeline.ipynb`
- Artifacts:
  - Metrics (ARI, NMI, Purity): `outputs/fragment_clustering_baseline/results/metrics.json`
  - Visualization (predicted vs. true groups): `outputs/fragment_clustering_baseline/results/visualization_real.png`

## Inference / Validation (CPU friendly)

You can run evaluation on CPU either via the notebook or a minimal Python script. The training loop supports GPU if available, but evaluation is intentionally lightweight and runs fine on CPU.

1) Notebook (recommended for exploration)

- Open `notebooks/fragment_reconstruction_pipeline.ipynb` in Jupyter or VS Code.
- Ensure you have the environment installed (see `requirements.txt` and `pyproject.toml`).
- The notebook demonstrates loading data, training (optional), and CPU-friendly evaluation and visualization.

2) Minimal Python example (CPU-only)

Use the `FragmentAutoencoderTrainer` from `src/study/trainer.py` to load a checkpoint and run validation on CPU. Adjust `data_path` and `checkpoint_path` to your setup.

```python
import pytorch_lightning as pl
import torch
from pathlib import Path
from src.study.trainer import FragmentAutoencoderTrainer

# Paths
data_path = "data/dev_data"  # or "data/train_data" for a larger run
checkpoint_path = Path("outputs/conv_autoencoder/checkpoints")

# Find the best checkpoint (pattern created by ModelCheckpoint callback)
ckpts = sorted(checkpoint_path.glob("best_model-*.ckpt"))
assert ckpts, f"No checkpoint found under {checkpoint_path}"

# Load model and run validation on CPU
model = FragmentAutoencoderTrainer.load_from_checkpoint(
    ckpts[0],
    map_location=torch.device("cpu"),
)

trainer = pl.Trainer(
    accelerator="cpu",
    devices=1,
    logger=False,
)

# Ensure datasets are set up (uses model.hparams.data_path)
model.data_path = Path(data_path)
model.setup(stage="fit")

val_results = trainer.validate(model)
print(val_results)
```

Notes:

- The trainer logs clustering metrics (ARI/NMI with KMeans and KMedoids) during validation. See `FragmentAutoencoderTrainer.validation_step()` in `src/study/trainer.py`.
- Checkpoints are saved under `outputs/<model_name>/checkpoints/` by the Lightning `ModelCheckpoint` callback configured in `FragmentAutoencoderTrainer.configure_callbacks()`.
- To reproduce a quick run end-to-end on CPU, you can also execute the `__main__` block in `src/study/trainer.py` (it defaults to the convolutional autoencoder).

## Notes

- Training can be performed on multi-GPU systems; evaluation and visualization are designed to be CPU-friendly.
- The notebook includes simple baselines (random, raw-pixel KMeans) for context.

## License

MIT License — see `LICENSE` for details.