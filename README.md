# Case Study: Image Fragment Reconstruction (Self-Supervised)

This repository implements a minimal, self-supervised pipeline that groups unordered 16×16 fragments back to their source 64×64 images. The goal is to learn fragment embeddings with a lightweight CNN autoencoder and recover the original grouping via clustering.

## What’s included

- Minimal Python modules under `src/study/` for data loading, fragment generation, and the model.
- A single public notebook with the full workflow (training, CPU-centric evaluation, baselines, visualization):
  - `notebooks/fragment_reconstruction_pipeline.ipynb`
- Final artifacts tracked with Git LFS:
  - `outputs/fragment_clustering_baseline/results/metrics.json`
  - `outputs/fragment_clustering_baseline/results/visualization_real.png`

## Dataset

The pipeline expects the legacy ImageNet-64 pickle batches at:

```text
/mnt/localssd/datasets/case/
├── train_data/
└── dev_data/
```
The code auto-detects this format and loads it without modification.

## Results & Report

- Final report: `docs/REPORT.md`
- Notebook: `notebooks/fragment_reconstruction_pipeline.ipynb`
- Artifacts:
  - Metrics (ARI, NMI, Purity): `outputs/fragment_clustering_baseline/results/metrics.json`
  - Visualization (predicted vs. true groups): `outputs/fragment_clustering_baseline/results/visualization_real.png`

## Notes

- Training can be performed on multi-GPU systems; evaluation and visualization are designed to be CPU-friendly.
- The notebook includes simple baselines (random, raw-pixel KMeans) for context.

## License

MIT License — see `LICENSE` for details.