# CoderGym New PyTorch Tasks

This repository contains four new `pytorch_task_v1` tasks for the CoderGym homework.
Each task is self-contained, uses PyTorch for model training/evaluation, saves artifacts,
and exits with status `0` when its validation thresholds pass.

## Tasks

- `mlp_lvl2_moons_mixup`: two-moons MLP classifier with mixup, AdamW, cosine LR, and gradient clipping.
- `cnn_lvl3_synthetic_shapes_cutout`: CNN classifier on generated 16x16 shape images with cutout and label smoothing.
- `rnn_lvl2_sine_gru_forecast`: GRU sequence forecaster for noisy sine waves with OneCycleLR.
- `ae_lvl3_breastcancer_sparse_denoising`: sparse denoising autoencoder on sklearn breast-cancer features.

## Run

```bash
python3 MLtasks/tasks/mlp_lvl2_moons_mixup/task.py
python3 MLtasks/tasks/cnn_lvl3_synthetic_shapes_cutout/task.py
python3 MLtasks/tasks/rnn_lvl2_sine_gru_forecast/task.py
python3 MLtasks/tasks/ae_lvl3_breastcancer_sparse_denoising/task.py
```

