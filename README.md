# EM Project — 1D CNN scaffold (MATLAB)

This folder contains a minimal MATLAB Deep Learning Toolbox scaffold for a 1D CNN suitable for vibration/current waveform classification.

Files added
- `create1DCNN.m` — builds a compact 1D CNN (implemented via 2D conv with width 1)
- `generate_synthetic_data.m` — fast synthetic dataset for local testing
- `augment.m` — simple augmentations (noise, scale, shift)
- `dataset.m` (loadDataset) — loads dataset or falls back to synthetic data
- `train.m` (trainModel) — training entry point using `trainNetwork`
- `evaluate.m` — compute accuracy, precision, recall, confusion chart

Quick start (MATLAB)

1. Open MATLAB and set current folder to this project directory.
2. Run training with default synthetic dataset:

```matlab
net = trainModel('inputLength',2000,'epochs',10,'augmentFactor',1);
```

3. To train with your real dataset, prepare a folder where each `.mat` contains
   variables `signal` (vector) and `label` (string) and call:

```matlab
net = trainModel('dataDir','path/to/your/matfiles','inputLength',2000,'epochs',30);
```

Notes & Requirements
- MATLAB R2020b or later recommended (Deep Learning Toolbox, Image Processing Toolbox for `imtranslate`).
- CPU training is supported; training on GPU requires Parallel Computing Toolbox and a compatible GPU and you can change `'ExecutionEnvironment'` in `train.m`.
- The scaffold uses an imageInputLayer approach so signals are shaped as `[L 1 1 N]`.

Next steps I can add
- a 2D-spectrogram path with a CNN using pretrained image models
- improved dataset loader (CSV, folder structure, streaming datastores)
- class weighting / focal loss for imbalance

If you'd like, I can also produce a MATLAB Live Script that walks through loading real data and visualizing spectrograms.
