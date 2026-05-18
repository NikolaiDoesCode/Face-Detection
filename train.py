"""
train.py — Training and validation loop for FaceDetectorCNN.

Pipeline
--------
1. Load config from config.yaml (learning rate, batch size, epochs, etc.).
2. Build train/val DataLoaders via dataset.get_loaders(config).
3. Instantiate FaceDetectorCNN and move it to the best available device
   (CUDA → MPS → CPU).
4. Optimiser: Adam with weight decay from config.
5. LR scheduler: ReduceLROnPlateau — halves the LR when val loss stalls for
   `patience` epochs.
6. Loss function: BCEWithLogitsLoss (numerically stabler than BCE + Sigmoid).

Each epoch:
  - Train pass: forward → loss → backward → optimiser step.
  - Val pass (no_grad): compute val loss and binary accuracy.
  - Save checkpoint to checkpoints/best_model.pth whenever val loss improves.
  - Log train loss, val loss, val accuracy to runs/metrics.csv for later plotting.

After training:
  - Plot train/val loss curves and accuracy curve, saved to runs/loss_curve.png
    and runs/accuracy_curve.png.
  - Print final test-set metrics (precision, recall, F1, AUC-ROC).

Entry point:
    python train.py            # uses config.yaml
    python train.py --config path/to/other.yaml
"""
