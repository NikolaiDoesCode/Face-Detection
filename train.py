"""
train.py — Training and validation loop for FaceDetectorCNN.

Pipeline
--------
1. Load config.yaml.
2. Build train/val DataLoaders  (dataset.get_loaders).
3. Instantiate FaceDetectorCNN  (model.build_model).
4. Optimiser: Adam + weight decay.
5. Scheduler: ReduceLROnPlateau — halves LR after `lr_patience` stagnant epochs.
6. Loss: BCEWithLogitsLoss.
7. Train for N epochs; save best checkpoint whenever val loss improves.
   Also save a "last" checkpoint every epoch so training can always be resumed.
8. After training: plot loss/accuracy curves, print precision/recall/F1/AUC-ROC.

Entry points
------------
    python train.py                              # fresh run with config.yaml
    python train.py --config other.yaml          # different config
    python train.py --resume checkpoints/best_model.pth   # resume from checkpoint
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in headless environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, roc_auc_score

from dataset import FaceDataset, get_loaders, _read_manifest, _manifest_path
from model import build_model, load_checkpoint


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Device selection
# ══════════════════════════════════════════════════════════════════════════════
#
# Device selection logic is intentionally duplicated from detect.py rather than
# imported from it.  detect.py imports from model.py; having train.py import
# from detect.py would create an indirect dependency between two files that are
# conceptually independent (training vs inference).  Five lines of duplication
# is a better trade than a confusing import graph.
# ══════════════════════════════════════════════════════════════════════════════

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Training epoch
# ══════════════════════════════════════════════════════════════════════════════
#
# One training epoch: iterate the loader, do forward → loss → backward →
# optimizer step for each mini-batch, return the mean loss over all samples.
#
# Loss reduction: we accumulate `loss.item() * batch_size` (sum) rather than
# averaging the per-batch mean losses.  These two are equivalent when all
# batches are the same size, but the last batch of an epoch is often smaller.
# Summing and dividing by the dataset length gives the true per-sample mean.
#
# optimizer.zero_grad() is called before each forward pass (not after), so
# stale gradients from the previous batch never pollute the current one.
# Calling it after would work too, but before is the canonical PyTorch style
# and makes the gradient lifecycle easier to reason about.
#
# model.train() must be set here (not just once at the start of training)
# because val_epoch calls model.eval().  If train_epoch doesn't reset the
# mode, Dropout stays disabled after the first validation pass.
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean BCEWithLogitsLoss over all samples."""
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)   # (B, 1) → (B,)  matches label shape
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Validation epoch
# ══════════════════════════════════════════════════════════════════════════════
#
# Validation mirrors training minus the backward pass.  Two extra things:
#
# torch.no_grad(): disables gradient tracking entirely.  This is faster than
#   model.eval() alone (which only affects Dropout/BatchNorm behaviour) because
#   autograd no longer builds the computation graph for each batch.
#
# Collecting all logits and labels: we accumulate raw logits (not predictions)
#   so the caller can compute threshold-independent metrics like AUC-ROC, which
#   needs a continuous score rather than a binary decision.  Accuracy (at
#   threshold=0.5) is computed here for the per-epoch log; full sklearn metrics
#   are computed once at the end of training.
#
# Threshold 0.5 vs detection_threshold: the 0.5 threshold used here for
#   training accuracy is symmetric around the decision boundary.  The higher
#   detection_threshold in config (0.85 by default) is an inference-time choice
#   that trades recall for precision to reduce false positives in the webcam
#   feed.  Training accuracy with 0.5 is what we use to monitor training health.
# ══════════════════════════════════════════════════════════════════════════════

def val_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """
    Run one validation epoch.

    Returns:
        mean_loss:  float, mean BCEWithLogitsLoss.
        accuracy:   float, binary accuracy at threshold=0.5.
        all_logits: (N,) float32 tensor — raw logits across the full val set.
        all_labels: (N,) float32 tensor — ground-truth labels.
    """
    model.eval()
    total_loss  = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).squeeze(1)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)   # (N,)
    all_labels = torch.cat(all_labels)   # (N,)

    probs    = torch.sigmoid(all_logits)
    preds    = (probs > 0.5).float()
    accuracy = (preds == all_labels).float().mean().item()

    return total_loss / len(loader.dataset), accuracy, all_logits, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Checkpoint management
# ══════════════════════════════════════════════════════════════════════════════
#
# We keep two checkpoints on disk at all times:
#
#   best_model.pth  — lowest val loss seen so far.  This is what detect.py and
#                     main.py will load.  It may be many epochs behind the most
#                     recent epoch if the model later overfit.
#
#   last_model.pth  — end of the most recently completed epoch, always
#                     overwritten.  Used for resuming training with --resume
#                     after an interruption.  Provides the optimizer state and
#                     epoch counter; best_model.pth may not have these if the
#                     best epoch was early in training.
#
# Checkpoint dict schema (mirrors model.load_checkpoint expectations):
#   epoch           int   — epoch that produced these weights
#   model_state     dict  — nn.Module.state_dict()
#   optimizer_state dict  — optimizer.state_dict() for momentum/variance
#   val_loss        float — validation loss at this epoch (or best so far)
#   config          dict  — full config snapshot for provenance
#
# The config snapshot in the checkpoint lets detect.py warn the user if they
# run inference with different normalisation constants than those used during
# training, which would silently degrade accuracy.
# ══════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss":        val_loss,
        "config":          config,
    }, path)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Metrics logging
# ══════════════════════════════════════════════════════════════════════════════
#
# Per-epoch metrics are appended to a CSV file so they survive process restarts
# and can be loaded by any tool (pandas, Excel, a second matplotlib script)
# without re-running training.
#
# When resuming with --resume, the CSV is appended to, not overwritten.
# The epoch number in each row is the authoritative key, so rows from a
# previous run that overlap in epoch number are duplicates — this won't
# happen in normal use because --resume starts from checkpoint["epoch"] + 1.
# ══════════════════════════════════════════════════════════════════════════════

def _log_metrics(
    csv_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_acc: float,
    lr: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "lr"])
        writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{lr:.2e}",
        ])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Loss and accuracy curves
# ══════════════════════════════════════════════════════════════════════════════
#
# Plots are generated from the metrics CSV rather than from in-memory lists.
# This means the plots can be regenerated at any time without re-training, and
# they correctly include all epochs from a resumed run.
#
# matplotlib.use("Agg") at the top of the file sets the non-interactive backend
# so plt.savefig() works on headless machines (CI, remote servers) that have no
# display.  It has no effect if a display is available.
# ══════════════════════════════════════════════════════════════════════════════

def _plot_curves(csv_path: Path, runs_dir: Path) -> None:
    """Read metrics CSV and write loss_curve.png and accuracy_curve.png."""
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    epochs      = [int(r["epoch"])         for r in rows]
    train_loss  = [float(r["train_loss"])  for r in rows]
    val_loss    = [float(r["val_loss"])    for r in rows]
    val_acc     = [float(r["val_accuracy"]) for r in rows]

    # ── Loss curve ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train loss",  linewidth=2)
    ax.plot(epochs, val_loss,   label="Val loss",    linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title("FaceDetectorCNN — Train / Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    loss_path = runs_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {loss_path}")

    # ── Accuracy curve ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_acc, color="#2ca02c", label="Val accuracy", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, label="Chance (0.5)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy  (threshold = 0.5)")
    ax.set_title("FaceDetectorCNN — Validation Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    acc_path = runs_dir / "accuracy_curve.png"
    fig.savefig(acc_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {acc_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Final evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════
#
# After training we run one clean validation pass on the BEST checkpoint (not
# the last epoch) to report the metrics that correspond to the weights
# detect.py will actually use.  If the model overfit in later epochs, the
# last-epoch val metrics would be misleading.
#
# Metrics reported:
#   precision / recall / F1  — per class and macro-average via sklearn's
#     classification_report.  Threshold is 0.5 (not detection_threshold);
#     this is a model-quality report, not a detector-configuration report.
#
#   AUC-ROC — area under the receiver operating characteristic curve.
#     Threshold-independent: measures how well the model separates the two
#     classes across all possible thresholds.  A perfect classifier scores
#     1.0; random chance scores 0.5.  Anything above ~0.95 is excellent for
#     this task.
# ══════════════════════════════════════════════════════════════════════════════

def _print_final_metrics(all_logits: torch.Tensor, all_labels: torch.Tensor) -> None:
    probs  = torch.sigmoid(all_logits).numpy().astype(np.float64)
    labels = all_labels.numpy().astype(int)
    preds  = (probs > 0.5).astype(int)

    print("\n=== Final Validation Metrics (best checkpoint) ===")
    print(classification_report(
        labels, preds,
        target_names=["no-face", "face"],
        digits=4,
    ))

    try:
        auc = roc_auc_score(labels, probs)
        print(f"  AUC-ROC : {auc:.4f}")
    except ValueError as e:
        # roc_auc_score raises if only one class is present in labels
        print(f"  AUC-ROC : could not compute — {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Smoke-test DataLoader
# ══════════════════════════════════════════════════════════════════════════════
#
# The smoke test is designed to answer one question as fast as possible:
# "does the entire pipeline run without crashing?"  It is not meant to produce
# a usable model.
#
# 100 samples (50 face + 50 no-face), 2 epochs, batch size 16:
#   - 80 train / 20 val after the 80/20 split
#   - ~5 train batches per epoch — enough to touch every code path (forward,
#     backward, scheduler step, checkpoint save, val pass, metrics log)
#   - Completes in under a minute on MPS
#
# num_workers=0: multiprocessing startup adds ~2s per run; pointless for 100
#   samples.  The real training run uses config num_workers (4) for throughput.
#
# Separate output dirs (checkpoints/smoke_test/, runs/smoke_test/) keep smoke
#   artifacts out of the real training outputs.  Delete them freely.
# ══════════════════════════════════════════════════════════════════════════════

def _get_smoke_loaders(config: dict) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build tiny 100-sample loaders (batch=16, num_workers=0) for smoke testing."""
    import random as _random
    from sklearn.model_selection import train_test_split as _tts

    mp = _manifest_path(config)
    if not mp.exists():
        raise FileNotFoundError(
            f"Manifest not found at '{mp}'. Run 'python dataset.py' first."
        )

    all_rows = _read_manifest(config)
    pos_rows = [r for r in all_rows if r["label"] == "1"]
    neg_rows = [r for r in all_rows if r["label"] == "0"]

    n_each = min(50, len(pos_rows), len(neg_rows))
    if n_each < 10:
        raise ValueError(
            f"Not enough samples for smoke test "
            f"(have {len(pos_rows)} pos / {len(neg_rows)} neg, need ≥10 each). "
            "Run 'python dataset.py' first."
        )

    _random.seed(42)
    sampled = _random.sample(pos_rows, n_each) + _random.sample(neg_rows, n_each)
    labels  = [int(r["label"]) for r in sampled]

    train_rows, val_rows = _tts(
        sampled, test_size=0.2, stratify=labels, random_state=42
    )

    train_ds = FaceDataset(train_rows, config, augment=True)
    val_ds   = FaceDataset(val_rows,   config, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True,  num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,   batch_size=16, shuffle=False, num_workers=0
    )

    print(f"  Smoke loaders: {len(train_ds)} train / {len(val_ds)} val  "
          f"({n_each} face + {n_each} no-face, batch=16)")
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Main: ties everything together
# ══════════════════════════════════════════════════════════════════════════════
#
# Execution order:
#   setup → data → model → loss/optim/scheduler → (optional resume) →
#   epoch loop → post-training plots → final metrics on best checkpoint
#
# Resuming: --resume loads optimizer state so Adam's moment estimates carry
#   over.  Without them, the first few epochs after resuming behave as if the
#   optimizer is cold-started: larger-than-expected gradient steps until the
#   estimates re-warm.  Reloading the state avoids this.
#
# LR logging: we capture the current LR before scheduler.step() because
#   ReduceLROnPlateau updates the LR AFTER the step call.  Logging before
#   records the LR that was active during the epoch just completed.
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Train FaceDetectorCNN")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT",
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 2 epochs on 100 samples to validate the pipeline end-to-end")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke_test:
        print("=" * 60)
        print("  SMOKE TEST — 2 epochs, 100 samples, isolated outputs")
        print("  Purpose: verify the full pipeline runs without errors.")
        print("  Do not use the resulting checkpoint for real detection.")
        print("=" * 60 + "\n")
        config["training"]["epochs"] = 2

    runs_dir       = Path(config["training"]["runs_dir"])       / ("smoke_test" if args.smoke_test else "")
    checkpoint_dir = Path(config["training"]["checkpoint_dir"]) / ("smoke_test" if args.smoke_test else "")
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv    = runs_dir / "metrics.csv"
    best_ckpt_path = checkpoint_dir / "best_model.pth"
    last_ckpt_path = checkpoint_dir / "last_model.pth"

    device = _best_device()
    print(f"Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n=== Dataset ===")
    if args.smoke_test:
        train_loader, val_loader = _get_smoke_loaders(config)
    else:
        train_loader, val_loader = get_loaders(config)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n=== Model ===")
    model = build_model(config, device)

    # ── Loss / optimiser / scheduler ──────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # ReduceLROnPlateau: multiply LR by lr_factor when val loss has not
    # improved for lr_patience consecutive epochs.  "min" mode because we want
    # the loss to decrease.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["lr_factor"],
        patience=config["training"]["lr_patience"],
    )

    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")

    if args.resume:
        print(f"\n=== Resuming from {args.resume} ===")
        checkpoint = load_checkpoint(args.resume, model, device)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch   = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"  Continuing from epoch {start_epoch}  "
              f"(best val_loss so far: {best_val_loss:.4f})")

    # ── Training loop ─────────────────────────────────────────────────────────
    total_epochs = config["training"]["epochs"]
    print(f"\n=== Training  (epochs {start_epoch}–{total_epochs}) ===")

    last_val_logits: torch.Tensor | None = None
    last_val_labels: torch.Tensor | None = None

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_logits, val_labels = val_epoch(
            model, val_loader, criterion, device
        )

        # Capture LR before scheduler mutates it so the log reflects this epoch
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"  [{epoch:3d}/{total_epochs}]  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"acc={val_acc:.4f}  lr={current_lr:.2e}  "
            f"({elapsed:.1f}s)"
        )

        _log_metrics(metrics_csv, epoch, train_loss, val_loss, val_acc, current_lr)

        # Always overwrite last checkpoint so --resume can pick up here
        _save_checkpoint(last_ckpt_path, model, optimizer, epoch, val_loss, config)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(best_ckpt_path, model, optimizer, epoch, val_loss, config)
            print(f"           ↑ best val_loss {val_loss:.4f} — saved to {best_ckpt_path}")

        last_val_logits = val_logits
        last_val_labels = val_labels

    # ── Post-training ─────────────────────────────────────────────────────────
    print(f"\n=== Training complete  (best val_loss: {best_val_loss:.4f}) ===")

    if args.smoke_test:
        print("\n✓ Smoke test passed — full pipeline ran without errors.")
        print(f"  Checkpoint saved to : {best_ckpt_path}")
        print("\n  Next steps:")
        print("    python train.py          ← full 20-epoch training run")
        print("    python main.py           ← live detection (needs full checkpoint)")
        return

    print("\n--- Loss and accuracy curves ---")
    _plot_curves(metrics_csv, runs_dir)

    # Final metrics: reload the BEST checkpoint so the report reflects the
    # weights detect.py will actually use, not the potentially-overfit last epoch
    print("\n--- Reloading best checkpoint for final metrics ---")
    load_checkpoint(best_ckpt_path, model, device)
    _, _, best_val_logits, best_val_labels = val_epoch(
        model, val_loader, criterion, device
    )
    _print_final_metrics(best_val_logits, best_val_labels)

    print(f"\nCheckpoint : {best_ckpt_path}")
    print("Next step  : python main.py")


if __name__ == "__main__":
    main()
