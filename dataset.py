"""
dataset.py — Data pipeline for the face/no-face binary classifier.

Handles everything between raw images and ready-to-use DataLoaders:
  1. Download LFW positive samples and save as 64×64 RGB PNGs.
  2. Mine random 64×64 negative patches from user-supplied background images.
  3. Expose a manifest-backed FaceDataset and a get_loaders() factory.
  4. Provide an append_hard_negatives() hook for the second training phase.

Run directly to build the dataset before training:
    python dataset.py                   # uses config.yaml
    python dataset.py --config other.yaml
    python dataset.py --fresh           # delete existing manifest and rebuild
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
# Fallback normalisation constants. config.yaml is the authoritative source;
# these are used only if the config keys are absent (e.g. in unit tests).
# Both dataset.py and detect.py import from config so they are always in sync —
# never hard-code these values anywhere else.
# ──────────────────────────────────────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Manifest helpers
# ══════════════════════════════════════════════════════════════════════════════
#
# Every sample in the dataset — positive, negative, or hard-negative — is
# tracked as a single row in a CSV manifest (path, label). This is the central
# design choice that makes hard-negative mining possible without restructuring
# the codebase: adding new samples is a simple CSV append, and every downstream
# component (FaceDataset, get_loaders) just reads the same file.
#
# Alternatives considered:
#   • Folder scan (glob positives/ + negatives/) — simple, but you can't
#     distinguish original negatives from hard-mined ones, and the ratio can
#     silently drift between runs.
#   • SQLite — overkill for a list of file paths and a binary label.
#   • HDF5 / LMDB — faster I/O at scale, but complicates inspection and
#     portability. PNG + CSV is human-readable and git-diffable.
# ══════════════════════════════════════════════════════════════════════════════

def _manifest_path(config: dict) -> Path:
    return Path(config["data"]["manifest_file"])


def _read_manifest(config: dict) -> list[dict]:
    """Return all rows from the manifest as a list of {path, label} dicts."""
    mp = _manifest_path(config)
    if not mp.exists():
        return []
    with open(mp, newline="") as f:
        return list(csv.DictReader(f))


def _append_to_manifest(rows: list[list], config: dict) -> None:
    """Append (path, label) rows to the manifest, writing the header if new."""
    mp = _manifest_path(config)
    mp.parent.mkdir(parents=True, exist_ok=True)
    write_header = not mp.exists()
    with open(mp, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["path", "label"])
        writer.writerows(rows)


def _count_by_label(config: dict) -> dict[int, int]:
    """Return {0: n_negatives, 1: n_positives} from the current manifest."""
    counts: dict[int, int] = {0: 0, 1: 0}
    for row in _read_manifest(config):
        counts[int(row["label"])] += 1
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Positive samples: LFW download
# ══════════════════════════════════════════════════════════════════════════════
#
# Labeled Faces in the Wild (LFW) is the standard benchmark for unconstrained
# face recognition. sklearn's fetch_lfw_people downloads it automatically and
# returns pre-cropped face images — these are ideal positive samples because
# the face is always centred and fills most of the crop, which mirrors how the
# sliding-window detector will present patches at inference.
#
# Colour: fetch_lfw_people returns float32 RGB in [0, 1] when color=True.
# We save as uint8 PNG (lossless, compact) and resize to patch_size×patch_size
# here so that FaceDataset never needs to resize at load time — keeping I/O
# in the DataLoader workers cheap.
#
# Idempotency: if positives are already in the manifest we skip the download.
# Pass --fresh on the CLI to force a full rebuild.
# ══════════════════════════════════════════════════════════════════════════════

def download_lfw(config: dict) -> int:
    """
    Fetch LFW, resize crops to patch_size×patch_size RGB, save PNGs, update manifest.

    Returns the number of positive samples saved.
    """
    existing = _count_by_label(config)[1]
    if existing > 0:
        print(f"  Skipping LFW download — {existing} positives already in manifest.")
        return existing

    patch_size    = config["data"]["patch_size"]
    positives_dir = Path(config["data"]["positives_dir"])
    positives_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching LFW dataset (first run downloads ~200 MB) …")
    # min_faces_per_person=1  → include everyone, maximises dataset size (~13 k)
    # resize=None             → get the native 62×47 crops; we resize ourselves
    #                           so patch_size in config.yaml is the only knob
    lfw = fetch_lfw_people(color=True, min_faces_per_person=1, resize=None)
    images = lfw.images  # shape: (n, h, w, 3), float32, [0, 1]

    rows = []
    for i, img_float in enumerate(images):
        img_uint8 = (img_float * 255).astype(np.uint8)
        pil_img   = Image.fromarray(img_uint8, mode="RGB")
        pil_img   = pil_img.resize((patch_size, patch_size), Image.BILINEAR)

        out_path = positives_dir / f"lfw_{i:05d}.png"
        pil_img.save(out_path)
        rows.append([str(out_path), 1])

    _append_to_manifest(rows, config)
    print(f"  Saved {len(rows)} positive samples → {positives_dir}")
    return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Negative samples: background patch mining
# ══════════════════════════════════════════════════════════════════════════════
#
# Negatives are random 64×64 crops from background images that contain no
# faces. The quality of negatives matters as much as positives: if all
# negatives are plain sky or blank walls the classifier learns a trivial
# shortcut ("does it look textured?") instead of learning face structure.
# Use varied indoor/outdoor scenes — COCO unlabelled or SUN397 work well.
#
# Random sampling (rather than a dense grid) is used for two reasons:
#   1. Diversity: each crop has a unique spatial context.
#   2. Avoiding texture memorisation: a grid over the same image would give
#      many highly correlated patches, effectively reducing sample diversity.
#
# Multiscale crops: before taking each 64×64 crop we randomly scale the source
# image between 0.5× and 2.0×. This means the same background pixel can appear
# at different effective resolutions, mimicking the variety the sliding-window
# detector will encounter at different pyramid scales.
# ══════════════════════════════════════════════════════════════════════════════

def generate_negatives(config: dict, n_negatives: Optional[int] = None) -> int:
    """
    Mine random 64×64 patches from background images as negative samples.

    If n_negatives is None, generates enough to match the current positive count
    (1:1 balance). Appends to the manifest; does not overwrite existing negatives.

    Returns the number of new negative samples saved.
    """
    existing_neg = _count_by_label(config)[0]
    existing_pos = _count_by_label(config)[1]

    if n_negatives is None:
        n_negatives = existing_pos

    still_needed = n_negatives - existing_neg
    if still_needed <= 0:
        print(f"  Skipping negative mining — {existing_neg} negatives already in manifest.")
        return existing_neg

    patch_size      = config["data"]["patch_size"]
    negatives_dir   = Path(config["data"]["negatives_dir"])
    backgrounds_dir = Path(config["data"]["backgrounds_dir"])
    negatives_dir.mkdir(parents=True, exist_ok=True)

    if not backgrounds_dir.exists() or not any(backgrounds_dir.iterdir()):
        raise FileNotFoundError(
            f"No background images found in '{backgrounds_dir}'.\n"
            "Add face-free scene images there (COCO, SUN397, or your own)."
        )

    bg_files = [
        p for p in backgrounds_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if not bg_files:
        raise FileNotFoundError(f"No supported image files found in '{backgrounds_dir}'.")

    rows: list[list] = []
    saved   = 0
    idx     = existing_neg   # continue numbering from where we left off
    attempts = 0
    max_attempts = still_needed * 20

    while saved < still_needed and attempts < max_attempts:
        bg_path = random.choice(bg_files)
        try:
            img = Image.open(bg_path).convert("RGB")
        except Exception:
            attempts += 1
            continue

        # Random scale between 0.5× and 2.0× before cropping
        scale   = random.uniform(0.5, 2.0)
        new_w   = max(patch_size, int(img.width  * scale))
        new_h   = max(patch_size, int(img.height * scale))
        img     = img.resize((new_w, new_h), Image.BILINEAR)

        x = random.randint(0, img.width  - patch_size)
        y = random.randint(0, img.height - patch_size)
        patch = img.crop((x, y, x + patch_size, y + patch_size))

        out_path = negatives_dir / f"neg_{idx:06d}.png"
        patch.save(out_path)
        rows.append([str(out_path), 0])
        saved += 1
        idx   += 1
        attempts += 1

    if saved < still_needed:
        print(
            f"  Warning: only generated {saved}/{still_needed} negatives "
            f"(hit attempt limit). Add more background images for a full 1:1 ratio."
        )

    _append_to_manifest(rows, config)
    print(f"  Saved {saved} negative samples → {negatives_dir}")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Hard-negative mining hook
# ══════════════════════════════════════════════════════════════════════════════
#
# After the first training phase, a mining script runs the trained model over
# a large set of background images and collects patches where the model was
# confidently wrong (predicted face with high confidence on a non-face patch).
# These "hard negatives" are the most useful training signal for eliminating
# false positives in the final detector.
#
# This function is intentionally thin — it only updates the manifest. The
# mining logic lives in the script that calls it, keeping concerns separate:
#   • Mining script: knows about the model, thresholds, which images to scan.
#   • append_hard_negatives: knows about the dataset format.
#
# The two-phase workflow is:
#   1. python dataset.py && python train.py          → base model
#   2. python mine_hard_negatives.py                 → finds FP patches
#      (calls append_hard_negatives internally)
#   3. python train.py --resume checkpoints/best_model.pth → fine-tune
# ══════════════════════════════════════════════════════════════════════════════

def append_hard_negatives(hard_neg_paths: list[str], config: dict) -> None:
    """
    Register confirmed false-positive patches as label=0 rows in the manifest.

    hard_neg_paths: absolute paths to 64×64 PNG patches already saved to disk.
                    The caller (mine_hard_negatives.py) is responsible for
                    saving the images; this function only updates the manifest.
    """
    if not hard_neg_paths:
        return
    rows = [[path, 0] for path in hard_neg_paths]
    _append_to_manifest(rows, config)
    print(f"  Appended {len(rows)} hard negatives to manifest.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FaceDataset
# ══════════════════════════════════════════════════════════════════════════════
#
# FaceDataset is a standard torch Dataset that loads patches on demand from the
# manifest. The key design decisions:
#
# Colour: PIL .convert("RGB") is called on every load even though we saved
#   everything as RGB. This is a safety net: if a PNG was accidentally written
#   as RGBA or L (e.g. a greyscale background image), the tensor is still
#   always 3×H×W — the model never sees the wrong number of channels.
#
# Label dtype: float32 scalar (not long). BCEWithLogitsLoss expects targets
#   that match the model output shape — a float scalar per sample, not a class
#   index. Using long would silently produce wrong gradient magnitudes.
#
# Augmentations (training only):
#   • RandomHorizontalFlip: faces can be left- or right-facing. Safe.
#   • ColorJitter: mild brightness/contrast shifts simulate lighting.
#     Saturation and hue are kept very small — aggressive colour distortion
#     makes patches look unlike any real-world face.
#   • RandomRotation ±15°: slight head tilts. Beyond ±15° the crop starts
#     missing chin/forehead and no longer looks like the training distribution.
#   No vertical flip — upside-down faces are not a real-world detection target
#   and would add misleading signal.
# ══════════════════════════════════════════════════════════════════════════════

class FaceDataset(Dataset):
    """
    Binary face/no-face patch dataset backed by a manifest CSV.

    Args:
        rows:    list of dicts with keys 'path' (str) and 'label' (str '0'|'1').
        config:  full config dict; reads normalize_mean / normalize_std.
        augment: apply training-time augmentations when True.
    """

    def __init__(self, rows: list[dict], config: dict, augment: bool = False):
        self.rows = rows
        mean = config["data"].get("normalize_mean", _IMAGENET_MEAN)
        std  = config["data"].get("normalize_std",  _IMAGENET_STD)
        normalize = transforms.Normalize(mean=mean, std=std)

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05
                ),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),   # [0,255] uint8 PIL → [0.0,1.0] float32
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row   = self.rows[idx]
        image = Image.open(row["path"]).convert("RGB")
        tensor = self.transform(image)
        label  = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DataLoader factory
# ══════════════════════════════════════════════════════════════════════════════
#
# get_loaders handles two concerns beyond simply wrapping FaceDataset:
#
# Stratified split: sklearn's train_test_split with stratify= ensures the
#   80/20 ratio holds separately for each class. Without stratification a
#   random split could accidentally put 90 % of rare hard negatives in one
#   split, making validation metrics unreliable.
#
# WeightedRandomSampler (training only): after hard-negative mining the
#   manifest may no longer be 1:1. Rather than discarding samples to rebalance,
#   WeightedRandomSampler draws each sample with probability proportional to
#   the inverse frequency of its class. Every sample contributes to training;
#   each mini-batch is approximately balanced.
#
#   We do NOT apply this to the validation loader — val metrics should reflect
#   the true (possibly imbalanced) distribution so that precision and recall
#   numbers are meaningful.
#
# pin_memory=True: pre-pins CPU tensors so the transfer to GPU is async and
#   faster. Has no effect on CPU-only runs.
# ══════════════════════════════════════════════════════════════════════════════

def get_loaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Build (train_loader, val_loader) from the manifest CSV.

    Raises FileNotFoundError if the manifest does not exist — run dataset.py first.
    """
    mp = _manifest_path(config)
    if not mp.exists():
        raise FileNotFoundError(
            f"Manifest not found at '{mp}'. "
            "Run 'python dataset.py' to build the dataset first."
        )

    all_rows = _read_manifest(config)
    if len(all_rows) < 10:
        raise ValueError(
            f"Manifest has only {len(all_rows)} rows — too few to split. "
            "Re-run 'python dataset.py' to generate samples."
        )

    labels = [int(r["label"]) for r in all_rows]

    train_rows, val_rows = train_test_split(
        all_rows,
        test_size=config["data"]["val_split"],
        stratify=labels,
        random_state=42,
    )

    train_ds = FaceDataset(train_rows, config, augment=True)
    val_ds   = FaceDataset(val_rows,   config, augment=False)

    train_labels = [int(r["label"]) for r in train_rows]
    sampler = _make_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,          # mutually exclusive with shuffle=True
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    n_pos = labels.count(1)
    n_neg = labels.count(0)
    print(f"  Dataset: {len(train_ds)} train / {len(val_ds)} val "
          f"({n_pos} positives, {n_neg} negatives total)")
    return train_loader, val_loader


def _make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Return a sampler that gives each class equal expected weight per batch."""
    class_counts = [labels.count(0), labels.count(1)]
    # If either class has 0 samples this will raise ZeroDivisionError, which is
    # the correct behaviour — the dataset is unusable in that state.
    weights = [1.0 / class_counts[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build face detection dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing manifest and rebuild from scratch",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.fresh:
        mp = _manifest_path(cfg)
        if mp.exists():
            mp.unlink()
            print(f"Deleted existing manifest at {mp}")

    print("=== Step 1: Positive samples (LFW) ===")
    n_pos = download_lfw(cfg)

    print(f"\n=== Step 2: Negative samples (targeting {n_pos} to match positives) ===")
    n_neg = generate_negatives(cfg, n_negatives=n_pos)

    print(f"\n=== Done ===")
    print(f"  {n_pos} positives  +  {n_neg} negatives  →  {_manifest_path(cfg)}")
    print("  Run 'python train.py' to start training.")
