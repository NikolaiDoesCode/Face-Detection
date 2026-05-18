"""
detect.py — Sliding-window face detector using a trained FaceDetectorCNN.

Full pipeline per frame:
  1. BGR→RGB conversion  (OpenCV frames arrive as BGR; model trained on RGB)
  2. ImageNet normalisation  (must match dataset.py exactly)
  3. Image pyramid  (resize frame to each scale in config)
  4. Batched patch extraction via torch.unfold()  (all positions at once, ~10× faster
     than a Python loop)
  5. Chunked GPU inference  (prevents OOM on high-res frames or small GPUs)
  6. Non-maximum suppression  (IoU-based greedy suppression collapses overlapping boxes)

Public API:
  Detector(checkpoint_path, config)  — load model once, reuse across frames
  Detector.detect(frame) -> list[tuple[int,int,int,int,float]]
      frame : BGR uint8 numpy array (straight from cv2.VideoCapture.read)
      return: [(x, y, w, h, confidence), ...] in original frame coordinates,
              sorted highest-confidence first, NMS-filtered
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torchvision import transforms

from model import build_model, load_checkpoint


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Device selection
# ══════════════════════════════════════════════════════════════════════════════
#
# Detection runs on the best device available at startup and stays there for
# the lifetime of the Detector object.  We check in order of speed:
#   CUDA  (NVIDIA GPU) → MPS (Apple Silicon) → CPU
#
# MPS is PyTorch's Metal Performance Shaders backend, available on M1/M2/M3
# Macs since PyTorch 1.12.  It gives a meaningful speedup for the batched
# forward passes in the sliding window (typically 3–5× over CPU on M-series).
#
# The device is resolved once here and passed to build_model, so all tensors
# — patches, logits, the model weights — live on the same device throughout.
# ══════════════════════════════════════════════════════════════════════════════

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Frame preprocessing
# ══════════════════════════════════════════════════════════════════════════════
#
# OpenCV reads webcam frames as BGR uint8.  The model was trained on RGB
# float32 tensors normalised with ImageNet mean/std.  Closing this gap is the
# single most important correctness requirement in the whole inference path —
# a channel-order mismatch produces silently wrong (but numerically valid)
# outputs and no error.
#
# The preprocessing function:
#   1. cvtColor BGR→RGB: swap red and blue channels.
#   2. permute (H,W,3)→(3,H,W): PyTorch convention is channels-first.
#   3. .float().div(255): uint8 [0,255] → float32 [0.0, 1.0].
#   4. normalize: subtract ImageNet mean, divide by std, per channel.
#      The mean and std are read from config so they are provably identical
#      to what dataset.py used when building the training set.
#
# The result is a (3, H, W) float32 tensor that the model's Conv layers
# expect.  Scaling to a pyramid scale happens AFTER this step, on the tensor.
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_frame(
    frame: np.ndarray,
    normalize: transforms.Normalize,
) -> torch.Tensor:
    """
    BGR uint8 numpy array → normalised float32 tensor (3, H, W) on CPU.

    Stays on CPU here; individual patch batches are moved to the target
    device just before the forward pass (see _run_inference).
    """
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # (H, W, 3) uint8
    t     = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, H, W) float32
    t     = t.div(255.0)
    t     = normalize(t)                                     # ImageNet-normalised
    return t


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Batched patch extraction with torch.unfold()
# ══════════════════════════════════════════════════════════════════════════════
#
# A naive sliding window loops over every (row, col) position, crops a patch,
# and runs a forward pass — roughly 2 000 forward passes per frame for a
# 640×480 image across four pyramid scales.  That is 2 000 Python→GPU round
# trips per frame, which dominates wall-clock time.
#
# torch.Tensor.unfold(dim, size, step) extracts all windows along one
# dimension as a new trailing axis, using the tensor's stride mechanism —
# no data is copied until the final reshape forces materialisation:
#
#   t            : (3, H, W)
#   t.unfold(1, P, S)        → (3, n_rows, W, P)      [slide along height]
#   .unfold(2, P, S)         → (3, n_rows, n_cols, P, P)  [slide along width]
#   .permute(1,2,0,3,4)      → (n_rows, n_cols, 3, P, P)
#   .reshape(-1, 3, P, P)    → (n_rows*n_cols, 3, P, P)  ← one contiguous batch
#
# where P = patch_size, S = step_size,
#   n_rows = floor((H - P) / S) + 1
#   n_cols = floor((W - P) / S) + 1
#
# The (row_idx, col_idx) grid position maps back to the scaled-frame pixel
# coordinate (col_idx*S, row_idx*S), which then rescales to the original
# frame by dividing by the pyramid scale factor.
# ══════════════════════════════════════════════════════════════════════════════

def _extract_patches(
    t: torch.Tensor,
    patch_size: int,
    step_size: int,
) -> tuple[torch.Tensor, int, int]:
    """
    Extract all sliding-window patches from a preprocessed frame tensor.

    Args:
        t:          (3, H, W) float32 tensor, ImageNet-normalised.
        patch_size: side length of each square patch (must equal model input size).
        step_size:  stride of the sliding window in pixels.

    Returns:
        patches:  (N, 3, patch_size, patch_size) where N = n_rows * n_cols.
        n_rows:   number of window positions along the height axis.
        n_cols:   number of window positions along the width axis.

    Returns (None, 0, 0) if the frame is smaller than patch_size in either
    dimension at the current pyramid scale.
    """
    _, H, W = t.shape
    if H < patch_size or W < patch_size:
        return None, 0, 0

    windows = (
        t
        .unfold(1, patch_size, step_size)   # (3, n_rows, W, P)
        .unfold(2, patch_size, step_size)   # (3, n_rows, n_cols, P, P)
    )
    n_rows = windows.shape[1]
    n_cols = windows.shape[2]

    # contiguous() forces materialisation of the strided view before reshape.
    # Without it, reshape can fail with a non-contiguous memory error.
    patches = (
        windows
        .permute(1, 2, 0, 3, 4)            # (n_rows, n_cols, 3, P, P)
        .contiguous()
        .reshape(-1, 3, patch_size, patch_size)  # (N, 3, P, P)
    )
    return patches, n_rows, n_cols


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Chunked GPU inference
# ══════════════════════════════════════════════════════════════════════════════
#
# Even with batched extraction, a 1280×720 frame at scale 1.0 with step 16
# produces ~3 600 patches.  Sending all of them to the GPU in one call can
# exhaust VRAM on smaller cards (4 GB) or the unified memory on base M1 Macs.
#
# torch.split(patches, chunk_size) divides the batch into slices of at most
# chunk_size without copying data.  Each slice gets a forward pass, the logits
# are collected on CPU, then concatenated.  The GPU only ever holds one chunk
# at a time, keeping peak VRAM usage bounded to:
#   chunk_size × 3 × 64 × 64 × 4 bytes ≈ 12 MB for chunk_size=256.
#
# inference_batch_size in config.yaml (default 256) is the knob the user
# turns if they see OOM errors.
# ══════════════════════════════════════════════════════════════════════════════

def _run_inference(
    patches: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    """
    Run the model on all patches, returning per-patch face probabilities.

    Args:
        patches:    (N, 3, P, P) float32 tensor on CPU.
        model:      FaceDetectorCNN in eval() mode.
        device:     target device for computation.
        chunk_size: max patches per forward pass.

    Returns:
        confidences: (N,) float32 tensor on CPU, values in [0, 1].
    """
    results = []
    for chunk in torch.split(patches, chunk_size):
        chunk = chunk.to(device)
        with torch.no_grad():
            logits = model(chunk)                     # (chunk, 1), raw logits
        probs = torch.sigmoid(logits).squeeze(1)      # (chunk,)  probabilities
        results.append(probs.cpu())
    return torch.cat(results)                         # (N,)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Non-maximum suppression (NMS)
# ══════════════════════════════════════════════════════════════════════════════
#
# The sliding window at multiple scales produces many overlapping detections
# for the same face.  NMS collapses them into a single box per face.
#
# Algorithm (greedy, O(n²) in the number of candidates):
#   1. Sort candidates by confidence, highest first.
#   2. Initialise an empty "kept" list.
#   3. For each candidate in sorted order:
#        if its IoU with EVERY already-kept box is below nms_iou_threshold:
#          keep it; otherwise discard it.
#   4. Return kept boxes.
#
# IoU (Intersection over Union): the ratio of the overlapping area of two
# boxes to their combined area.  Two boxes with IoU > 0.3 are almost
# certainly detections of the same face and the lower-confidence one is
# suppressed.
#
# Why greedy rather than soft-NMS or learned NMS?
#   Greedy NMS is simple, deterministic, and fast enough for real-time use.
#   The number of candidates per frame is small (typically < 100 above
#   threshold), so the O(n²) cost is negligible.  Soft-NMS (which decays
#   rather than removes overlapping boxes) is better when two faces are very
#   close together, but that is an edge case we can address in a later pass.
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a: tuple, b: tuple) -> float:
    """
    Intersection over Union for two (x, y, w, h, conf) boxes.

    Uses (x, y) as the top-left corner.  Returns 0.0 if either box has
    zero area (degenerate patch).
    """
    ax1, ay1, aw, ah = a[0], a[1], a[2], a[3]
    bx1, by1, bw, bh = b[0], b[1], b[2], b[3]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = aw * ah
    area_b = bw * bh
    union  = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _nms(
    candidates: list[tuple],
    iou_threshold: float,
) -> list[tuple]:
    """
    Greedy NMS: suppress lower-confidence boxes that overlap a kept box.

    Args:
        candidates:    list of (x, y, w, h, confidence) tuples.
        iou_threshold: IoU above which two boxes are considered the same face.

    Returns:
        Filtered list of (x, y, w, h, confidence) sorted by confidence desc.
    """
    if not candidates:
        return []

    sorted_boxes = sorted(candidates, key=lambda b: b[4], reverse=True)
    kept: list[tuple] = []

    for box in sorted_boxes:
        if all(_iou(box, k) < iou_threshold for k in kept):
            kept.append(box)

    return kept


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Detector class: the public API
# ══════════════════════════════════════════════════════════════════════════════
#
# Detector is the single object main.py creates at startup and then calls
# .detect() on every frame.  All the per-frame cost is inside detect(); the
# one-time setup cost (loading weights, building the normalisation transform,
# resolving the device) is paid in __init__().
#
# The normalisation transform is built once and stored as self.normalize.
# This is the critical coupling point between training and inference: if you
# retrain with different mean/std values and load the new checkpoint without
# updating config.yaml, the normalisation will silently mismatch and accuracy
# will degrade.  The checkpoint dict contains a config snapshot for exactly
# this reason — __init__ logs a warning if the training-time normalisation
# doesn't match the current config.
#
# Why does detect() convert coordinates to int?
#   OpenCV's cv2.rectangle expects integer pixel coordinates.  Keeping floats
#   through the entire pipeline (for precise IoU calculations in NMS) and
#   rounding only at the final output step avoids accumulated rounding errors
#   from scale × step arithmetic.
# ══════════════════════════════════════════════════════════════════════════════

class Detector:
    """
    Sliding-window face detector backed by a trained FaceDetectorCNN.

    Usage:
        detector = Detector("checkpoints/best_model.pth", config)
        boxes = detector.detect(bgr_frame)   # list of (x, y, w, h, confidence)
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config: dict,
    ):
        self.config     = config
        self.device     = _best_device()
        self.patch_size = config["data"]["patch_size"]
        self.scales     = config["detection"]["scales"]
        self.step_size  = config["detection"]["step_size"]
        self.threshold  = config["detection"]["detection_threshold"]
        self.nms_iou    = config["detection"]["nms_iou_threshold"]
        self.chunk_size = config["detection"].get("inference_batch_size", 256)

        print(f"Detector: device={self.device}, "
              f"scales={self.scales}, step={self.step_size}px, "
              f"threshold={self.threshold}")

        # Load model — build_model runs the architecture smoke test
        self.model = build_model(config, self.device)
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.eval()  # disable Dropout; switch BN to running statistics

        # Warn if the training-time normalisation differs from the current config.
        # A mismatch won't crash but will silently degrade accuracy.
        ckpt_cfg = checkpoint.get("config", {})
        ckpt_mean = ckpt_cfg.get("data", {}).get("normalize_mean")
        curr_mean = config["data"]["normalize_mean"]
        if ckpt_mean and ckpt_mean != curr_mean:
            print(
                f"  WARNING: checkpoint normalisation mean {ckpt_mean} "
                f"differs from config {curr_mean}. "
                "Detection accuracy may be degraded."
            )

        # Build the normalisation transform — stored once, applied every frame.
        mean = config["data"]["normalize_mean"]
        std  = config["data"]["normalize_std"]
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def detect(self, frame: np.ndarray) -> list[tuple]:
        """
        Run the full detection pipeline on one BGR frame.

        Args:
            frame: (H, W, 3) BGR uint8 numpy array from cv2.VideoCapture.

        Returns:
            List of (x, y, w, h, confidence) tuples in original frame pixel
            coordinates, sorted by confidence descending, NMS-filtered.
            x, y, w, h are integers; confidence is a float in [0, 1].
        """
        # Preprocess once — normalization is scale-invariant so we do it on
        # the full frame before resizing, not on each patch individually.
        t = _preprocess_frame(frame, self.normalize)  # (3, H, W) float32

        candidates: list[tuple] = []

        for scale in self.scales:
            scaled_t = self._scale_frame(t, scale)
            patches, n_rows, n_cols = _extract_patches(
                scaled_t, self.patch_size, self.step_size
            )
            if patches is None:
                continue  # frame too small at this scale

            confidences = _run_inference(
                patches, self.model, self.device, self.chunk_size
            )  # (n_rows * n_cols,)

            conf_grid = confidences.reshape(n_rows, n_cols)
            hot = (conf_grid > self.threshold).nonzero(as_tuple=False)

            for pos in hot:
                row_idx, col_idx = pos[0].item(), pos[1].item()

                # Pixel position in the scaled frame
                x_s = col_idx * self.step_size
                y_s = row_idx * self.step_size

                # Rescale to original frame coordinates.
                # The patch covers patch_size pixels in the SCALED frame,
                # which corresponds to patch_size/scale pixels in the original.
                x = x_s / scale
                y = y_s / scale
                w = self.patch_size / scale
                h = self.patch_size / scale
                conf = conf_grid[row_idx, col_idx].item()

                candidates.append((x, y, w, h, conf))

        kept = _nms(candidates, self.nms_iou)

        # Convert coordinates to int for OpenCV drawing in main.py
        return [
            (int(x), int(y), int(w), int(h), conf)
            for x, y, w, h, conf in kept
        ]

    @staticmethod
    def _scale_frame(t: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Resize a (3, H, W) tensor to (3, int(H*scale), int(W*scale)).

        Uses bilinear interpolation — the same as the training resize in
        dataset.py — so the texture statistics the model learned at 64×64
        are consistent with what it sees at inference.
        """
        if scale == 1.0:
            return t
        _, H, W = t.shape
        new_h = max(1, int(H * scale))
        new_w = max(1, int(W * scale))
        # unsqueeze adds the batch dim required by interpolate, then squeeze removes it
        scaled = torch.nn.functional.interpolate(
            t.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return scaled
