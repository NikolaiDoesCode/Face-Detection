"""
model.py — FaceDetectorCNN architecture for binary face/no-face classification.

Spatial dimension flow (no padding, default floor-mode MaxPool):
  Input          →  3 × 64 × 64
  ConvBlock 1    → 32 × 31 × 31   (conv trims 1px/edge → 62, pool halves → 31)
  ConvBlock 2    → 64 × 14 × 14   (conv → 29, pool floor(29/2) → 14)
  ConvBlock 3    → 128 ×  6 ×  6  (conv → 12, pool → 6)
  Flatten        → 4 608           (128 × 6 × 6)
  Linear         → 512
  Linear         → 1  (raw logit, NO sigmoid)

Output contract:
  forward() returns a raw logit (unbounded float), NOT a probability.
  • train.py uses BCEWithLogitsLoss — applies sigmoid internally; passing a
    probability instead of a logit would double-apply sigmoid and silently
    produce wrong gradients.
  • detect.py applies torch.sigmoid() explicitly to convert to a confidence
    score at inference time.
  Always call model.eval() before inference to disable Dropout and switch
  BatchNorm to running-statistics mode.

Public API:
  FaceDetectorCNN(dropout)         — nn.Module
  build_model(config, device)      — instantiate + validate + print param count
  load_checkpoint(path, model, device) → checkpoint dict
"""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ConvBlock: reusable building block
# ══════════════════════════════════════════════════════════════════════════════
#
# Every convolutional stage of FaceDetectorCNN uses the same four-operation
# sequence: Conv → BatchNorm → ReLU → MaxPool. Extracting it as its own module
# has two benefits:
#
#   1. The feature extractor reads like a list of architectural decisions
#      ("three blocks, doubling channels each time") rather than a wall of
#      twelve layer constructors.
#   2. If the block recipe changes (e.g. adding a second conv per block),
#      the change happens in one place.
#
# Why this specific ordering?
#
#   Conv: extracts spatial features, produces unnormalised activations.
#
#   BatchNorm directly after Conv: normalises the pre-activation distribution
#   to zero mean and unit variance. This keeps the signal well-scaled before
#   ReLU — without it, very negative pre-activations would be permanently
#   killed by ReLU early in training ("dying ReLU" problem). BatchNorm also
#   acts as a regulariser, reducing sensitivity to weight initialisation and
#   allowing a higher learning rate.
#
#   ReLU after BatchNorm: introduces non-linearity. inplace=True saves the
#   memory of an intermediate tensor; safe here because nothing else holds a
#   reference to the pre-ReLU values.
#
#   MaxPool(2×2) last: halves both spatial dimensions, building translation
#   invariance — a face shifted by 1px should produce the same features as
#   the unshifted version. MaxPool is placed last (after the non-linearity)
#   because pooling over rectified features preserves the strongest activations
#   rather than averaging raw values that may cancel out.
#
# Why 3×3 kernels with no padding?
#   3×3 is the smallest filter that captures 2D spatial context. No padding
#   means each conv slightly shrinks the feature map (by 1px/edge), which is
#   intentional: the edge pixels of a 64×64 patch often contain background
#   rather than face structure and don't need to be preserved symmetrically.
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv2d(3×3, no pad) → BatchNorm2d → ReLU → MaxPool2d(2×2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
            # bias=False: BatchNorm already has a learnable bias (beta parameter),
            # so the Conv bias would be redundant and just waste memory.
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FaceDetectorCNN: feature extractor
# ══════════════════════════════════════════════════════════════════════════════
#
# The feature extractor is three ConvBlocks stacked sequentially. The two
# design choices that drive accuracy here are:
#
# Channel doubling (32 → 64 → 128):
#   As each MaxPool halves the spatial resolution, the number of channels
#   doubles. This keeps the total information capacity of each layer's output
#   roughly constant (halving H and W loses 4× spatial positions; doubling
#   channels recovers 2× of that). It also follows the intuition that later
#   layers need more channels to represent more abstract combinations of the
#   lower-level features produced by earlier layers.
#
# Three blocks instead of two or four:
#   Two blocks would leave too much spatial detail (16×16 maps), giving the
#   classifier head too large an input and making it prone to memorising
#   spatial positions rather than face structure. Four blocks would reduce
#   the maps to 2×2 — too coarse to retain meaningful local features from a
#   64×64 face crop. Three blocks landing on 6×6 is the right balance for
#   this input size.
# ══════════════════════════════════════════════════════════════════════════════

# Flat feature count after the three ConvBlocks: 128 channels × 6 × 6 spatial.
# This constant is defined at module level so load_checkpoint and build_model
# can reference it without instantiating the model first.
_FLAT_FEATURES = 128 * 6 * 6  # = 4608


class FaceDetectorCNN(nn.Module):
    """
    Binary face/no-face classifier for 64×64 RGB patches.

    Returns a raw logit per sample — apply torch.sigmoid() externally to get
    a probability. See module docstring for the full output contract.

    Args:
        dropout: dropout probability for the classifier head (default 0.5).
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32),   # 3×64×64  → 32×31×31
            ConvBlock(32,  64),   # 32×31×31 → 64×14×14
            ConvBlock(64, 128),   # 64×14×14 → 128×6×6
        )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 3 — Classifier head
        # ══════════════════════════════════════════════════════════════════════
        #
        # The classifier head converts the 4608-element feature vector from the
        # extractor into a single number that represents "how much does this
        # patch look like a face?"
        #
        # Linear(4608 → 512):
        #   A compression step. 4608 features → 512 gives the head enough
        #   capacity to learn non-linear combinations of the spatial features
        #   without the final prediction layer being overwhelmed with inputs.
        #   512 is a practical midpoint: large enough for expressiveness, small
        #   enough that Dropout below has room to regularise meaningfully.
        #
        # ReLU after the first linear: the linear projection is itself a linear
        #   operation. Without a non-linearity here the two linear layers would
        #   collapse into a single matrix multiply and the head would have no
        #   more capacity than a logistic regression on the raw 4608 features.
        #
        # Dropout(p=0.5):
        #   Randomly zeroes half the 512 units during each training forward
        #   pass. This forces the head to learn redundant representations —
        #   no single unit can "carry" the prediction alone — which dramatically
        #   reduces overfitting on the relatively small LFW dataset.
        #   Dropout is ONLY active during model.train(); model.eval() disables
        #   it deterministically, which is why detect.py must call model.eval()
        #   before running the sliding window.
        #
        # Linear(512 → 1), NO Sigmoid:
        #   Raw logit output. See module docstring for the full reasoning.
        #   The key point: BCEWithLogitsLoss in train.py uses the log-sum-exp
        #   trick to fuse sigmoid + cross-entropy in a single numerically stable
        #   operation. Adding Sigmoid here first would break that.
        # ══════════════════════════════════════════════════════════════════════

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_FLAT_FEATURES, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
            # Intentionally no Sigmoid — see module docstring.
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Forward pass and output contract
    # ══════════════════════════════════════════════════════════════════════════
    #
    # The forward pass is deliberately minimal: features → classifier → return.
    # No sigmoid, no thresholding, no post-processing — all of that belongs in
    # the callers (train.py and detect.py) where the intent is explicit.
    #
    # Input contract:
    #   x must be a float32 tensor of shape (B, 3, 64, 64) where the three
    #   channels are RGB (not BGR) and values are ImageNet-normalised.
    #   Violating either condition (wrong channel order, unnormalised) produces
    #   valid-looking output with degraded accuracy and no error.
    #
    # Output contract:
    #   Shape (B, 1), dtype float32, values are unbounded raw logits.
    #   Positive logit  → model leans toward "face".
    #   Negative logit  → model leans toward "no face".
    #   sigmoid(logit) > threshold is the detection decision in detect.py.
    # ══════════════════════════════════════════════════════════════════════════

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x  # shape: (B, 1), raw logits


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Factory and checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════
#
# build_model and load_checkpoint are the two entry points other files use.
# Neither train.py nor detect.py should import FaceDetectorCNN directly —
# going through these helpers ensures:
#
#   1. The smoke test in build_model catches dimension bugs immediately (at
#      startup) rather than mid-training when a shape mismatch would raise a
#      cryptic CUDA error after potentially hours of work.
#
#   2. load_checkpoint returns the full checkpoint dict, not just the weights.
#      train.py uses it to resume the optimizer state and epoch counter.
#      detect.py uses it to log which epoch and val_loss the weights came from,
#      giving a traceable link between a detection run and a training run.
#
#   3. If the architecture ever changes (e.g. adding a 4th conv block), only
#      build_model needs updating — not every caller.
# ══════════════════════════════════════════════════════════════════════════════

def build_model(config: dict, device: torch.device) -> "FaceDetectorCNN":
    """
    Instantiate FaceDetectorCNN from config, move to device, and validate.

    Runs a zero-tensor forward pass to confirm the spatial dimensions are
    consistent with the configured patch_size. Raises AssertionError at
    startup rather than mid-training if they are not.

    Prints the trainable parameter count, useful for portfolio documentation
    and for sanity-checking that a config change had the intended effect.
    """
    dropout = config["model"]["dropout"]
    model   = FaceDetectorCNN(dropout=dropout).to(device)

    patch_size = config["data"]["patch_size"]
    with torch.no_grad():
        dummy  = torch.zeros(1, 3, patch_size, patch_size, device=device)
        output = model(dummy)

    assert output.shape == (1, 1), (
        f"Smoke test failed: expected output shape (1, 1), got {output.shape}.\n"
        f"patch_size={patch_size} is incompatible with _FLAT_FEATURES={_FLAT_FEATURES}.\n"
        f"If you changed patch_size in config.yaml, update _FLAT_FEATURES in model.py."
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  FaceDetectorCNN ready — {n_params:,} trainable parameters "
          f"(dropout={dropout}, device={device})")
    return model


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: FaceDetectorCNN,
    device: torch.device,
) -> dict:
    """
    Load a full training checkpoint into an existing model instance.

    The checkpoint dict contains:
        epoch          — int, the epoch this checkpoint was saved at
        model_state    — OrderedDict of parameter tensors
        optimizer_state — dict for resuming Adam momentum and variance
        val_loss       — float, best validation loss achieved
        config         — dict snapshot of config.yaml at training time

    Returns the full dict so callers can access metadata beyond just the weights.
    train.py uses it to resume mid-training; detect.py uses it to log provenance.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(
        f"  Loaded checkpoint from '{checkpoint_path}' "
        f"(epoch {checkpoint.get('epoch', '?')}, "
        f"val_loss {checkpoint.get('val_loss', float('nan')):.4f})"
    )
    return checkpoint
