"""
model.py — CNN architecture for binary face/no-face patch classification.

Architecture
------------
Input: 3 × 64 × 64 RGB tensor, normalised to [0, 1].

Feature extractor (three convolutional blocks):
  Block 1 — Conv2d(3,  32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)   → 32 × 31 × 31
  Block 2 — Conv2d(32, 64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)   → 64 × 15 × 15
  Block 3 — Conv2d(64,128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)   → 128 × 6 × 6

Classifier head:
  Flatten → 128*6*6 = 4608
  Linear(4608, 512) → ReLU → Dropout(p=0.5)
  Linear(512, 1)    → Sigmoid

Output: scalar in (0, 1) — probability that the patch contains a face.

Design notes
------------
- BatchNorm after each conv stabilises training and allows a higher learning
  rate without the loss diverging.
- Dropout(0.5) before the final linear layer is the primary regulariser;
  weight-decay in the optimiser acts as a secondary one.
- Sigmoid on the single output pairs with BCELoss in train.py (or BCEWithLogitsLoss
  if the sigmoid is removed and raw logits are preferred).

Classes
-------
FaceDetectorCNN  — the nn.Module described above.
"""
