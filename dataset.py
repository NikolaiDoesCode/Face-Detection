"""
dataset.py — Data pipeline for the face/no-face binary classifier.

Responsibilities
----------------
1. Positive samples (faces):
   - Download the Labeled Faces in the Wild (LFW) dataset via sklearn's
     `fetch_lfw_people`, which pulls ~13,000 face crops from the web.
   - Resize every crop to 64x64 RGB and save as PNG under data/positives/.

2. Negative samples (non-faces):
   - Accept a folder of background images (indoor/outdoor scenes with no faces).
   - Slide a 64x64 window across each background image at random positions and
     scales, saving patches under data/negatives/.
   - Target roughly a 1:1 positive-to-negative ratio; excess negatives are
     discarded at sampling time to avoid class imbalance.

3. PyTorch Dataset:
   - `FaceDataset` subclasses `torch.utils.data.Dataset`.
   - Applies augmentations at training time: random horizontal flip, colour
     jitter, small rotations.
   - Returns (tensor, label) pairs where label=1 for face, label=0 for no-face.

4. DataLoader helpers:
   - `get_loaders(config)` returns (train_loader, val_loader) split 80/20 from
     the full dataset, using config values for batch size, num_workers, etc.

Usage (will be called from train.py):
    from dataset import get_loaders
    train_loader, val_loader = get_loaders(config)
"""
