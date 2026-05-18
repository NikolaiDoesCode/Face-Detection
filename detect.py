"""
detect.py — Sliding-window face detector using a trained FaceDetectorCNN.

Detection pipeline
------------------
1. Load FaceDetectorCNN weights from a checkpoint path.

2. Sliding window (per frame):
   - For each scale in a configurable pyramid (e.g. 1.0, 0.75, 0.5, 0.25):
       a. Resize the frame to that scale.
       b. Slide a 64×64 window across the resized frame with a configurable
          step size (e.g. 16px).
       c. Crop each patch, normalise, run through the CNN.
       d. If the model's output probability exceeds `detection_threshold`,
          record the bounding box (rescaled back to original frame coordinates)
          and its confidence score.

3. Non-maximum suppression (NMS):
   - Sort all candidate boxes by confidence (highest first).
   - Greedily keep a box if its IoU with every already-kept box is below
     `nms_iou_threshold`.
   - Returns a list of (x, y, w, h, confidence) tuples.

4. Public API:
   `Detector(checkpoint_path, config)` — loads the model once.
   `Detector.detect(frame) -> list[tuple]` — runs the full pipeline on one BGR
   frame and returns NMS-filtered bounding boxes.

Performance note
----------------
The sliding window is the bottleneck. Batching patches from all positions at a
given scale before a single model forward pass (instead of one-by-one inference)
cuts wall-clock time by ~10×; this batched approach will be used here.
"""
