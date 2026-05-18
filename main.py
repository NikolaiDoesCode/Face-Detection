"""
main.py — Entry point: live webcam feed with real-time face detection.

What it does
------------
1. Parse CLI args:
   --checkpoint  path to model weights (default: checkpoints/best_model.pth)
   --config      path to config file   (default: config.yaml)
   --camera      camera device index   (default: 0)
   --no-record   disable automatic video recording

2. Load config and instantiate `Detector` from detect.py.

3. Open the webcam via OpenCV.

4. Per-frame loop:
   a. Read frame from camera.
   b. Pass frame to `Detector.detect()` to get bounding boxes.
   c. Draw each box and its confidence percentage on the frame.
   d. Optionally record to an MP4 when at least one face is detected (same
      logic as the original Detection.py: keep recording for
      SECONDS_TO_RECORD_AFTER_DETECTION after the last detection).
   e. Display the annotated frame in an OpenCV window.
   f. Exit cleanly on 'q' keypress.

5. Release all resources (VideoWriter, VideoCapture) on exit.

Usage:
    python main.py
    python main.py --checkpoint checkpoints/best_model.pth --camera 1
    python main.py --no-record
"""
