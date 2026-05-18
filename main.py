"""
main.py — Live webcam face detection with optional video recording.

Ties together the full pipeline:
  Detector (detect.py) → OpenCV frame loop → annotated display + MP4 recording

Usage
-----
    python main.py                                         # defaults from config.yaml
    python main.py --camera 1                              # use a different camera
    python main.py --checkpoint checkpoints/best_model.pth # explicit weights path
    python main.py --no-record                             # display only, no files saved
    python main.py --config other.yaml                     # different config

Keys while running
------------------
    q   quit cleanly

Outputs
-------
    recordings/   timestamped MP4s, written while a face is present
    screenshots/  timestamped PNGs, one per detection event (debounced)
"""

import argparse
import collections
import datetime
import time
from pathlib import Path

import cv2
import yaml

from detect import Detector


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════
#
# All visual elements are drawn onto the frame in place (OpenCV mutates the
# numpy array).  We keep drawing logic in standalone functions so the webcam
# loop stays readable as a sequence of high-level steps.
#
# Colour conventions — BGR (OpenCV's channel order):
#   (0, 220, 0)    bright green  — bounding boxes and confidence labels
#   (0, 0, 220)    bright red    — REC indicator
#   (220, 220, 220) light grey   — FPS / face count overlay
#
# Font: FONT_HERSHEY_SIMPLEX is the most legible of OpenCV's built-in fonts
#   at small sizes.  All text uses a 1-pixel black outline (drawn first at
#   slightly higher thickness) so it remains readable against any background.
# ══════════════════════════════════════════════════════════════════════════════

_GREEN  = (0, 220, 0)
_RED    = (0, 0, 220)
_ORANGE = (0, 165, 220)   # screenshot flash indicator
_GREY   = (220, 220, 220)
_BLACK  = (0, 0, 0)
_FONT   = cv2.FONT_HERSHEY_SIMPLEX


def _put_text(
    frame,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.65,
    color: tuple = _GREY,
    thickness: int = 2,
) -> None:
    """Draw text with a thin black outline for readability on any background."""
    cv2.putText(frame, text, pos, _FONT, scale, _BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, _FONT, scale, color,  thickness,     cv2.LINE_AA)


def _draw_boxes(frame, boxes: list[tuple]) -> None:
    """
    Draw each detection box and its confidence percentage on the frame.

    boxes: list of (x, y, w, h, confidence) from Detector.detect().
    """
    for x, y, w, h, conf in boxes:
        # Bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), _GREEN, 2, cv2.LINE_AA)
        # Confidence label just above the box; clamp y so it never goes off-screen
        label    = f"{conf:.0%}"
        label_y  = max(y - 6, 14)
        _put_text(frame, label, (x, label_y), scale=0.6, color=_GREEN)


def _draw_hud(
    frame,
    fps: float,
    n_faces: int,
    recording: bool,
    screenshot_flash: bool = False,
) -> None:
    """
    Draw the heads-up display: face count, FPS, and status indicators.

    Indicators stack vertically so REC and CAM never overlap.
    screenshot_flash should be True for ~2 seconds after a screenshot is saved.
    """
    _put_text(frame, f"Faces : {n_faces}", (10, 26), color=_GREY)
    _put_text(frame, f"FPS   : {fps:.1f}",  (10, 52), color=_GREY)
    y = 78
    if recording:
        _put_text(frame, "● REC", (10, y), scale=0.75, color=_RED,    thickness=2)
        y += 26
    if screenshot_flash:
        _put_text(frame, "● CAM", (10, y), scale=0.75, color=_ORANGE, thickness=2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FPS tracker
# ══════════════════════════════════════════════════════════════════════════════
#
# A rolling window of the N most recent frame timestamps gives a smoothed FPS
# reading that doesn't spike wildly when a single heavy frame (e.g. one with
# many detections) takes longer than usual.
#
# collections.deque(maxlen=N) automatically evicts the oldest timestamp when a
# new one is appended, so the window is always exactly N frames wide.
#
# FPS = frames / elapsed = (maxlen - 1) / (newest_time - oldest_time).
# We need at least 2 timestamps for a meaningful rate; return 0.0 otherwise.
# ══════════════════════════════════════════════════════════════════════════════

class _FPSTracker:
    def __init__(self, window: int = 30):
        self._times: collections.deque = collections.deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Recording state machine
# ══════════════════════════════════════════════════════════════════════════════
#
# Recording follows the same logic as the original Detection.py, extracted into
# a class so the webcam loop doesn't carry six raw variables.
#
# State transitions:
#
#   IDLE  ──(face detected)──►  RECORDING  (VideoWriter opened, file started)
#
#   RECORDING  ──(face detected)──►  RECORDING  (timer reset; keep recording)
#   RECORDING  ──(no face, timer not running)──►  COOLING_DOWN  (countdown starts)
#
#   COOLING_DOWN  ──(face detected)──►  RECORDING  (cancel countdown, keep going)
#   COOLING_DOWN  ──(countdown expired)──►  IDLE  (VideoWriter released, file saved)
#
# The "cooling down" period avoids choppy recordings when a face briefly
# disappears between frames — without it, each momentary gap would split one
# activity into many short clips.
# ══════════════════════════════════════════════════════════════════════════════

class _Recorder:
    """
    Manages the VideoWriter lifecycle based on face presence.

    Args:
        output_dir:    directory to save MP4 files.
        frame_size:    (width, height) tuple from the camera.
        fourcc_str:    FourCC codec string (e.g. 'mp4v').
        fps:           output video frame rate.
        linger_secs:   seconds to keep recording after the last detection.
        enabled:       if False, record() is a no-op (--no-record mode).
    """

    def __init__(
        self,
        output_dir: Path,
        frame_size: tuple[int, int],
        fourcc_str: str,
        fps: float,
        linger_secs: float,
        enabled: bool,
    ):
        self._output_dir  = output_dir
        self._frame_size  = frame_size
        self._fourcc      = cv2.VideoWriter_fourcc(*fourcc_str)
        self._fps         = fps
        self._linger      = linger_secs
        self._enabled     = enabled

        self._writer: cv2.VideoWriter | None = None
        self._recording   = False
        self._cooling_down = False
        self._stopped_at: float = 0.0

        if enabled:
            output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_recording(self) -> bool:
        return self._recording

    def update(self, frame, face_detected: bool) -> None:
        """
        Advance the state machine and write the frame if currently recording.

        Must be called every frame in the webcam loop.
        """
        if not self._enabled:
            return

        if face_detected:
            if not self._recording:
                self._start()
            # Reset any in-progress countdown — face is back
            self._cooling_down = False

        else:  # no face this frame
            if self._recording:
                if not self._cooling_down:
                    # Begin countdown
                    self._cooling_down = True
                    self._stopped_at   = time.time()
                elif time.time() - self._stopped_at >= self._linger:
                    self._stop()

        if self._recording and self._writer is not None:
            self._writer.write(frame)

    def release(self) -> None:
        """Release the VideoWriter if open. Call on program exit."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _start(self) -> None:
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        path      = self._output_dir / f"{timestamp}.mp4"
        self._writer   = cv2.VideoWriter(
            str(path), self._fourcc, self._fps, self._frame_size
        )
        self._recording    = True
        self._cooling_down = False
        print(f"  [REC] Started  → {path}")

    def _stop(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._recording    = False
        self._cooling_down = False
        print("  [REC] Stopped")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Screenshot capture
# ══════════════════════════════════════════════════════════════════════════════
#
# Takes a PNG snapshot of the annotated frame whenever a face is detected,
# subject to a minimum interval so a face sitting in frame for 10 minutes
# doesn't fill the screenshots folder with thousands of near-identical images.
#
# The interval is intentionally coarse (default 10 seconds).  The goal is
# "one representative image per detection event", not a frame-by-frame log —
# the MP4 recording handles that.  If you want a tighter burst, lower
# screenshots.min_interval_secs in config.yaml.
#
# update() returns True on the frame a screenshot was actually saved so the
# HUD can flash a "● CAM" indicator for a couple of seconds.
# ══════════════════════════════════════════════════════════════════════════════

class _Screenshotter:
    """
    Debounced screenshot capture: saves a PNG when a face is detected,
    at most once every min_interval_secs.

    Args:
        output_dir:        directory to write PNG files.
        min_interval_secs: cooldown between saves.
        enabled:           if False, update() is a no-op (--no-screenshot).
    """

    def __init__(self, output_dir: Path, min_interval_secs: float, enabled: bool):
        self._output_dir = output_dir
        self._interval   = min_interval_secs
        self._enabled    = enabled
        self._last_shot  = 0.0   # epoch time of most recent save

        if enabled:
            output_dir.mkdir(parents=True, exist_ok=True)

    def update(self, frame, face_detected: bool) -> bool:
        """
        Conditionally save a screenshot.

        Returns True if a screenshot was written this call, False otherwise.
        Callers use the return value to drive the HUD flash indicator.
        """
        if not self._enabled or not face_detected:
            return False

        now = time.time()
        if now - self._last_shot < self._interval:
            return False   # still within the cooldown window

        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        path      = self._output_dir / f"{timestamp}.png"
        cv2.imwrite(str(path), frame)
        self._last_shot = now
        print(f"  [CAM] Screenshot → {path}")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Webcam loop
# ══════════════════════════════════════════════════════════════════════════════
#
# The loop is intentionally thin — all the heavy logic lives in Detector and
# _Recorder.  The loop's only jobs are:
#   1. Read a frame.
#   2. Detect faces.
#   3. Draw the result.
#   4. Hand the frame to the recorder.
#   5. Show the window.
#   6. Handle the quit key.
#
# Error handling on cap.read():
#   A False return from cap.read() usually means the camera was unplugged or
#   the driver dropped a frame.  We skip the frame and continue rather than
#   crashing — dropped frames are common on busy USB buses and don't indicate
#   a fatal error.
#
# Performance note:
#   The sliding window across four pyramid scales takes ~0.5–3 seconds per
#   frame on CPU.  On Apple M-series (MPS) or a CUDA GPU it is much faster.
#   To improve CPU performance, reduce the scales list in config.yaml
#   (e.g. [1.0, 0.5]) or increase step_size (e.g. 32).  Both reduce the
#   number of patches and therefore the number of CNN forward passes.
# ══════════════════════════════════════════════════════════════════════════════

def _run_loop(
    detector: Detector,
    cap: cv2.VideoCapture,
    recorder: _Recorder,
    screenshotter: _Screenshotter,
) -> None:
    """
    Main frame loop. Runs until the user presses 'q' or the camera disconnects.
    """
    fps_tracker       = _FPSTracker(window=30)
    last_shot_at: float = 0.0   # drives the 2-second HUD flash after each screenshot

    while True:
        ok, frame = cap.read()
        if not ok:
            print("  Camera read failed — frame skipped.")
            continue

        # ── Detection ─────────────────────────────────────────────────────────
        boxes        = detector.detect(frame)
        face_present = len(boxes) > 0

        # ── Annotate ──────────────────────────────────────────────────────────
        _draw_boxes(frame, boxes)
        fps_tracker.tick()

        # Flash "● CAM" for 2 seconds after the most recent screenshot
        screenshot_flash = (time.time() - last_shot_at) < 2.0
        _draw_hud(frame, fps_tracker.fps, len(boxes), recorder.is_recording,
                  screenshot_flash)

        # ── Screenshot ────────────────────────────────────────────────────────
        if screenshotter.update(frame, face_detected=face_present):
            last_shot_at = time.time()

        # ── Record ────────────────────────────────────────────────────────────
        recorder.update(frame, face_detected=face_present)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow("Face Detector  (q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main: argument parsing and startup
# ══════════════════════════════════════════════════════════════════════════════
#
# Startup sequence:
#   1. Parse args and load config.
#   2. Build Detector — this is the slow step (loads model weights, runs smoke
#      test).  Done before opening the camera so any weight/config errors are
#      reported immediately rather than after the camera stream has started.
#   3. Open the camera and verify it works.
#   4. Build the Recorder with the camera's actual frame size.
#   5. Enter the frame loop.
#   6. Clean up in a finally block so the camera and any open VideoWriter are
#      always released, even if the loop exits via an uncaught exception.
#
# Why open the camera after the Detector?
#   Camera streams keep the device busy.  If model loading fails (missing
#   checkpoint, wrong config), we want to exit without having locked the camera.
#   Reversing the order would leave the camera occupied until the OS times out.
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live face detection with a custom CNN"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to model checkpoint (default: detection.checkpoint in config.yaml)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera device index (default: 0)",
    )
    parser.add_argument(
        "--no-record", action="store_true",
        help="Disable automatic MP4 recording",
    )
    parser.add_argument(
        "--no-screenshot", action="store_true",
        help="Disable automatic PNG screenshots on detection",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoint_path = args.checkpoint or config["detection"]["checkpoint"]

    # ── Load model ────────────────────────────────────────────────────────────
    print("=== Loading detector ===")
    detector = Detector(checkpoint_path, config)

    # ── Open camera ───────────────────────────────────────────────────────────
    print(f"\n=== Opening camera {args.camera} ===")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. "
            "Check the --camera index or that no other app is using it."
        )

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Resolution: {frame_w}×{frame_h}")

    # ── Build recorder ────────────────────────────────────────────────────────
    rec_cfg  = config["recording"]
    recorder = _Recorder(
        output_dir  = Path(rec_cfg["output_dir"]),
        frame_size  = (frame_w, frame_h),
        fourcc_str  = rec_cfg["fourcc"],
        fps         = rec_cfg["fps"],
        linger_secs = rec_cfg["seconds_after_detection"],
        enabled     = not args.no_record,
    )

    if args.no_record:
        print("  Recording disabled (--no-record).")
    else:
        print(f"  Recordings  → {rec_cfg['output_dir']}/")

    # ── Build screenshotter ───────────────────────────────────────────────────
    ss_cfg       = config["screenshots"]
    screenshotter = _Screenshotter(
        output_dir        = Path(ss_cfg["output_dir"]),
        min_interval_secs = ss_cfg["min_interval_secs"],
        enabled           = not args.no_screenshot,
    )

    if args.no_screenshot:
        print("  Screenshots disabled (--no-screenshot).")
    else:
        print(f"  Screenshots → {ss_cfg['output_dir']}/  "
              f"(every {ss_cfg['min_interval_secs']}s min)")

    print("\nRunning — press 'q' to quit.\n")

    # ── Frame loop (always clean up) ──────────────────────────────────────────
    try:
        _run_loop(detector, cap, recorder, screenshotter)
    finally:
        recorder.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


if __name__ == "__main__":
    main()
