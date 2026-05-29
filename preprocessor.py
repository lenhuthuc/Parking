"""
M1 — Frame preprocessing.
Letterbox resize to 640×640 + CLAHE brightness normalisation.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from typing import Tuple


class Preprocessor:
    """
    Stateless preprocessor.  Call preprocess() to get a normalised tensor
    ready for YOLO, plus the inverse-mapping parameters needed by M2.
    """

    def __init__(self, target_size: int = 640, clahe_clip: float = 2.0,
                 clahe_tile: Tuple[int, int] = (8, 8)) -> None:
        self.target_size = target_size
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                      tileGridSize=clahe_tile)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def preprocess(self, frame_bgr: np.ndarray
                   ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Returns
        -------
        tensor    : uint8 RGB image of shape (target_size, target_size, 3)
        scale     : float — ratio used for the resize (same for x and y)
        pad       : (pad_x, pad_y) — pixels of padding added on each side
        """
        equalized = self._apply_clahe(frame_bgr)
        tensor, scale, pad = self._letterbox(equalized)
        return tensor, scale, pad

    def map_back(self, x: float, y: float,
                 scale: float, pad: Tuple[int, int]) -> Tuple[float, float]:
        """Convert a point from the 640×640 space back to original pixel space."""
        pad_x, pad_y = pad
        x_orig = (x - pad_x) / scale
        y_orig = (y - pad_y) / scale
        return x_orig, y_orig

    def map_back_bbox(self, bbox: Tuple[float, float, float, float],
                      scale: float, pad: Tuple[int, int]
                      ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        x1o, y1o = self.map_back(x1, y1, scale, pad)
        x2o, y2o = self.map_back(x2, y2, scale, pad)
        return (x1o, y1o, x2o, y2o)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """CLAHE on the L-channel of LAB, preserving hue/saturation."""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = self._clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _letterbox(self, frame_bgr: np.ndarray
                   ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize keeping aspect ratio; pad remainder with grey (114).
        Returns (image_640, scale, (pad_x, pad_y)).
        """
        h, w = frame_bgr.shape[:2]
        t = self.target_size
        scale = min(t / w, t / h)

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(frame_bgr, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        pad_x = (t - new_w) // 2
        pad_y = (t - new_h) // 2

        canvas = np.full((t, t, 3), 114, dtype=np.uint8)
        canvas[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return rgb, scale, (pad_x, pad_y)


class FrameSampler:
    """
    Wraps an OpenCV VideoCapture and yields frames at *target_fps*.
    Drops frames to match target rate without sleeping.
    """

    def __init__(self, source, target_fps: float = 2.0) -> None:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._step = max(1, int(round(native_fps / target_fps)))
        self._frame_idx = 0

    @property
    def fps(self) -> float:
        native = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        return native / self._step

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration
            idx = self._frame_idx
            self._frame_idx += 1
            if idx % self._step == 0:
                return frame

    def release(self) -> None:
        self.cap.release()
