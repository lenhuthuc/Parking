"""
M2 — Object detection wrapper around YOLOv8.
Returns detections as plain Python dicts; no framework objects leak out.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

# Lazy import so the rest of the codebase works even without ultralytics
try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


@dataclass
class Detection:
    """Single detection output (coordinates in the *original* pixel space)."""
    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2)
    confidence: float
    label: int  # class index (0 = motorbike in most configs)


class Detector:
    """
    Thin wrapper around YOLOv8.
    After construction call detect() with an RGB uint8 numpy array (640×640).
    """

    # YOLO class index for motorbike/motorcycle (COCO: 3)
    MOTORBIKE_CLASSES = {3}

    def __init__(self, model_path: str = "yolov8n.pt",
                 conf_threshold: float = 0.25,
                 iou_nms: float = 0.45,
                 device: str = "cpu") -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. "
                "Run: pip install ultralytics"
            )
        self.model = _YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_nms = iou_nms
        self.device = device

    def detect(self, tensor_rgb: np.ndarray,
               scale: float, pad: Tuple[int, int],
               target_classes: Optional[set] = None) -> List[Detection]:
        """
        Run inference on the pre-processed 640×640 RGB tensor and return
        detections mapped back to the original frame's pixel coordinate system.

        Parameters
        ----------
        tensor_rgb  : (640, 640, 3) uint8 array (output of Preprocessor)
        scale       : inverse-map scale from Preprocessor
        pad         : (pad_x, pad_y) from Preprocessor
        target_classes : set of class indices to keep; None = keep motorbikes
        """
        if target_classes is None:
            target_classes = self.MOTORBIKE_CLASSES

        results = self.model.predict(
            source=tensor_rgb,
            conf=self.conf_threshold,
            iou=self.iou_nms,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in target_classes:
                continue
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Map from 640-space back to original pixel space
            x1o, y1o = self._map_back(x1, y1, scale, pad)
            x2o, y2o = self._map_back(x2, y2, scale, pad)
            detections.append(Detection(
                bbox=(x1o, y1o, x2o, y2o),
                confidence=conf,
                label=cls_id,
            ))
        return detections

    @staticmethod
    def _map_back(x: float, y: float,
                  scale: float, pad: Tuple[int, int]) -> Tuple[float, float]:
        pad_x, pad_y = pad
        return (x - pad_x) / scale, (y - pad_y) / scale


# ---------------------------------------------------------------------------
# Stub detector (for testing without GPU / ultralytics)
# ---------------------------------------------------------------------------

class StubDetector:
    """Returns hand-crafted detections from a list; useful for unit tests."""

    def __init__(self, detections_per_frame: List[List[Detection]]) -> None:
        self._frames = detections_per_frame
        self._idx = 0

    def detect(self, tensor_rgb, scale, pad,
               target_classes=None) -> List[Detection]:
        if not self._frames:
            return []
        result = self._frames[self._idx % len(self._frames)]
        self._idx += 1
        return result
