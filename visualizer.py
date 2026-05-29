"""
M5 — Visualisation and alerts.
Renders the 2-D parking map, free-slot counter, and violation alerts
onto the live frame.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

from layout import Layout, SlotDef
from logic import SlotState, ViolationAlert

# ─────────────────────────── colours ────────────────────────────────

_C_EMPTY    = (0,   200,  0)     # green
_C_OCCUPIED = (0,   0,   220)   # red  (BGR)
_C_UNKNOWN  = (0,   165, 255)   # orange
_C_ZONE     = (0,   0,   180)   # dark red fill
_C_TEXT     = (255, 255, 255)
_C_HUD_BG   = (30,  30,  30)
_FONT       = cv2.FONT_HERSHEY_SIMPLEX

# Hatch pattern (for greyscale / printed posters)
_HATCH_EMPTY    = None         # solid fill with alpha
_HATCH_OCCUPIED = "cross"
_HATCH_UNKNOWN  = "diag"

_ALPHA_FILL = 0.35   # polygon fill transparency


def _state_colour(state: SlotState) -> Tuple[int, int, int]:
    return {
        SlotState.EMPTY:    _C_EMPTY,
        SlotState.OCCUPIED: _C_OCCUPIED,
        SlotState.UNKNOWN:  _C_UNKNOWN,
    }[state]


def _state_label(state: SlotState) -> str:
    return {
        SlotState.EMPTY:    "TRONG",
        SlotState.OCCUPIED: "CO XE",
        SlotState.UNKNOWN:  "N/A",
    }[state]


# ─────────────────────────── hatch helpers ──────────────────────────

def _draw_hatch(canvas: np.ndarray, pts: np.ndarray,
                hatch: str, colour: Tuple) -> None:
    """Draw a hatch pattern clipped to the polygon region."""
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    h, w = canvas.shape[:2]
    step = 10

    if hatch == "cross":
        for y in range(0, h, step):
            row = np.zeros((1, w), dtype=np.uint8)
            row[0, :] = 255
            combined = cv2.bitwise_and(mask, row)
            canvas[y, combined[y] > 0] = colour
        for x in range(0, w, step):
            col_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
            col_mask[:, x] = 255
            combined = cv2.bitwise_and(mask, col_mask)
            canvas[combined > 0] = colour
    elif hatch == "diag":
        for d in range(-h, w, step):
            pts_line = np.array(
                [[max(0, d), 0], [min(w - 1, d + h), min(h - 1, w - 1 - d)]],
                dtype=np.int32
            )
            line_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(line_mask,
                     (max(0, d), 0),
                     (min(w - 1, d + h - 1), min(h - 1, h - 1)),
                     255, 1)
            combined = cv2.bitwise_and(mask, line_mask)
            canvas[combined > 0] = colour


# ─────────────────────────── main class ─────────────────────────────

class Visualizer:
    """
    Renders annotations on a copy of the live frame.
    call render() each sampled frame.
    """

    def __init__(self, layout: Layout,
                 show_detections: bool = True,
                 show_ids: bool = True) -> None:
        self.layout = layout
        self.show_detections = show_detections
        self.show_ids = show_ids
        self._alert_log: List[str] = []   # recent text messages

    def render(self, frame: np.ndarray,
               states: List[SlotState],
               detections,
               alerts: List[ViolationAlert]) -> np.ndarray:
        """
        Returns a BGR annotated frame (does not modify the input).
        """
        canvas = frame.copy()

        self._draw_no_park_zones(canvas)
        self._draw_slots(canvas, states)

        if self.show_detections:
            self._draw_detections(canvas, detections)

        free = sum(1 for s in states if s == SlotState.EMPTY)
        total = len(states)
        self._draw_hud(canvas, free, total)

        for alert in alerts:
            msg = (f"[VI PHAM] Zone {alert.zone_id}  "
                   f"conf={alert.confidence:.2f}")
            self._alert_log.append(msg)
            self._draw_alert_banner(canvas, msg)

        self._draw_alert_log(canvas)
        return canvas

    # ------------------------------------------------------------------ #
    # Drawing helpers                                                      #
    # ------------------------------------------------------------------ #

    def _draw_slots(self, canvas: np.ndarray,
                    states: List[SlotState]) -> None:
        overlay = canvas.copy()
        for slot, state in zip(self.layout.slots, states):
            pts = np.array(slot.polygon, dtype=np.int32)
            colour = _state_colour(state)

            cv2.fillPoly(overlay, [pts], colour)
            cv2.polylines(canvas, [pts], True, colour, 2)

            if self.show_ids:
                cx = int(sum(p[0] for p in slot.polygon) / len(slot.polygon))
                cy = int(sum(p[1] for p in slot.polygon) / len(slot.polygon))
                label = f"{slot.id}\n{_state_label(state)}"
                _put_text_centered(canvas, slot.id, (cx, cy - 8),
                                   _FONT, 0.40, colour, 1)
                _put_text_centered(canvas, _state_label(state),
                                   (cx, cy + 8), _FONT, 0.35, _C_TEXT, 1)

        cv2.addWeighted(overlay, _ALPHA_FILL, canvas, 1 - _ALPHA_FILL,
                        0, canvas)

    def _draw_no_park_zones(self, canvas: np.ndarray) -> None:
        overlay = canvas.copy()
        for zone in self.layout.zones:
            pts = np.array(zone.polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], _C_ZONE)
            cv2.polylines(canvas, [pts], True, (0, 0, 255), 2)
            cx = int(sum(p[0] for p in zone.polygon) / len(zone.polygon))
            cy = int(sum(p[1] for p in zone.polygon) / len(zone.polygon))
            cv2.putText(canvas, f"NO PARK {zone.id}",
                        (cx - 30, cy), _FONT, 0.4, (0, 0, 255), 1)
        cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)

    def _draw_detections(self, canvas: np.ndarray, detections) -> None:
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(canvas, f"{det.confidence:.2f}",
                        (x1, max(0, y1 - 4)),
                        _FONT, 0.35, (255, 255, 0), 1)

    def _draw_hud(self, canvas: np.ndarray,
                  free: int, total: int) -> None:
        h, w = canvas.shape[:2]
        bar_h = 44
        cv2.rectangle(canvas, (0, 0), (w, bar_h), _C_HUD_BG, -1)

        occupied = total - free
        # Simple inline bar
        bar_w = min(w - 20, 300)
        ratio = free / total if total else 0
        cv2.rectangle(canvas, (10, 10), (10 + bar_w, 34), (60, 60, 60), -1)
        cv2.rectangle(canvas, (10, 10),
                      (10 + int(bar_w * ratio), 34), _C_EMPTY, -1)
        cv2.rectangle(canvas, (10, 10), (10 + bar_w, 34), (180, 180, 180), 1)

        cv2.putText(canvas,
                    f"TRONG: {free}   CO XE: {occupied}   TONG: {total}",
                    (14 + bar_w + 8, 28), _FONT, 0.55, _C_TEXT, 1)

    def _draw_alert_banner(self, canvas: np.ndarray, msg: str) -> None:
        h, w = canvas.shape[:2]
        cv2.rectangle(canvas, (0, h - 50), (w, h), (0, 0, 160), -1)
        cv2.putText(canvas, msg, (8, h - 18), _FONT, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_alert_log(self, canvas: np.ndarray) -> None:
        keep = self._alert_log[-6:]
        self._alert_log = keep
        h, w = canvas.shape[:2]
        for i, msg in enumerate(reversed(keep)):
            y = h - 60 - i * 18
            if y < 50:
                break
            cv2.putText(canvas, msg, (8, y), _FONT, 0.35,
                        (200, 120, 120), 1, cv2.LINE_AA)


def _put_text_centered(img: np.ndarray, text: str, center: Tuple[int, int],
                        font, scale: float, colour, thickness: int) -> None:
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = center[0] - tw // 2
    y = center[1] + th // 2
    cv2.putText(img, text, (x, y), font, scale, colour, thickness,
                cv2.LINE_AA)
