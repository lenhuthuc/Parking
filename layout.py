"""
M3 — Spatial layout management.
Interactive OpenCV UI to draw parking slots and no-parking zones.
Saves / loads layout.json with pre-computed area and AABB.
"""
from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from geometry import (
    shoelace_area, aabb_of_polygon, Polygon, Point
)

# ------------------------------------------------------------------ #
# Data structures                                                      #
# ------------------------------------------------------------------ #

class SlotDef:
    __slots__ = ("id", "polygon", "area", "aabb")

    def __init__(self, slot_id: str, polygon: Polygon) -> None:
        self.id = slot_id
        self.polygon = polygon
        self.area = shoelace_area(polygon)
        self.aabb: Tuple[float, float, float, float] = aabb_of_polygon(polygon)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "polygon": self.polygon,
            "area": self.area,
            "aabb": list(self.aabb),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SlotDef":
        pts = [tuple(p) for p in d["polygon"]]
        obj = cls.__new__(cls)
        obj.id = d["id"]
        obj.polygon = pts
        obj.area = d.get("area", shoelace_area(pts))
        obj.aabb = tuple(d["aabb"]) if "aabb" in d else aabb_of_polygon(pts)
        return obj


class ZoneDef:
    __slots__ = ("id", "polygon", "area", "aabb")

    def __init__(self, zone_id: str, polygon: Polygon) -> None:
        self.id = zone_id
        self.polygon = polygon
        self.area = shoelace_area(polygon)
        self.aabb: Tuple[float, float, float, float] = aabb_of_polygon(polygon)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "polygon": self.polygon,
            "area": self.area,
            "aabb": list(self.aabb),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ZoneDef":
        pts = [tuple(p) for p in d["polygon"]]
        obj = cls.__new__(cls)
        obj.id = d["id"]
        obj.polygon = pts
        obj.area = d.get("area", shoelace_area(pts))
        obj.aabb = tuple(d["aabb"]) if "aabb" in d else aabb_of_polygon(pts)
        return obj


class Layout:
    def __init__(self, slots: List[SlotDef], zones: List[ZoneDef]) -> None:
        self.slots = slots
        self.zones = zones

    def save(self, path: str) -> None:
        data = {
            "parking_slots": [s.to_dict() for s in self.slots],
            "no_parking_zones": [z.to_dict() for z in self.zones],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Layout":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        slots = [SlotDef.from_dict(d) for d in data.get("parking_slots", [])]
        zones = [ZoneDef.from_dict(d) for d in data.get("no_parking_zones", [])]
        return cls(slots, zones)


# ------------------------------------------------------------------ #
# Interactive drawing UI                                              #
# ------------------------------------------------------------------ #

_COLOR_SLOT = (0, 255, 0)        # green
_COLOR_ZONE = (0, 0, 255)        # red
_COLOR_WIP = (255, 165, 0)       # orange — polygon being drawn
_COLOR_VERTEX = (255, 255, 0)    # yellow dots
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class LayoutEditor:
    """
    OpenCV-based interactive editor.

    Controls
    --------
    Left-click   : add vertex to current polygon
    Right-click  : undo last vertex
    Enter / s    : finish current polygon as a SLOT
    z            : finish current polygon as a NO-PARKING ZONE
    Backspace    : delete the last completed polygon
    Escape / q   : quit and save
    """

    def __init__(self, frame: np.ndarray,
                 existing_layout: Optional[Layout] = None) -> None:
        self._base = frame.copy()
        self._h, self._w = frame.shape[:2]

        self._slots: List[SlotDef] = (
            list(existing_layout.slots) if existing_layout else []
        )
        self._zones: List[ZoneDef] = (
            list(existing_layout.zones) if existing_layout else []
        )

        self._wip: List[Point] = []          # current polygon vertices
        self._mouse: Optional[Point] = None  # cursor position
        self._win = "Layout Editor"

    def run(self) -> Layout:
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._win, self._on_mouse)
        self._redraw()

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):   # Escape or q → done
                break
            elif key in (13, ord("s")): # Enter or s → save as slot
                self._finish_slot()
            elif key == ord("z"):        # z → save as zone
                self._finish_zone()
            elif key in (8, 127):        # Backspace/Delete → undo last polygon
                self._undo_last()
            self._redraw()

        cv2.destroyWindow(self._win)
        return Layout(self._slots, self._zones)

    # ------------------------------------------------------------------ #

    def _on_mouse(self, event, x, y, flags, param):
        self._mouse = (float(x), float(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            self._wip.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and self._wip:
            self._wip.pop()
        self._redraw()

    def _finish_slot(self):
        if len(self._wip) < 3:
            return
        slot_id = f"S{len(self._slots) + 1:02d}"
        self._slots.append(SlotDef(slot_id, list(self._wip)))
        self._wip = []

    def _finish_zone(self):
        if len(self._wip) < 3:
            return
        zone_id = f"Z{len(self._zones) + 1:02d}"
        self._zones.append(ZoneDef(zone_id, list(self._wip)))
        self._wip = []

    def _undo_last(self):
        if self._slots:
            self._slots.pop()
        elif self._zones:
            self._zones.pop()

    def _redraw(self):
        canvas = self._base.copy()

        # Completed slots
        for slot in self._slots:
            pts = np.array(slot.polygon, dtype=np.int32)
            cv2.polylines(canvas, [pts], True, _COLOR_SLOT, 2)
            cx, cy = _centroid(slot.polygon)
            cv2.putText(canvas, slot.id, (int(cx) - 10, int(cy)),
                        _FONT, 0.5, _COLOR_SLOT, 1)

        # Completed no-parking zones
        overlay = canvas.copy()
        for zone in self._zones:
            pts = np.array(zone.polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 0, 80))
            cv2.polylines(canvas, [pts], True, _COLOR_ZONE, 2)
            cx, cy = _centroid(zone.polygon)
            cv2.putText(canvas, zone.id, (int(cx) - 10, int(cy)),
                        _FONT, 0.5, _COLOR_ZONE, 1)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        # Work-in-progress polygon
        if self._wip:
            for pt in self._wip:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4,
                           _COLOR_VERTEX, -1)
            if len(self._wip) > 1:
                pts = np.array(self._wip, dtype=np.int32)
                cv2.polylines(canvas, [pts], False, _COLOR_WIP, 1)
            if self._mouse:
                cv2.line(canvas,
                         (int(self._wip[-1][0]), int(self._wip[-1][1])),
                         (int(self._mouse[0]), int(self._mouse[1])),
                         _COLOR_WIP, 1)

        # HUD
        lines = [
            "Left-click: add vertex   Right-click: undo vertex",
            "Enter/S: finish SLOT     Z: finish NO-PARK ZONE",
            "Backspace: undo last polygon   Q/Esc: save & quit",
            f"Slots: {len(self._slots)}   Zones: {len(self._zones)}   "
            f"WIP vertices: {len(self._wip)}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (8, 18 + i * 18),
                        _FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow(self._win, canvas)


def _centroid(polygon: Polygon) -> Point:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ------------------------------------------------------------------ #
# Convenience entry points                                            #
# ------------------------------------------------------------------ #

def create_layout(frame: np.ndarray,
                  save_path: str = "layout.json") -> Layout:
    """Open the editor, return the layout and save to *save_path*."""
    editor = LayoutEditor(frame)
    layout = editor.run()
    layout.save(save_path)
    print(f"Saved {len(layout.slots)} slots + {len(layout.zones)} zones "
          f"to {save_path}")
    return layout


def load_or_create_layout(frame: np.ndarray,
                           path: str = "layout.json") -> Layout:
    """Load existing layout or open editor if the file does not exist."""
    if os.path.exists(path):
        layout = Layout.load(path)
        print(f"Loaded layout: {len(layout.slots)} slots, "
              f"{len(layout.zones)} zones from {path}")
        return layout
    return create_layout(frame, path)
