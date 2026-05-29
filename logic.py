"""
M4 — Core decision logic.
Covers:
  • Spatial grid for AABB broad-phase filtering
  • Slot-detection matching (greedy by IoU + anchor tie-break)
  • Raw-state classifier (3-class threshold rule)
  • Per-slot finite state machine with dwell + hysteresis
  • No-parking violation detector
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from geometry import (
    polygon_iou, point_in_polygon, anchor_point,
    aabbs_overlap, shoelace_area, bbox_area,
    Polygon
)
from detector import Detection
from layout import SlotDef, ZoneDef

# ─────────────────────────── constants ──────────────────────────────

IOU_LO   = 0.30
IOU_HI   = 0.50
CONF_LO  = 0.50
CONF_HI  = 0.70
OCC_THR  = 0.60   # occlusion threshold

TAU_PARK  = 5.0   # seconds: consecutive "occupied" before slot → OCCUPIED
TAU_LEAVE = 3.0   # seconds: consecutive "empty"    before slot → EMPTY
TAU_UNK   = 1.5   # seconds: consecutive "unknown"  before slot → UNKNOWN
TAU_VIOL  = 30.0  # seconds: dwell in no-park zone before alert


# ─────────────────────────── enums ──────────────────────────────────

class SlotState(Enum):
    EMPTY   = "TRỐNG"
    OCCUPIED = "CÓ XE"
    UNKNOWN  = "KHÔNG XÁC ĐỊNH"


class RawState(Enum):
    EMPTY   = "TRỐNG"
    OCCUPIED = "CÓ XE"
    UNKNOWN  = "KHÔNG XÁC ĐỊNH"


# ─────────────────────────── alert ──────────────────────────────────

@dataclass
class ViolationAlert:
    zone_id: str
    snapshot: Optional[np.ndarray]   # cropped frame evidence
    confidence: float
    bbox: Tuple[float, float, float, float]


# ─────────────────────────── spatial grid ───────────────────────────

class SpatialGrid:
    """
    Divides the image into a coarse grid; each slot AABB is registered in
    every cell it overlaps.  grid.query(aabb) returns only the detection
    indices whose AABBs overlap *aabb* — O(1) broad-phase filter.
    """

    def __init__(self, img_w: int, img_h: int, cell_size: int = 100) -> None:
        self.img_w = img_w
        self.img_h = img_h
        self.cell_size = cell_size
        self.cols = math.ceil(img_w / cell_size)
        self.rows = math.ceil(img_h / cell_size)
        self._cells: Dict[Tuple[int, int], List[int]] = {}

    def _cells_for_aabb(self, aabb: Tuple[float, float, float, float]):
        xmin, ymin, xmax, ymax = aabb
        c0 = max(0, int(xmin // self.cell_size))
        c1 = min(self.cols - 1, int(xmax // self.cell_size))
        r0 = max(0, int(ymin // self.cell_size))
        r1 = min(self.rows - 1, int(ymax // self.cell_size))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                yield (r, c)

    def build(self, aabbs: List[Tuple[float, float, float, float]]) -> None:
        """Register a list of AABBs (indexed by position in the list)."""
        self._cells.clear()
        for idx, aabb in enumerate(aabbs):
            for cell in self._cells_for_aabb(aabb):
                self._cells.setdefault(cell, []).append(idx)

    def query(self, aabb: Tuple[float, float, float, float]) -> List[int]:
        """Return indices registered in cells overlapping *aabb* (no duplicates)."""
        found = set()
        for cell in self._cells_for_aabb(aabb):
            found.update(self._cells.get(cell, []))
        return list(found)


# ─────────────────────────── matching ───────────────────────────────

@dataclass
class MatchResult:
    best_det: Optional[Detection]
    best_iou: float
    occlusion: float   # fraction of slot area covered by *other* boxes


def match_slot(slot: SlotDef,
               detections: List[Detection],
               candidate_indices: List[int]) -> MatchResult:
    """
    Find the detection best associated with *slot*.
    - candidate_indices: pre-filtered by spatial grid (broad-phase)
    - Greedy: pick the box with highest IoU; use anchor as tie-breaker
    - Accumulate occlusion from non-winning boxes that overlap the slot
    """
    best_det: Optional[Detection] = None
    best_iou = 0.0
    occlusion_area = 0.0

    for idx in candidate_indices:
        det = detections[idx]
        iou = polygon_iou(det.bbox, slot.polygon, slot.area)
        anchor = anchor_point(det.bbox)
        anchor_inside = point_in_polygon(anchor[0], anchor[1], slot.polygon)

        if anchor_inside or iou > 0:
            if iou > best_iou:
                # Demote current best to occlusion if it exists
                if best_det is not None:
                    from geometry import sutherland_hodgman, bbox_to_polygon
                    inter = sutherland_hodgman(
                        bbox_to_polygon(best_det.bbox), slot.polygon
                    )
                    if len(inter) >= 3:
                        from geometry import shoelace_area as _sa
                        occlusion_area += _sa(inter)
                best_iou = iou
                best_det = det
            else:
                # This box occludes the slot but is not the winner
                from geometry import sutherland_hodgman, bbox_to_polygon
                inter = sutherland_hodgman(
                    bbox_to_polygon(det.bbox), slot.polygon
                )
                if len(inter) >= 3:
                    from geometry import shoelace_area as _sa
                    occlusion_area += _sa(inter)

    occ_ratio = min(occlusion_area / slot.area, 1.0) if slot.area > 0 else 0.0
    return MatchResult(best_det, best_iou, occ_ratio)


# ─────────────────────────── raw-state rule ─────────────────────────

def decide_raw_state(iou: float, conf: float, occ: float) -> RawState:
    """
    3-class threshold classifier.
    Order of checks is deliberate (CÓ XE first, TRỐNG last).
    """
    if iou >= IOU_HI and conf >= CONF_HI:
        return RawState.OCCUPIED

    if (occ > OCC_THR
            or (CONF_LO <= conf < CONF_HI)
            or (IOU_LO <= iou < IOU_HI)):
        return RawState.UNKNOWN

    if iou < IOU_LO or conf < CONF_LO:
        return RawState.EMPTY

    return RawState.UNKNOWN   # catch-all safety net


# ─────────────────────────── per-slot FSM ───────────────────────────

@dataclass
class SlotMemory:
    state: SlotState = SlotState.EMPTY
    hold: int = 0          # frames raw has been different from state


def _threshold(current: SlotState, raw: RawState,
               n_park: int, n_leave: int, n_unk: int) -> int:
    if raw == RawState.UNKNOWN:
        return n_unk
    if raw == RawState.OCCUPIED:
        # Coming from UNKNOWN is treated like EMPTY for safety
        return n_park
    # raw == EMPTY
    return n_leave


def update_state_machine(mem: SlotMemory, raw: RawState,
                         n_park: int, n_leave: int, n_unk: int) -> SlotMemory:
    """
    Hysteresis FSM.  Returns a new SlotMemory.
    - raw matches current state → reset hold counter.
    - raw differs → increment hold; flip state once hold ≥ threshold.
    """
    raw_as_slot = {
        RawState.EMPTY:    SlotState.EMPTY,
        RawState.OCCUPIED: SlotState.OCCUPIED,
        RawState.UNKNOWN:  SlotState.UNKNOWN,
    }[raw]

    if raw_as_slot == mem.state:
        return SlotMemory(state=mem.state, hold=0)

    new_hold = mem.hold + 1
    threshold = _threshold(mem.state, raw, n_park, n_leave, n_unk)

    if new_hold >= threshold:
        return SlotMemory(state=raw_as_slot, hold=0)

    return SlotMemory(state=mem.state, hold=new_hold)


# ─────────────────────────── violation check ────────────────────────

@dataclass
class ViolationTimer:
    zone_id: str
    count: int = 0


def check_violation(zone: ZoneDef,
                    detections: List[Detection],
                    vtimer: ViolationTimer,
                    n_viol: int,
                    frame: Optional[np.ndarray] = None
                    ) -> Optional[ViolationAlert]:
    """
    Returns an alert exactly once when the vehicle first reaches n_viol frames.
    """
    occupied = False
    trigger_det: Optional[Detection] = None

    for det in detections:
        if det.confidence < CONF_HI:
            continue
        anchor = anchor_point(det.bbox)
        inside = point_in_polygon(anchor[0], anchor[1], zone.polygon)
        if not inside:
            iou = polygon_iou(det.bbox, zone.polygon, zone.area)
            inside = iou >= IOU_HI
        if inside:
            occupied = True
            trigger_det = det
            break

    if occupied:
        vtimer.count += 1
    else:
        vtimer.count = 0

    # Alert fires exactly when counter == n_viol (not every frame after)
    if vtimer.count == n_viol and trigger_det is not None:
        snapshot = None
        if frame is not None:
            x1, y1, x2, y2 = [int(v) for v in trigger_det.bbox]
            snapshot = frame[max(0, y1):y2, max(0, x1):x2].copy()
        return ViolationAlert(
            zone_id=zone.id,
            snapshot=snapshot,
            confidence=trigger_det.confidence,
            bbox=trigger_det.bbox,
        )
    return None


# ─────────────────────────── main M4 class ──────────────────────────

class ParkingLogic:
    """
    Stateful M4 processor.  Call process_frame() once per sampled frame.
    """

    def __init__(self, layout, sample_fps: float = 2.0) -> None:
        from layout import Layout
        self.layout: Layout = layout
        self.sample_fps = sample_fps

        # Pre-compute frame thresholds from time constants
        self.n_park  = max(1, math.ceil(TAU_PARK  * sample_fps))
        self.n_leave = max(1, math.ceil(TAU_LEAVE * sample_fps))
        self.n_unk   = max(1, math.ceil(TAU_UNK   * sample_fps))
        self.n_viol  = max(1, math.ceil(TAU_VIOL  * sample_fps))

        # Per-slot state memory
        self._slot_mem: List[SlotMemory] = [
            SlotMemory() for _ in layout.slots
        ]

        # Per-zone violation timers
        self._vtimers: List[ViolationTimer] = [
            ViolationTimer(z.id) for z in layout.zones
        ]

        # Spatial grid (built from detection AABBs each frame — lightweight)
        self._grid = SpatialGrid(img_w=4000, img_h=4000)

    def process_frame(self, detections: List[Detection],
                      frame: Optional[np.ndarray] = None
                      ) -> Tuple[List[SlotState], List[ViolationAlert]]:
        """
        Parameters
        ----------
        detections : output of M2 (in original pixel coordinates)
        frame      : current raw frame (for snapshot cropping in alerts)

        Returns
        -------
        states  : list of SlotState for each slot in layout.slots
        alerts  : list of ViolationAlert (may be empty)
        """
        det_aabbs = [
            (d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3])
            for d in detections
        ]
        self._grid.build(det_aabbs)

        states: List[SlotState] = []
        for i, slot in enumerate(self.layout.slots):
            cand_idx = self._grid.query(slot.aabb)
            # Filter further by actual AABB overlap
            cand_idx = [
                j for j in cand_idx
                if aabbs_overlap(slot.aabb, det_aabbs[j])
            ]

            match = match_slot(slot, detections, cand_idx)

            conf = match.best_det.confidence if match.best_det else 0.0
            raw = decide_raw_state(match.best_iou, conf, match.occlusion)

            self._slot_mem[i] = update_state_machine(
                self._slot_mem[i], raw,
                self.n_park, self.n_leave, self.n_unk
            )
            states.append(self._slot_mem[i].state)

        alerts: List[ViolationAlert] = []
        for k, zone in enumerate(self.layout.zones):
            alert = check_violation(zone, detections,
                                    self._vtimers[k], self.n_viol, frame)
            if alert:
                alerts.append(alert)

        return states, alerts

    @property
    def slot_states(self) -> List[SlotState]:
        return [m.state for m in self._slot_mem]
