"""
Detection service — runs the full M1→M2→M4→M5 pipeline in a background thread.
Pushes state snapshots to a shared queue consumed by the WebSocket broadcaster.
"""
from __future__ import annotations
import asyncio
import datetime
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from config import cfg
from database.session import get_db
from database import crud
from layout import Layout, load_or_create_layout
from preprocessor import Preprocessor, FrameSampler
from detector import Detector
from logic import ParkingLogic, SlotState
from visualizer import Visualizer

log = logging.getLogger("parking.detection")


# ─────────────────────── shared state snapshot ──────────────────

@dataclass
class FrameSnapshot:
    camera_id: int
    timestamp: datetime.datetime
    states: List[str]          # SlotState.value per slot
    free: int
    occupied: int
    unknown: int
    total: int
    fps: float
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    jpeg_bytes: Optional[bytes] = None   # annotated frame as JPEG


# ─────────────────────── worker ─────────────────────────────────

class DetectionWorker:
    """
    Runs in a daemon thread.  Writes FrameSnapshot objects to *out_queue*
    and persists events / violations to the database.
    """

    def __init__(self, camera_id: int,
                 out_queue: "queue.Queue[FrameSnapshot]",
                 jpeg_quality: int = 60) -> None:
        self.camera_id    = camera_id
        self.out_queue    = out_queue
        self.jpeg_quality = jpeg_quality

        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Will be initialised inside the thread
        self._layout:     Optional[Layout]        = None
        self._detector:   Optional[Detector]      = None
        self._logic:      Optional[ParkingLogic]  = None
        self._viz:        Optional[Visualizer]    = None
        self._preprocessor = Preprocessor()

        # Track previous states to detect transitions
        self._prev_states: List[Optional[SlotState]] = []

        # Hourly stats accumulator
        self._stat_accumulator: List[FrameSnapshot] = []
        self._last_stat_flush: datetime.datetime = datetime.datetime.utcnow()

    # ── public ────────────────────────────────────────────────── #

    def start(self) -> None:
        with get_db() as db:
            cam = crud.get_camera(db, self.camera_id)
            if cam is None:
                raise ValueError(f"Camera {self.camera_id} not found in DB")
            source      = cam.source
            layout_path = cam.layout_path

        # Try converting to int (camera index)
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass

        self._thread = threading.Thread(
            target=self._run,
            args=(source, layout_path),
            daemon=True,
            name=f"detection-cam{self.camera_id}",
        )
        self._thread.start()
        log.info("Detection worker started for camera %d", self.camera_id)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        log.info("Detection worker stopped for camera %d", self.camera_id)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── private ───────────────────────────────────────────────── #

    def _run(self, source, layout_path: str) -> None:
        try:
            self._setup(source, layout_path)
            self._loop(source)
        except Exception as exc:
            log.exception("Detection worker error (cam %d): %s",
                          self.camera_id, exc)

    def _setup(self, source, layout_path: str) -> None:
        # Load layout
        cap = cv2.VideoCapture(source)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read first frame from {source}")

        self._layout = load_or_create_layout(frame, layout_path)

        # Sync slots to DB
        with get_db() as db:
            slot_defs = [
                {"id": s.id, "polygon": s.polygon,
                 "area": s.area, "zone": "default"}
                for s in self._layout.slots
            ]
            crud.upsert_slots(db, self.camera_id, slot_defs)

        self._prev_states = [None] * len(self._layout.slots)

        # Models
        self._detector = Detector(
            model_path=cfg.model.path,
            device=cfg.model.device,
            conf_threshold=cfg.model.confidence_threshold,
            iou_nms=cfg.model.iou_nms,
        )
        self._logic = ParkingLogic(self._layout,
                                   sample_fps=cfg.camera.sample_fps)
        self._viz   = Visualizer(self._layout)

    def _loop(self, source) -> None:
        sampler = FrameSampler(source, target_fps=cfg.camera.sample_fps)
        fps_times: List[float] = []

        try:
            for frame in sampler:
                if self._stop_event.is_set():
                    break

                t0 = time.perf_counter()

                tensor, scale, pad = self._preprocessor.preprocess(frame)
                detections = self._detector.detect(tensor, scale, pad)
                states, alerts = self._logic.process_frame(detections, frame)

                annotated = self._viz.render(frame, states, detections, alerts)
                _, jpeg_buf = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )

                fps_times.append(time.perf_counter())
                if len(fps_times) > 30:
                    fps_times.pop(0)
                fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) \
                    if len(fps_times) > 1 else 0.0

                free     = sum(1 for s in states if s == SlotState.EMPTY)
                occupied = sum(1 for s in states if s == SlotState.OCCUPIED)
                unknown  = sum(1 for s in states if s == SlotState.UNKNOWN)

                snap = FrameSnapshot(
                    camera_id  = self.camera_id,
                    timestamp  = datetime.datetime.utcnow(),
                    states     = [s.value for s in states],
                    free       = free,
                    occupied   = occupied,
                    unknown    = unknown,
                    total      = len(states),
                    fps        = fps,
                    alerts     = [
                        {"zone_id": a.zone_id, "confidence": a.confidence,
                         "bbox": list(a.bbox)}
                        for a in alerts
                    ],
                    jpeg_bytes = jpeg_buf.tobytes(),
                )

                # Non-blocking push; drop frame if consumer is slow
                try:
                    self.out_queue.put_nowait(snap)
                except queue.Full:
                    pass

                self._persist(states, alerts, snap)

        finally:
            sampler.release()

    def _persist(self, states: List[SlotState],
                 alerts, snap: FrameSnapshot) -> None:
        with get_db() as db:
            db_slots = crud.get_slots(db, self.camera_id)
            id_map   = {s.slot_id: s.id for s in db_slots}

            # Record state-change events
            for i, (new_state, slot_def) in enumerate(
                    zip(states, self._layout.slots)):
                prev = self._prev_states[i]
                if prev != new_state:
                    pk = id_map.get(slot_def.id)
                    if pk:
                        crud.record_event(db, pk, new_state.value)
                    self._prev_states[i] = new_state

            # Persist violations
            for alert in alerts:
                snap_path = None
                if alert.snapshot is not None:
                    import os
                    from config import cfg as _cfg
                    os.makedirs(_cfg.export.violation_snapshots_dir, exist_ok=True)
                    ts = snap.timestamp.strftime("%Y%m%d_%H%M%S")
                    snap_path = os.path.join(
                        _cfg.export.violation_snapshots_dir,
                        f"viol_cam{self.camera_id}_{alert.zone_id}_{ts}.jpg"
                    )
                    cv2.imwrite(snap_path, alert.snapshot)
                crud.create_violation(
                    db, self.camera_id, alert.zone_id,
                    alert.confidence, snap_path, list(alert.bbox)
                )

        # Accumulate hourly stats (flush once per minute for efficiency)
        self._stat_accumulator.append(snap)
        now = datetime.datetime.utcnow()
        if (now - self._last_stat_flush).total_seconds() >= 60:
            self._flush_stats()
            self._last_stat_flush = now

    def _flush_stats(self) -> None:
        if not self._stat_accumulator:
            return
        snaps = self._stat_accumulator
        self._stat_accumulator = []

        bucket = snaps[-1].timestamp.replace(minute=0, second=0, microsecond=0)
        n = len(snaps)
        avg_occ = sum(s.occupied / max(s.total, 1) for s in snaps) / n
        avg_emp = sum(s.free     / max(s.total, 1) for s in snaps) / n
        avg_unk = sum(s.unknown  / max(s.total, 1) for s in snaps) / n
        total   = snaps[-1].total

        with get_db() as db:
            crud.upsert_hourly_stat(
                db, self.camera_id, bucket, total,
                avg_occ, avg_emp, avg_unk
            )


# ─────────────────────── manager ────────────────────────────────

class DetectionManager:
    """Singleton — manages all DetectionWorkers."""

    def __init__(self) -> None:
        self._workers: Dict[int, DetectionWorker] = {}
        self._queues:  Dict[int, "queue.Queue[FrameSnapshot]"] = {}

    def start_camera(self, camera_id: int,
                     queue_size: int = 10) -> "queue.Queue[FrameSnapshot]":
        if camera_id in self._workers and self._workers[camera_id].running:
            return self._queues[camera_id]

        q: queue.Queue[FrameSnapshot] = queue.Queue(maxsize=queue_size)
        worker = DetectionWorker(camera_id, q)
        worker.start()
        self._workers[camera_id] = worker
        self._queues[camera_id]  = q
        return q

    def stop_camera(self, camera_id: int) -> None:
        worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()
        self._queues.pop(camera_id, None)

    def stop_all(self) -> None:
        for cid in list(self._workers):
            self.stop_camera(cid)

    def get_queue(self, camera_id: int) -> Optional["queue.Queue[FrameSnapshot]"]:
        return self._queues.get(camera_id)

    def status(self) -> Dict[int, bool]:
        return {cid: w.running for cid, w in self._workers.items()}


# Global singleton
detection_manager = DetectionManager()
