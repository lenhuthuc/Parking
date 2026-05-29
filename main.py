"""
Parking Detection System — standalone CV entry point (no web server).
Runs the full pipeline in a single process with an OpenCV display window.

For the full web dashboard, use:  python run_server.py

Usage:
  python main.py                          # webcam + layout editor
  python main.py video.mp4               # video file
  python main.py --setup-only            # draw layout only, then exit
  python main.py --layout custom.json    # use a specific layout file
"""
from __future__ import annotations
import argparse
import os
import time
from typing import Optional

import cv2
import numpy as np

from config import cfg
from database.session import init_db
from database import crud
from database.session import get_db
from preprocessor import Preprocessor, FrameSampler
from detector import Detector
from layout import Layout, load_or_create_layout
from logic import ParkingLogic, SlotState
from visualizer import Visualizer


def _first_frame(source) -> np.ndarray:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read first frame.")
    return frame


class _FPS:
    def __init__(self, n: int = 30) -> None:
        self._t: list = []
        self._n = n

    def tick(self) -> float:
        self._t.append(time.perf_counter())
        if len(self._t) > self._n:
            self._t.pop(0)
        return (len(self._t) - 1) / (self._t[-1] - self._t[0]) \
            if len(self._t) > 1 else 0.0


class ParkingDetectionSystem:
    """Standalone pipeline (no web/DB overhead)."""

    def __init__(self, source=0, layout_path: str = "layout.json",
                 model_path: str = "yolov8n.pt",
                 sample_fps: float = 2.0,
                 device: str = "cpu",
                 display_scale: float = 1.0,
                 camera_name: str = "Camera 1") -> None:
        self.source        = source
        self.layout_path   = layout_path
        self.model_path    = model_path
        self.sample_fps    = sample_fps
        self.device        = device
        self.display_scale = display_scale
        self.camera_name   = camera_name

        self._prep   = Preprocessor()
        self._layout: Optional[Layout]   = None
        self._det:    Optional[Detector]  = None
        self._logic:  Optional[ParkingLogic] = None
        self._viz:    Optional[Visualizer]   = None
        self._camera_db_id: Optional[int]    = None

    # ── Setup ────────────────────────────────────────────────── #

    def setup(self) -> None:
        init_db()

        print("[SETUP] Reading sample frame …")
        sample = _first_frame(self.source)

        print("[SETUP] Loading / creating layout …")
        self._layout = load_or_create_layout(sample, self.layout_path)
        print(f"[SETUP] {len(self._layout.slots)} slots, "
              f"{len(self._layout.zones)} no-park zones.")

        # Register camera in DB
        with get_db() as db:
            cams = crud.list_cameras(db)
            match = next((c for c in cams if c.source == str(self.source)), None)
            if match is None:
                cam = crud.create_camera(
                    db, self.camera_name, str(self.source), self.layout_path
                )
                self._camera_db_id = cam.id
            else:
                self._camera_db_id = match.id

            slot_defs = [
                {"id": s.id, "polygon": s.polygon,
                 "area": s.area, "zone": "default"}
                for s in self._layout.slots
            ]
            crud.upsert_slots(db, self._camera_db_id, slot_defs)

        print("[SETUP] Loading YOLO model …")
        self._det   = Detector(self.model_path, device=self.device)
        self._logic = ParkingLogic(self._layout, sample_fps=self.sample_fps)
        self._viz   = Visualizer(self._layout)
        print("[SETUP] Done.")

    # ── Online loop ──────────────────────────────────────────── #

    def run(self) -> None:
        if self._layout is None:
            self.setup()

        sampler = FrameSampler(self.source, target_fps=self.sample_fps)
        fps     = _FPS()
        win     = "Parking Detection  [Q] quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        prev_states = [None] * len(self._layout.slots)

        try:
            for frame in sampler:
                # M1
                tensor, scale, pad = self._prep.preprocess(frame)
                # M2
                dets = self._det.detect(tensor, scale, pad)
                # M4
                states, alerts = self._logic.process_frame(dets, frame)
                # M5
                annotated = self._viz.render(frame, states, dets, alerts)

                f = fps.tick()
                cv2.putText(annotated, f"FPS:{f:.1f}",
                            (annotated.shape[1] - 90, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

                if self.display_scale != 1.0:
                    h, w = annotated.shape[:2]
                    annotated = cv2.resize(annotated, (
                        int(w * self.display_scale),
                        int(h * self.display_scale)
                    ))
                cv2.imshow(win, annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

                # Persist to DB
                self._persist(states, alerts, frame, prev_states)
                prev_states = list(states)

                if alerts:
                    for a in alerts:
                        print(f"[ALERT] Zone {a.zone_id}  "
                              f"conf={a.confidence:.0%}")

        except StopIteration:
            print("[INFO] End of stream.")
        finally:
            sampler.release()
            cv2.destroyAllWindows()

    def _persist(self, states, alerts, frame, prev_states) -> None:
        if self._camera_db_id is None:
            return
        with get_db() as db:
            db_slots = crud.get_slots(db, self._camera_db_id)
            id_map   = {s.slot_id: s.id for s in db_slots}

            for i, (state, slot_def) in enumerate(
                    zip(states, self._layout.slots)):
                if prev_states[i] != state:
                    pk = id_map.get(slot_def.id)
                    if pk:
                        crud.record_event(db, pk, state.value)

            for alert in alerts:
                snap_path = None
                if alert.snapshot is not None:
                    import datetime
                    os.makedirs(cfg.export.violation_snapshots_dir,
                                exist_ok=True)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = os.path.join(
                        cfg.export.violation_snapshots_dir,
                        f"viol_{alert.zone_id}_{ts}.jpg"
                    )
                    cv2.imwrite(snap_path, alert.snapshot)
                crud.create_violation(
                    db, self._camera_db_id, alert.zone_id,
                    alert.confidence, snap_path, list(alert.bbox)
                )


# ── CLI ──────────────────────────────────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser(
        description="Parking Detection — standalone mode"
    )
    p.add_argument("source", nargs="?", default=cfg.camera.default_source,
                   help="Video file / RTSP URL / camera index")
    p.add_argument("--layout",     default=cfg.layout.path)
    p.add_argument("--model",      default=cfg.model.path)
    p.add_argument("--fps",        type=float, default=cfg.camera.sample_fps)
    p.add_argument("--device",     default=cfg.model.device)
    p.add_argument("--scale",      type=float, default=cfg.camera.display_scale)
    p.add_argument("--name",       default="Camera 1")
    p.add_argument("--setup-only", action="store_true")
    args = p.parse_args()

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    sys = ParkingDetectionSystem(
        source=source,
        layout_path=args.layout,
        model_path=args.model,
        sample_fps=args.fps,
        device=args.device,
        display_scale=args.scale,
        camera_name=args.name,
    )

    sys.setup()
    if not args.setup_only:
        sys.run()
    else:
        print("[SETUP-ONLY] Layout saved. Exiting.")


if __name__ == "__main__":
    main()
