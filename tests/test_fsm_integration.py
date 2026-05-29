"""
Integration test: full M4 pipeline with stub detections.
Run: pytest tests/test_fsm_integration.py -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detector import Detection, StubDetector
from layout import Layout, SlotDef, ZoneDef
from logic import ParkingLogic, SlotState


def _make_layout():
    slot = SlotDef("A01", [(100, 100), (200, 100), (200, 200), (100, 200)])
    zone = ZoneDef("Z01", [(300, 300), (400, 300), (400, 400), (300, 400)])
    return Layout([slot], [zone])


def _det(x1, y1, x2, y2, conf=0.85):
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf, label=3)


class TestParkingLogicIntegration:
    """Simulate frame-by-frame processing without a real model."""

    def test_slot_becomes_occupied_after_dwell(self):
        layout = _make_layout()
        # sample_fps=1 → n_park=5 frames
        logic  = ParkingLogic(layout, sample_fps=1.0)

        # Box fully overlaps slot A01
        det = _det(100, 100, 200, 200, conf=0.90)

        for _ in range(5):
            states, _ = logic.process_frame([det])
        assert states[0] == SlotState.OCCUPIED

    def test_slot_stays_empty_below_dwell(self):
        layout = _make_layout()
        logic  = ParkingLogic(layout, sample_fps=1.0)

        det = _det(100, 100, 200, 200, conf=0.90)
        for _ in range(4):
            states, _ = logic.process_frame([det])
        assert states[0] == SlotState.EMPTY

    def test_slot_returns_empty_after_car_leaves(self):
        layout = _make_layout()
        logic  = ParkingLogic(layout, sample_fps=1.0)

        det = _det(100, 100, 200, 200, conf=0.90)
        for _ in range(5):
            logic.process_frame([det])

        for _ in range(3):
            states, _ = logic.process_frame([])
        assert states[0] == SlotState.EMPTY

    def test_violation_alert_fires_at_30s(self):
        layout = _make_layout()
        logic  = ParkingLogic(layout, sample_fps=1.0)

        # Det inside no-park zone Z01
        det = _det(320, 320, 380, 380, conf=0.90)
        alerts_seen = []
        for _ in range(31):
            _, alerts = logic.process_frame([det])
            alerts_seen.extend(alerts)

        assert len(alerts_seen) == 1
        assert alerts_seen[0].zone_id == "Z01"

    def test_violation_does_not_spam(self):
        layout = _make_layout()
        logic  = ParkingLogic(layout, sample_fps=1.0)

        det = _det(320, 320, 380, 380, conf=0.90)
        total_alerts = 0
        for _ in range(60):
            _, alerts = logic.process_frame([det])
            total_alerts += len(alerts)

        assert total_alerts == 1   # fires exactly once

    def test_passthrough_car_does_not_trigger_violation(self):
        layout = _make_layout()
        logic  = ParkingLogic(layout, sample_fps=1.0)

        det = _det(320, 320, 380, 380, conf=0.90)
        # Only 10 frames (< 30s threshold)
        for _ in range(10):
            _, alerts = logic.process_frame([det])
        assert alerts == []

    def test_low_confidence_detection_gives_unknown(self):
        layout = _make_layout()
        logic  = ParkingLogic(layout, sample_fps=1.0)

        # conf between CONF_LO and CONF_HI → UNKNOWN raw
        det = _det(100, 100, 200, 200, conf=0.60)
        for _ in range(2):   # n_unk=2 frames at fps=1
            states, _ = logic.process_frame([det])
        assert states[0] == SlotState.UNKNOWN
