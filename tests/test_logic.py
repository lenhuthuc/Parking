"""
Unit tests for logic.py — raw-state classifier and FSM.
Run: pytest tests/test_logic.py -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from logic import (
    RawState, SlotState, SlotMemory,
    decide_raw_state, update_state_machine,
    IOU_LO, IOU_HI, CONF_LO, CONF_HI, OCC_THR
)


# ─── decide_raw_state ───────────────────────────────────────────

class TestDecideRawState:
    def test_occupied_high_confidence(self):
        assert decide_raw_state(IOU_HI, CONF_HI, 0.0) == RawState.OCCUPIED

    def test_occupied_exact_boundary(self):
        assert decide_raw_state(IOU_HI, CONF_HI, 0.0) == RawState.OCCUPIED

    def test_empty_no_detection(self):
        assert decide_raw_state(0.0, 0.0, 0.0) == RawState.EMPTY

    def test_empty_low_conf(self):
        assert decide_raw_state(0.0, CONF_LO - 0.01, 0.0) == RawState.EMPTY

    def test_unknown_mid_iou(self):
        result = decide_raw_state((IOU_LO + IOU_HI) / 2, CONF_HI, 0.0)
        assert result == RawState.UNKNOWN

    def test_unknown_mid_conf(self):
        result = decide_raw_state(IOU_HI, (CONF_LO + CONF_HI) / 2, 0.0)
        assert result == RawState.UNKNOWN

    def test_unknown_high_occlusion(self):
        result = decide_raw_state(0.0, 0.0, OCC_THR + 0.01)
        assert result == RawState.UNKNOWN

    def test_catchall_is_unknown(self):
        # edge case: iou between LO and HI, conf above HI → UNKNOWN (not OCCUPIED)
        result = decide_raw_state((IOU_LO + IOU_HI) / 2, CONF_HI + 0.1, 0.0)
        assert result == RawState.UNKNOWN


# ─── update_state_machine ────────────────────────────────────────

N_PARK  = 5
N_LEAVE = 3
N_UNK   = 2


class TestFSM:
    def _run(self, initial: SlotState, raws: list) -> list:
        mem = SlotMemory(state=initial, hold=0)
        history = []
        for raw in raws:
            mem = update_state_machine(mem, raw, N_PARK, N_LEAVE, N_UNK)
            history.append(mem.state)
        return history

    def test_stays_empty_below_threshold(self):
        raws    = [RawState.OCCUPIED] * (N_PARK - 1)
        history = self._run(SlotState.EMPTY, raws)
        assert all(s == SlotState.EMPTY for s in history)

    def test_transitions_to_occupied_at_threshold(self):
        raws    = [RawState.OCCUPIED] * N_PARK
        history = self._run(SlotState.EMPTY, raws)
        assert history[-1] == SlotState.OCCUPIED

    def test_transitions_back_to_empty(self):
        raws    = [RawState.OCCUPIED] * N_PARK + [RawState.EMPTY] * N_LEAVE
        history = self._run(SlotState.EMPTY, raws)
        assert history[-1] == SlotState.EMPTY

    def test_hold_reset_on_same_raw(self):
        # 3 occupied, then 1 empty (resets hold), then 3 more occupied → no transition
        raws = ([RawState.OCCUPIED] * 3
                + [RawState.EMPTY]
                + [RawState.OCCUPIED] * (N_PARK - 1))
        history = self._run(SlotState.EMPTY, raws)
        assert history[-1] == SlotState.EMPTY

    def test_unknown_transition(self):
        raws    = [RawState.UNKNOWN] * N_UNK
        history = self._run(SlotState.EMPTY, raws)
        assert history[-1] == SlotState.UNKNOWN

    def test_exit_unknown_to_occupied(self):
        raws = ([RawState.UNKNOWN] * N_UNK
                + [RawState.OCCUPIED] * N_PARK)
        history = self._run(SlotState.EMPTY, raws)
        assert history[-1] == SlotState.OCCUPIED

    def test_single_noise_frame_absorbed(self):
        """A single empty frame should not flip OCCUPIED → EMPTY."""
        raws = [RawState.EMPTY]
        mem  = SlotMemory(state=SlotState.OCCUPIED, hold=0)
        mem  = update_state_machine(mem, RawState.EMPTY, N_PARK, N_LEAVE, N_UNK)
        assert mem.state == SlotState.OCCUPIED
        assert mem.hold  == 1
