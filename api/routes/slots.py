"""Slot state endpoints."""
from __future__ import annotations
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.session import get_db_fastapi
from database import crud
from services.detection_service import detection_manager

router = APIRouter(prefix="/cameras/{camera_id}/slots", tags=["slots"])


class SlotStateOut(BaseModel):
    slot_id: str
    state: str
    zone: str
    polygon: list = []


class SlotHistoryItem(BaseModel):
    state: str
    recorded_at: str


@router.get("/", response_model=List[SlotStateOut])
def list_slot_states(camera_id: int, db: Session = Depends(get_db_fastapi)):
    """Current state of every slot for a camera."""
    slots = crud.get_slots(db, camera_id)
    if not slots:
        raise HTTPException(404, "Camera or slots not found")

    # Try to get live state from detection manager
    q = detection_manager.get_queue(camera_id)
    live_states: dict = {}
    if q is not None:
        import queue as _q
        try:
            snap = q.queue[-1]  # peek last item without consuming
            live_states = dict(zip(
                [s.slot_id for s in slots],
                snap.states
            ))
        except (IndexError, AttributeError):
            pass

    result = []
    for slot in slots:
        result.append(SlotStateOut(
            slot_id=slot.slot_id,
            state=live_states.get(slot.slot_id, "KHÔNG XÁC ĐỊNH"),
            zone=slot.zone,
            polygon=slot.polygon or [],
        ))
    return result


@router.get("/{slot_id}/history", response_model=List[SlotHistoryItem])
def slot_history(camera_id: int, slot_id: str, limit: int = 50,
                 db: Session = Depends(get_db_fastapi)):
    slots = crud.get_slots(db, camera_id)
    match = next((s for s in slots if s.slot_id == slot_id), None)
    if match is None:
        raise HTTPException(404, f"Slot {slot_id} not found")

    events = crud.get_slot_history(db, match.id, limit=limit)
    return [
        SlotHistoryItem(state=e.state, recorded_at=e.recorded_at.isoformat())
        for e in events
    ]
