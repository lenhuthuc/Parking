"""Violation endpoints."""
from __future__ import annotations
import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.session import get_db_fastapi
from database import crud

router = APIRouter(prefix="/violations", tags=["violations"])


class ViolationOut(BaseModel):
    id: int
    camera_id: int
    zone_id: str
    confidence: float
    snapshot_path: Optional[str]
    recorded_at: str
    acknowledged: bool
    notes: Optional[str]


class AckRequest(BaseModel):
    notes: Optional[str] = None


@router.get("/", response_model=List[ViolationOut])
def list_violations(
    camera_id: Optional[int] = None,
    since_hours: int = Query(24, ge=1, le=720),
    unacked_only: bool = False,
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db_fastapi),
):
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=since_hours)
    rows = crud.list_violations(
        db, camera_id=camera_id, since=since,
        unacked_only=unacked_only, limit=limit
    )
    return [
        ViolationOut(
            id=v.id, camera_id=v.camera_id, zone_id=v.zone_id,
            confidence=v.confidence, snapshot_path=v.snapshot_path,
            recorded_at=v.recorded_at.isoformat(),
            acknowledged=v.acknowledged, notes=v.notes,
        )
        for v in rows
    ]


@router.post("/{violation_id}/acknowledge")
def acknowledge(violation_id: int, body: AckRequest,
                db: Session = Depends(get_db_fastapi)):
    v = crud.acknowledge_violation(db, violation_id, body.notes)
    if v is None:
        raise HTTPException(404, "Violation not found")
    return {"status": "ok", "id": violation_id}


@router.get("/{violation_id}/snapshot")
def get_snapshot(violation_id: int, db: Session = Depends(get_db_fastapi)):
    import os
    row = (db.query(crud.Violation)
             .filter(crud.Violation.id == violation_id).first())
    if row is None:
        raise HTTPException(404, "Violation not found")
    if not row.snapshot_path or not os.path.exists(row.snapshot_path):
        raise HTTPException(404, "Snapshot file not found")
    return FileResponse(row.snapshot_path, media_type="image/jpeg")
