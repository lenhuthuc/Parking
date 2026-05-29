"""Statistics and export endpoints."""
from __future__ import annotations
import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.session import get_db_fastapi
from database import crud
from services.export_service import (
    export_violations_csv, export_occupancy_csv, export_occupancy_excel
)

router = APIRouter(prefix="/stats", tags=["stats"])


class HourlyStatOut(BaseModel):
    hour: str
    total_slots: int
    occupied_pct: float
    empty_pct: float
    unknown_pct: float


class SummaryOut(BaseModel):
    camera_id: int
    total_slots: int
    free: int
    occupied: int
    unknown: int
    violations_today: int


@router.get("/cameras/{camera_id}/summary", response_model=SummaryOut)
def summary(camera_id: int, db: Session = Depends(get_db_fastapi)):
    from services.detection_service import detection_manager

    total = len(crud.get_slots(db, camera_id))
    viol_today = crud.violation_count_today(db, camera_id)

    free = occupied = unknown = 0
    q = detection_manager.get_queue(camera_id)
    if q is not None:
        try:
            snap = list(q.queue)[-1]
            free     = snap.free
            occupied = snap.occupied
            unknown  = snap.unknown
        except (IndexError, AttributeError):
            pass

    return SummaryOut(
        camera_id=camera_id,
        total_slots=total,
        free=free, occupied=occupied, unknown=unknown,
        violations_today=viol_today,
    )


@router.get("/cameras/{camera_id}/hourly",
            response_model=List[HourlyStatOut])
def hourly_stats(
    camera_id: int,
    since_hours: int = Query(24, ge=1, le=720),
    zone: str = "all",
    db: Session = Depends(get_db_fastapi),
):
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=since_hours)
    rows = crud.get_hourly_stats(db, camera_id, since, zone=zone)
    return [
        HourlyStatOut(
            hour=r.hour_bucket.isoformat(),
            total_slots=r.total_slots,
            occupied_pct=round(r.avg_occupied * 100, 1),
            empty_pct=round(r.avg_empty * 100, 1),
            unknown_pct=round(r.avg_unknown * 100, 1),
        )
        for r in rows
    ]


@router.get("/cameras/{camera_id}/export/violations.csv")
def export_violations(
    camera_id: int,
    since_hours: int = Query(24, ge=1, le=8760),
):
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=since_hours)
    data = export_violations_csv(camera_id=camera_id, since=since)
    return Response(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=violations.csv"},
    )


@router.get("/cameras/{camera_id}/export/occupancy.csv")
def export_occupancy_csv_endpoint(
    camera_id: int,
    since_hours: int = Query(24, ge=1, le=8760),
):
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=since_hours)
    data = export_occupancy_csv(camera_id=camera_id, since=since)
    return Response(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=occupancy.csv"},
    )


@router.get("/cameras/{camera_id}/export/report.xlsx")
def export_excel(
    camera_id: int,
    since_hours: int = Query(24, ge=1, le=8760),
):
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=since_hours)
    data = export_occupancy_excel(camera_id=camera_id, since=since)
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=report.xlsx"},
    )
