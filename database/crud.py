"""
CRUD helpers — thin wrappers around SQLAlchemy queries.
All functions accept a Session; callers manage commit/rollback.
"""
from __future__ import annotations
import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from database.models import (
    Camera, ParkingSlot, SlotEvent, OccupancyStat, Violation
)


# ─────────────────────── Camera ──────────────────────────────────

def create_camera(db: Session, name: str, source: str,
                  layout_path: str = "layout.json") -> Camera:
    cam = Camera(name=name, source=str(source), layout_path=layout_path)
    db.add(cam)
    db.flush()
    return cam


def get_camera(db: Session, camera_id: int) -> Optional[Camera]:
    return db.query(Camera).filter(Camera.id == camera_id).first()


def list_cameras(db: Session) -> List[Camera]:
    return db.query(Camera).filter(Camera.active == True).all()


def deactivate_camera(db: Session, camera_id: int) -> None:
    db.query(Camera).filter(Camera.id == camera_id).update({"active": False})


# ─────────────────────── ParkingSlot ─────────────────────────────

def upsert_slots(db: Session, camera_id: int,
                 slot_defs: List[Dict[str, Any]]) -> List[ParkingSlot]:
    """Replace all slots for a camera (called after layout reload)."""
    db.query(ParkingSlot).filter(
        ParkingSlot.camera_id == camera_id
    ).delete(synchronize_session=False)

    slots = []
    for d in slot_defs:
        s = ParkingSlot(
            slot_id=d["id"],
            camera_id=camera_id,
            polygon=d["polygon"],
            area=d.get("area", 0.0),
            zone=d.get("zone", "default"),
        )
        db.add(s)
        slots.append(s)
    db.flush()
    return slots


def get_slots(db: Session, camera_id: int) -> List[ParkingSlot]:
    return (db.query(ParkingSlot)
              .filter(ParkingSlot.camera_id == camera_id)
              .all())


# ─────────────────────── SlotEvent ───────────────────────────────

def record_event(db: Session, slot_pk: int, state: str) -> SlotEvent:
    ev = SlotEvent(slot_id=slot_pk, state=state)
    db.add(ev)
    db.flush()
    return ev


def get_slot_history(db: Session, slot_pk: int,
                     since: Optional[datetime.datetime] = None,
                     limit: int = 200) -> List[SlotEvent]:
    q = (db.query(SlotEvent)
           .filter(SlotEvent.slot_id == slot_pk)
           .order_by(desc(SlotEvent.recorded_at)))
    if since:
        q = q.filter(SlotEvent.recorded_at >= since)
    return q.limit(limit).all()


# ─────────────────────── OccupancyStat ───────────────────────────

def upsert_hourly_stat(db: Session, camera_id: int,
                       hour_bucket: datetime.datetime,
                       total: int, occupied: float,
                       empty: float, unknown: float,
                       zone: str = "all") -> OccupancyStat:
    row = (db.query(OccupancyStat)
             .filter(OccupancyStat.camera_id == camera_id,
                     OccupancyStat.hour_bucket == hour_bucket,
                     OccupancyStat.zone == zone)
             .first())
    if row is None:
        row = OccupancyStat(camera_id=camera_id, hour_bucket=hour_bucket,
                            zone=zone)
        db.add(row)
    row.total_slots  = total
    row.avg_occupied = occupied
    row.avg_empty    = empty
    row.avg_unknown  = unknown
    db.flush()
    return row


def get_hourly_stats(db: Session, camera_id: int,
                     since: datetime.datetime,
                     until: Optional[datetime.datetime] = None,
                     zone: str = "all") -> List[OccupancyStat]:
    q = (db.query(OccupancyStat)
           .filter(OccupancyStat.camera_id == camera_id,
                   OccupancyStat.zone == zone,
                   OccupancyStat.hour_bucket >= since)
           .order_by(OccupancyStat.hour_bucket))
    if until:
        q = q.filter(OccupancyStat.hour_bucket <= until)
    return q.all()


# ─────────────────────── Violation ───────────────────────────────

def create_violation(db: Session, camera_id: int, zone_id: str,
                     confidence: float,
                     snapshot_path: Optional[str] = None,
                     bbox: Optional[List[float]] = None) -> Violation:
    v = Violation(
        camera_id=camera_id,
        zone_id=zone_id,
        confidence=confidence,
        snapshot_path=snapshot_path,
        bbox=bbox,
    )
    db.add(v)
    db.flush()
    return v


def list_violations(db: Session, camera_id: Optional[int] = None,
                    since: Optional[datetime.datetime] = None,
                    unacked_only: bool = False,
                    limit: int = 100) -> List[Violation]:
    q = db.query(Violation).order_by(desc(Violation.recorded_at))
    if camera_id is not None:
        q = q.filter(Violation.camera_id == camera_id)
    if since:
        q = q.filter(Violation.recorded_at >= since)
    if unacked_only:
        q = q.filter(Violation.acknowledged == False)
    return q.limit(limit).all()


def acknowledge_violation(db: Session, violation_id: int,
                           notes: Optional[str] = None) -> Optional[Violation]:
    v = db.query(Violation).filter(Violation.id == violation_id).first()
    if v:
        v.acknowledged = True
        if notes:
            v.notes = notes
        db.flush()
    return v


def violation_count_today(db: Session, camera_id: int) -> int:
    today = datetime.datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return (db.query(func.count(Violation.id))
              .filter(Violation.camera_id == camera_id,
                      Violation.recorded_at >= today)
              .scalar() or 0)
