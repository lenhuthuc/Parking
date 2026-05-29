"""
SQLAlchemy ORM models.
Tables:
  - parking_slots     : static slot definitions (mirrors layout.json)
  - slot_events       : every state-change event per slot
  - occupancy_stats   : hourly aggregated occupancy percentages
  - violations        : no-parking zone violation records
  - cameras           : registered camera sources
"""
from __future__ import annotations
import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Camera(Base):
    __tablename__ = "cameras"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(80), nullable=False)
    source     = Column(String(256), nullable=False)   # path / RTSP URL / index
    layout_path = Column(String(256), default="layout.json")
    active     = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    slots      = relationship("ParkingSlot", back_populates="camera",
                              cascade="all, delete-orphan")
    violations = relationship("Violation", back_populates="camera",
                              cascade="all, delete-orphan")


class ParkingSlot(Base):
    __tablename__ = "parking_slots"

    id         = Column(Integer, primary_key=True, index=True)
    slot_id    = Column(String(20), nullable=False, index=True)
    camera_id  = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    polygon    = Column(JSON, nullable=False)   # list of [x, y] pairs
    area       = Column(Float, default=0.0)
    zone       = Column(String(40), default="default")   # e.g. "A", "B", "floor1"

    camera     = relationship("Camera", back_populates="slots")
    events     = relationship("SlotEvent", back_populates="slot",
                              cascade="all, delete-orphan")


class SlotEvent(Base):
    """Recorded every time a slot changes state."""
    __tablename__ = "slot_events"

    id         = Column(Integer, primary_key=True, index=True)
    slot_id    = Column(Integer, ForeignKey("parking_slots.id"), nullable=False)
    state      = Column(String(20), nullable=False)   # EMPTY / OCCUPIED / UNKNOWN
    recorded_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    slot       = relationship("ParkingSlot", back_populates="events")


class OccupancyStat(Base):
    """
    Hourly aggregated stats — pre-computed by the detection service
    to keep dashboard queries fast.
    """
    __tablename__ = "occupancy_stats"

    id           = Column(Integer, primary_key=True, index=True)
    camera_id    = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    hour_bucket  = Column(DateTime, nullable=False, index=True)  # truncated to hour
    total_slots  = Column(Integer, default=0)
    avg_occupied = Column(Float, default=0.0)   # 0.0 – 1.0
    avg_empty    = Column(Float, default=0.0)
    avg_unknown  = Column(Float, default=0.0)
    zone         = Column(String(40), default="all")


class Violation(Base):
    __tablename__ = "violations"

    id           = Column(Integer, primary_key=True, index=True)
    camera_id    = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    zone_id      = Column(String(20), nullable=False)
    confidence   = Column(Float, default=0.0)
    snapshot_path = Column(String(256), nullable=True)
    bbox         = Column(JSON, nullable=True)    # [x1, y1, x2, y2]
    recorded_at  = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    acknowledged = Column(Boolean, default=False)
    notes        = Column(Text, nullable=True)

    camera       = relationship("Camera", back_populates="violations")
