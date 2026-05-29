"""Camera management endpoints."""
from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.session import get_db_fastapi
from database import crud
from services.detection_service import detection_manager

router = APIRouter(prefix="/cameras", tags=["cameras"])


class CameraIn(BaseModel):
    name: str
    source: str
    layout_path: str = "layout.json"


class CameraOut(BaseModel):
    id: int
    name: str
    source: str
    layout_path: str
    active: bool
    running: bool


@router.get("/", response_model=List[CameraOut])
def list_cameras(db: Session = Depends(get_db_fastapi)):
    cameras = crud.list_cameras(db)
    status  = detection_manager.status()
    return [
        CameraOut(
            id=c.id, name=c.name, source=c.source,
            layout_path=c.layout_path, active=c.active,
            running=status.get(c.id, False),
        )
        for c in cameras
    ]


@router.post("/", response_model=CameraOut, status_code=201)
def create_camera(body: CameraIn, db: Session = Depends(get_db_fastapi)):
    cam = crud.create_camera(db, body.name, body.source, body.layout_path)
    db.commit()
    return CameraOut(
        id=cam.id, name=cam.name, source=cam.source,
        layout_path=cam.layout_path, active=cam.active, running=False,
    )


@router.post("/{camera_id}/start")
def start_camera(camera_id: int, db: Session = Depends(get_db_fastapi)):
    cam = crud.get_camera(db, camera_id)
    if cam is None:
        raise HTTPException(404, "Camera not found")
    detection_manager.start_camera(camera_id)
    return {"status": "started", "camera_id": camera_id}


@router.post("/{camera_id}/stop")
def stop_camera(camera_id: int):
    detection_manager.stop_camera(camera_id)
    return {"status": "stopped", "camera_id": camera_id}


@router.delete("/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_db_fastapi)):
    detection_manager.stop_camera(camera_id)
    crud.deactivate_camera(db, camera_id)
    return {"status": "deactivated", "camera_id": camera_id}
