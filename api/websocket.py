"""
WebSocket broadcaster.
Each connected client subscribes to a camera feed and receives
JSON snapshots as the detection worker produces them.
"""
from __future__ import annotations
import asyncio
import base64
import json
import logging
import queue
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

from services.detection_service import detection_manager, FrameSnapshot

log = logging.getLogger("parking.ws")


class ConnectionManager:
    """Keeps track of active WebSocket connections per camera."""

    def __init__(self) -> None:
        # camera_id → set of connected websockets
        self._connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, ws: WebSocket, camera_id: int) -> None:
        await ws.accept()
        self._connections.setdefault(camera_id, set()).add(ws)
        log.debug("WS connected: cam=%d total=%d",
                  camera_id, len(self._connections[camera_id]))

    def disconnect(self, ws: WebSocket, camera_id: int) -> None:
        conns = self._connections.get(camera_id, set())
        conns.discard(ws)
        log.debug("WS disconnected: cam=%d total=%d", camera_id, len(conns))

    async def broadcast(self, camera_id: int, data: str) -> None:
        conns = list(self._connections.get(camera_id, set()))
        dead = []
        for ws in conns:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, camera_id)

    def subscriber_count(self, camera_id: int) -> int:
        return len(self._connections.get(camera_id, set()))


ws_manager = ConnectionManager()


async def camera_ws_endpoint(websocket: WebSocket, camera_id: int) -> None:
    """
    FastAPI endpoint handler.
    Reads the latest snapshot (non-destructive) so it does not compete
    with the video WebSocket which consumes the queue for JPEG frames.
    """
    await ws_manager.connect(websocket, camera_id)

    # Ensure worker is running
    if detection_manager.get_queue(camera_id) is None:
        detection_manager.start_camera(camera_id)

    last_ts = None
    try:
        while True:
            snap = detection_manager.get_latest_snapshot(camera_id)
            if snap is None or snap.timestamp == last_ts:
                await asyncio.sleep(0.1)
                continue

            last_ts = snap.timestamp
            payload = {
                "camera_id": snap.camera_id,
                "timestamp": snap.timestamp.isoformat(),
                "states":    snap.states,
                "free":      snap.free,
                "occupied":  snap.occupied,
                "unknown":   snap.unknown,
                "total":     snap.total,
                "fps":       round(snap.fps, 1),
                "alerts":    snap.alerts,
            }
            await ws_manager.broadcast(camera_id, json.dumps(payload))

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, camera_id)


async def mjpeg_stream_endpoint(websocket: WebSocket, camera_id: int) -> None:
    """
    Streams full-FPS annotated JPEG frames as base64 over WebSocket.
    Video display is decoupled from inference rate: frames are rendered
    at display_fps (settings.yaml) while YOLO runs at sample_fps.
    """
    await ws_manager.connect(websocket, camera_id)

    if detection_manager.get_queue(camera_id) is None:
        detection_manager.start_camera(camera_id)

    vq = detection_manager.get_video_queue(camera_id)

    try:
        while True:
            if vq is None:
                vq = detection_manager.get_video_queue(camera_id)
                await asyncio.sleep(0.05)
                continue

            try:
                jpeg_bytes: bytes = vq.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue

            snap = detection_manager.get_latest_snapshot(camera_id)
            fps = round(snap.fps, 1) if snap else 0.0

            await websocket.send_text(json.dumps({
                "type":      "frame",
                "camera_id": camera_id,
                "fps":       fps,
                "data":      base64.b64encode(jpeg_bytes).decode(),
            }))

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, camera_id)
