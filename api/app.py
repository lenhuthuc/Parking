"""
FastAPI application factory.
Mounts:
  - REST API routes under /api/v1/
  - WebSocket endpoints under /ws/
  - Static files and Jinja2 templates for the web dashboard
"""
from __future__ import annotations
import os
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database.session import init_db
from api.routes.cameras    import router as cameras_router
from api.routes.slots      import router as slots_router
from api.routes.violations import router as violations_router
from api.routes.stats      import router as stats_router
from api.websocket import camera_ws_endpoint, mjpeg_stream_endpoint

_ROOT = Path(__file__).parent.parent
_TEMPLATES = Jinja2Templates(directory=str(_ROOT / "web" / "templates"))


def create_app() -> FastAPI:
    app = FastAPI(
        title="Parking Detection System",
        description="Hệ thống giám sát bãi đỗ xe máy bằng camera",
        version="1.0.0",
    )

    # ── Startup / shutdown ─────────────────────────────────────
    @app.on_event("startup")
    async def startup():
        init_db()

    @app.on_event("shutdown")
    async def shutdown():
        from services.detection_service import detection_manager
        detection_manager.stop_all()

    # ── Static files ───────────────────────────────────────────
    static_path = _ROOT / "web" / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)),
                  name="static")

    # ── REST API ───────────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(cameras_router,    prefix=prefix)
    app.include_router(slots_router,      prefix=prefix)
    app.include_router(violations_router, prefix=prefix)
    app.include_router(stats_router,      prefix=prefix)

    # ── WebSocket ──────────────────────────────────────────────
    @app.websocket("/ws/cameras/{camera_id}/state")
    async def ws_state(websocket: WebSocket, camera_id: int):
        await camera_ws_endpoint(websocket, camera_id)

    @app.websocket("/ws/cameras/{camera_id}/video")
    async def ws_video(websocket: WebSocket, camera_id: int):
        await mjpeg_stream_endpoint(websocket, camera_id)

    # ── Web pages ──────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return _TEMPLATES.TemplateResponse(
            "dashboard.html", {"request": request}
        )

    @app.get("/violations", response_class=HTMLResponse)
    async def violations_page(request: Request):
        return _TEMPLATES.TemplateResponse(
            "violations.html", {"request": request}
        )

    @app.get("/statistics", response_class=HTMLResponse)
    async def statistics_page(request: Request):
        return _TEMPLATES.TemplateResponse(
            "statistics.html", {"request": request}
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        return _TEMPLATES.TemplateResponse(
            "settings.html", {"request": request}
        )

    # ── Health check ───────────────────────────────────────────
    @app.get("/health")
    async def health():
        from services.detection_service import detection_manager
        return {"status": "ok", "workers": detection_manager.status()}

    return app


app = create_app()
