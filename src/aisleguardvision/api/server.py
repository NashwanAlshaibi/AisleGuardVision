"""FastAPI service.

A minimal foundation for the eventual dashboard. Phase 1 deliberately ships no
frontend: an unnecessary UI built before the detection logic is trustworthy is
effort spent in the wrong place.

    GET  /health
    GET  /metrics                    (JSON; ?format=prometheus for text)
    GET  /cameras
    GET  /cameras/{camera_id}
    POST /cameras/{camera_id}/enable
    POST /cameras/{camera_id}/disable
    GET  /events
    GET  /events/{event_id}

Auth is a shared secret in ``X-API-Key``, sourced from the environment. An
empty key disables auth, which is acceptable only on a trusted loopback
interface -- and the server logs a warning at startup when that is the case.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..core.config import AppConfig
from ..core.logging import get_logger
from ..events.models import SecurityEventModel
from ..events.recorder import IncidentStore
from .schemas import (
    CameraActionResponse,
    CameraListResponse,
    CameraResponse,
    EventListResponse,
    EventSummary,
    HealthResponse,
    MetricsResponse,
)
from .state import RuntimeState, get_state

logger = get_logger(__name__)


def require_state() -> RuntimeState:
    state = get_state()
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AisleGuard Vision runtime is not initialized",
        )
    return state


def api_key_guard(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    state: RuntimeState = Depends(require_state),
) -> None:
    """Validate the shared secret, if one is configured."""
    expected = state.config.api.api_key
    if not expected:
        return
    # Constant-time comparison: an API key check that leaks timing is a real,
    # if slow, way to recover the key.
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing X-API-Key"
        )


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="AisleGuard Vision",
        version=__version__,
        description=(
            "Retail loss-prevention behavioral analysis. This API exposes camera "
            "health, pipeline metrics and incidents recorded for human review. "
            "Incidents are behavioral risk signals, never determinations that a "
            "theft occurred. No facial, identity or demographic recognition is "
            "performed anywhere in this system; person identifiers are temporary "
            "per-camera tracking ids."
        ),
    )

    effective = config or (get_state().config if get_state() else AppConfig())
    if effective.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=effective.api.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    guard = [Depends(api_key_guard)]

    # -- health ------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(state: RuntimeState = Depends(require_state)) -> HealthResponse:
        """Liveness and camera health. Intentionally unauthenticated.

        A load balancer or supervisor must be able to probe this without
        holding a credential.
        """
        summary = state.health()
        return HealthResponse(
            status=str(summary.get("status", "starting")),
            version=__version__,
            uptime_seconds=round(state.uptime_seconds, 1),
            cameras_total=int(summary.get("cameras_total", 0)),
            cameras_streaming=int(summary.get("cameras_streaming", 0)),
            cameras_reconnecting=int(summary.get("cameras_reconnecting", 0)),
            cameras_failed=int(summary.get("cameras_failed", 0)),
            cameras_stalled=list(summary.get("cameras_stalled", [])),
            device=state.device_label,
            item_detection_available=state.item_detection_available,
        )

    # -- metrics -----------------------------------------------------------
    @app.get("/metrics", tags=["system"], dependencies=guard)
    def metrics(
        response_format: str = Query("json", alias="format", pattern="^(json|prometheus)$"),
        state: RuntimeState = Depends(require_state),
    ):
        """Pipeline metrics as JSON, or Prometheus text exposition."""
        if response_format == "prometheus":
            return Response(
                content=state.metrics.to_prometheus(),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )
        return MetricsResponse(**state.metrics.snapshot())

    # -- cameras -----------------------------------------------------------
    @app.get("/cameras", response_model=CameraListResponse, tags=["cameras"], dependencies=guard)
    def list_cameras(state: RuntimeState = Depends(require_state)) -> CameraListResponse:
        cameras = [_to_camera_response(s, state) for s in state.statuses()]
        return CameraListResponse(cameras=cameras, count=len(cameras))

    @app.get(
        "/cameras/{camera_id}",
        response_model=CameraResponse,
        tags=["cameras"],
        dependencies=guard,
    )
    def get_camera(camera_id: str, state: RuntimeState = Depends(require_state)) -> CameraResponse:
        camera_status = state.status(camera_id)
        if camera_status is None:
            raise HTTPException(status_code=404, detail=f"unknown camera {camera_id!r}")
        return _to_camera_response(camera_status, state)

    @app.post(
        "/cameras/{camera_id}/enable",
        response_model=CameraActionResponse,
        tags=["cameras"],
        dependencies=guard,
    )
    def enable_camera(
        camera_id: str, state: RuntimeState = Depends(require_state)
    ) -> CameraActionResponse:
        return _set_camera_enabled(camera_id, True, state)

    @app.post(
        "/cameras/{camera_id}/disable",
        response_model=CameraActionResponse,
        tags=["cameras"],
        dependencies=guard,
    )
    def disable_camera(
        camera_id: str, state: RuntimeState = Depends(require_state)
    ) -> CameraActionResponse:
        return _set_camera_enabled(camera_id, False, state)

    # -- events ------------------------------------------------------------
    @app.get("/events", response_model=EventListResponse, tags=["events"], dependencies=guard)
    def list_events(
        camera_id: str | None = Query(None),
        limit: int | None = Query(None, ge=1, le=1000),
        min_risk: float = Query(0.0, ge=0.0, le=100.0),
        state: RuntimeState = Depends(require_state),
    ) -> EventListResponse:
        store = _store(state)
        effective_limit = limit or state.config.api.default_event_limit
        events = store.list_events(camera_id=camera_id, limit=effective_limit, min_risk=min_risk)
        return EventListResponse(
            events=[_to_summary(e) for e in events],
            count=len(events),
            total_stored=store.count(),
        )

    @app.get(
        "/events/{event_id}",
        response_model=SecurityEventModel,
        tags=["events"],
        dependencies=guard,
    )
    def get_event(
        event_id: str, state: RuntimeState = Depends(require_state)
    ) -> SecurityEventModel:
        """Full incident record, including the complete evidence breakdown."""
        event = _store(state).get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail=f"unknown event {event_id!r}")
        return event

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(state: RuntimeState) -> IncidentStore:
    if state.incident_store is None:
        state.incident_store = IncidentStore(state.config.storage)
    return state.incident_store


def _to_camera_response(camera_status, state: RuntimeState) -> CameraResponse:
    return CameraResponse(
        camera_id=camera_status.camera_id,
        name=camera_status.name,
        state=camera_status.state,
        enabled=camera_status.enabled,
        source_label=camera_status.source_label,
        resolution=camera_status.resolution,
        decode_fps=camera_status.decode_fps,
        inference_fps=camera_status.inference_fps,
        pose_fps=camera_status.pose_fps,
        frames_received=camera_status.frames_received,
        frames_dropped=camera_status.frames_dropped,
        reconnect_count=camera_status.reconnect_count,
        active_tracks=camera_status.active_tracks,
        last_frame_timestamp=camera_status.last_frame_timestamp,
        last_error=camera_status.last_error,
        zone_count=state.zone_count(camera_status.camera_id),
    )


def _set_camera_enabled(camera_id: str, enabled: bool, state: RuntimeState) -> CameraActionResponse:
    manager = state.camera_manager
    if manager is None:
        raise HTTPException(status_code=503, detail="camera manager is not running")
    ok = manager.enable(camera_id) if enabled else manager.disable(camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"unknown camera {camera_id!r}")
    camera_status = state.status(camera_id)
    action = "enabled" if enabled else "disabled"
    logger.info("camera %s via API", action, extra={"fields": {"camera_id": camera_id}})
    return CameraActionResponse(
        camera_id=camera_id,
        enabled=enabled,
        state=camera_status.state if camera_status else "idle",
        message=f"camera {camera_id} {action}",
    )


def _to_summary(event: SecurityEventModel) -> EventSummary:
    return EventSummary(
        event_id=event.event_id,
        camera_id=event.camera_id,
        person_id=event.person_id,
        timestamp=event.timestamp,
        timestamp_iso=event.timestamp_iso,
        risk_score=event.risk_score,
        threat_level=event.threat_level,
        behavior_state=event.behavior_state.value,
        event_type=event.event_type.value,
        positive_evidence_count=len(event.positive_evidence),
        negative_evidence_count=len(event.negative_evidence),
        has_clip=bool(event.clip_path),
        has_snapshot=bool(event.snapshot_path),
        review_status=event.review_status,
    )


def run_api(state: RuntimeState, host: str | None = None, port: int | None = None) -> None:
    """Run the API server in the current thread (blocking)."""
    import uvicorn

    from .state import set_state

    set_state(state)
    config = state.config.api
    if not config.api_key:
        logger.warning(
            "API authentication is disabled (no api_key configured). Bind to a "
            "loopback interface only, or set AISLEGUARD_API_KEY."
        )
    uvicorn.run(
        create_app(state.config),
        host=host or config.host,
        port=port or config.port,
        log_level=state.config.logging.level.lower(),
    )


def start_api_thread(state: RuntimeState) -> None:
    """Run the API on a daemon thread alongside the video pipeline."""
    import threading

    import uvicorn

    from .state import set_state

    set_state(state)
    config = state.config.api
    if not config.api_key:
        logger.warning(
            "API authentication is disabled (no api_key configured). Bind to a "
            "loopback interface only, or set AISLEGUARD_API_KEY."
        )

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(state.config),
            host=config.host,
            port=config.port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, name="api", daemon=True)
    thread.start()
    logger.info(
        "API listening",
        extra={
            "fields": {
                "host": config.host,
                "port": config.port,
                "auth": "api-key" if config.api_key else "disabled",
            }
        },
    )
