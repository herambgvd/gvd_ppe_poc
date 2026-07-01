"""
# ======================================
# SOCKET EVENTS
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module registers Socket.IO events for live dashboard communication.
- It allows backend camera workers to push active violations, metrics, and camera status without page refresh.
- Enterprise dashboards use event-driven updates for live operational awareness.
"""

from __future__ import annotations

from .alert_manager import ALERTS
from .analytics_engine import ANALYTICS
from .database import DB
from .stream_manager import STREAMS


def register_socket_events(socketio):
    ALERTS.bind_socketio(socketio)

    @socketio.on("connect")
    def handle_connect():
        socketio.emit("metrics_update", {"analytics": ANALYTICS.summary(), "runtime": STREAMS.list_runtime_stats()})
        socketio.emit("active_events_snapshot", DB.list_active_events())

    @socketio.on("request_metrics")
    def handle_request_metrics():
        socketio.emit("metrics_update", {"analytics": ANALYTICS.summary(), "runtime": STREAMS.list_runtime_stats()})

    @socketio.on("start_camera")
    def handle_start_camera(data):
        camera_id = data.get("camera_id")
        ok = STREAMS.start_camera(camera_id)
        socketio.emit("camera_action_result", {"camera_id": camera_id, "action": "start", "ok": ok})

    @socketio.on("stop_camera")
    def handle_stop_camera(data):
        camera_id = data.get("camera_id")
        ok = STREAMS.stop_camera(camera_id)
        socketio.emit("camera_action_result", {"camera_id": camera_id, "action": "stop", "ok": ok})
