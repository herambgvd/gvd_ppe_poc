"""
# ======================================
# ALERT MANAGER
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module centralizes outgoing notifications to the dashboard.
- It prevents camera processing threads from knowing UI implementation details.
- In enterprise deployments, this can later fan out to email, WhatsApp, siren, webhook, SMS, or VMS integrations.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .logger import get_logger

logger = get_logger("ppe.alerts")


class AlertManager:
    def __init__(self):
        self.socketio = None

    def bind_socketio(self, socketio) -> None:
        self.socketio = socketio

    def emit(self, event_name: str, payload: Dict[str, Any], namespace: str = "/") -> None:
        if not self.socketio:
            return
        try:
            self.socketio.emit(event_name, payload, namespace=namespace)
        except Exception as exc:
            logger.exception("Socket emit failed for %s: %s", event_name, exc)

    def emit_events(self, events: Iterable[Dict[str, Any]]) -> None:
        for event in events:
            self.emit("violation_event", event)
            if event.get("state") in {"NEW", "ACTIVE"}:
                self.emit("active_violation", event)

    def emit_camera_status(self, camera_id: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.emit("camera_status", {"camera_id": camera_id, "status": status, "metadata": metadata or {}})

    def emit_metrics(self, metrics: Dict[str, Any]) -> None:
        self.emit("metrics_update", metrics)


ALERTS = AlertManager()
