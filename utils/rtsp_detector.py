"""
# ======================================
# RTSP DETECTOR
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module registers RTSP CCTV streams into the same camera management pipeline.
- It solves the operational requirement of onboarding many IP cameras without changing code.
- In enterprise deployments, this module can be expanded to validate credentials, ONVIF metadata, camera health, and network reachability.
"""

from __future__ import annotations

import uuid
from typing import Dict

from .database import DB


def register_rtsp_camera(name: str, rtsp_url: str, rules: Dict | None = None, zones: list | None = None) -> Dict:
    camera = {
        "camera_id": f"rtsp_{uuid.uuid4().hex[:8]}",
        "name": name,
        "source_type": "rtsp",
        "source_uri": rtsp_url,
        "enabled": True,
        "rules": rules or {},
        "zones": zones or [],
    }
    DB.insert_camera(camera)
    return camera
