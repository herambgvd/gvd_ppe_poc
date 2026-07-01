"""
# ======================================
# WEBCAM DETECTOR
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module provides a small service wrapper to register local webcams as managed camera sources.
- It keeps webcam handling consistent with RTSP sources: both are handled by StreamManager workers.
- Enterprise systems avoid separate one-off webcam code paths because they become hard to maintain.
"""

from __future__ import annotations

import uuid
from typing import Dict

from .database import DB


def register_webcam(name: str = "Local Webcam", device_index: int = 0, rules: Dict | None = None) -> Dict:
    camera = {
        "camera_id": f"webcam_{uuid.uuid4().hex[:8]}",
        "name": name,
        "source_type": "webcam",
        "source_uri": str(device_index),
        "enabled": True,
        "rules": rules or {},
        "zones": [],
    }
    DB.insert_camera(camera)
    return camera
