"""
# ======================================
# VIDEO DETECTOR
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module processes uploaded videos offline using the same detection, tracking, ReID, association, compliance, and event logic as live streams.
- It solves the audit/review use case where safety teams upload historical CCTV clips.
- Enterprise design reuses the core pipeline instead of implementing a separate simplified video path.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict

import cv2

from .association_engine import AssociationEngine
from .compliance_engine import ComplianceEngine
from .config import CONFIG
from .detector import PPEDetector, draw_detections
from .event_manager import EventManager
from .logger import get_logger
from .reid_manager import ReIDManager
from .tracker_manager import TrackerManager

logger = get_logger("ppe.video_detector")


def process_uploaded_video(input_path: Path, rules: Dict | None = None) -> Dict:
    job_id = "video_" + uuid.uuid4().hex[:10]
    output_path = CONFIG.OUTPUT_VIDEO_DIR / f"{job_id}.mp4"
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector = PPEDetector()
    tracker = TrackerManager(job_id)
    reid = ReIDManager(job_id)
    association = AssociationEngine(job_id)
    compliance = ComplianceEngine(rules or {})
    events = EventManager(job_id, "video")

    frame_id = 0
    violation_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % max(1, CONFIG.DEFAULT_FRAME_SKIP) == 0:
            detections = detector.track_frame(frame, persist=True)
            detections = tracker.update(frame, detections)
            people = [d for d in detections if d.canonical_class == CONFIG.PERSON_CLASS]
            reid.update_person_identities(frame, people)
            associations = association.associate(detections)
            violations = compliance.evaluate(associations)
            for v in violations:
                v["camera_id"] = job_id
            violation_count += len(violations)
            annotated = draw_detections(frame, detections, violations)
            events.update(violations, annotated, frame)
            writer.write(annotated)
        else:
            writer.write(frame)

    cap.release()
    writer.release()
    return {
        "job_id": job_id,
        "input_path": str(input_path),
        "output_path": f"/static/outputs/videos/{output_path.name}",
        "frames": frame_id,
        "violation_count": violation_count,
    }
