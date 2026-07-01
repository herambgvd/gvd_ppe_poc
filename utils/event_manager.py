"""
# ======================================
# EVENT MANAGER
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- Converts frame-by-frame violations into enterprise lifecycle events.
- Prevents duplicate alerts from continuous detections.
- Tracks:
    - NEW
    - ACTIVE
    - RESOLVED
    - EXPIRED
- Uses ReID global identities instead of unstable ByteTrack IDs.

Enterprise architecture:
- ByteTrack IDs are temporary.
- ReID identities are persistent.
- Events must follow worker identity,
  not temporary tracker IDs.
"""

from __future__ import annotations

import uuid

from dataclasses import dataclass, field

from datetime import datetime, timedelta, timezone

from pathlib import Path

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import CONFIG
from .database import DB, utc_now_iso
from .logger import get_logger

logger = get_logger("ppe.event_manager")


# ======================================
# TIME HELPERS
# ======================================

def now_dt() -> datetime:

    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:

    return dt.isoformat()


# ======================================
# EVENT MEMORY
# ======================================

@dataclass
class EventMemory:

    key: Tuple[str, str, str]

    event_id: Optional[str] = None

    first_seen: datetime = field(
        default_factory=now_dt
    )

    last_seen: datetime = field(
        default_factory=now_dt
    )

    observation_count: int = 0

    state: str = "OBSERVED"

    screenshot_path: Optional[str] = None

    crop_path: Optional[str] = None

    confidence: float = 0.0

    cooldown_until: Optional[datetime] = None


# ======================================
# EVENT MANAGER
# ======================================

class EventManager:

    # ======================================
    # INIT
    # ======================================

    def __init__(

        self,

        camera_id: str,

        source_type: str = "rtsp"

    ):

        self.camera_id = camera_id

        self.source_type = source_type

        # ======================================
        # ACTIVE EVENT MEMORY
        # ======================================

        self.events: Dict[
            Tuple[str, str, str],
            EventMemory
        ] = {}

    # ======================================
    # UPDATE EVENTS
    # ======================================

    def update(

        self,

        violations: List[Dict],

        annotated_frame: np.ndarray,

        raw_frame: np.ndarray

    ) -> List[Dict]:

        """
        Update lifecycle from current-frame violations.

        Uses:
        - ReID identities
        - persistence logic
        - cooldown logic
        """

        now = now_dt()

        seen_keys = set()

        emitted: List[Dict] = []

        # ======================================
        # PROCESS VIOLATIONS
        # ======================================

        for v in violations:

            v["camera_id"] = self.camera_id

            # ======================================
            # REID IDENTITY
            # ======================================

            identity_id = (

                v.get("reid_global_id")

                or

                f"track_{v.get('track_id')}"
            )

            # ======================================
            # EVENT KEY
            # ======================================

            key = (

                self.camera_id,

                identity_id,

                v["violation_type"]
            )

            seen_keys.add(key)

            mem = self.events.get(key)

            # ======================================
            # NEW MEMORY
            # ======================================

            if mem is None:

                mem = EventMemory(
                    key=key
                )

                self.events[key] = mem

            # ======================================
            # UPDATE MEMORY
            # ======================================

            mem.last_seen = now

            mem.observation_count += 1

            mem.confidence = max(

                mem.confidence,

                float(
                    v.get(
                        "confidence",
                        0
                    )
                )
            )

            # ======================================
            # CREATE EVENT
            # ======================================

            if (

                mem.event_id is None

                and

                mem.observation_count
                >=
                CONFIG.VIOLATION_PERSISTENCE_FRAMES

            ):

                mem.event_id = (
                    self._create_event_id()
                )

                mem.state = "NEW"

                # ======================================
                # SAVE EVIDENCE
                # ======================================

                (

                    mem.screenshot_path,

                    mem.crop_path

                ) = self._save_evidence(

                    mem.event_id,

                    v,

                    annotated_frame,

                    raw_frame
                )

                # ======================================
                # EVENT PAYLOAD
                # ======================================

                event_payload = (
                    self._event_payload(
                        mem,
                        v,
                        now
                    )
                )

                # ======================================
                # DATABASE
                # ======================================

                DB.upsert_event(
                    event_payload
                )

                DB.insert_violation({

                    "event_id": mem.event_id,

                    "camera_id": self.camera_id,

                    # ======================================
                    # STORE REID IDENTITY
                    # ======================================

                    "track_id": identity_id,

                    "violation_type": (
                        v["violation_type"]
                    ),

                    "confidence": mem.confidence,

                    "screenshot_path": (
                        mem.screenshot_path
                    ),

                    "crop_path": (
                        mem.crop_path
                    ),

                    "metadata": v,
                })

                emitted.append(
                    event_payload
                )

                logger.info(

                    "Created event %s for %s",

                    mem.event_id,

                    key
                )

            # ======================================
            # ACTIVE EVENT
            # ======================================

            elif mem.event_id is not None:

                mem.state = "ACTIVE"

                event_payload = (
                    self._event_payload(
                        mem,
                        v,
                        now
                    )
                )

                DB.upsert_event(
                    event_payload
                )

                emitted.append(
                    event_payload
                )

        # ======================================
        # RESOLVE OLD EVENTS
        # ======================================

        for key, mem in list(
            self.events.items()
        ):

            if (
                key in seen_keys
                or
                mem.event_id is None
            ):

                continue

            seconds_missing = (

                now - mem.last_seen

            ).total_seconds()

            # ======================================
            # RESOLVE
            # ======================================

            if (

                mem.state in {
                    "NEW",
                    "ACTIVE"
                }

                and

                seconds_missing
                >=
                CONFIG.EVENT_RESOLVE_AFTER_SECONDS
            ):

                mem.state = "RESOLVED"

                payload = (
                    self._event_payload(

                        mem,

                        {

                            "track_id": key[1],

                            "reid_global_id": key[1],

                            "violation_type": key[2],

                            "confidence": mem.confidence
                        },

                        now,

                        resolved=True
                    )
                )

                DB.upsert_event(
                    payload
                )

                emitted.append(
                    payload
                )

            # ======================================
            # EXPIRE
            # ======================================

            if (

                now - mem.last_seen

            ).total_seconds() >= (
                CONFIG.EVENT_EXPIRE_AFTER_SECONDS
            ):

                mem.state = "EXPIRED"

                self.events.pop(
                    key,
                    None
                )

        return emitted

    # ======================================
    # FINALIZE (called when a stream / video job ends)
    # ======================================

    def finalize(self) -> List[Dict]:
        """
        Resolve every still-open event.

        Event resolution normally happens inside update() while frames are being
        processed. When a video job finishes or is stopped, update() is no longer
        called, so any event that was ACTIVE at the last frame would stay ONGOING
        forever. Calling finalize() at the end closes them out as RESOLVED.
        """
        now = now_dt()
        closed: List[Dict] = []
        for key, mem in list(self.events.items()):
            if mem.event_id is not None and mem.state in {"NEW", "ACTIVE"}:
                mem.state = "RESOLVED"
                payload = self._event_payload(
                    mem,
                    {
                        "track_id": key[1],
                        "reid_global_id": key[1],
                        "violation_type": key[2],
                        "confidence": mem.confidence,
                    },
                    now,
                    resolved=True,
                )
                DB.upsert_event(payload)
                closed.append(payload)
        return closed

    # ======================================
    # EVENT PAYLOAD
    # ======================================

    def _event_payload(

        self,

        mem: EventMemory,

        violation: Dict,

        now: datetime,

        resolved: bool = False

    ) -> Dict:

        end_ts = (
            now
            if resolved
            else None
        )

        duration = (

            now - mem.first_seen

        ).total_seconds()

        cooldown_until = now + timedelta(

            seconds=CONFIG.DUPLICATE_COOLDOWN_SECONDS
        )

        return {

            "event_id": mem.event_id,

            # ======================================
            # STORE REID IDENTITY
            # ======================================

            "track_id": (

                violation.get(
                    "reid_global_id"
                )

                or

                f"track_{violation.get('track_id')}"
            ),

            "camera_id": self.camera_id,

            "violation_type": violation.get(
                "violation_type"
            ),

            "state": mem.state,

            "timestamp_start": iso(
                mem.first_seen
            ),

            "timestamp_end": (
                iso(end_ts)
                if end_ts
                else None
            ),

            "duration": duration,

            "screenshot_path": (
                mem.screenshot_path
            ),

            "crop_path": (
                mem.crop_path
            ),

            "confidence": float(
                mem.confidence
            ),

            "evidence": violation,

            "last_seen_ts": iso(
                mem.last_seen
            ),

            "cooldown_until": iso(
                cooldown_until
            ),

            "created_at": iso(
                mem.first_seen
            ),
        }

    # ======================================
    # SAVE EVIDENCE
    # ======================================

    def _save_evidence(

        self,

        event_id: str,

        violation: Dict,

        annotated_frame: np.ndarray,

        raw_frame: np.ndarray

    ):

        folder = (
            CONFIG.VIOLATION_DIR
            /
            self.source_type
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        screenshot_rel = (

            f"violations/"
            f"{self.source_type}/"
            f"{event_id}.jpg"
        )

        crop_rel = (

            f"violations/"
            f"{self.source_type}/"
            f"{event_id}_crop.jpg"
        )

        screenshot_path = (
            CONFIG.STATIC_DIR
            /
            screenshot_rel
        )

        crop_path = (
            CONFIG.STATIC_DIR
            /
            crop_rel
        )

        # ======================================
        # SAVE SCREENSHOT
        # ======================================

        cv2.imwrite(

            str(screenshot_path),

            annotated_frame
        )

        # ======================================
        # SAVE PERSON CROP
        # ======================================

        crop = self._crop(

            raw_frame,

            violation.get(
                "person_bbox"
            )
        )

        if crop is not None and crop.size:

            cv2.imwrite(

                str(crop_path),

                crop
            )

        return (

            f"/static/{screenshot_rel}",

            f"/static/{crop_rel}"
        )

    # ======================================
    # CROP PERSON
    # ======================================

    @staticmethod
    def _crop(

        frame: np.ndarray,

        bbox

    ) -> Optional[np.ndarray]:

        if not bbox:
            return None

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        pad_x = int(
            (x2 - x1) * 0.08
        )

        pad_y = int(
            (y2 - y1) * 0.08
        )

        x1, y1 = (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y)
        )

        x2, y2 = (
            min(w - 1, x2 + pad_x),
            min(h - 1, y2 + pad_y)
        )

        return frame[
            y1:y2,
            x1:x2
        ]

    # ======================================
    # CREATE EVENT ID
    # ======================================

    @staticmethod
    def _create_event_id() -> str:

        return (
            "evt_"
            +
            uuid.uuid4().hex[:16]
        )