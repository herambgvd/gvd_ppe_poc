"""
# ======================================
# COMPLIANCE ENGINE / RULE ENGINE
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- Converts PPE associations into business-rule decisions.
- Separates:
    detection
        from
    safety policy.
- Supports:
    - per-camera PPE rules
    - future zone-based rules
    - identity-aware violations
    - enterprise analytics

Enterprise architecture:
- Detection models should NEVER contain business logic.
- Safety rules must remain configurable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional

from .association_engine import PersonAssociation
from .config import CONFIG


# ======================================
# RULE SET
# ======================================

@dataclass
class ComplianceRuleSet:

    mandatory_ppe: List[str]

    optional_ppe: List[str]

    min_person_confidence: float = 0.50

    # ======================================
    # LOAD CAMERA RULES
    # ======================================

    @classmethod
    def from_camera_rules(

        cls,

        rules: Optional[Dict]

    ) -> "ComplianceRuleSet":

        rules = rules or {}

        return cls(

            mandatory_ppe=rules.get(

                "mandatory_ppe",

                CONFIG.DEFAULT_MANDATORY_PPE
            ),

            optional_ppe=rules.get(

                "optional_ppe",

                CONFIG.DEFAULT_OPTIONAL_PPE
            ),

            min_person_confidence=float(

                rules.get(
                    "min_person_confidence",
                    0.50
                )
            ),
        )


# ======================================
# COMPLIANCE ENGINE
# ======================================

class ComplianceEngine:

    # ======================================
    # INIT
    # ======================================

    def __init__(

        self,

        camera_rules: Optional[Dict] = None

    ):

        self.rules = (

            ComplianceRuleSet.from_camera_rules(
                camera_rules
            )
        )

        # Per-track temporal history of missing PPE, for confirmation smoothing.
        # track_id -> ppe -> deque[bool]  (True = PPE was missing that frame)
        self._ppe_history: Dict[object, Dict[str, Deque[bool]]] = {}
        self._track_last_seen: Dict[object, int] = {}
        self._frame_no: int = 0

    # ======================================
    # TEMPORAL CONFIRMATION
    # ======================================

    def _confirm_missing(self, track_id, ppe: str, missing_now: bool) -> bool:
        """
        Record this frame's observation and decide whether ``ppe`` should be
        reported as missing for ``track_id`` right now.

        Returns True only when the PPE is currently missing AND has been missing
        in at least PPE_CONFIRM_MIN_MISSING of the last PPE_CONFIRM_WINDOW
        observed frames — suppressing single-frame association dropouts.
        """
        window = max(1, CONFIG.PPE_CONFIRM_WINDOW)
        need = max(1, CONFIG.PPE_CONFIRM_MIN_MISSING)

        by_ppe = self._ppe_history.setdefault(track_id, {})
        hist = by_ppe.get(ppe)
        if hist is None or hist.maxlen != window:
            hist = deque(hist or (), maxlen=window)
            by_ppe[ppe] = hist
        hist.append(bool(missing_now))

        if not missing_now:
            return False
        return sum(hist) >= min(need, len(hist)) and len(hist) >= min(need, window)

    def _prune_history(self, active_tracks: set) -> None:
        """Drop history for tracks not seen recently to bound memory."""
        for tid in active_tracks:
            self._track_last_seen[tid] = self._frame_no
        if len(self._ppe_history) <= 512:
            return
        stale = [
            tid for tid, seen in self._track_last_seen.items()
            if self._frame_no - seen > 120
        ]
        for tid in stale:
            self._ppe_history.pop(tid, None)
            self._track_last_seen.pop(tid, None)

    # ======================================
    # EVALUATE COMPLIANCE
    # ======================================

    def evaluate(

        self,

        associations: List[PersonAssociation]

    ) -> List[Dict]:

        violations: List[Dict] = []

        self._frame_no += 1
        active_tracks: set = set()

        # ======================================
        # PROCESS WORKERS
        # ======================================

        for assoc in associations:

            person = assoc.person

            # ======================================
            # LOW CONFIDENCE FILTER
            # ======================================

            if (

                person.conf
                <
                self.rules.min_person_confidence

                or

                person.track_id is None
            ):

                continue

            # ======================================
            # SIZE FILTER
            # Skip far-away / tiny persons: PPE cannot be judged reliably,
            # so we must not raise (false) violations for them.
            # ======================================

            px1, py1, px2, py2 = person.bbox

            person_area = (
                max(0.0, px2 - px1)
                *
                max(0.0, py2 - py1)
            )

            if person_area < CONFIG.MIN_PERSON_BBOX_AREA:

                continue

            # ======================================
            # REID IDENTITY
            # ======================================

            reid_global_id = (

                person.metadata.get(
                    "reid_global_id"
                )
            )

            # Temporal smoothing must key on the most STABLE identity available.
            # track_id churns heavily in crowded scenes, which would reset the
            # confirmation window every few frames; the ReID global id survives
            # track breaks, so prefer it.
            smooth_key = reid_global_id or f"track_{person.track_id}"
            active_tracks.add(smooth_key)

            # ======================================
            # PPE RULE CHECKS
            # ======================================

            for ppe in self.rules.mandatory_ppe:

                negative_class = (
                    f"no_{ppe}"
                )

                has_positive = (
                    assoc.has_ppe(ppe)
                )

                has_negative = (
                    assoc.has_negative(
                        negative_class
                    )
                )

                # ======================================
                # TEMPORAL CONFIRMATION
                # A single dropped frame must not raise a violation; only flag
                # PPE that is *consistently* missing across recent frames.
                # ======================================

                missing_now = has_negative or (not has_positive)

                if self._confirm_missing(
                    smooth_key, ppe, missing_now
                ):

                    confidence = (

                        self._violation_confidence(

                            assoc,

                            ppe,

                            negative_class
                        )
                    )

                    # ======================================
                    # VIOLATION PAYLOAD
                    # ======================================

                    violation = {

                        # ======================================
                        # TRACKING
                        # ======================================

                        "track_id": (
                            person.track_id
                        ),

                        "reid_global_id": (
                            reid_global_id
                        ),

                        # ======================================
                        # CAMERA
                        # ======================================

                        "camera_id": None,

                        # ======================================
                        # VIOLATION
                        # ======================================

                        "violation_type": (
                            f"missing_{ppe}"
                        ),

                        "required_ppe": ppe,

                        "confidence": confidence,

                        # ======================================
                        # PERSON DATA
                        # ======================================

                        "person_bbox": list(
                            person.bbox
                        ),

                        "person_confidence": (
                            float(person.conf)
                        ),

                        # ======================================
                        # COMPLIANCE STATUS
                        # ======================================

                        "compliance": (
                            assoc.compliance_summary()
                        ),

                        # ======================================
                        # ASSOCIATION DATA
                        # ======================================

                        "association": (
                            assoc.as_dict()
                        ),

                        # ======================================
                        # VIOLATION TIMESTAMP
                        # ======================================

                        "timestamp": (
                            datetime.utcnow().isoformat()
                        ),

                        # ======================================
                        # ENTERPRISE SEVERITY
                        # ======================================

                        "severity": (
                            self._severity_level(
                                confidence
                            )
                        ),
                    }

                    violations.append(
                        violation
                    )

        self._prune_history(active_tracks)

        return violations

    # ======================================
    # VIOLATION CONFIDENCE
    # ======================================

    def _violation_confidence(

        self,

        assoc: PersonAssociation,

        ppe: str,

        negative_class: str

    ) -> float:

        # ======================================
        # NEGATIVE PPE DETECTED
        # ======================================

        if assoc.has_negative(
            negative_class
        ):

            neg_conf = max([

                d.conf

                for d in assoc.negative_ppe[
                    negative_class
                ]

            ] or [0.75])

            return float(

                min(
                    0.98,
                    neg_conf + 0.10
                )
            )

        # ======================================
        # PPE ABSENCE
        # ======================================

        person_conf = (
            assoc.person.conf
        )

        assoc_penalty = (
            assoc.scores.get(
                ppe,
                0
            )
        )

        return float(

            max(

                0.45,

                min(

                    0.82,

                    person_conf
                    -
                    0.10
                    +
                    assoc_penalty * 0.2
                )
            )
        )

    # ======================================
    # SEVERITY LEVEL
    # ======================================

    @staticmethod
    def _severity_level(
        confidence: float
    ) -> str:

        if confidence >= 0.90:
            return "critical"

        if confidence >= 0.75:
            return "high"

        if confidence >= 0.60:
            return "medium"

        return "low"