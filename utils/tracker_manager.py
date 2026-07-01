"""
# ======================================
# TRACKER MANAGER
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- Maintains persistent tracking state above ByteTrack.
- Combines:
    - motion tracking
    - appearance ReID
    - temporal memory
    - identity stabilization
- Bridges:
    YOLO
        →
    ByteTrack
        →
    ReID
        →
    enterprise event lifecycle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CONFIG
from .detector import Detection

# ======================================
# REID IMPORTS
# ======================================

from .feature_extractor import FeatureExtractor
from .reid_matcher import ReIDMatcher


# ======================================
# TRACK STATE
# ======================================

@dataclass
class TrackState:

    # ======================================
    # BASIC TRACK DATA
    # ======================================

    track_id: int

    camera_id: str

    bbox: Tuple[float, float, float, float]

    first_seen: float = field(default_factory=time)

    last_seen: float = field(default_factory=time)

    age_frames: int = 0

    missed_frames: int = 0

    canonical_class: str = CONFIG.PERSON_CLASS

    # ======================================
    # REID DATA
    # ======================================

    reid_global_id: Optional[str] = None

    embedding: Optional[np.ndarray] = None

    reid_similarity: float = 0.0

    # ======================================
    # UPDATE TRACK
    # ======================================

    def update(self, det: Detection) -> None:

        self.bbox = det.bbox

        self.last_seen = time()

        self.age_frames += 1

        self.missed_frames = 0


# ======================================
# TRACKER MANAGER
# ======================================

class TrackerManager:

    # ======================================
    # INIT
    # ======================================

    def __init__(self, camera_id: str):

        self.camera_id = camera_id

        # ======================================
        # ACTIVE TRACKS
        # ======================================

        self.tracks: Dict[
            int,
            TrackState
        ] = {}

        # ======================================
        # FALLBACK TRACK IDS
        # ======================================

        self._next_fallback_id = 1_000_000

        # ======================================
        # REID SYSTEM
        # ======================================

        self.feature_extractor = (
            FeatureExtractor()
        )

        self.reid_matcher = (
            ReIDMatcher()
        )

        print(
            f"[TrackerManager] Initialized "
            f"for camera: {camera_id}"
        )

    # ======================================
    # UPDATE TRACKER
    # ======================================

    def update(

        self,

        frame,

        detections: List[Detection]

    ) -> List[Detection]:

        """
        Adds:
        - stable tracking
        - appearance embeddings
        - persistent ReID identities
        - temporal stabilization
        """

        seen_ids = set()

        # ======================================
        # PROCESS DETECTIONS
        # ======================================

        for det in detections:

            # ======================================
            # ONLY PERSON CLASS
            # ======================================

            if (
                det.canonical_class
                !=
                CONFIG.PERSON_CLASS
            ):

                continue

            # ======================================
            # ENSURE TRACK ID
            # ======================================

            if det.track_id is None:

                det.track_id = (
                    self._assign_fallback_id(
                        det
                    )
                )

            seen_ids.add(
                det.track_id
            )

            # ======================================
            # TRACK STATE
            # ======================================

            state = self.tracks.get(
                det.track_id
            )

            # ======================================
            # PERSON CROP
            # ======================================

            x1, y1, x2, y2 = map(
                int,
                det.bbox
            )

            h, w = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)

            if (
                x2 <= x1
                or
                y2 <= y1
            ):

                continue

            person_crop = frame[
                y1:y2,
                x1:x2
            ]

            # ======================================
            # EMBEDDING
            # ======================================

            embedding = None

            try:

                # ======================================
                # SPARSE REID EXTRACTION
                # ======================================

                if (

                    state is None

                    or

                    state.age_frames % 15 == 0

                    or

                    state.embedding is None
                ):

                    embedding = (

                        self.feature_extractor.extract(
                            person_crop
                        )
                    )

                else:

                    embedding = (
                        state.embedding
                    )

            except Exception as e:

                print(
                    f"[Embedding ERROR] {e}"
                )

                embedding = None

            # ======================================
            # REID MATCHING
            # ======================================

            try:

                reid_result = (

                    self.reid_matcher.match(
                        embedding
                    )
                )

            except Exception as e:

                print(
                    f"[ReID Match ERROR] {e}"
                )

                reid_result = None

            # ======================================
            # METADATA SAFETY
            # ======================================

            if not hasattr(
                det,
                "metadata"
            ):

                det.metadata = {}

            # ======================================
            # SAFE REID HANDLING
            # ======================================

            if reid_result is not None:

                reid_identity = (
                    reid_result.get(
                        "identity_id",
                        "unknown"
                    )
                )

                # ======================================
                # IGNORE TEMPORARY UNKNOWN
                # ======================================

                if (
                    reid_identity
                    ==
                    "temporary_unknown"
                ):

                    # ======================================
                    # KEEP PREVIOUS IDENTITY
                    # ======================================

                    if state is not None:

                        reid_identity = (
                            state.reid_global_id
                        )

                    else:

                        reid_identity = "unknown"

                det.metadata[
                    "reid_global_id"
                ] = reid_identity

                det.metadata[
                    "reid_similarity"
                ] = round(

                    reid_result.get(
                        "similarity",
                        0.0
                    ),

                    3
                )

            else:

                det.metadata[
                    "reid_global_id"
                ] = "unknown"

                det.metadata[
                    "reid_similarity"
                ] = 0.0

            # ======================================
            # DEBUG OUTPUT
            # ======================================

            print(

                "[ReID]",

                det.track_id,

                det.metadata.get(
                    "reid_global_id"
                ),

                det.metadata.get(
                    "reid_similarity"
                )

            )

            # ======================================
            # NEW TRACK
            # ======================================

            if state is None:

                self.tracks[
                    det.track_id
                ] = TrackState(

                    track_id=det.track_id,

                    camera_id=self.camera_id,

                    bbox=det.bbox,

                    canonical_class=(
                        det.canonical_class
                    ),

                    reid_global_id=(
                        det.metadata.get(
                            "reid_global_id"
                        )
                    ),

                    embedding=embedding,

                    reid_similarity=(

                        det.metadata.get(
                            "reid_similarity",
                            0.0
                        )
                    )
                )

            # ======================================
            # EXISTING TRACK
            # ======================================

            else:

                state.update(det)

                # ======================================
                # EMBEDDING SMOOTHING
                # ======================================

                if (

                    embedding is not None

                    and

                    state.embedding is not None
                ):

                    try:

                        state.embedding = (

                            0.8
                            *
                            state.embedding

                            +

                            0.2
                            *
                            embedding
                        )

                    except Exception:

                        state.embedding = (
                            embedding
                        )

                elif embedding is not None:

                    state.embedding = (
                        embedding
                    )

                # ======================================
                # UPDATE REID
                # ======================================

                if (

                    det.metadata.get(
                        "reid_global_id"
                    )

                    !=

                    "unknown"
                ):

                    state.reid_global_id = (

                        det.metadata.get(
                            "reid_global_id"
                        )
                    )

                state.reid_similarity = (

                    det.metadata.get(
                        "reid_similarity",
                        0.0
                    )
                )

        # ======================================
        # HANDLE MISSING TRACKS
        # ======================================

        stale = []

        for tid, state in (
            self.tracks.items()
        ):

            if tid not in seen_ids:

                state.missed_frames += 1

                # ======================================
                # REMOVE OLD TRACKS
                # ======================================

                if (

                    time() - state.last_seen

                    >

                    CONFIG.REID_MAX_AGE_SECONDS
                    * 2
                ):

                    stale.append(tid)

        # ======================================
        # CLEANUP
        # ======================================

        for tid in stale:

            self.tracks.pop(
                tid,
                None
            )

        return detections

    # ======================================
    # FALLBACK TRACK ID
    # ======================================

    def _assign_fallback_id(

        self,

        det: Detection

    ) -> int:

        """
        Approximate tracking fallback
        when ByteTrack temporarily fails.
        """

        cx, cy = det.center

        best_id = None

        best_dist = float("inf")

        # ======================================
        # SEARCH EXISTING TRACKS
        # ======================================

        for tid, state in (
            self.tracks.items()
        ):

            sx = (
                state.bbox[0]
                +
                state.bbox[2]
            ) / 2

            sy = (
                state.bbox[1]
                +
                state.bbox[3]
            ) / 2

            dist = (

                (
                    (cx - sx) ** 2
                    +
                    (cy - sy) ** 2
                ) ** 0.5
            )

            if (

                dist < best_dist

                and

                dist < 60
            ):

                best_id = tid

                best_dist = dist

        # ======================================
        # REUSE TRACK
        # ======================================

        if best_id is not None:

            return best_id

        # ======================================
        # CREATE NEW TRACK
        # ======================================

        self._next_fallback_id += 1

        return self._next_fallback_id