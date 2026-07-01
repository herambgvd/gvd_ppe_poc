"""
# ======================================
# REID MANAGER
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module provides appearance-based person re-identification beyond ByteTrack.
- ByteTrack is strong for short-term tracking, but IDs can change after occlusion, camera shake, or temporary disappearance.
- ReID stores visual embeddings and reconnects a new local track ID to an older global identity when similarity is high.

Enterprise design:
- The interface is compatible with TorchReID/FastReID style embedding backends.
- A lightweight HSV histogram fallback is included so the system remains runnable on day one.
- Cross-camera ReID is intentionally future-ready but conservative; production cross-camera matching needs calibration, camera topology, and privacy/legal review.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from time import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import CONFIG
from .detector import Detection
from .logger import get_logger

logger = get_logger("ppe.reid")


# Shared OSNet extractor (loaded once, GPU). Falls back to histogram if unavailable.
_OSNET = None


def _get_osnet():
    global _OSNET
    if _OSNET is None:
        try:
            from .feature_extractor import FeatureExtractor
            _OSNET = FeatureExtractor()
            logger.info("ReID using OSNet appearance embeddings")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OSNet unavailable, falling back to histogram ReID: %s", exc)
            _OSNET = False
    return _OSNET


@dataclass
class ReIDRecord:
    global_id: str
    camera_id: str
    local_track_id: int
    embedding: np.ndarray
    bbox: Tuple[float, float, float, float]
    first_seen: float = field(default_factory=time)
    last_seen: float = field(default_factory=time)
    hits: int = 1


class ReIDManager:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.records: Dict[str, ReIDRecord] = {}
        self.local_to_global: Dict[int, str] = {}
        self._frame_no = 0

    def update_person_identities(self, frame: np.ndarray, people: List[Detection]) -> Dict[int, str]:
        if not CONFIG.REID_ENABLED:
            return {}

        assignments: Dict[int, str] = {}
        now = time()
        self._frame_no += 1
        refresh = self._frame_no % max(1, CONFIG.REID_EMBED_REFRESH_FRAMES) == 0
        self._prune(now)

        for person in people:
            if person.track_id is None:
                continue

            existing_gid = self.local_to_global.get(person.track_id)

            # ---- known ByteTrack id: keep its identity, embed only occasionally (fast) ----
            if existing_gid and existing_gid in self.records:
                rec = self.records[existing_gid]
                rec.bbox = person.bbox
                rec.last_seen = now
                rec.hits += 1
                if refresh:
                    crop = self._safe_crop(frame, person.bbox)
                    if crop.size:
                        rec.embedding = self._smooth_embedding(rec.embedding, self.extract_embedding(crop))
                assignments[person.track_id] = existing_gid
                person.metadata["reid_global_id"] = existing_gid
                person.metadata["reid_similarity"] = 1.0
                continue

            # ---- new ByteTrack id: extract appearance and try to re-identify ----
            crop = self._safe_crop(frame, person.bbox)
            if crop.size == 0:
                continue
            emb = self.extract_embedding(crop)

            gid, score = self._match_existing(emb, person.bbox)
            if gid is None:
                gid = self._new_global_id(person.track_id, emb)
                self.records[gid] = ReIDRecord(gid, self.camera_id, person.track_id, emb, person.bbox)
                score = 1.0
            else:
                rec = self.records[gid]
                rec.local_track_id = person.track_id
                rec.embedding = self._smooth_embedding(rec.embedding, emb)
                rec.bbox = person.bbox
                rec.last_seen = now
                rec.hits += 1

            self.local_to_global[person.track_id] = gid
            assignments[person.track_id] = gid
            person.metadata["reid_global_id"] = gid
            person.metadata["reid_similarity"] = round(float(score), 2)
        return assignments

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Appearance embedding. Prefers OSNet; falls back to a color+texture histogram."""
        ext = _get_osnet()
        if ext:
            try:
                emb = np.asarray(ext.extract(crop), dtype=np.float32).flatten()
                n = np.linalg.norm(emb) + 1e-8
                return (emb / n).astype(np.float32)
            except Exception as exc:  # noqa: BLE001
                logger.debug("OSNet extract failed, using histogram: %s", exc)
        return self._extract_hist(crop)

    def _extract_hist(self, crop: np.ndarray) -> np.ndarray:
        crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 80, 160)
        edge_hist = cv2.resize(edge, (16, 32)).flatten().astype(np.float32) / 255.0
        emb = np.concatenate([hist_h, hist_s, edge_hist.astype(np.float32)])
        norm = np.linalg.norm(emb) + 1e-8
        return (emb / norm).astype(np.float32)

    def _match_existing(self, embedding: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[Optional[str], float]:
        best_gid = None
        best_score = -1.0
        for gid, rec in self.records.items():
            score = self.cosine_similarity(embedding, rec.embedding)
            if score > best_score:
                best_gid = gid
                best_score = score
        if best_score >= CONFIG.REID_SIMILARITY_THRESHOLD:
            return best_gid, best_score
        return None, best_score

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))

    @staticmethod
    def _smooth_embedding(old: np.ndarray, new: np.ndarray, alpha: float = 0.75) -> np.ndarray:
        emb = alpha * old + (1 - alpha) * new
        return emb / (np.linalg.norm(emb) + 1e-8)

    def _safe_crop(self, frame: np.ndarray, bbox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        return frame[y1:y2, x1:x2]

    def _new_global_id(self, local_track_id: int, emb: np.ndarray) -> str:
        seed = f"{self.camera_id}:{local_track_id}:{time()}:{emb[:5].tolist()}".encode()
        return "gid_" + hashlib.sha1(seed).hexdigest()[:12]

    def _prune(self, now: float) -> None:
        old = [gid for gid, rec in self.records.items() if now - rec.last_seen > CONFIG.REID_MAX_AGE_SECONDS]
        for gid in old:
            rec = self.records.pop(gid, None)
            if rec:
                self.local_to_global.pop(rec.local_track_id, None)
