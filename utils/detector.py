"""
# ======================================
# DETECTOR
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module wraps the trained YOLO model (`best.pt`) behind a clean inference API.
- It converts raw Ultralytics outputs into normalized enterprise detection objects used by:
    - tracking
    - association
    - compliance engine
    - analytics
    - UI
- The rest of the system should not know about YOLO tensor internals.
- This isolation makes:
    - model swaps
    - TensorRT export
    - ONNX deployment
    - future upgrades
  much easier.

Enterprise architecture reason:
- Production CV systems frequently swap models.
- The business logic layer should remain stable.
- Each camera owns an isolated detector instance so ByteTrack state does not leak across streams.
"""

from __future__ import annotations

import threading

from dataclasses import dataclass, field

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .config import CONFIG, normalize_class_name
from .logger import get_logger

logger = get_logger("ppe.detector")


# ======================================
# DETECTION OBJECT
# ======================================

@dataclass
class Detection:

    x1: float
    y1: float
    x2: float
    y2: float

    conf: float

    class_id: int

    class_name: str

    canonical_class: str

    track_id: Optional[int] = None

    metadata: Dict = field(default_factory=dict)

    # ======================================
    # BBOX
    # ======================================

    @property
    def bbox(self) -> Tuple[float, float, float, float]:

        return (
            self.x1,
            self.y1,
            self.x2,
            self.y2
        )

    # ======================================
    # AREA
    # ======================================

    @property
    def area(self) -> float:

        return (

            max(0.0, self.x2 - self.x1)

            *

            max(0.0, self.y2 - self.y1)

        )

    # ======================================
    # CENTER
    # ======================================

    @property
    def center(self) -> Tuple[float, float]:

        return (

            (self.x1 + self.x2) / 2.0,

            (self.y1 + self.y2) / 2.0

        )

    # ======================================
    # SERIALIZATION
    # ======================================

    def as_dict(self) -> Dict:

        return {

            "bbox": [

                round(self.x1, 2),

                round(self.y1, 2),

                round(self.x2, 2),

                round(self.y2, 2)

            ],

            "conf": round(float(self.conf), 4),

            "class_id": int(self.class_id),

            "class_name": self.class_name,

            "canonical_class": self.canonical_class,

            "track_id": self.track_id,

            "metadata": self.metadata,
        }


# ======================================
# PPE DETECTOR
# ======================================

class PPEDetector:

    # ======================================
    # INIT
    # ======================================

    def __init__(

        self,

        model_path=CONFIG.MODEL_PATH,

        tracker_path=CONFIG.TRACKER_CONFIG_PATH

    ):

        self.model_path = str(model_path)

        self.tracker_path = str(tracker_path)

        self._model = None

        self._model_lock = threading.RLock()

        self.names: Dict[int, str] = {}

        self._load_model()

    # ======================================
    # LOAD MODEL
    # ======================================

    def _load_model(self) -> None:

        try:

            from ultralytics import YOLO

            self._model = YOLO(
                self.model_path
            )

            self.names = (
                self._extract_names()
            )

            logger.info(

                "Loaded YOLO model from %s with classes: %s",

                self.model_path,

                self.names

            )

        except Exception as exc:

            logger.exception(

                "Failed to load YOLO model: %s",

                exc

            )

            raise RuntimeError(

                "Could not load best.pt. "
                "Install ultralytics/torch "
                "and confirm best.pt exists."

            ) from exc

    # ======================================
    # EXTRACT CLASS NAMES
    # ======================================

    def _extract_names(self) -> Dict[int, str]:

        names = (
            getattr(self._model, "names", {})
            or
            {}
        )

        if isinstance(names, list):

            return {

                i: n

                for i, n in enumerate(names)

            }

        return {

            int(k): str(v)

            for k, v in names.items()

        }

    # ======================================
    # IMAGE DETECTION
    # ======================================

    def predict_image(
        self,
        frame: np.ndarray
    ) -> List[Detection]:

        """
        Plain detection for:
        - images
        - snapshots
        - offline analysis
        """

        with self._model_lock:

            results = self._model.predict(

                source=frame,

                conf=CONFIG.CONF_THRESHOLD,

                iou=CONFIG.IOU_THRESHOLD,

                imgsz=CONFIG.IMG_SIZE,

                max_det=CONFIG.MAX_DETECTIONS,

                device=(
                    None
                    if CONFIG.DEVICE == "auto"
                    else CONFIG.DEVICE
                ),

                verbose=False
            )

        return self._parse_results(results)

    # ======================================
    # VIDEO TRACKING
    # ======================================

    def track_frame(

        self,

        frame: np.ndarray,

        persist: bool = True

    ) -> List[Detection]:

        """
        YOLO + ByteTrack pipeline.

        persist=True:
        keeps tracker state across frames.
        """

        with self._model_lock:

            results = self._model.track(

                source=frame,

                persist=persist,

                tracker=self.tracker_path,

                conf=CONFIG.CONF_THRESHOLD,

                iou=CONFIG.IOU_THRESHOLD,

                imgsz=CONFIG.IMG_SIZE,

                max_det=CONFIG.MAX_DETECTIONS,

                device=(

                    None

                    if CONFIG.DEVICE == "auto"

                    else CONFIG.DEVICE
                ),

                verbose=False
            )

        return self._parse_results(results)

    # ======================================
    # PARSE YOLO RESULTS
    # ======================================

    def _parse_results(
        self,
        results
    ) -> List[Detection]:

        detections: List[Detection] = []

        if not results:
            return detections

        result = results[0]

        boxes = getattr(
            result,
            "boxes",
            None
        )

        if boxes is None or len(boxes) == 0:
            return detections

        # ======================================
        # BOX DATA
        # ======================================

        xyxy = (

            boxes.xyxy.detach().cpu().numpy()

            if hasattr(boxes.xyxy, "detach")

            else np.asarray(boxes.xyxy)

        )

        confs = (

            boxes.conf.detach().cpu().numpy()

            if hasattr(boxes.conf, "detach")

            else np.asarray(boxes.conf)

        )

        clss = (

            boxes.cls.detach().cpu().numpy().astype(int)

            if hasattr(boxes.cls, "detach")

            else np.asarray(boxes.cls).astype(int)

        )

        # ======================================
        # TRACK IDS
        # ======================================

        ids = None

        if getattr(boxes, "id", None) is not None:

            ids = (

                boxes.id.detach().cpu().numpy().astype(int)

                if hasattr(boxes.id, "detach")

                else np.asarray(boxes.id).astype(int)

            )

        # ======================================
        # BUILD DETECTIONS
        # ======================================

        for i, bbox in enumerate(xyxy):

            cls_id = int(clss[i])

            class_name = self.names.get(
                cls_id,
                str(cls_id)
            )

            canonical = normalize_class_name(
                class_name
            )

            # ======================================
            # PER-CLASS CONFIDENCE GATE
            # persons -> high recall, PPE -> stricter (kills false positives)
            # ======================================

            conf_i = float(confs[i])

            if canonical == CONFIG.PERSON_CLASS:
                thr = CONFIG.PERSON_CONF_THRESHOLD
            elif canonical in CONFIG.NEGATIVE_PPE_CLASSES:
                thr = CONFIG.NEG_PPE_CONF_THRESHOLD
            elif canonical in CONFIG.PPE_CLASSES:
                thr = CONFIG.PPE_CONF_THRESHOLD
            else:
                thr = CONFIG.PPE_CONF_THRESHOLD

            if conf_i < thr:
                continue

            detections.append(

                Detection(

                    x1=float(bbox[0]),

                    y1=float(bbox[1]),

                    x2=float(bbox[2]),

                    y2=float(bbox[3]),

                    conf=float(confs[i]),

                    class_id=cls_id,

                    class_name=class_name,

                    canonical_class=canonical,

                    track_id=(
                        int(ids[i])
                        if ids is not None
                        else None
                    ),
                )
            )

        return detections


# ======================================
# DRAW DETECTIONS  (enterprise annotation renderer)
# ======================================

# Enterprise palette (BGR)
_COLOR_VIOLATION = (60, 60, 240)     # red
_COLOR_COMPLIANT = (110, 210, 90)    # green
_COLOR_PERSON = (255, 170, 40)       # amber/blue for neutral person
_COLOR_PPE = (200, 200, 70)          # teal for positive PPE items
_COLOR_NEG = (60, 60, 240)           # red for no_* classes
_COLOR_HUD_BG = (24, 18, 12)         # dark panel
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_rounded(img, p1, p2, color, thickness=1, radius=10, filled=False):
    """Draw a rectangle with rounded corners (approximation)."""
    x1, y1 = p1
    x2, y2 = p2
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if filled:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def _draw_corner_box(img, p1, p2, color, thickness=2, length=18):
    """Bounding box drawn with emphasized L-shaped corners + faint full outline."""
    x1, y1 = p1
    x2, y2 = p2
    # faint full outline for context
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    L = max(6, min(length, (x2 - x1) // 3, (y2 - y1) // 3))
    for (cx, cy, dx, dy) in (
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1),
    ):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, thickness, cv2.LINE_AA)


def _draw_chip(img, text, org, bg_color, text_color=_WHITE,
               scale=0.5, thickness=1, pad=6, alpha=1.0):
    """Draw a filled rounded label chip with centered text; returns its height."""
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, thickness)
    x, y = org
    h = th + base + pad
    w = tw + pad * 2
    # keep chip inside the frame
    x = max(0, min(x, img.shape[1] - w))
    y = max(h, y)
    p1 = (x, y - h)
    p2 = (x + w, y)
    if alpha < 1.0:
        overlay = img.copy()
        _draw_rounded(overlay, p1, p2, bg_color, radius=6, filled=True)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        _draw_rounded(img, p1, p2, bg_color, radius=6, filled=True)
    cv2.putText(img, text, (x + pad, y - base - pad // 2 + 1),
                _FONT, scale, text_color, thickness, cv2.LINE_AA)
    return h


def _draw_hud(img, persons, violations, compliant, s=1.0):
    """Top-left translucent summary banner."""
    lines = [
        ("neubit.ai", _WHITE, 0.6 * s, max(1, round(2 * s))),
        (f"Persons: {persons}   Compliant: {compliant}   Violations: {violations}",
         (170, 220, 255) if violations == 0 else (120, 160, 255), 0.5 * s, max(1, round(s))),
    ]
    pad = round(12 * s)
    widths, heights = [], []
    for text, _, sc, th in lines:
        (tw, thh), base = cv2.getTextSize(text, _FONT, sc, th)
        widths.append(tw)
        heights.append(thh + base)
    w = max(widths) + pad * 2
    h = sum(heights) + pad * 2 + (len(lines) - 1) * 6
    overlay = img.copy()
    _draw_rounded(overlay, (12, 12), (12 + w, 12 + h), _COLOR_HUD_BG,
                  radius=round(12 * s), filled=True)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    # accent bar
    accent = _COLOR_VIOLATION if violations else _COLOR_COMPLIANT
    cv2.rectangle(img, (12, 12), (12 + max(4, round(5 * s)), 12 + h), accent, -1)
    y = 12 + pad
    for (text, color, sc, th), hh in zip(lines, heights):
        y += hh
        cv2.putText(img, text, (12 + pad + 6, y - 4), _FONT, sc, color, th, cv2.LINE_AA)
        y += 6


# ======================================
# REGION OF INTEREST (ROI)
# ======================================

def _roi_polygon_px(roi, w, h):
    return np.array([[int(x * w), int(y * h)] for x, y in roi], dtype=np.int32)


def filter_by_roi(detections, roi, w, h):
    """
    Keep only detections whose anchor point lies inside the ROI polygon.
    - persons: anchored at feet (bottom-centre) — best for a ground zone
    - other classes: anchored at box centre
    `roi` is a list of normalised [x, y] points (0..1). Falsy / < 3 points = no filtering.
    """
    if not roi or len(roi) < 3:
        return list(detections)
    poly = _roi_polygon_px(roi, w, h)
    kept = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        if d.canonical_class == CONFIG.PERSON_CLASS:
            ax, ay = (x1 + x2) / 2.0, y2
        else:
            ax, ay = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if cv2.pointPolygonTest(poly, (float(ax), float(ay)), False) >= 0:
            kept.append(d)
    return kept


def draw_roi(frame, roi):
    """Draw the ROI polygon as a translucent green zone."""
    if not roi or len(roi) < 3:
        return frame
    h, w = frame.shape[:2]
    poly = _roi_polygon_px(roi, w, h)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], (70, 170, 90))
    cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
    cv2.polylines(frame, [poly], True, (110, 210, 90),
                  max(2, round(h / 540)), cv2.LINE_AA)
    return frame


# Single-letter glyphs for each PPE type shown as status icons.
_PPE_ICON = {
    "helmet": "H", "vest": "V", "gloves": "G",
    "boots": "B", "goggles": "O", "mask": "M",
}


def _draw_check(img, cx, cy, r, color, th):
    """Small check-mark centred at (cx, cy)."""
    cv2.line(img, (cx - r, cy), (cx - r // 3, cy + r), color, th, cv2.LINE_AA)
    cv2.line(img, (cx - r // 3, cy + r), (cx + r, cy - r), color, th, cv2.LINE_AA)


def _draw_cross(img, cx, cy, r, color, th):
    """Small cross centred at (cx, cy)."""
    cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), color, th, cv2.LINE_AA)
    cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), color, th, cv2.LINE_AA)


def _draw_ppe_icons(img, x, y, statuses, s):
    """
    Horizontal row of PPE status icons.
    Each icon: rounded chip, green = present, red = missing, with the PPE
    letter and a check / cross mark.
    """
    size = max(18, round(26 * s))
    gap = round(6 * s)
    fs = 0.42 * s
    th = max(1, round(1.4 * s))
    cx = x
    for letter, present in statuses:
        color = _COLOR_COMPLIANT if present else _COLOR_VIOLATION
        _draw_rounded(img, (cx, y), (cx + size, y + size), color, radius=round(6 * s), filled=True)
        # letter on the left half
        cv2.putText(img, letter, (cx + round(4 * s), y + size - round(7 * s)),
                    _FONT, fs, _WHITE, th, cv2.LINE_AA)
        # check / cross mark on the right half
        mcx = cx + size - round(7 * s)
        mcy = y + size // 2
        r = max(2, round(3.5 * s))
        if present:
            _draw_check(img, mcx, mcy, r, _WHITE, th)
        else:
            _draw_cross(img, mcx, mcy, r, _WHITE, th)
        cx += size + gap
    return size


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    violations: Optional[List[Dict]] = None,
    mandatory_ppe: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Icon-based annotation renderer:
    - person boxes coloured green (compliant) / red (violation)
    - a per-person PPE status icon row (helmet / vest ...): green = present, red = missing
    - PPE item / negative detections drawn as thin outlines (no clutter labels)
    - translucent HUD summary banner
    """
    out = frame.copy()
    detections = list(detections)
    ppe_items = mandatory_ppe or CONFIG.DEFAULT_MANDATORY_PPE

    s = max(0.5, min(2.2, out.shape[0] / 720.0))
    box_th = max(2, round(2 * s))
    corner_len = round(20 * s)

    # track_id -> set of missing PPE (from compliance violations)
    missing_by_track: Dict[str, set] = {}
    for v in (violations or []):
        tid = str(v.get("track_id"))
        vt = str(v.get("violation_type", ""))
        if vt.startswith("missing_"):
            missing_by_track.setdefault(tid, set()).add(vt[len("missing_"):])
    violation_track_ids = {str(v.get("track_id")) for v in (violations or [])}

    persons = compliant = 0

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        canonical = det.canonical_class
        is_person = canonical == CONFIG.PERSON_CLASS
        is_negative = canonical.startswith("no_")

        # ---- PPE items / negatives: not drawn; status is shown via person icons ----
        if not is_person:
            continue

        # ---- person ----
        persons += 1
        tid = str(det.track_id)
        is_viol = tid in violation_track_ids
        color = _COLOR_VIOLATION if is_viol else _COLOR_COMPLIANT
        if not is_viol:
            compliant += 1

        _draw_corner_box(out, (x1, y1), (x2, y2), color,
                         thickness=box_th, length=corner_len)

        # compact id chip
        label = f"#{det.track_id}" if det.track_id is not None else "PERSON"
        _draw_chip(out, label, (x1, y1), color, _WHITE,
                   scale=0.48 * s, thickness=max(1, round(s)), pad=round(6 * s))

        # PPE status icons (mandatory PPE): green = present, red = missing
        missing = missing_by_track.get(tid, set())
        statuses = [
            (_PPE_ICON.get(ppe, ppe[:1].upper()), ppe not in missing)
            for ppe in ppe_items
        ]
        icon_size = _draw_ppe_icons(out, x1, y2 + round(6 * s), statuses, s)

        # small ReID identity chip below the icons (no SIM, keeps it clean)
        gid = det.metadata.get("reid_global_id", "unknown")
        if gid and gid != "unknown":
            short = gid[-6:] if len(str(gid)) > 6 else str(gid)
            _draw_chip(out, f"ID {short}",
                       (x1, y2 + round(6 * s) + icon_size + round(20 * s)),
                       (40, 40, 40), (0, 255, 255), scale=0.4 * s,
                       thickness=max(1, round(s)), pad=round(5 * s), alpha=0.85)

    _draw_hud(out, persons, len(violation_track_ids), compliant, s=s)
    return out
