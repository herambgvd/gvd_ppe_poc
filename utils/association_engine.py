"""
# ======================================
# ASSOCIATION ENGINE
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- Decides which PPE item belongs to which tracked person.
- Converts raw detections into worker-level PPE state.
- Uses:
    - body-region logic
    - IoU scoring
    - center-distance scoring
    - temporal consistency
    - confidence weighting
    - ReID-aware person metadata

Enterprise architecture:
- YOLO detects objects independently.
- Association layer converts:
    raw detections
        →
    worker understanding.
"""

from __future__ import annotations

from collections import defaultdict, deque

from dataclasses import dataclass, field

from typing import Deque, Dict, Iterable, List, Optional, Tuple

from .config import CONFIG
from .detector import Detection


# ======================================
# TYPES
# ======================================

BBox = Tuple[float, float, float, float]


# ======================================
# PERSON ASSOCIATION
# ======================================

@dataclass
class PersonAssociation:

    # ======================================
    # PERSON
    # ======================================

    person: Detection

    # ======================================
    # POSITIVE PPE
    # ======================================

    ppe: Dict[str, List[Detection]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # ======================================
    # NEGATIVE PPE
    # ======================================

    negative_ppe: Dict[str, List[Detection]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # ======================================
    # ASSOCIATION SCORES
    # ======================================

    scores: Dict[str, float] = field(
        default_factory=dict
    )

    # ======================================
    # PPE CHECK
    # ======================================

    def has_ppe(
        self,
        canonical_class: str
    ) -> bool:

        return (

            len(
                self.ppe.get(
                    canonical_class,
                    []
                )
            ) > 0
        )

    # ======================================
    # NEGATIVE PPE CHECK
    # ======================================

    def has_negative(
        self,
        missing_class: str
    ) -> bool:

        return (

            len(
                self.negative_ppe.get(
                    missing_class,
                    []
                )
            ) > 0
        )

    # ======================================
    # COMPLIANCE SUMMARY
    # ======================================

    def compliance_summary(self):

        return {

            "helmet": self.has_ppe(
                "helmet"
            ),

            "vest": self.has_ppe(
                "vest"
            ),

            "gloves": self.has_ppe(
                "gloves"
            ),

            "goggles": self.has_ppe(
                "goggles"
            ),

            "boots": self.has_ppe(
                "boots"
            ),

            "mask": self.has_ppe(
                "mask"
            ),
        }

    # ======================================
    # SERIALIZATION
    # ======================================

    def as_dict(self) -> Dict:

        return {

            # ======================================
            # TRACK ID
            # ======================================

            "track_id": (
                self.person.track_id
            ),

            # ======================================
            # REID GLOBAL ID
            # ======================================

            "reid_global_id": (

                self.person.metadata.get(
                    "reid_global_id"
                )
            ),

            # ======================================
            # PERSON DATA
            # ======================================

            "person": (
                self.person.as_dict()
            ),

            # ======================================
            # POSITIVE PPE
            # ======================================

            "ppe": {

                k: [
                    d.as_dict()
                    for d in v
                ]

                for k, v in (
                    self.ppe.items()
                )
            },

            # ======================================
            # NEGATIVE PPE
            # ======================================

            "negative_ppe": {

                k: [
                    d.as_dict()
                    for d in v
                ]

                for k, v in (
                    self.negative_ppe.items()
                )
            },

            # ======================================
            # SCORES
            # ======================================

            "scores": (
                self.scores
            ),

            # ======================================
            # COMPLIANCE
            # ======================================

            "compliance": (
                self.compliance_summary()
            ),
        }


# ======================================
# ASSOCIATION ENGINE
# ======================================

class AssociationEngine:

    # ======================================
    # BODY REGIONS
    # ======================================

    BODY_REGION_BY_PPE = {

        "helmet": "head",

        "no_helmet": "head",

        "goggles": "head",

        "no_goggles": "head",

        "mask": "head",

        "no_mask": "head",

        "vest": "torso",

        "no_vest": "torso",

        "gloves": "hands",

        "no_gloves": "hands",

        "boots": "feet",

        "no_boots": "feet",
    }

    # ======================================
    # INIT
    # ======================================

    def __init__(
        self,
        camera_id: str
    ):

        self.camera_id = camera_id

        # ======================================
        # TEMPORAL ASSOCIATION CACHE
        # ======================================

        self.temporal_cache: Dict[
            str,
            int
        ] = {}

        # ======================================
        # FRAME HISTORY
        # ======================================

        self.history: Deque[
            Dict[str, int]
        ] = deque(maxlen=30)

    # ======================================
    # ASSOCIATE PPE
    # ======================================

    def associate(

        self,

        detections: List[Detection]

    ) -> List[PersonAssociation]:

        # ======================================
        # PERSONS
        # ======================================

        people = [

            d

            for d in detections

            if (
                d.canonical_class
                ==
                CONFIG.PERSON_CLASS
            )
        ]

        # ======================================
        # PPE ITEMS
        # ======================================

        items = [

            d

            for d in detections

            if (

                d.canonical_class
                in
                CONFIG.PPE_CLASSES

                or

                d.canonical_class
                in
                CONFIG.NEGATIVE_PPE_CLASSES
            )
        ]

        # ======================================
        # PERSON ASSOCIATIONS
        # ======================================

        associations = {

            p.track_id: PersonAssociation(
                person=p
            )

            for p in people

            if p.track_id is not None
        }

        if not people or not items:

            return list(
                associations.values()
            )

        # ======================================
        # FRAME ASSIGNMENTS
        # ======================================

        frame_assignments: Dict[
            str,
            int
        ] = {}

        # ======================================
        # MATCH PPE ITEMS
        # ======================================

        for item in items:

            best_person: Optional[
                Detection
            ] = None

            best_score = 0.0

            # ======================================
            # SCORE ALL PERSONS
            # ======================================

            for person in people:

                score = self._score_pair(
                    person,
                    item
                )

                if score > best_score:

                    best_score = score

                    best_person = person

            # ======================================
            # ACCEPT MATCH
            # ======================================

            if (

                best_person is not None

                and

                best_person.track_id is not None

                and

                best_score
                >=
                CONFIG.ASSOCIATION_MIN_SCORE
            ):

                pa = associations.get(
                    best_person.track_id
                )

                if pa is None:

                    pa = PersonAssociation(
                        best_person
                    )

                    associations[
                        best_person.track_id
                    ] = pa

                # ======================================
                # NEGATIVE PPE
                # ======================================

                if item.canonical_class.startswith(
                    "no_"
                ):

                    pa.negative_ppe[
                        item.canonical_class
                    ].append(item)

                # ======================================
                # POSITIVE PPE
                # ======================================

                else:

                    pa.ppe[
                        item.canonical_class
                    ].append(item)

                # ======================================
                # STORE SCORE
                # ======================================

                pa.scores[
                    item.canonical_class
                ] = max(

                    pa.scores.get(
                        item.canonical_class,
                        0
                    ),

                    round(
                        best_score,
                        4
                    )
                )

                # ======================================
                # TEMPORAL CACHE
                # ======================================

                cache_key = (
                    self._cache_key(item)
                )

                frame_assignments[
                    cache_key
                ] = best_person.track_id

                self.temporal_cache[
                    cache_key
                ] = best_person.track_id

        # ======================================
        # SAVE HISTORY
        # ======================================

        self.history.append(
            frame_assignments
        )

        return list(
            associations.values()
        )

    # ======================================
    # SCORE PERSON + PPE
    # ======================================

    def _score_pair(

        self,

        person: Detection,

        item: Detection

    ) -> float:

        region_name = (

            self.BODY_REGION_BY_PPE.get(
                item.canonical_class,
                "torso"
            )
        )

        region_box = self.body_region(
            person.bbox,
            region_name
        )

        # ======================================
        # REGION IOU
        # ======================================

        region_iou = iou(
            region_box,
            item.bbox
        )

        # ======================================
        # PERSON OVERLAP
        # ======================================

        person_overlap = (

            intersection_area(
                person.bbox,
                item.bbox
            )

            /

            max(item.area, 1.0)
        )

        # ======================================
        # DISTANCE SCORE
        # ======================================

        distance_score = (
            normalized_center_distance_score(
                region_box,
                item.bbox
            )
        )

        # ======================================
        # TEMPORAL BONUS
        # ======================================

        temporal_bonus = (

            CONFIG.TEMPORAL_ASSOC_BONUS

            if (
                self.temporal_cache.get(
                    self._cache_key(item)
                )
                ==
                person.track_id
            )

            else 0.0
        )

        # ======================================
        # FINAL SCORE
        # ======================================

        score = (

            0.42 * region_iou

            +

            0.22 * person_overlap

            +

            0.18 * distance_score

            +

            0.10 * item.conf

            +

            temporal_bonus
        )

        return float(
            min(score, 1.0)
        )

    # ======================================
    # CACHE KEY
    # ======================================

    def _cache_key(
        self,
        item: Detection
    ) -> str:

        if item.track_id is not None:

            return (

                f"{self.camera_id}:"

                f"{item.canonical_class}:"

                f"tid:{item.track_id}"
            )

        cx, cy = item.center

        return (

            f"{self.camera_id}:"

            f"{item.canonical_class}:"

            f"grid:{int(cx//32)}:"

            f"{int(cy//32)}"
        )

    # ======================================
    # BODY REGIONS
    # ======================================

    @staticmethod
    def body_region(

        person_box: BBox,

        region: str

    ) -> BBox:

        x1, y1, x2, y2 = person_box

        w, h = (
            x2 - x1,
            y2 - y1
        )

        # ======================================
        # HEAD
        # ======================================

        if region == "head":

            return (

                x1 + 0.18 * w,

                y1,

                x2 - 0.18 * w,

                y1 + 0.28 * h
            )

        # ======================================
        # TORSO
        # ======================================

        if region == "torso":

            return (

                x1 + 0.08 * w,

                y1 + 0.22 * h,

                x2 - 0.08 * w,

                y1 + 0.72 * h
            )

        # ======================================
        # HANDS
        # ======================================

        if region == "hands":

            return (

                x1 - 0.05 * w,

                y1 + 0.25 * h,

                x2 + 0.05 * w,

                y1 + 0.78 * h
            )

        # ======================================
        # FEET
        # ======================================

        if region == "feet":

            return (

                x1 + 0.05 * w,

                y1 + 0.68 * h,

                x2 - 0.05 * w,

                y2
            )

        return person_box


# ======================================
# INTERSECTION AREA
# ======================================

def intersection_area(

    a: BBox,

    b: BBox

) -> float:

    x1 = max(a[0], b[0])

    y1 = max(a[1], b[1])

    x2 = min(a[2], b[2])

    y2 = min(a[3], b[3])

    return (

        max(0.0, x2 - x1)

        *

        max(0.0, y2 - y1)
    )


# ======================================
# IOU
# ======================================

def iou(

    a: BBox,

    b: BBox

) -> float:

    inter = intersection_area(a, b)

    area_a = (

        max(0.0, a[2] - a[0])

        *

        max(0.0, a[3] - a[1])
    )

    area_b = (

        max(0.0, b[2] - b[0])

        *

        max(0.0, b[3] - b[1])
    )

    union = area_a + area_b - inter

    return (

        float(inter / union)

        if union > 0

        else 0.0
    )


# ======================================
# DISTANCE SCORE
# ======================================

def normalized_center_distance_score(

    region: BBox,

    item: BBox

) -> float:

    rx = (
        region[0] + region[2]
    ) / 2

    ry = (
        region[1] + region[3]
    ) / 2

    ix = (
        item[0] + item[2]
    ) / 2

    iy = (
        item[1] + item[3]
    ) / 2

    rw = max(
        1.0,
        region[2] - region[0]
    )

    rh = max(
        1.0,
        region[3] - region[1]
    )

    norm_dist = (

        (
            ((rx - ix) / rw) ** 2
            +
            ((ry - iy) / rh) ** 2
        ) ** 0.5
    )

    return float(

        max(
            0.0,
            1.0 - norm_dist
        )
    )