"""
# ======================================
# CONFIG
# ======================================

# ======================================
# PURPOSE
# ======================================
This module centralizes all application configuration.

Why this module exists:
- Enterprise applications must not scatter thresholds, paths, and business rules across code files.
- Centralized config makes deployment predictable across laptop, GPU workstation, and server environments.
- SQLite can later be replaced by PostgreSQL without changing the detection pipeline modules.

What problem it solves:
- Prevents hard-coded model paths, queue sizes, confidence thresholds, and compliance rules.
- Gives one place to tune detection confidence, tracking, persistence, cooldown, and streaming behavior.

Why this architecture is used in enterprise systems:
- Separates environment-specific configuration from business logic.
- Makes CI/CD, staging, production, and client-specific deployments easier to manage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


BASE_DIR = Path(__file__).resolve().parents[1]

# Load environment overrides from a local .env file (production-friendly).
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except Exception:  # noqa: BLE001 - dotenv is optional
    pass


@dataclass(frozen=True)
class AppConfig:
    # Core paths
    BASE_DIR: Path = BASE_DIR
    MODEL_PATH: Path = BASE_DIR / "best.pt"
    TRACKER_CONFIG_PATH: Path = BASE_DIR / "custom_bytetrack.yaml"
    DB_PATH: Path = BASE_DIR / "ppe.db"
    LOG_DIR: Path = BASE_DIR / "logs"
    STATIC_DIR: Path = BASE_DIR / "static"

    UPLOAD_IMAGE_DIR: Path = BASE_DIR / "static" / "uploads" / "images"
    UPLOAD_VIDEO_DIR: Path = BASE_DIR / "static" / "uploads" / "videos"
    OUTPUT_IMAGE_DIR: Path = BASE_DIR / "static" / "outputs" / "images"
    OUTPUT_VIDEO_DIR: Path = BASE_DIR / "static" / "outputs" / "videos"
    VIOLATION_DIR: Path = BASE_DIR / "static" / "violations"

    # Runtime
    SECRET_KEY: str = os.getenv("PPE_SECRET_KEY", "change-this-in-production")
    HOST: str = os.getenv("PPE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PPE_PORT", "5000"))
    DEBUG: bool = os.getenv("PPE_DEBUG", "false").lower() == "true"

    # Detection
    DEVICE: str = os.getenv("PPE_DEVICE", "auto")  # auto, cpu, 0, cuda:0
    # Higher inference resolution greatly improves recall on distant / crowded
    # workers: at 960px most far people score <0.45 confidence and get dropped;
    # at 1536px their confidence roughly doubles so they are actually detected
    # and tracked. Costs ~2.5x compute (fine on the RTX 5070 with frame-skip).
    IMG_SIZE: int = int(os.getenv("PPE_IMG_SIZE", "1536"))
    # Base floor passed to the model (kept low for recall); real gating is per-class below.
    CONF_THRESHOLD: float = float(os.getenv("PPE_CONF_THRESHOLD", "0.25"))
    IOU_THRESHOLD: float = float(os.getenv("PPE_IOU_THRESHOLD", "0.55"))
    MAX_DETECTIONS: int = int(os.getenv("PPE_MAX_DETECTIONS", "300"))

    # Per-class confidence gates applied after inference:
    # - persons: kept lower so occluded / multiple people are still detected
    # - PPE: kept higher to suppress false positives (e.g. a helmet on a door handle)
    PERSON_CONF_THRESHOLD: float = float(os.getenv("PPE_PERSON_CONF", "0.35"))
    PPE_CONF_THRESHOLD: float = float(os.getenv("PPE_PPE_CONF", "0.50"))
    NEG_PPE_CONF_THRESHOLD: float = float(os.getenv("PPE_NEG_CONF", "0.50"))

    # Minimum person bounding-box area (px^2) required before a compliance
    # violation is raised — stops far-away / tiny detections from creating events.
    MIN_PERSON_BBOX_AREA: int = int(os.getenv("PPE_MIN_PERSON_AREA", "9000"))

    # Temporal confirmation of missing PPE (per tracked worker). A single frame
    # where a vest/helmet fails to associate (worker turned sideways, briefly
    # occluded) must NOT raise a violation. We only flag missing PPE when it is
    # absent in at least MIN_MISSING of the last WINDOW observed frames AND it is
    # currently absent — this kills transient false "missing_vest" events.
    PPE_CONFIRM_WINDOW: int = int(os.getenv("PPE_CONFIRM_WINDOW", "8"))
    PPE_CONFIRM_MIN_MISSING: int = int(os.getenv("PPE_CONFIRM_MIN_MISSING", "6"))

    # Streaming / performance
    FRAME_BUFFER_SIZE: int = int(os.getenv("PPE_FRAME_BUFFER_SIZE", "8"))
    DEFAULT_FRAME_SKIP: int = int(os.getenv("PPE_FRAME_SKIP", "2"))
    TARGET_STREAM_FPS: int = int(os.getenv("PPE_STREAM_FPS", "12"))
    JPEG_QUALITY: int = int(os.getenv("PPE_JPEG_QUALITY", "80"))
    # Live MJPEG preview is downscaled + lighter-compressed for a smooth stream
    # (the saved output video keeps full resolution / quality).
    STREAM_PREVIEW_WIDTH: int = int(os.getenv("PPE_STREAM_PREVIEW_WIDTH", "960"))
    STREAM_JPEG_QUALITY: int = int(os.getenv("PPE_STREAM_JPEG_QUALITY", "70"))
    CAPTURE_RECONNECT_SECONDS: float = float(os.getenv("PPE_CAPTURE_RECONNECT_SECONDS", "5"))

    # Event lifecycle
    VIOLATION_PERSISTENCE_FRAMES: int = int(os.getenv("PPE_PERSISTENCE_FRAMES", "5"))
    EVENT_RESOLVE_AFTER_SECONDS: float = float(os.getenv("PPE_RESOLVE_AFTER_SECONDS", "4"))
    EVENT_EXPIRE_AFTER_SECONDS: float = float(os.getenv("PPE_EXPIRE_AFTER_SECONDS", "120"))
    DUPLICATE_COOLDOWN_SECONDS: float = float(os.getenv("PPE_DUPLICATE_COOLDOWN_SECONDS", "30"))

    # Association thresholds
    ASSOCIATION_MIN_SCORE: float = float(os.getenv("PPE_ASSOCIATION_MIN_SCORE", "0.28"))
    TEMPORAL_ASSOC_BONUS: float = float(os.getenv("PPE_TEMPORAL_ASSOC_BONUS", "0.12"))

    # ReID thresholds
    REID_ENABLED: bool = os.getenv("PPE_REID_ENABLED", "true").lower() == "true"
    REID_SIMILARITY_THRESHOLD: float = float(os.getenv("PPE_REID_SIMILARITY_THRESHOLD", "0.90"))
    REID_MAX_AGE_SECONDS: float = float(os.getenv("PPE_REID_MAX_AGE_SECONDS", "12"))
    # Re-extract the OSNet embedding for an already-tracked person only every N
    # processed frames (keeps ReID accurate while staying fast).
    REID_EMBED_REFRESH_FRAMES: int = int(os.getenv("PPE_REID_EMBED_REFRESH", "15"))

    # Compliance rules; these can be overridden per camera via DB rules_json.
    DEFAULT_MANDATORY_PPE: List[str] = field(default_factory=lambda: ["helmet", "vest"])
    DEFAULT_OPTIONAL_PPE: List[str] = field(default_factory=lambda: ["gloves", "goggles", "boots", "mask"])

    # Full set of PPE the UI lets you enable per camera / video. Add a type here
    # (and ensure the model detects it) to scale up to an 8-PPE scenario — every
    # form, icon row and compliance check is driven off this list.
    SELECTABLE_PPE: List[str] = field(default_factory=lambda: ["helmet", "vest", "gloves", "goggles", "boots", "mask"])

    # Canonical classes expected by the pipeline.
    PERSON_CLASS: str = "person"
    PPE_CLASSES: Set[str] = field(default_factory=lambda: {"helmet", "vest", "gloves", "goggles", "boots", "mask"})
    NEGATIVE_PPE_CLASSES: Set[str] = field(default_factory=lambda: {"no_helmet", "no_vest", "no_gloves", "no_goggles", "no_boots", "no_mask"})

    # Alias map supports common YOLO dataset label names.
    CLASS_ALIASES: Dict[str, str] = field(default_factory=lambda: {
        "person": "person",
        "worker": "person",
        "hardhat": "helmet",
        "hard_hat": "helmet",
        "helmet": "helmet",
        "safety_helmet": "helmet",
        "no-hardhat": "no_helmet",
        "no_hardhat": "no_helmet",
        "no-helmet": "no_helmet",
        "no_helmet": "no_helmet",
        "safety vest": "vest",
        "safety_vest": "vest",
        "vest": "vest",
        "reflective_vest": "vest",
        "no-safety vest": "no_vest",
        "no_safety_vest": "no_vest",
        "no-vest": "no_vest",
        "no_vest": "no_vest",
        "glove": "gloves",
        "gloves": "gloves",
        "no-gloves": "no_gloves",
        "no_gloves": "no_gloves",
        "goggle": "goggles",
        "goggles": "goggles",
        "no-goggles": "no_goggles",
        "no_goggles": "no_goggles",
        "boot": "boots",
        "boots": "boots",
        "safety_boot": "boots",
        "safety_boots": "boots",
        "no-boots": "no_boots",
        "no_boots": "no_boots",
        "mask": "mask",
        "face_mask": "mask",
        "no-mask": "no_mask",
        "no_mask": "no_mask",
    })

    def ensure_dirs(self) -> None:
        for path in [
            self.LOG_DIR,
            self.UPLOAD_IMAGE_DIR,
            self.UPLOAD_VIDEO_DIR,
            self.OUTPUT_IMAGE_DIR,
            self.OUTPUT_VIDEO_DIR,
            self.VIOLATION_DIR / "webcam",
            self.VIOLATION_DIR / "rtsp",
            self.VIOLATION_DIR / "video",
            self.STATIC_DIR / "sounds",
        ]:
            path.mkdir(parents=True, exist_ok=True)


CONFIG = AppConfig()
CONFIG.ensure_dirs()


def normalize_class_name(raw_name: str) -> str:
    """Normalize model class labels to enterprise canonical class names."""
    key = str(raw_name).strip().lower().replace(" ", "_")
    key_hyphen = str(raw_name).strip().lower()
    return CONFIG.CLASS_ALIASES.get(key, CONFIG.CLASS_ALIASES.get(key_hyphen, key))
