"""
# ======================================
# VIDEO JOB MANAGER
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- Processes an uploaded video asynchronously in a background worker thread
  using the same detection / tracking / ReID / association / compliance
  pipeline as live streams.
- While the video is being analysed it exposes the latest annotated frame as
  an MJPEG stream so the browser can watch detection happen "live", side by
  side with real-time PPE violation alerts.
- When processing finishes it stores the rendered output video plus an
  aggregated, whole-video PPE compliance summary so the final result page can
  replay the annotated footage and show per-worker compliance.

Why this architecture:
- The previous implementation processed the upload synchronously inside the
  request, so the user saw nothing until the whole clip finished and the final
  page had no compliance summary. Moving the work to a tracked background job
  unlocks the live view and a rich result page without changing the core CV
  modules.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional

import cv2
import numpy as np

from .association_engine import AssociationEngine
from .compliance_engine import ComplianceEngine
from .config import CONFIG
from .detector import PPEDetector, draw_detections, draw_roi, filter_by_roi
from .event_manager import EventManager
from .logger import get_logger
from .reid_manager import ReIDManager
from .tracker_manager import TrackerManager

logger = get_logger("ppe.video_job")

REQUIRED_PPE = ["helmet", "vest"]

# A live alert is considered "active" while its person+violation was seen within
# this many processed frames; older ones drop off so the panel shows only the
# currently-violating workers (mirrors the Active Events concept).
ACTIVE_ALERT_WINDOW_FRAMES = 60


def _ffmpeg_exe() -> Optional[str]:
    """Locate an ffmpeg binary (system, or the bundled imageio-ffmpeg static build)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


class _FFmpegWriter:
    """
    Streams BGR frames to ffmpeg (libx264) and produces a browser-playable
    H.264 MP4 with a fast-start moov atom.

    OpenCV's own H.264 path on this host only exposes the hardware
    ``h264_v4l2m2m`` encoder, which fails to open and silently falls back to
    ``mp4v`` — a codec Chrome cannot decode (blank player). Encoding through
    ffmpeg's libx264 avoids that and also yields far smaller files. Closing the
    pipe early (a user stop) still finalises a valid, seekable partial video.
    """

    def __init__(self, exe: str, out_path: Path, fps: float, w: int, h: int):
        self.out_path = out_path
        self.w, self.h = w, h
        self.ok = False
        fps = float(fps) if fps and fps > 0 else 20.0
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps:.4f}",
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            # yuv420p needs even dimensions for broad browser support.
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to start ffmpeg writer: %s", exc)
            self.proc = None

    def write(self, frame: np.ndarray) -> None:
        if not self.ok or self.proc is None or self.proc.stdin is None:
            return
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h))
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, ValueError):
            self.ok = False

    def release(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=120)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass


# ======================================
# SINGLE VIDEO JOB
# ======================================

class VideoJob:

    def __init__(self, job_id: str, input_path: Path):
        self.job_id = job_id
        self.input_path = Path(input_path)

        # lifecycle
        self.state = "processing"  # processing | done | error | cancelled
        self.cancelled = False
        self.error: Optional[str] = None
        self.started_at = datetime.utcnow().isoformat()

        # progress
        self.total_frames = 0
        self.processed_frames = 0
        self.violation_count = 0

        # results
        self.output_path: Optional[str] = None     # web path to final video
        self.persons_summary: List[Dict] = []

        # live frame buffer
        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()

        # PPE enforced for this job (set from rules; defaults to helmet+vest)
        self.mandatory_ppe: List[str] = list(REQUIRED_PPE)

        # aggregation state
        self._person_state: Dict[str, Dict] = {}
        # Active live-alert store: key -> {payload..., "last_frame": int}
        self._alerts: Dict[tuple, Dict] = {}

    # ------------------------------------
    # LIVE FRAME
    # ------------------------------------

    def set_frame(self, frame: np.ndarray) -> None:
        # Downscale for a light, smooth live preview (output file keeps full res).
        h, w = frame.shape[:2]
        max_w = CONFIG.STREAM_PREVIEW_WIDTH
        if max_w and w > max_w:
            scale = max_w / float(w)
            frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, CONFIG.STREAM_JPEG_QUALITY],
        )
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    # ------------------------------------
    # STATUS SNAPSHOT
    # ------------------------------------

    @property
    def alerts(self) -> List[Dict]:
        """Currently-active violations only (recently seen), newest first."""
        cutoff = self.processed_frames - ACTIVE_ALERT_WINDOW_FRAMES
        active = [
            a for a in self._alerts.values()
            if a.get("last_frame", 0) >= cutoff
        ]
        active.sort(key=lambda a: a.get("last_frame", 0), reverse=True)
        return [
            {k: v for k, v in a.items() if k != "last_frame"}
            for a in active
        ]

    @property
    def progress(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return round(
            min(100.0, (self.processed_frames / self.total_frames) * 100.0),
            1,
        )

    def status(self) -> Dict:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "error": self.error,
            "progress": self.progress,
            "processed_frames": self.processed_frames,
            "total_frames": self.total_frames,
            "violation_count": self.violation_count,
            "alerts": self.alerts[:12],
            "result_url": (
                f"/video/result/{self.job_id}"
                if self.state in ("done", "cancelled")
                else None
            ),
        }

    def result(self) -> Dict:
        return {
            "job_id": self.job_id,
            "output_path": self.output_path,
            "frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "violation_count": self.violation_count,
            "persons_summary": self.persons_summary,
        }


# ======================================
# JOB MANAGER (SINGLETON)
# ======================================

class VideoJobManager:

    def __init__(self):
        self.jobs: Dict[str, VideoJob] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Optional[VideoJob]:
        with self._lock:
            return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Flag a running job to stop; the worker loop exits on the next frame."""
        job = self.get(job_id)
        if job and job.state == "processing":
            job.cancelled = True
            return True
        return False

    # ------------------------------------
    # START A JOB
    # ------------------------------------

    def start(self, input_path: Path, rules: Optional[Dict] = None) -> VideoJob:
        job_id = "video_" + uuid.uuid4().hex[:10]
        job = VideoJob(job_id, input_path)

        with self._lock:
            self.jobs[job_id] = job

        worker = threading.Thread(
            target=self._run,
            args=(job, rules or {}),
            daemon=True,
        )
        worker.start()

        return job

    # ------------------------------------
    # WORKER LOOP
    # ------------------------------------

    def _run(self, job: VideoJob, rules: Dict) -> None:
        try:
            self._process(job, rules)
            job.persons_summary = self._build_summary(job)
            job.state = "cancelled" if job.cancelled else "done"
            logger.info(
                "Video job %s %s (%s violations, %s persons)",
                job.job_id,
                "cancelled" if job.cancelled else "finished",
                job.violation_count,
                len(job.persons_summary),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Video job %s failed: %s", job.job_id, exc)
            job.error = str(exc)
            job.state = "error"

    def _process(self, job: VideoJob, rules: Dict) -> None:
        output_path = CONFIG.OUTPUT_VIDEO_DIR / f"{job.job_id}.mp4"

        cap = cv2.VideoCapture(str(job.input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {job.input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        job.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        writer = self._open_writer(output_path, fps, w, h)

        roi = (rules or {}).get("roi")
        mandatory = [p for p in ((rules or {}).get("mandatory_ppe") or REQUIRED_PPE)]
        job.mandatory_ppe = mandatory

        detector = PPEDetector()
        tracker = TrackerManager(job.job_id)
        reid = ReIDManager(job.job_id)
        association = AssociationEngine(job.job_id)
        compliance = ComplianceEngine(rules or {})
        events = EventManager(job.job_id, "video")

        frame_id = 0
        last_detections: list = []
        last_violations: list = []
        try:
            while True:
                if job.cancelled:
                    logger.info("Video job %s cancelled by user at frame %s", job.job_id, frame_id)
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                frame_id += 1
                job.processed_frames = frame_id

                if frame_id % max(1, CONFIG.DEFAULT_FRAME_SKIP) == 0:
                    detections = detector.track_frame(frame, persist=True)
                    detections = tracker.update(frame, detections)

                    if roi:
                        detections = filter_by_roi(detections, roi, w, h)

                    people = [
                        d for d in detections
                        if d.canonical_class == CONFIG.PERSON_CLASS
                    ]
                    reid.update_person_identities(frame, people)

                    associations = association.associate(detections)
                    violations = compliance.evaluate(associations)
                    for v in violations:
                        v["camera_id"] = job.job_id

                    annotated = draw_detections(frame, detections, violations, mandatory_ppe=mandatory)
                    if roi:
                        draw_roi(annotated, roi)
                    events.update(violations, annotated, frame)

                    self._register_persons(job, associations)
                    self._register_alerts(job, violations, frame, frame_id)
                    # Distinct violating (person + violation-type) count — a
                    # meaningful incident number, not a per-frame running total.
                    job.violation_count = len(job._alerts)

                    last_detections = detections
                    last_violations = violations

                    writer.write(annotated)
                    job.set_frame(annotated)
                else:
                    # Re-draw the most recent detections so annotations PERSIST on
                    # skipped frames instead of blinking off (the flicker cause).
                    annotated = draw_detections(frame, last_detections, last_violations)
                    if roi:
                        draw_roi(annotated, roi)
                    writer.write(annotated)
                    if frame_id % 2 == 0:
                        job.set_frame(annotated)
        finally:
            cap.release()
            writer.release()
            # Close out any events still ACTIVE at the last processed frame,
            # otherwise they stay stuck as ONGOING forever (no more update()).
            try:
                events.finalize()
            except Exception:  # noqa: BLE001
                logger.exception("Event finalize failed for job %s", job.job_id)

        if job.total_frames <= 0:
            job.total_frames = frame_id

        job.output_path = f"/static/outputs/videos/{output_path.name}"

    # ------------------------------------
    # VIDEO WRITER (browser-friendly codec)
    # ------------------------------------

    def _open_writer(self, output_path: Path, fps, w, h):
        # Prefer ffmpeg/libx264 so the result reliably plays in Chrome.
        exe = _ffmpeg_exe()
        if exe:
            writer = _FFmpegWriter(exe, output_path, fps, w, h)
            if writer.ok:
                logger.info("Video writer using ffmpeg libx264 (H.264)")
                return writer
            logger.warning("ffmpeg writer unavailable; falling back to OpenCV.")

        # Fallback: OpenCV writer (avc1 then mp4v). mp4v may not play in Chrome.
        for fourcc_name in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if writer.isOpened():
                logger.info("Video writer using codec %s", fourcc_name)
                return writer
            writer.release()
        raise RuntimeError("Could not open a video writer for the output file.")

    # ------------------------------------
    # PERSON AGGREGATION
    # ------------------------------------

    def _person_key(self, track_id, reid_global_id) -> str:
        if reid_global_id:
            return str(reid_global_id)
        return f"track_{track_id}"

    def _register_persons(self, job: VideoJob, associations) -> None:
        for assoc in associations:
            data = assoc.as_dict()
            reid_global_id = data.get("reid_global_id")
            track_id = data.get("track_id")
            key = self._person_key(track_id, reid_global_id)

            state = job._person_state.setdefault(
                key,
                {
                    "display_id": reid_global_id or track_id,
                    "missing": set(),
                    "order": len(job._person_state),
                },
            )

            ppe = data.get("ppe", {}) or {}
            for item in job.mandatory_ppe:
                has_item = item in ppe and len(ppe[item]) > 0
                if not has_item:
                    state["missing"].add(item)

    def _build_summary(self, job: VideoJob) -> List[Dict]:
        persons = []
        ordered = sorted(
            job._person_state.items(),
            key=lambda kv: kv[1]["order"],
        )
        for index, (_key, state) in enumerate(ordered):
            items = {}
            violations = []
            for item in job.mandatory_ppe:
                has_item = item not in state["missing"]
                items[item] = has_item
                if not has_item:
                    violations.append(f"{item.title()} Missing")

            persons.append(
                {
                    "person_number": index + 1,
                    "track_id": state["display_id"],
                    "items": items,
                    "violations": violations,
                    "compliant": len(violations) == 0,
                }
            )
        return persons

    # ------------------------------------
    # LIVE ALERTS
    # ------------------------------------

    def _register_alerts(self, job: VideoJob, violations, frame, frame_id) -> None:
        for v in violations:
            reid_global_id = v.get("reid_global_id")
            track_id = v.get("track_id")
            key = (
                self._person_key(track_id, reid_global_id),
                v.get("violation_type"),
            )

            existing = job._alerts.get(key)
            if existing is not None:
                # Same person still violating — keep it active, refresh the crop.
                existing["last_frame"] = frame_id
                snapshot = self._save_crop(job, frame, v.get("person_bbox"), key)
                if snapshot:
                    existing["snapshot"] = snapshot
                existing["created_at"] = v.get("timestamp", existing.get("created_at", ""))
                continue

            snapshot = self._save_crop(job, frame, v.get("person_bbox"), key)
            job._alerts[key] = {
                "track_id": reid_global_id or track_id,
                "violation_type": (
                    v.get("violation_type", "PPE Violation")
                    .replace("_", " ")
                    .title()
                ),
                "created_at": v.get("timestamp", ""),
                "snapshot": snapshot,
                "last_frame": frame_id,
            }

    def _save_crop(self, job, frame, bbox, key) -> Optional[str]:
        if not bbox:
            return None
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            out_dir = CONFIG.VIOLATION_DIR / "video"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Stable, filesystem-safe name per person+violation.
            slug = "".join(c if c.isalnum() else "_" for c in "_".join(str(k) for k in key))
            fname = f"{job.job_id}_{slug}.jpg"
            cv2.imwrite(str(out_dir / fname), crop)
            return f"/static/violations/video/{fname}"
        except Exception:  # noqa: BLE001
            return None


# ======================================
# SINGLETON INSTANCE
# ======================================

VIDEO_JOBS = VideoJobManager()


def frame_generator(job_id: str) -> Generator[bytes, None, None]:
    """MJPEG generator that streams the latest annotated frame of a job."""

    boundary_open = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"

    placeholder = _placeholder_jpeg()
    last = None
    idle = 0

    # prime the <img> immediately
    yield boundary_open + placeholder + b"\r\n"

    while True:
        job = VIDEO_JOBS.get(job_id)
        if job is None:
            break

        frame = job.get_frame()
        if frame is not None and frame is not last:
            # push new annotated frames the moment they are ready
            yield boundary_open + frame + b"\r\n"
            last = frame
            idle = 0
        else:
            idle += 1
            if idle >= 25:  # ~1s keep-alive so the connection stays warm
                yield boundary_open + (last or placeholder) + b"\r\n"
                idle = 0

        if job.state != "processing":
            final = job.get_frame() or placeholder
            yield boundary_open + final + b"\r\n"
            break

        time.sleep(0.04)


def _placeholder_jpeg() -> bytes:
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[:] = (17, 24, 39)
    cv2.putText(
        canvas,
        "Preparing video...",
        (130, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (148, 163, 184),
        2,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", canvas)
    return buf.tobytes() if ok else b""
