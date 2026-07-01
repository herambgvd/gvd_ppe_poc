"""
# ======================================
# STATS
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module contains small runtime counters used by camera workers and dashboards.
- It keeps performance metrics consistent: FPS, processed frames, dropped frames, and last error.
- Enterprise monitoring requires health stats to identify camera, GPU, or network bottlenecks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Dict, Optional


@dataclass
class RuntimeStats:
    camera_id: str
    frames_read: int = 0
    frames_processed: int = 0
    frames_streamed: int = 0
    dropped_frames: int = 0
    last_error: Optional[str] = None
    started_at: float = field(default_factory=time)
    last_frame_ts: Optional[float] = None
    last_process_ts: Optional[float] = None

    def fps_read(self) -> float:
        elapsed = max(1e-6, time() - self.started_at)
        return round(self.frames_read / elapsed, 2)

    def fps_processed(self) -> float:
        elapsed = max(1e-6, time() - self.started_at)
        return round(self.frames_processed / elapsed, 2)

    def as_dict(self) -> Dict:
        return {
            "camera_id": self.camera_id,
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_streamed": self.frames_streamed,
            "dropped_frames": self.dropped_frames,
            "fps_read": self.fps_read(),
            "fps_processed": self.fps_processed(),
            "last_error": self.last_error,
            "uptime_seconds": round(time() - self.started_at, 2),
            "last_frame_ts": self.last_frame_ts,
            "last_process_ts": self.last_process_ts,
        }
