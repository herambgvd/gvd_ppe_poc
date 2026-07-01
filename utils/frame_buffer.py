"""
# ======================================
# FRAME BUFFER
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module provides a thread-safe queue between frame acquisition and AI processing.
- It solves the production problem where RTSP/Webcam capture speed and AI inference speed are different.
- Enterprise streaming systems use bounded queues to avoid memory leaks during camera spikes or GPU slowdown.

Design decision:
- When the queue is full, the oldest frame is dropped. For live safety monitoring, recent frames are more valuable than stale frames.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from time import time
from typing import Optional

import numpy as np

from .config import CONFIG


@dataclass
class FramePacket:
    frame: np.ndarray
    frame_id: int
    timestamp: float


class FrameBuffer:
    def __init__(self, maxsize: int = CONFIG.FRAME_BUFFER_SIZE):
        self.queue: "queue.Queue[FramePacket]" = queue.Queue(maxsize=maxsize)
        self.dropped_frames = 0
        self._lock = threading.Lock()

    def put(self, packet: FramePacket) -> None:
        with self._lock:
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass
            self.queue.put_nowait(packet)

    def get(self, timeout: float = 0.5) -> Optional[FramePacket]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def size(self) -> int:
        return self.queue.qsize()

    def clear(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
