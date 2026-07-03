"""
# ======================================
# DATABASE
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module owns persistence for cameras, events, violations, analytics, and system logs.
- It hides SQLite implementation details behind a repository-style API.
- This architecture is enterprise-friendly because PostgreSQL migration later only requires replacing this module, not rewriting CV logic.

Key enterprise decisions:
- Uses WAL mode for better read/write concurrency in SQLite.
- Uses a process-level write lock because SQLite writes must be serialized when accessed by multiple camera threads.
- Stores JSON columns as TEXT today so the same data can map cleanly to PostgreSQL JSONB later.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import CONFIG
from .logger import get_logger

logger = get_logger("ppe.database")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: Path = CONFIG.DB_PATH):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self.init_db()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    rules_json TEXT,
                    zones_json TEXT,
                    status TEXT DEFAULT 'STOPPED',
                    last_seen TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    track_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    timestamp_start TEXT NOT NULL,
                    timestamp_end TEXT,
                    duration REAL DEFAULT 0,
                    screenshot_path TEXT,
                    crop_path TEXT,
                    confidence REAL DEFAULT 0,
                    evidence_json TEXT,
                    last_seen_ts TEXT,
                    cooldown_until TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_camera_state ON events(camera_id, state);
                CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(violation_type, timestamp_start);

                CREATE TABLE IF NOT EXISTS violations (
                    violation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    camera_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL DEFAULT 0,
                    screenshot_path TEXT,
                    crop_path TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );

                CREATE INDEX IF NOT EXISTS idx_violations_camera_time ON violations(camera_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_violations_type_time ON violations(violation_type, timestamp);

                CREATE TABLE IF NOT EXISTS analytics (
                    analytics_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bucket_start TEXT NOT NULL,
                    camera_id TEXT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS system_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata_json TEXT,
                    timestamp TEXT NOT NULL
                );
                """
            )
            # --- lightweight migrations for existing databases ---
            cols = {r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()}
            if "review_status" not in cols:
                # pending = not yet reviewed, confirmed = genuine, false = false alarm
                conn.execute("ALTER TABLE events ADD COLUMN review_status TEXT DEFAULT 'pending'")
        logger.info("Database initialized at %s", self.db_path)

    def reset_runtime_state(self) -> Dict[str, int]:
        """
        Reconcile persisted state at startup.

        On a fresh process no camera workers are running and no EventManager holds
        the previous session's events, so anything left RUNNING / ACTIVE in the DB
        is stale. Mark all cameras stopped and close out orphaned open events so
        the UI doesn't show forever-ONGOING incidents after a restart/crash.
        """
        now = utc_now_iso()
        with self._lock, self.connect() as conn:
            cams = conn.execute(
                "UPDATE cameras SET status='STOPPED', updated_at=? WHERE UPPER(status) != 'STOPPED'",
                (now,),
            ).rowcount
            evts = conn.execute(
                """
                UPDATE events
                SET state='RESOLVED',
                    timestamp_end=COALESCE(timestamp_end, last_seen_ts, updated_at, ?),
                    updated_at=?
                WHERE state IN ('NEW','ACTIVE')
                """,
                (now, now),
            ).rowcount
        return {"cameras_stopped": cams, "events_resolved": evts}

    def purge_events_older_than(self, days: float) -> Dict[str, Any]:
        """
        Delete events + violations older than `days`, returning the evidence
        image web-paths that the caller should unlink from disk. Timestamps are
        UTC ISO (utc_now_iso / event_manager iso), so a lexicographic compare
        against an isoformat cutoff is correct.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        files = set()
        with self._lock, self.connect() as conn:
            for tbl, tcol in (("events", "timestamp_start"), ("violations", "timestamp")):
                for r in conn.execute(
                    f"SELECT screenshot_path, crop_path FROM {tbl} WHERE {tcol} < ?", (cutoff,)
                ).fetchall():
                    for p in (r["screenshot_path"], r["crop_path"]):
                        if p:
                            files.add(p)
            ev = conn.execute("DELETE FROM events WHERE timestamp_start < ?", (cutoff,)).rowcount
            vi = conn.execute("DELETE FROM violations WHERE timestamp < ?", (cutoff,)).rowcount
        return {"events": ev, "violations": vi, "files": list(files), "cutoff": cutoff}

    def set_event_review(self, event_id: str, verdict: str) -> bool:
        """Mark an event as human-verified: 'confirmed', 'false', or 'pending'."""
        if verdict not in ("confirmed", "false", "pending"):
            return False
        now = utc_now_iso()
        with self._lock, self.connect() as conn:
            cur = conn.execute(
                "UPDATE events SET review_status=?, updated_at=? WHERE event_id=?",
                (verdict, now, event_id),
            )
            return cur.rowcount > 0

    def insert_camera(self, camera: Dict[str, Any]) -> None:
        now = utc_now_iso()
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cameras
                (camera_id, name, source_type, source_uri, enabled, rules_json, zones_json, status, last_seen, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT status FROM cameras WHERE camera_id=?), 'STOPPED'), NULL,
                        COALESCE((SELECT created_at FROM cameras WHERE camera_id=?), ?), ?)
                """,
                (
                    camera["camera_id"], camera["name"], camera["source_type"], camera["source_uri"],
                    int(camera.get("enabled", 1)), json.dumps(camera.get("rules", {})), json.dumps(camera.get("zones", [])),
                    camera["camera_id"], camera["camera_id"], now, now
                ),
            )

    def list_cameras(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM cameras ORDER BY created_at DESC").fetchall()
        return [self._row_to_camera(r) for r in rows]

    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM cameras WHERE camera_id=?", (camera_id,)).fetchone()
        return self._row_to_camera(row) if row else None

    def delete_camera(self, camera_id: str) -> None:
        with self._lock, self.connect() as conn:
            conn.execute("DELETE FROM cameras WHERE camera_id=?", (camera_id,))

    def update_camera(self, camera_id: str, name: str, source_type: str,
                      source_uri: str, rules: Dict[str, Any]) -> bool:
        """Edit an existing camera's details (name / source / rules)."""
        now = utc_now_iso()
        with self._lock, self.connect() as conn:
            cur = conn.execute(
                "UPDATE cameras SET name=?, source_type=?, source_uri=?, rules_json=?, updated_at=? WHERE camera_id=?",
                (name, source_type, source_uri, json.dumps(rules), now, camera_id),
            )
            return cur.rowcount > 0

    def update_camera_status(self, camera_id: str, status: str) -> None:
        now = utc_now_iso()
        with self._lock, self.connect() as conn:
            conn.execute(
                "UPDATE cameras SET status=?, last_seen=?, updated_at=? WHERE camera_id=?",
                (status, now, now, camera_id),
            )

    def set_camera_roi(self, camera_id: str, roi) -> bool:
        """Persist an ROI polygon (list of normalised [x, y] points) into rules_json."""
        cam = self.get_camera(camera_id)
        if not cam:
            return False
        rules = cam.get("rules") or {}
        if roi:
            rules["roi"] = roi
        else:
            rules.pop("roi", None)
        now = utc_now_iso()
        with self._lock, self.connect() as conn:
            conn.execute(
                "UPDATE cameras SET rules_json=?, updated_at=? WHERE camera_id=?",
                (json.dumps(rules), now, camera_id),
            )
        return True

    def upsert_event(self, event: Dict[str, Any]) -> None:
        now = utc_now_iso()
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO events
                (event_id, track_id, camera_id, violation_type, state, timestamp_start, timestamp_end,
                 duration, screenshot_path, crop_path, confidence, evidence_json, last_seen_ts,
                 cooldown_until, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    state=excluded.state,
                    timestamp_end=excluded.timestamp_end,
                    duration=excluded.duration,
                    screenshot_path=COALESCE(excluded.screenshot_path, events.screenshot_path),
                    crop_path=COALESCE(excluded.crop_path, events.crop_path),
                    confidence=excluded.confidence,
                    evidence_json=COALESCE(excluded.evidence_json, events.evidence_json),
                    last_seen_ts=excluded.last_seen_ts,
                    cooldown_until=excluded.cooldown_until,
                    updated_at=excluded.updated_at
                """,
                (
                    event["event_id"], str(event["track_id"]), event["camera_id"], event["violation_type"],
                    event["state"], event["timestamp_start"], event.get("timestamp_end"), float(event.get("duration", 0)),
                    event.get("screenshot_path"), event.get("crop_path"), float(event.get("confidence", 0)),
                    json.dumps(event.get("evidence", {})), event.get("last_seen_ts"), event.get("cooldown_until"),
                    event.get("created_at", now), now,
                ),
            )

    def insert_violation(self, violation: Dict[str, Any]) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO violations
                (event_id, camera_id, track_id, violation_type, timestamp, confidence, screenshot_path, crop_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    violation.get("event_id"), violation["camera_id"], str(violation["track_id"]),
                    violation["violation_type"], violation.get("timestamp", utc_now_iso()),
                    float(violation.get("confidence", 0)), violation.get("screenshot_path"), violation.get("crop_path"),
                    json.dumps(violation.get("metadata", {})),
                ),
            )

    def list_active_events(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE state IN ('NEW','ACTIVE') ORDER BY timestamp_start DESC LIMIT 200"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_events(self, limit: int = 300) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY timestamp_start DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    def list_violations(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM violations ORDER BY timestamp DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_analytics(self, metric: Dict[str, Any]) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO analytics(bucket_start, camera_id, metric_name, metric_value, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.get("bucket_start", utc_now_iso()), metric.get("camera_id"), metric["metric_name"],
                    float(metric["metric_value"]), json.dumps(metric.get("metadata", {})), utc_now_iso(),
                ),
            )

    def log_system_event(self, level: str, component: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO system_logs(level, component, message, metadata_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                (level, component, message, json.dumps(metadata or {}), utc_now_iso()),
            )

    def _row_to_camera(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["rules"] = json.loads(data.pop("rules_json") or "{}")
        data["zones"] = json.loads(data.pop("zones_json") or "[]")
        data["enabled"] = bool(data.get("enabled", 1))
        # Normalise status casing (workers write "RUNNING"/"STARTING"/"STOPPED";
        # templates & API compare against lowercase). "starting" counts as live
        # so the button doesn't flash "Start" during warm-up.
        raw_status = (data.get("status") or "").lower()
        data["status"] = "running" if raw_status in ("running", "starting") else raw_status
        return data


DB = Database()
