"""
# ======================================
# MAIN
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This is the Flask application entry point for the enterprise PPE Detection System.
- It wires together routes, Socket.IO, database, camera manager, upload processing, live MJPEG streaming, and dashboard pages.
- The app.py file intentionally remains an orchestration layer; heavy computer-vision logic lives inside utils/ modules.

Why this architecture is used in enterprise systems:
- Flask handles HTTP/API/UI concerns.
- StreamManager handles camera workers.
- Detector/Association/ReID/Event modules handle domain-specific safety intelligence.
- This separation prevents a monolithic app.py that becomes impossible to maintain.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict

import cv2
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, url_for
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

from utils.analytics_engine import ANALYTICS
from utils.association_engine import AssociationEngine
from utils.compliance_engine import ComplianceEngine
from utils.config import CONFIG
from utils.database import DB
from utils.detector import PPEDetector, draw_detections
from utils.event_manager import EventManager
from utils.logger import get_logger
from utils.reid_manager import ReIDManager
from utils.rtsp_detector import register_rtsp_camera
from utils.socket_events import register_socket_events
from utils.stream_manager import STREAMS
from utils.tracker_manager import TrackerManager
from utils.video_detector import process_uploaded_video
from utils.video_job_manager import VIDEO_JOBS, frame_generator as video_job_frames
from utils.webcam_detector import register_webcam

logger = get_logger("ppe.app")

app = Flask(__name__)
app.config["SECRET_KEY"] = CONFIG.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB video uploads
# Reload templates on change so UI updates do not require a full restart.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)

register_socket_events(socketio)

# Reconcile stale state from a previous session: no workers run at startup, so
# any camera left "running" or event left ACTIVE in the DB is orphaned.
try:
    _reset = DB.reset_runtime_state()
    if _reset["cameras_stopped"] or _reset["events_resolved"]:
        logger.info(
            "Startup reconcile: stopped %s camera(s), resolved %s orphaned event(s)",
            _reset["cameras_stopped"], _reset["events_resolved"],
        )
except Exception:  # noqa: BLE001
    logger.exception("Startup state reconcile failed")


# ======================================
# RETENTION — auto-delete old events + evidence images
# ======================================
def _purge_old_events() -> None:
    """Delete events/violations older than the retention window and unlink their
    evidence images (the DB is tiny; the image files are the real disk cost)."""
    res = DB.purge_events_older_than(CONFIG.EVENT_RETENTION_DAYS)
    removed = 0
    guard = str(CONFIG.VIOLATION_DIR.resolve())
    for web in res["files"]:
        try:
            disk = (CONFIG.BASE_DIR / str(web).lstrip("/")).resolve()
            # only ever delete inside the violations dir
            if str(disk).startswith(guard) and disk.is_file():
                disk.unlink()
                removed += 1
        except Exception:  # noqa: BLE001
            pass
    if res["events"] or res["violations"] or removed:
        logger.info(
            "Retention purge (> %s days): %s events, %s violations, %s evidence files removed",
            CONFIG.EVENT_RETENTION_DAYS, res["events"], res["violations"], removed,
        )


def _cleanup_loop() -> None:
    while True:
        try:
            _purge_old_events()
        except Exception:  # noqa: BLE001
            logger.exception("Retention purge failed")
        time.sleep(6 * 3600)  # every 6 hours


# run once now, then periodically in the background
try:
    _purge_old_events()
except Exception:  # noqa: BLE001
    logger.exception("Initial retention purge failed")
threading.Thread(target=_cleanup_loop, daemon=True).start()


# ======================================
# FILE VALIDATION
# ======================================

def allowed_file(filename: str, allowed: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


# ======================================
# PERSON SUMMARY BUILDER
# ======================================

def build_person_summary(associations):

    persons = []

    required_ppe = [
        "helmet",
        "vest",
    ]

    for index, assoc in enumerate(associations):

        assoc_dict = assoc.as_dict()

        ppe = assoc_dict.get("ppe", {})

        violations = []

        person_data = {
            "person_number": index + 1,
            "track_id": assoc_dict.get("track_id"),
            "items": {},
            "violations": [],
            "compliant": True
        }

        for item in required_ppe:

            has_item = item in ppe and len(ppe[item]) > 0

            person_data["items"][item] = has_item

            if not has_item:
                violations.append(f"{item.title()} Missing")

        person_data["violations"] = violations

        if len(violations) > 0:
            person_data["compliant"] = False

        persons.append(person_data)

    return persons


# ======================================
# ROUTES
# ======================================

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        cameras=DB.list_cameras(),
        analytics=ANALYTICS.summary()
    )


@app.route("/image", methods=["GET", "POST"])
def image_page():

    if request.method == "GET":
        return render_template("image.html")

    file = request.files.get("image")

    if not file or file.filename == "":
        flash("Please upload an image.", "danger")
        return redirect(url_for("image_page"))

    if not allowed_file(file.filename, {"jpg", "jpeg", "png", "bmp", "webp"}):
        flash("Unsupported image format.", "danger")
        return redirect(url_for("image_page"))

    filename = secure_filename(file.filename)

    input_path = CONFIG.UPLOAD_IMAGE_DIR / f"{uuid.uuid4().hex}_{filename}"

    file.save(input_path)

    result = process_image(input_path)

    return render_template(
        "result.html",
        result=result,
        mode="image"
    )


@app.route("/video", methods=["GET", "POST"])
def video_page():

    if request.method == "GET":
        return render_template("video.html", selectable_ppe=CONFIG.SELECTABLE_PPE)

    file = request.files.get("video")

    if not file or file.filename == "":
        flash("Please upload a video.", "danger")
        return redirect(url_for("video_page"))

    if not allowed_file(file.filename, {"mp4", "avi", "mov", "mkv", "webm"}):
        flash("Unsupported video format.", "danger")
        return redirect(url_for("video_page"))

    filename = secure_filename(file.filename)

    input_path = CONFIG.UPLOAD_VIDEO_DIR / f"{uuid.uuid4().hex}_{filename}"

    file.save(input_path)

    # PPE the user wants detected / enforced for this clip (defaults to helmet+vest).
    selected_ppe = request.form.getlist("mandatory_ppe") or ["helmet", "vest"]
    rules = {"mandatory_ppe": selected_ppe}

    # Process asynchronously so the browser can watch live detection while
    # the clip is analysed, then jump to the final result page.
    job = VIDEO_JOBS.start(input_path, rules=rules)

    return render_template(
        "video_processing.html",
        job_id=job.job_id,
        filename=filename
    )


@app.route("/video/result/<job_id>")
def video_result_page(job_id: str):

    job = VIDEO_JOBS.get(job_id)

    if job is None:
        flash("Video job not found or expired.", "danger")
        return redirect(url_for("video_page"))

    if job.state == "error":
        return render_template(
            "result.html",
            result={"error": job.error},
            mode="error"
        ), 500

    if job.state == "processing":
        # Still running — send the user back to the live view.
        return render_template(
            "video_processing.html",
            job_id=job.job_id,
            filename=job.input_path.name
        )

    # "done" or "cancelled": show the (partial) result of whatever was processed.
    return render_template(
        "result.html",
        result=job.result(),
        mode="video",
        partial=(job.state == "cancelled")
    )


@app.route("/video_feed_upload/<job_id>")
def video_feed_upload(job_id: str):

    return Response(
        video_job_frames(job_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/video-job/<job_id>")
def api_video_job(job_id: str):

    job = VIDEO_JOBS.get(job_id)

    if job is None:
        return jsonify({"ok": False, "error": "Job not found"}), 404

    return jsonify(job.status())


@app.route("/api/video-job/<job_id>/stop", methods=["POST"])
def api_stop_video_job(job_id: str):
    return jsonify({"ok": VIDEO_JOBS.cancel(job_id)})


@app.route("/webcam")
def webcam_page():

    cameras = [
        c for c in DB.list_cameras()
        if c["source_type"] == "webcam"
    ]

    return render_template(
        "webcam.html",
        cameras=cameras
    )


@app.route("/cctv")
def cctv_page():

    cameras = [
        c for c in DB.list_cameras()
        if c["source_type"] == "rtsp"
    ]

    return render_template(
        "cctv.html",
        cameras=cameras
    )


@app.route("/history")
def history_page():

    events = DB.list_events(limit=500)

    # Club a worker's multiple simultaneous violations into one row (like the
    # Live Alerts feed): one card per person listing all missing PPE.
    groups = {}
    order = []
    for e in events:  # newest-first
        # Key by (camera, identity): track_id is usually a globally-unique ReID
        # gid, but can fall back to a per-camera "track_N" that collides across
        # cameras — including camera_id keeps two cameras' workers separate.
        key = (e.get("camera_id"), e.get("track_id") or "unknown")
        g = groups.get(key)
        if g is None:
            g = {
                "track_id": key,
                "camera_id": e.get("camera_id"),
                "missing": [],
                "confidence": 0.0,
                "timestamp_start": e.get("timestamp_start", ""),
                "crop_path": e.get("crop_path"),
                "screenshot_path": e.get("screenshot_path"),
                "event_ids": [],
                "reviews": set(),
            }
            groups[key] = g
            order.append(key)
        ppe = (e.get("violation_type") or "").replace("missing_", "").replace("no_", "").strip()
        if ppe and ppe not in g["missing"]:
            g["missing"].append(ppe)
        g["confidence"] = max(g["confidence"], float(e.get("confidence") or 0))
        g["event_ids"].append(e.get("event_id"))
        g["reviews"].add(e.get("review_status") or "pending")

    grouped = []
    for k in order:
        g = groups[k]
        rv = g.pop("reviews")
        # group verdict: all-confirmed / all-false, else pending
        g["review_status"] = ("confirmed" if rv == {"confirmed"}
                              else "false" if rv == {"false"}
                              else "pending")
        g["ids_csv"] = ",".join(str(i) for i in g["event_ids"])
        grouped.append(g)

    return render_template(
        "history.html",
        groups=grouped,
        violations=DB.list_violations(limit=500)
    )


@app.route("/analytics")
def analytics_page():

    return render_template(
        "analytics.html",
        analytics=ANALYTICS.summary()
    )


@app.route("/active-events")
def active_events_page():

    return render_template(
        "active_events.html",
        events=DB.list_active_events()
    )


@app.route("/camera-manager", methods=["GET", "POST"])
def camera_manager_page():

    if request.method == "POST":

        name = request.form.get("name", "Camera")

        source_type = request.form.get("source_type", "rtsp")

        source_uri = request.form.get("source_uri", "0")

        mandatory_ppe = (
            request.form.getlist("mandatory_ppe")
            or CONFIG.DEFAULT_MANDATORY_PPE
        )

        rules = {
            "mandatory_ppe": mandatory_ppe
        }

        if source_type == "webcam":

            register_webcam(
                name=name,
                device_index=int(source_uri or 0),
                rules=rules
            )

        else:

            register_rtsp_camera(
                name=name,
                rtsp_url=source_uri,
                rules=rules
            )

        flash("Camera registered successfully.", "success")

        return redirect(url_for("camera_manager_page"))

    return render_template(
        "camera_manager.html",
        cameras=DB.list_cameras(),
        default_ppe=CONFIG.DEFAULT_MANDATORY_PPE,
        selectable_ppe=CONFIG.SELECTABLE_PPE
    )


@app.route("/settings")
def settings_page():
    return render_template("settings.html", config=CONFIG)


# ======================================
# VIDEO FEED
# ======================================

@app.route("/video_feed/<camera_id>")
def video_feed(camera_id: str):

    return Response(
        STREAMS.frame_generator(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ======================================
# API ROUTES
# ======================================

@app.route("/api/events/<event_id>", methods=["DELETE"])
def delete_event(event_id):

    with DB.connect() as conn:

        conn.execute(
            "DELETE FROM violations WHERE event_id=?",
            (event_id,)
        )

        conn.execute(
            "DELETE FROM events WHERE event_id=?",
            (event_id,)
        )

        conn.commit()

    return jsonify({"ok": True})

@app.route("/api/events/<event_id>/review", methods=["POST"])
def review_event(event_id):
    """Record a supervisor's genuine/false verdict on an event."""
    data = request.get_json(silent=True) or {}
    verdict = data.get("verdict", "")
    ok = DB.set_event_review(event_id, verdict)
    return jsonify({"ok": ok, "verdict": verdict if ok else None})


@app.route("/api/events/delete-selected", methods=["POST"])
def delete_selected_events():

    data = request.get_json()

    ids = data.get("ids", [])

    with DB.connect() as conn:

        for event_id in ids:

            conn.execute(
                "DELETE FROM violations WHERE event_id=?",
                (event_id,)
            )

            conn.execute(
                "DELETE FROM events WHERE event_id=?",
                (event_id,)
            )

        conn.commit()

    return jsonify({"ok": True})


@app.route("/api/events/delete-all", methods=["DELETE"])
def delete_all_events():

    with DB.connect() as conn:

        conn.execute("DELETE FROM violations")

        conn.execute("DELETE FROM events")

        conn.commit()

    return jsonify({"ok": True})



@app.route("/api/health")
def api_health():

    return jsonify({
        "ok": True,
        "model": str(CONFIG.MODEL_PATH),
        "cameras": len(DB.list_cameras())
    })


@app.route("/api/cameras", methods=["GET", "POST"])
def api_cameras():

    if request.method == "GET":
        return jsonify(DB.list_cameras())

    data = request.get_json(force=True)

    source_type = data.get("source_type", "rtsp")

    if source_type == "webcam":

        camera = register_webcam(
            data.get("name", "Webcam"),
            int(data.get("source_uri", 0)),
            data.get("rules", {})
        )

    else:

        camera = register_rtsp_camera(
            data.get("name", "RTSP Camera"),
            data["source_uri"],
            data.get("rules", {}),
            data.get("zones", [])
        )

    return jsonify(camera), 201


@app.route("/api/cameras/<camera_id>", methods=["DELETE"])
def api_delete_camera(camera_id: str):

    STREAMS.stop_camera(camera_id)

    DB.delete_camera(camera_id)

    return jsonify({"ok": True})


@app.route("/api/cameras/<camera_id>", methods=["PUT"])
def api_update_camera(camera_id: str):

    cam = DB.get_camera(camera_id)
    if not cam:
        return jsonify({"ok": False, "error": "Camera not found"}), 404

    # Editing is only allowed while the camera is stopped (clean workflow:
    # stop -> edit -> start). A running worker holds its own config in memory.
    if cam.get("status") == "running":
        return jsonify({"ok": False, "error": "Stop the camera before editing."}), 409

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or cam.get("name") or "Camera").strip()
    source_type = data.get("source_type") or cam.get("source_type") or "rtsp"
    source_uri = (data.get("source_uri") or cam.get("source_uri") or "0").strip()
    mandatory_ppe = data.get("mandatory_ppe") or ["helmet", "vest"]

    # Preserve any existing ROI, update the mandatory PPE.
    rules = cam.get("rules") or {}
    rules["mandatory_ppe"] = mandatory_ppe

    ok = DB.update_camera(camera_id, name, source_type, source_uri, rules)

    # Applied on next start — start_camera rebuilds the worker from latest rules.
    return jsonify({"ok": ok})


@app.route("/api/cameras/<camera_id>/start", methods=["POST"])
def api_start_camera(camera_id: str):

    return jsonify({
        "ok": STREAMS.start_camera(camera_id)
    })


@app.route("/api/cameras/<camera_id>/stop", methods=["POST"])
def api_stop_camera(camera_id: str):

    return jsonify({
        "ok": STREAMS.stop_camera(camera_id)
    })


@app.route("/api/cameras/<camera_id>/snapshot")
def api_camera_snapshot(camera_id: str):
    # Prefer a live frame from a running worker.
    worker = STREAMS.get_worker(camera_id)
    if worker is not None:
        jpeg = worker.get_latest_jpeg()
        if jpeg:
            return Response(jpeg, mimetype="image/jpeg")

    cam = DB.get_camera(camera_id)
    if not cam:
        return jsonify({"ok": False, "error": "Camera not found"}), 404

    src = cam.get("source_uri", "0")
    source = int(src) if str(src).isdigit() else src
    cap = cv2.VideoCapture(source)
    ok, frame = False, None
    if cap.isOpened():
        for _ in range(3):
            ok, frame = cap.read()
            if ok:
                break
    cap.release()
    if not ok or frame is None:
        return jsonify({"ok": False, "error": "Could not grab a frame"}), 503

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/cameras/<camera_id>/roi", methods=["POST"])
def api_set_camera_roi(camera_id: str):
    data = request.get_json(silent=True) or {}
    clean = []
    for p in (data.get("points") or []):
        try:
            x, y = float(p[0]), float(p[1])
            clean.append([min(1.0, max(0.0, x)), min(1.0, max(0.0, y))])
        except (TypeError, ValueError, IndexError):
            continue
    roi = clean if len(clean) >= 3 else None
    ok = DB.set_camera_roi(camera_id, roi)
    # Apply live if the camera is running.
    worker = STREAMS.get_worker(camera_id)
    if worker is not None:
        worker.rules["roi"] = roi
    return jsonify({"ok": ok, "points": len(clean)})


@app.route("/api/events/active")
def api_active_events():
    return jsonify(DB.list_active_events())


@app.route("/api/history")
def api_history():

    return jsonify({
        "events": DB.list_events(limit=500),
        "violations": DB.list_violations(limit=500)
    })


@app.route("/api/analytics/summary")
def api_analytics_summary():
    return jsonify(ANALYTICS.summary())


@app.route("/api/runtime")
def api_runtime():
    return jsonify(STREAMS.list_runtime_stats())

@app.route("/api/latest-alerts")
def api_latest_alerts():

    # Per-worker compliance feed: group recent events by person so each card
    # shows one worker + exactly which PPE they are missing (helmet, vest, ...).
    # Simple for a supervisor — "is this worker compliant, and what's missing".
    events = DB.list_events(limit=60)

    persons = {}
    order = []
    for e in events:
        # (camera, identity) key — see history_page: avoids cross-camera collision
        # of the per-camera "track_N" fallback identity.
        key = (e.get("camera_id"), e.get("track_id") or "unknown")
        vt = (e.get("violation_type") or "")
        ppe = vt.replace("missing_", "").replace("no_", "").strip()
        ent = persons.get(key)
        if ent is None:
            # events are newest-first, so the first one seen carries the latest crop/time
            ent = {
                "track_id": key,
                "missing": [],
                "created_at": e.get("timestamp_start", ""),
                "snapshot": e.get("crop_path") or e.get("screenshot_path", ""),
            }
            persons[key] = ent
            order.append(key)
        if ppe and ppe not in ent["missing"]:
            ent["missing"].append(ppe)

    # Only non-compliant workers (with at least one missing PPE) belong in alerts.
    return jsonify([persons[k] for k in order if persons[k]["missing"]][:10])

# ======================================
# IMAGE PROCESSING
# ======================================

def process_image(input_path: Path) -> Dict:

    frame = cv2.imread(str(input_path))

    if frame is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    job_id = "image_" + uuid.uuid4().hex[:10]

    detector = PPEDetector()

    tracker = TrackerManager(job_id)

    reid = ReIDManager(job_id)

    association = AssociationEngine(job_id)

    compliance = ComplianceEngine()

    events = EventManager(job_id, "video")

    # ======================================
    # DETECTION
    # ======================================

    detections = detector.predict_image(frame)

    # ======================================
    # TRACKING
    # ======================================

    detections = tracker.update(frame, detections)

    # ======================================
    # PERSON FILTER
    # ======================================

    people = [
        d for d in detections
        if d.canonical_class == CONFIG.PERSON_CLASS
    ]

    # ======================================
    # RE-IDENTITY
    # ======================================

    reid.update_person_identities(frame, people)

    # ======================================
    # ASSOCIATION ENGINE
    # ======================================

    associations = association.associate(detections)

    # ======================================
    # COMPLIANCE ENGINE
    # ======================================

    violations = compliance.evaluate(associations)

    # ======================================
    # HUMAN READABLE PPE SUMMARY
    # ======================================

    persons_summary = build_person_summary(associations)

    # ======================================
    # CAMERA ID
    # ======================================

    for v in violations:
        v["camera_id"] = job_id

    # ======================================
    # DRAW DETECTIONS
    # ======================================

    annotated = draw_detections(
        frame,
        detections,
        violations
    )

    # ======================================
    # EVENT UPDATE
    # ======================================

    events.update(
        violations,
        annotated,
        frame
    )

    # ======================================
    # SAVE OUTPUT IMAGE
    # ======================================

    output_path = CONFIG.OUTPUT_IMAGE_DIR / f"{job_id}.jpg"

    cv2.imwrite(str(output_path), annotated)

    # ======================================
    # FINAL RESPONSE
    # ======================================

    return {

        "job_id": job_id,

        "input_path": f"/static/uploads/images/{input_path.name}",

        "output_path": f"/static/outputs/images/{output_path.name}",

        "detections": [
            d.as_dict()
            for d in detections
        ],

        "associations": [
            a.as_dict()
            for a in associations
        ],

        "persons_summary": persons_summary,

        "violations": violations,

        "violation_count": len(violations),
    }


# ======================================
# ERROR HANDLER
# ======================================

@app.errorhandler(Exception)
def handle_error(exc):

    logger.exception("Unhandled error: %s", exc)

    if request.path.startswith("/api/"):

        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500

    flash(str(exc), "danger")

    return render_template(
        "result.html",
        result={"error": str(exc)},
        mode="error"
    ), 500


@app.route('/favicon.ico')
def favicon():
    return '', 204



# ======================================
# MAIN RUNNER
# ======================================

if __name__ == "__main__":

    import socket

    def _lan_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:  # noqa: BLE001
            return "127.0.0.1"

    ip = _lan_ip()
    logger.info("=" * 56)
    logger.info(" neubit.ai — PPE Detection System")
    logger.info(" Local:   http://127.0.0.1:%s", CONFIG.PORT)
    if CONFIG.HOST == "0.0.0.0":
        logger.info(" Network: http://%s:%s   (share this on the LAN)", ip, CONFIG.PORT)
    logger.info(" Device:  %s | imgsz=%s | debug=%s", CONFIG.DEVICE, CONFIG.IMG_SIZE, CONFIG.DEBUG)
    logger.info("=" * 56)

    socketio.run(
        app,
        host=CONFIG.HOST,
        port=CONFIG.PORT,
        debug=CONFIG.DEBUG,
        allow_unsafe_werkzeug=True
    )