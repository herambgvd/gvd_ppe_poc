# 🦺 neubit.ai — AI-Powered PPE Detection System

An Enterprise-Level AI-Powered PPE (Personal Protective Equipment) Detection & Compliance Monitoring System built using **Flask, YOLO, ByteTrack, OSNet ReID, OpenCV, SQLite, and Real-Time Video Analytics**.

---

## ⚡ Quick Start

```bash
# 1. create venv + install deps
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. configure (optional) — copy and edit
cp .env.example .env

# 3. run (serves on the machine's LAN IP, not just localhost)
./run.sh
# or:  PPE_HOST=0.0.0.0 PPE_PORT=5000 PPE_DEVICE=0 .venv/bin/python app.py
```

On start the console prints the reachable URLs:

```
 Local:   http://127.0.0.1:5000
 Network: http://192.168.1.3:5000   (share this on the LAN)
```

### Run as a production service (systemd)

```bash
sudo cp deploy/neubit-ppe.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now neubit-ppe
sudo systemctl status neubit-ppe      # logs: journalctl -u neubit-ppe -f
```

`PPE_HOST=0.0.0.0` makes the app available on the server's IP (e.g. `http://192.168.1.3:5000`), `debug=false`, with auto-restart on failure.

### Recommended camera (RTSP/CCTV) settings

| Setting | Value | Why |
|---|---|---|
| Codec | H.264 (AVC) | Best OpenCV/FFmpeg compatibility |
| Resolution | 1080p | Model runs at imgsz 1536 internally |
| FPS | 10–15 | Enough for PPE; less GPU load |
| Bitrate | 4–8 Mbps CBR | Clear frames, no artifacts |
| Transport | RTSP over TCP | Avoids packet-loss corruption |

> If a live stream lags, raise `PPE_FRAME_SKIP` (2→3) or lower `PPE_IMG_SIZE` (1536→1280).

### 7-camera site deployment (e.g. SMCC)

Each camera runs its own detector+tracker (7 model instances on one GPU). Apply the
"7-CAMERA PROFILE" block in `.env.example` (`PPE_FRAME_SKIP=3`, `PPE_STREAM_FPS=10`,
`PPE_STREAM_PREVIEW_WIDTH=800`; keep `PPE_IMG_SIZE=1536` — proven accuracy). When a
go2rtc restream runs on the same machine (vizor stack), register cameras as
`rtsp://127.0.0.1:8554/<camera-uuid>` so each physical camera keeps ONE upstream RTSP
session. Watch `GET /api/runtime` (`fps_processed`, `dropped_frames`) and `nvidia-smi`
after starting all 7. Fallbacks if saturated: `PPE_FRAME_SKIP=4`, `PPE_IMG_SIZE=1280`,
or `_sub` stream URLs for far-view cameras.

### Public wall dashboard — `/public`

`http://<server>:5000/public` (alias `/wall`) is a read-only dark dashboard for a site
TV/monitor: live camera grid, "Violations today" KPIs, hourly trend, live alerts feed.
No controls — safe on an unattended screen. Realtime over the same Socket.IO events.

### Spoken alerts (TTS)

Every page (operator + wall) speaks new violations via the browser's SpeechSynthesis —
"No helmet detected on camera CS-3" — with a chime, a 4s anti-burst throttle, and a
🔊/🔇 toggle (top-right, persisted). Browser autoplay policy requires ONE click on the
page (the toggle) before audio is allowed — click it once after opening the wall page.

This project detects whether workers are wearing required safety equipment such as:

- 🪖 Helmet
- 🦺 Safety Vest
- 😷 Mask
- 👢 Safety Shoes *(future support)*
- 🧤 Gloves *(future support)*
- 🥽 Goggles *(future support)*

The system supports:

- 📷 Image Detection
- 🎥 Video Detection
- 📹 Live Webcam Monitoring
- 📡 RTSP / CCTV Stream Monitoring
- 📊 Dashboard Analytics
- 🚨 Real-Time Violation Alerts
- 🧠 Person-PPE Association Layer
- 🛰️ Multi-Person Tracking using ByteTrack
- 🧬 Person Re-Identification (ReID) using OSNet
- 🔁 Persistent Worker Identity Recovery
- 🗂️ Event History & Screenshot Logging

---

# 🚀 Features

## ✅ Real-Time PPE Detection
This project uses a custom-trained YOLOv26s model (`best.pt`) for real-time PPE detection.

The model detects:

- 👷 Person
- 🪖 Helmet
- 🦺 Safety Vest
- 😷 Mask
- ⛑️ Additional PPE Classes
- ❌ Missing PPE Violations

### Base Model

```bash
yolo26s.pt
```

After training:

```bash
best.pt
```

---

# 🧬 Enterprise ReID System (NEW)

## ✅ OSNet Person Re-Identification
The project now includes:

- OSNet-based feature extraction
- Persistent worker identity tracking
- Temporal embedding memory
- Cross-track identity recovery
- Stable worker global IDs

This upgrades the system from:

```text
Basic object tracking
```

into:

```text
Enterprise intelligent surveillance
```

---

## 🔥 Why ReID Is Important

Normal trackers like ByteTrack only maintain:

```text
Temporary Track IDs
```

Example:

```text
Track 1 → Track 17 → Track 43
```

when:

- worker turns
- occlusion happens
- person exits frame
- lighting changes
- motion blur occurs

This causes:

❌ Duplicate alerts  
❌ Broken identity tracking  
❌ Event fragmentation  
❌ Unstable monitoring

---

## ✅ Our Solution

The system now combines:

```text
ByteTrack + OSNet ReID
```

to create:

```text
Persistent Global Worker Identity
```

Example:

```text
Track 1
Track 17
Track 1000001
```

all become:

```text
gid_d074b67dab3d
```

This means:

✅ Same worker identity recovered  
✅ Stable event history  
✅ Enterprise-grade monitoring  
✅ Intelligent incident lifecycle

---

# 🧠 How ReID Works

## ReID Pipeline

```text
Camera Frame
      ↓
YOLO Detection
      ↓
ByteTrack Tracking
      ↓
Person Crop Extraction
      ↓
OSNet Feature Extractor
      ↓
Embedding Vector
      ↓
Cosine Similarity Matching
      ↓
Persistent Global Identity
```

---

## 🔹 OSNet Feature Extractor

OSNet converts a person image into:

```text
Mathematical Appearance Embedding
```

It learns:

- clothing texture
- body shape
- appearance patterns
- visual identity
- color distribution

Example embedding:

```python
[0.172, -0.552, 0.913, ...]
```

---

## 🔹 Cosine Similarity Matching

Embeddings are compared using:

```text
Cosine Similarity
```

High similarity:

```text
Same worker
```

Low similarity:

```text
Different worker
```

---

## 🔹 Temporal Embedding Memory

The project includes:

```python
embedding_history = deque(maxlen=50)
```

This stabilizes identities across frames.

Benefits:

✅ Reduces identity switching  
✅ Handles webcam noise  
✅ Handles motion blur  
✅ Handles temporary occlusion

---

## 🔹 Unknown Counter Protection

The system prevents false identity creation.

Instead of:

```text
1 bad frame → new identity
```

it uses:

```text
multi-frame validation
```

This dramatically improves:

- identity consistency
- event reliability
- enterprise stability

---

# ✅ Association Layer (Human + PPE Logic)

The system intelligently associates PPE items with detected persons.

Instead of only detecting objects independently, the project:

1. Detects persons
2. Detects PPE objects
3. Matches PPE objects to nearby persons
4. Determines compliance status

### Example:

| Person | Helmet | Vest | Result |
|---|---|---|---|
| Worker A | ❌ | ✅ | Violation |
| Worker B | ✅ | ✅ | Safe |

---

# ✅ ByteTrack Multi-Object Tracking

Each person receives a tracking ID.

### Benefits:

- Prevents duplicate alerts
- Tracks workers across frames
- Maintains temporal continuity
- Supports ReID recovery
- Improves event management

---

# ✅ Event Lifecycle Engine

The project includes enterprise event management.

### Event States

```text
NEW
 ↓
ACTIVE
 ↓
RESOLVED
 ↓
EXPIRED
```

This prevents:

❌ Alert flooding  
❌ Duplicate screenshots  
❌ Continuous repeated alerts

---

# ✅ Dashboard Analytics

Interactive dashboard showing:

- Total detections
- PPE compliance count
- Violations count
- Active streams
- Recent alerts
- Event history
- Live alert sidebar
- Stable worker identities
- Compliance trends

---

# ✅ Event Screenshot Storage

Whenever a PPE violation occurs:

- Screenshot captured automatically
- Event stored in SQLite database
- Timestamp recorded
- Compliance information saved
- Worker identity linked
- Evidence image generated

---

# ✅ RTSP / CCTV Support

Supports:

- CCTV Cameras
- IP Cameras
- Mobile IP Camera Apps
- RTSP Streams
- Webcam Devices

Example:

```bash
rtsp://username:password@ip:554/stream
```

---

# 🏗️ Enterprise System Architecture

```text
Video/Input Stream
        ↓
YOLO Detection Engine
        ↓
ByteTrack Tracking
        ↓
OSNet ReID Engine
        ↓
Global Identity Recovery
        ↓
Association Layer
(Person ↔ PPE Mapping)
        ↓
Compliance Rule Engine
        ↓
Violation Detection
        ↓
Event Lifecycle Manager
        ↓
Alert + Screenshot + Database Logging
        ↓
Dashboard Analytics
```

---

# 📂 Project Structure

```bash
PPE-DETECTION-SYSTEM/
│
├── app.py                      # Flask entry point + routes
├── best.pt                     # Custom-trained YOLO model (yolo26s.pt → trained on Colab)
├── custom_bytetrack.yaml       # ByteTrack tracker config
├── requirements.txt
├── ppe.db                      # SQLite database
│
├── static/
│   ├── css/
│   ├── js/
│   ├── sounds/
│   ├── uploads/                # Uploaded images/videos
│   ├── outputs/                # Annotated detection results
│   └── violations/             # Violation evidence snapshots
│
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── image.html
│   ├── video.html
│   ├── video_processing.html   # Live video analysis (side-by-side + alerts)
│   ├── result.html             # Final detection result + compliance summary
│   ├── webcam.html
│   ├── cctv.html
│   ├── camera_manager.html
│   ├── active_events.html
│   ├── analytics.html
│   ├── history.html
│   └── settings.html
│
└── utils/
    ├── config.py               # Centralized configuration
    ├── logger.py
    ├── database.py
    ├── detector.py             # YOLO inference wrapper
    ├── tracker_manager.py      # ByteTrack + ReID tracking
    ├── reid_manager.py
    ├── reid_matcher.py
    ├── feature_extractor.py    # OSNet embeddings
    ├── association_engine.py   # Person ↔ PPE mapping
    ├── compliance_engine.py    # PPE rule evaluation
    ├── event_manager.py        # Event lifecycle + screenshots
    ├── alert_manager.py
    ├── analytics_engine.py
    ├── stream_manager.py       # Live webcam/RTSP MJPEG workers
    ├── webcam_detector.py
    ├── rtsp_detector.py
    ├── video_detector.py       # Offline video processing
    ├── video_job_manager.py    # Async video jobs + live MJPEG + summary
    ├── frame_buffer.py
    ├── stats.py
    └── socket_events.py
```

---

# ⚙️ Technologies Used

## Backend

- Python
- Flask
- Flask-SocketIO
- SQLite
- OpenCV

## AI / Computer Vision

- YOLO26s
- ByteTrack
- OSNet
- TorchReID
- Multi-Object Tracking
- Person Re-Identification
- Association Logic
- Temporal Embedding Memory

## Frontend

- HTML
- CSS
- JavaScript
- Bootstrap
- Chart.js

---

# 🧠 Core Detection Flow

## 1️⃣ Frame Capture

Frames captured from:

- Webcam
- RTSP stream
- Uploaded video
- Uploaded image

---

## 2️⃣ YOLO Detection

YOLO detects:

```python
{
    "class": "person",
    "confidence": 0.95,
    "bbox": [x1, y1, x2, y2]
}
```

---

## 3️⃣ ByteTrack Tracking

Workers are tracked frame-by-frame.

Example:

```python
track_id = 17
```

---

## 4️⃣ OSNet ReID

Person crops are extracted.

OSNet generates embeddings.

Embeddings are matched using cosine similarity.

Global identity example:

```text
gid_a8d1e0ee5324
```

---

## 5️⃣ Association Engine

PPE items are mapped to workers.

Example:

```python
if helmet_inside_person_box:
    assign_helmet()
```

---

## 6️⃣ Compliance Engine

Rules evaluated.

Example:

```python
if not helmet:
    violation = True
```

---

## 7️⃣ Event Lifecycle Engine

Creates intelligent incidents.

Stores:

- ACTIVE events
- RESOLVED events
- evidence images
- timestamps
- worker identities

---

# 📊 Dashboard Features

## Dashboard Includes

- Live counters
- PPE summaries
- Violation trends
- Event timeline
- Camera feed preview
- Detection statistics
- Live alert sidebar
- Screenshot evidence
- Worker ReID identities

---

# 📦 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/PPE-DETECTION-SYSTEM.git
cd PPE-DETECTION-SYSTEM
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Application

```bash
python app.py
```

---

## 5️⃣ Open Browser

```bash
http://127.0.0.1:5000
```

---

# 📈 Performance Optimizations

- Frame skipping
- Confidence thresholding
- Efficient ByteTrack tracking
- Temporal embedding smoothing
- Identity memory caching
- Stream buffering
- Lightweight OSNet model

---

# 👨‍💻 Author

Developed by:

**Saurabh Kumar Mohanka**