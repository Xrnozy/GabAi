# GabAi: Real-Time Vision-Based Accessibility Assistant

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Android API 21+](https://img.shields.io/badge/Android-API%2021%2B-green)](https://developer.android.com/)
[![WebRTC](https://img.shields.io/badge/WebRTC-P2P%20Video-informational)](https://webrtc.org/)

*A mobile-first, real-time camera accessibility system that provides hands-free guidance for navigation, text recognition, gesture understanding, and color identification.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Core Technologies](#core-technologies)
- [Directional Safety Field Algorithm](#directional-safety-field-algorithm)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Performance & Optimization](#performance--optimization)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

**GabAi** is a real-time, camera-first accessibility platform designed to empower users with low vision and blind users through intelligent scene understanding and multimodal guidance. The system comprises three integrated components:

1. **Mobile App** (Android): Low-bandwidth video streaming, hands-free voice interface, text-to-speech feedback, and haptic confirmation.
2. **Backend Server** (Python): Real-time multi-modal inference pipeline with dynamic mode selection.
3. **Web Dashboard** (Frontend): Live monitoring, telemetry visualization, and debugging interface.

The platform uses a **Directional Safety Field**—a continuous, five-region risk model—to replace simplistic collision-zone detection. This approach fuses depth, optical flow, object detection, and segmentation to produce stable, context-aware navigation commands.

---

## Motivation

### The Problem

- **Navigation hazards** are difficult to detect reliably in real time without computer vision, leading to injury and reduced independence.
- **Text access** (signs, menus, labels) requires real-time, accurate recognition with minimal latency.
- **Gesture-based communication** (ASL, pointing, color distinction) needs immediate feedback in conversational contexts.
- **Bandwidth constraints** make streaming high-resolution video impractical on mobile data networks.
- **Cognitive load** must remain low—users need concise, non-repetitive audio cues, not a constant stream of alerts.

### Our Solution

GabAi addresses these challenges through:
- **Perspective-aware scene interpretation** via the Directional Safety Field algorithm.
- **Modular, mode-specific pipelines** that run only the inference needed.
- **Aggressive compression and optimization** for 250 kbps video streaming.
- **Temporal stabilization** with EMA smoothing and hysteresis-based state transitions.
- **Multimodal output** combining TTS, haptics, and real-time visual feedback.

---

## Features

### Navigation Mode
- **Directional Safety Field**: Five-region risk assessment (left, slight-left, forward, slight-right, right).
- **Multi-signal fusion**: Depth, YOLO detections, optical flow, road segmentation.
- **Adaptive urgency levels**: SAFE, WARNING, URGENT, CRITICAL with corresponding audio and haptic patterns.
- **Collision avoidance**: Real-time obstacle detection with distance-to-contact estimation.
- **Road walkability assessment**: Pothole detection and surface quality inference.
- **Steering strength & lane confidence**: Quantitative metrics for downstream planning.

### OCR Mode
- **Tesseract OCR**: Fast, lightweight, CPU-friendly text recognition.
- **Frame preprocessing**: Automatic contrast enhancement and blur sharpening for improved legibility.
- **Stability control**: Cooldown and multi-frame voting to reduce false positives.

### ASL Recognition
- **Real-time gesture detection** using YOLO-based hand pose estimation.
- **Per-word confidence thresholding** to reject low-confidence predictions.
- **Cooldown and stability hold** to minimize repeated announcements.

### Color Recognition
- **Dominant color extraction** from center-of-frame ROI.
- **Named color classification** with confidence scoring.
- **RGB value reporting** for granular color data.

### Dashboard & Monitoring
- **Live camera feeds** (raw, YOLO, depth, optical flow, fusion, pothole, road-path).
- **Real-time navigation telemetry** (risk scores, direction recommendations, collision probability).
- **SSE-based source change notifications** for instant viewer reconnection.
- **Adaptive bitrate handling** for variable network conditions.

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Android Mobile App                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Mode Selector  │  Camera Stream  │  TTS/Haptics Output │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────┬────────────────────────────────────────────────┘
                 │ WebRTC (320x240@10fps, 250 kbps)
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                Python Backend Server (port 8080)                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   VideoStreamTrack Pipeline (per-mode)                  │   │
│  │  ├─ NavigationTrack: Depth + YOLO + Flow + Segmentation │   │
│  │  ├─ OCRTrack: Tesseract text extraction                  │   │
│  │  ├─ ASLTrack: Gesture recognition                        │   │
│  │  └─ ColorTrack: Color classification                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Data Channels (WebRTC Negotiated)                      │   │
│  │  ├─ detection (id=0): Object/gesture results             │   │
│  │  ├─ depth (id=1): Depth statistics                       │   │
│  │  └─ navigation (id=2): Unified navigation commands       │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────┬────────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ↓                 ↓
  ┌──────────────┐  ┌──────────────┐
  │  Mobile App  │  │ Web Dashboard│
  │  (TTS/Audio) │  │  (Telemetry) │
  └──────────────┘  └──────────────┘
```

### Component Breakdown

| Component | File(s) | Purpose |
|-----------|---------|---------|
| **Server** | `servertest.py` | WebRTC signaling, mode routing, track subscription |
| **Config** | `config.py` | Model loading, shared state, constants |
| **Depth** | `depth.py` | Depth Anything V2 inference, metric conversion |
| **Detection** | `detection.py` | YOLO general detection, pothole detection |
| **Navigation Fusion** | `directional_safety.py`, `navigation.py`, `video_tracks.py` | Directional safety field, risk fusion |
| **OCR** | `ocr_detection.py`, `ocr_video_tracks.py` | Tesseract-based text extraction |
| **ASL** | `asl_detection.py`, `asl_video_tracks.py` | Gesture recognition |
| **Color** | `color_detection.py`, `color_video_tracks.py` | Color classification |
| **Visualization** | `visualization.py` | Overlay rendering, HUD graphics |
| **Mobile** | `mobile/app/src/main/java` | Android UI, WebRTC client, TTS/haptics |
| **Frontend** | `frontend/src/index.html` | Web dashboard, SSE listener, canvas rendering |

---

## Core Technologies

### Deep Learning & Computer Vision
- **[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)** (encoder: vitl): Universal monocular depth estimation.
- **[YOLOv8](https://github.com/ultralytics/ultralytics)**: Real-time object detection for obstacle/hazard identification.
- **[Tesseract OCR](https://github.com/UB-Mannheim/tesseract)**: Lightweight CPU-based text recognition.
- **[OpenCV](https://opencv.org/)**: Image processing, optical flow (Lucas–Kanade), morphological operations.

### Streaming & Communication
- **[WebRTC](https://webrtc.org/) (aiortc)**: P2P video streaming, negotiated data channels.
- **[aiohttp](https://docs.aiohttp.org/)**: Async HTTP server for signaling and SSE.
- **[Android WebRTC](https://github.com/google/webrtc/tree/master/sdk/android)**: Native Android peer connection.

### Accessibility
- **[pyttsx3](https://pypi.org/project/pyttsx3/)**: Server-side text-to-speech (fallback/server monitoring).
- **[Android TextToSpeech API](https://developer.android.com/reference/android/speech/tts/TextToSpeech)**: Mobile TTS.
- **[Android Vibrator API](https://developer.android.com/reference/android/os/Vibrator)**: Haptic feedback.
- **[Android SpeechRecognizer API](https://developer.android.com/reference/android/speech/SpeechRecognizer)**: Voice commands.
- **[Tesseract OCR](https://github.com/UB-Mannheim/tesseract)**: Lightweight CPU-based text recognition.

---

## Directional Safety Field Algorithm

### Overview
Instead of simplistic near/far/close zones, GabAi divides the camera view into five perspective-aware fan regions and computes a continuous risk score for each using multi-modal sensor fusion.

### Region Definition
- **Angular splits**: [-60°, -30°, -12°, 12°, 30°, 60°] from camera center (±60° horizontal FOV).
- **Depth perspective**: Vanishing point at ~55% down frame; regions narrow at top (far) and widen at bottom (near).
- **Geometry**: Each region is a trapezoid representing the user's visual corridor at that direction.

### Risk Computation (Per Direction)

For each region, the backend computes four independent risk signals and fuses them:

1. **Depth Risk** (weight: 0.35)
   - Uses robust percentile sampling (5th, 20th) to avoid noise.
   - Three zones: close (<0.6m, risk=1.0), near (0.6–2.0m), far (>2.0m).
   - Occupancy-like score: fraction of pixels in each zone.
   - Output: `depth_risk ∈ [0, 1]`

2. **Object Risk** (weight: 0.25)
   - YOLO detections filtered by region membership.
   - Per-class risk weights: human (1.0), car (0.9), pothole (0.8), obstacle (0.5).
   - Proximity scaling: `risk = class_risk × (1 – dist_m / DEPTH_FAR_M)`.
   - Weighted average of top-N objects by distance.
   - Output: `object_risk ∈ [0, 1]`

3. **Optical Flow Risk** (weight: 0.25)
   - Lucas–Kanade tracking extracts motion vectors.
   - Closing speed (points moving toward camera) indicates approach.
   - Magnitude thresholds: FAST (8 px/frame), MED (3 px/frame).
   - Output: `flow_risk ∈ [0, 1]`

4. **Walkability Risk** (weight: 0.15)
   - Road-path segmentation yields walkable/blocked mask.
   - Unwalkable fraction becomes the risk score.
   - Output: `walk_risk ∈ [0, 1]`

### Final Risk Score
```
direction_risk = 0.35 × depth_risk + 0.25 × object_risk + 0.25 × flow_risk + 0.15 × walk_risk
```

### Temporal Smoothing
- **EMA filter**: `smoothed[t] = α × raw[t] + (1 – α) × smoothed[t–1]`, α = 0.35.
- Balances responsiveness and stability; prevents flicker.

### Navigation Decision
- Compute risk for all five directions.
- Apply hysteresis (minimum risk delta = 0.04) to lock in "safest" direction.
- Hold for N frames (default: 2) before allowing transition.
- Threshold application:
  - If `forward_risk ≥ 0.65`: URGENT (consider "Left", "Right", or "Stop").
  - Else if `any_risk ≥ 0.35`: WARNING ("Caution ahead").
  - Else: SAFE ("Path clear").

---

## Project Structure

```
GabAi/
├── README.md                           # This file
├── design.md                           # Brand & design system reference
├── backend/
│   ├── config.py                       # Model loading, constants, shared state
│   ├── server.py                       # Minimal WebRTC relay (video only)
│   ├── servertest.py                   # Full server with all inference pipelines
│   ├── depth.py                        # Depth estimation pipeline
│   ├── detection.py                    # YOLO + pothole + label utilities
│   ├── optical_flow.py                 # Optical flow tracking
│   ├── directional_safety.py           # Directional Safety Field engine
│   ├── navigation.py                   # Navigation fusion & unified detection
│   ├── visualization.py                # Overlay rendering
│   ├── video_tracks.py                 # WebRTC VideoStreamTrack subclasses
│   ├── asl_detection.py                # ASL recognition pipeline
│   ├── asl_video_tracks.py             # ASL WebRTC track
│   ├── ocr_detection.py                # Tesseract OCR pipeline
│   ├── ocr_video_tracks.py             # OCR WebRTC track
│   ├── color_detection.py              # Color classification pipeline
│   ├── color_video_tracks.py           # Color WebRTC track
│   ├── tts_manager.py                  # Server-side TTS queue (fallback)
│   ├── road_seg.py                     # Road/path segmentation
│   ├── requirements.txt                # Python dependencies
│   ├── models/
│   │   ├── yolo/                       # YOLO checkpoint files
│   │   ├── depth/                      # Depth Anything V2 checkpoint
│   │   ├── asl/                        # ASL model checkpoint
│   │   ├── segmentation/               # Road & pothole segmentation models
│   │   └── ocr/                        # OCR model weights (if needed)
│   └── depth_anything_v2/              # Depth Anything V2 module
│       ├── dpt.py                      # DPT depth model
│       ├── run.py                      # Single-image inference script
│       └── run_video.py                # Video inference script
├── mobile/
│   ├── build.gradle.kts                # Gradle build config
│   ├── app/
│   │   ├── build.gradle.kts            # App module build config
│   │   └── src/main/java/com/example/gabai/
│   │       ├── MainActivity.kt         # Home screen with mode selection
│   │       ├── NavigationActivity.kt   # Navigation mode UI
│   │       ├── AslActivity.kt          # ASL mode UI
│   │       ├── OcrActivity.kt          # OCR mode UI
│   │       ├── ColorActivity.kt        # Color mode UI
│   │       ├── WebRTCClient.kt         # WebRTC peer connection
│   │       ├── services/
│   │       │   ├── TtsService.kt       # Text-to-speech wrapper
│   │       │   ├── VoiceCommandService.kt  # Speech recognition
│   │       │   └── Haptics.kt          # Vibration feedback
│   │       └── modules/
│   │           ├── NavigationModule.kt # Navigation listener
│   │           ├── AslModule.kt        # ASL listener
│   │           ├── OcrModule.kt        # OCR listener
│   │           ├── ColorModule.kt      # Color listener
│   │           ├── ModelManager.kt     # Model initialization
│   │           └── ServerManager.kt    # WebRTC lifecycle
│   └── res/                            # Android resources (layouts, strings, colors)
├── frontend/
│   └── src/
│       └── index.html                  # Web dashboard
└── landing_page/
    └── index.html                      # Public landing page
```

---

## Requirements

### System Requirements

#### Backend (Python Server)
- **OS**: Linux (recommended), macOS, or Windows with CUDA support.
- **Python**: 3.8 or higher.
- **GPU**: NVIDIA GPU with CUDA 11.x (recommended for depth & YOLO). CPU fallback available but slower.
- **RAM**: ≥8 GB (16 GB+ recommended for concurrent inference).
- **Storage**: ~5 GB for model weights.

#### Mobile (Android)
- **Android Version**: API 21 (Android 5.0) or higher.
- **RAM**: ≥2 GB recommended.
- **Camera**: Front or rear camera (user preference via user memory).

#### Frontend (Web Dashboard)
- **Browser**: Modern browser with WebRTC support (Chrome, Firefox, Edge, Safari 11+).
- **Network**: Must have access to backend server.

### Dependencies

#### Python

See `backend/requirements.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
pytesseract>=0.3.10
aiohttp>=3.8.0
aiortc>=1.5.0
av>=10.0.0
pyttsx3>=2.90
```

#### Android

- Android Studio 2021.3+
- Android SDK 21+
- WebRTC library (org.webrtc:google-webrtc)

---

## Installation

### 1. Backend Setup

#### Clone the Repository
```bash
git clone https://github.com/yourusername/gabai.git
cd gabai/backend
```

#### Install Python Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Download Model Weights

Ensure model weights are placed in `backend/models/`:

```bash
# YOLO detection model
mkdir -p models/yolo
# Download yolov8x.pt from ultralytics or Hugging Face

# Depth Anything V2
mkdir -p models/depth
# Download depth_anything_v2_vitl.pth

# Pothole detection
mkdir -p models/segmentation
# Add Yolov8-fintuned-on-potholes.pt, yolo11m-road-seg.pt, etc.

# ASL recognition
mkdir -p models/asl
# Add yolov8x1.pt
```

See `backend/config.py` for exact paths and model loading logic.

#### Install Tesseract (for OCR mode)

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download the installer from [GitHub - UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Update the path in `backend/ocr_detection.py` line 8.

#### Verify Installation
```bash
python backend/config.py  # Should load all models without errors
```

### 2. Backend Server Launch

#### Full Server (All Pipelines)
```bash
cd backend
python servertest.py
```
Runs on `http://0.0.0.0:8080`. Supports navigation, OCR, ASL, and color modes.

#### Minimal Relay Server
```bash
python server.py
```
WebRTC relay only, no inference. Useful for testing connectivity.

### 3. Mobile Setup

#### Build and Deploy Android App
```bash
cd mobile
./gradlew build
./gradlew installDebug  # Deploy to connected device or emulator
```

#### Configure Server URL
In `mobile/app/src/main/java/com/example/gabai/WebRTCClient.kt`, line ~24:
```kotlin
private val serverUrl = "http://YOUR_SERVER_IP:8080"
```

### 4. Frontend Dashboard

Open your browser and navigate to:
```
http://localhost:8080
```

The dashboard will connect to the backend server automatically.

---

## Configuration

### Backend Configuration (`backend/config.py`)

Key configurable parameters:

```python
# Inference input sizes
YOLO_INPUT_SIZE = 320        # Smaller = faster, less accurate
DEPTH_INPUT_SIZE = 256
POTHOLE_INPUT_SIZE = 416

# Depth conversion (meters)
DEPTH_NEAR_M = 0.4           # Closest depth (m)
DEPTH_FAR_M = 10.0           # Farthest depth (m)

# Directional Safety Field
RISK_W_DEPTH = 0.35          # Depth signal weight
RISK_W_OBJECT = 0.25         # Object detection weight
RISK_W_FLOW = 0.25           # Optical flow weight
RISK_W_WALKABILITY = 0.15    # Walkability weight
RISK_TEMPORAL_ALPHA = 0.35   # EMA smoothing (lower = smoother)

# Risk thresholds (0–1)
RISK_FAR_THRESHOLD = 0.35    # "Warning" threshold
RISK_NEAR_THRESHOLD = 0.65   # "Urgent" threshold

# Navigation
NAV_COOLDOWN_SECONDS = 2.0   # Min interval between announcements
NAV_POTHOLE_AREA_RATIO = 0.03  # Pothole size trigger (fraction of frame)
NAV_WALKABLE_LOW_THRESHOLD = 30.0  # Low walkability % (STOP)
NAV_WALKABLE_MED_THRESHOLD = 55.0  # Medium walkability % (CAUTION)
```

### Mobile Configuration

**Camera Selection** (`mobile/app/src/main/java/com/example/gabai/WebRTCClient.kt`):
```kotlin
// Prefer external camera (set by user memory in preferences)
val cameraEnumerator = Camera2Enumerator(context)
val deviceName = cameraEnumerator.deviceNames.first()  // Change index for rear camera
```

**Bandwidth Tuning** (line ~150):
```kotlin
videoCapturer?.startCapture(320, 240, 10)  // width, height, fps
val parameters = sender?.parameters
parameters?.encodings?.forEach {
    it.maxBitrateBps = 250_000   // 250 kbps
    it.minBitrateBps = 100_000
    it.maxFramerate = 10
}
```

---

## Usage

### Running the Full System

#### 1. Start Backend Server
```bash
cd backend
python servertest.py
# Output: [2026-05-09 14:30:00] Serving on http://0.0.0.0:8080
```

#### 2. Launch Android App
- Select a mode: Navigation, ASL, OCR, or Color.
- Grant camera and microphone permissions.
- The app will connect to the server via WebRTC.

#### 3. Open Web Dashboard
- Navigate to `http://localhost:8080` in your browser.
- You should see live camera feeds and processed tracks.
- Navigate data and risk scores appear in the left panel.

### Mode-Specific Workflows

#### Navigation Mode
1. Start the app in Navigation mode.
2. Backend streams depth + YOLO + optical flow continuously.
3. Directional Safety Field computes risk every frame.
4. Audio cues: "Move left," "Stop," "Path clear," etc.
5. Haptics confirm each instruction.

#### OCR Mode
1. Select OCR mode from menu.
2. Point camera at text.
3. Backend preprocesses frame, runs Tesseract OCR, and stabilizes output.
4. App speaks: "The sign says: Walk here."
5. User can request repeat or translation.

#### ASL Mode
1. Perform ASL gesture in front of camera.
2. Backend runs YOLO hand detection + classification.
3. Stable result triggers TTS: "Hello," "Thank you," etc.
4. Cooldown prevents spam; user can speak immediately.

#### Color Mode
1. Point camera at object of interest.
2. Center ROI is extracted and analyzed.
3. Dominant color is named and announced: "Red," "Blue," etc.
4. RGB values displayed on mobile and dashboard.

---

## API Reference

### WebRTC Data Channels

All data is JSON-formatted, sent over negotiated WebRTC data channels:

#### Navigation Channel (id=2)

**Navigation Data (sent by server)**:
```json
{
  "primary": "MOVE_LEFT",
  "secondary": "person close",
  "urgency": "urgent",
  "nav_command": "Left, person close",
  "scene_risk": 0.68,
  "nav_angle_deg": -35.2,
  "confidence": 0.92,
  "direction_risks": {
    "left": 0.42,
    "slight_left": 0.51,
    "forward": 0.75,
    "slight_right": 0.68,
    "right": 0.59
  },
  "safest_direction": "left",
  "steering_strength": -0.65,
  "lane_confidence": 0.84,
  "collision_probability": 0.12,
  "path_continuity_score": 0.91,
  "future_risk_score": 0.28,
  "pothole_present": false,
  "should_speak": true
}
```

#### Detection Channel (id=0)

**YOLO Detections**:
```json
{
  "type": "detection_stats",
  "detections": [
    {
      "class": "person",
      "confidence": 0.95
    },
    {
      "class": "car",
      "confidence": 0.87
    }
  ],
  "fps": 12.3
}
```

#### Depth Channel (id=1)

**Depth Statistics**:
```json
{
  "type": "depth_stats",
  "min_depth": 0.35,
  "max_depth": 9.2,
  "fps": 14.1
}
```

#### OCR Channel

**Tesseract Output**:
```json
{
  "nav_command": "The sign says: Walk here",
  "confidence": 0.92,
  "should_speak": true,
  "mode": "ocr"
}
```

#### ASL Channel

**ASL Gesture Output**:
```json
{
  "nav_command": "Hello",
  "confidence": 0.87,
  "should_speak": true
}
```

#### Color Channel

**Color Output**:
```json
{
  "nav_command": "Red",
  "confidence": 0.78,
  "should_speak": true,
  "rgb": [220, 20, 60]
}
```

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves web dashboard (index.html) |
| `/offer` | POST | WebRTC signaling (client sends SDP offer) |
| `/events` | GET | Server-Sent Events (SSE) for source change notifications |
| `/stream-status` | GET | Returns `{"source_ready": bool, "generation": int}` |
| `/nav-state` | GET | Current navigation state snapshot |

---

## Performance & Optimization

### Latency Breakdown (End-to-End)

| Stage | Time (ms) |
|-------|-----------|
| Camera capture → encode | 33 (1 frame @ 30 fps) |
| WebRTC network latency | 20–100 (typical) |
| Frame decode | 10–15 |
| Inference (all models) | 50–200 (GPU) / 300–1000 (CPU) |
| Risk fusion & decision | 5 |
| TTS initialization (first time) | 500–1500 |
| TTS playback | Variable |
| **Total** | **500–2500 ms** |

### Optimization Tips

1. **Reduce input size** (`YOLO_INPUT_SIZE = 256` instead of 320).
2. **Use GPU**: CUDA support dramatically speeds up depth + YOLO.
3. **Skip frames selectively** (OCR/ASL can run every 200ms instead of 30ms).
4. **Disable non-essential tracks** on the viewer (only request navigation + raw feed).
5. **Tune EMA alpha**: Higher values (0.5–0.7) respond faster but are noisier.

### Bandwidth Optimization

- **Video encoding**: H.264 at 320x240, 10 fps, 250 kbps (adaptive bitrate).
- **Data channels**: Compact JSON; avoid floating-point bloat.
- **Typical usage**: 0.5–2 MB per minute depending on inference verbosity.

---

## Development

### Setting Up a Development Environment

```bash
git clone https://github.com/yourusername/gabai.git
cd gabai

# Backend dev setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest pylint black  # Optional: dev tools

# Mobile dev setup
cd ../mobile
# Open in Android Studio

# Frontend dev
cd ../frontend
# Use Live Server or python -m http.server 8000
```

### Code Style & Linting

```bash
# Format Python code
black backend/*.py

# Lint
pylint backend/*.py

# Type checking (optional)
mypy backend/config.py --ignore-missing-imports
```

### Testing

```bash
# Unit test depth inference
pytest backend/test_depth.py

# Integration test server connectivity
pytest backend/test_server.py
```

### Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit changes with clear messages.
4. Push to your fork and open a Pull Request.

---

## Troubleshooting

### Backend Server Won't Start

**Issue**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue**: Model file not found at `models/yolo/yolov8x.pt`

**Solution**:
```bash
# Download manually or let the YOLO library auto-download
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
```

### Mobile App Won't Connect

**Issue**: "Cannot reach server"

**Solution**:
1. Verify backend is running: `curl http://YOUR_SERVER_IP:8080`
2. Check firewall allows port 8080.
3. Update `serverUrl` in `WebRTCClient.kt` to correct IP.
4. Use `adb logcat` to view detailed connection errors.

### Web Dashboard Shows Black/No Video

**Issue**: Video track not streaming

**Solution**:
1. Check browser console for WebRTC errors.
2. Ensure backend has an active sender (Android app connected).
3. Inspect `/stream-status` endpoint: should return `"source_ready": true`.

### High Latency / Choppy Video

**Issue**: Inference too slow

**Solution**:
1. Use GPU: Verify CUDA support (`torch.cuda.is_available()`).
2. Reduce input size: `YOLO_INPUT_SIZE = 256`.
3. Disable non-critical tracks (e.g., pothole detection).
4. Increase bitrate if network allows.

### Tesseract OCR Not Working

**Issue**: `pytesseract.TesseractNotFoundError: tesseract is not installed`

**Solution**:
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: Install [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and update path in `ocr_detection.py`.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)**: Universal monocular depth estimation.
- **[YOLOv8](https://github.com/ultralytics/ultralytics)**: Object detection framework.
- **[Tesseract OCR](https://github.com/UB-Mannheim/tesseract)**: Lightweight text recognition.
- **[OpenCV](https://opencv.org/)**: Computer vision library.
- **[WebRTC](https://webrtc.org/)**: Real-time communication standard.
- **[Android WebRTC](https://github.com/google/webrtc/tree/master/sdk/android)**: Mobile WebRTC implementation.

---

**Questions or Issues?** Please open an issue on [GitHub Issues](https://github.com/yourusername/gabai/issues).

---

*Last Updated: May 9, 2026*
