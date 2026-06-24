<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask 3.0"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/EasyOCR-1.7-FF6F00?style=for-the-badge" alt="EasyOCR"/>
</p>

# 🧿 GuideX — AI Navigation Assistant for the Visually Impaired

**GuideX** is a real-time, web-based AI navigation assistant designed to help blind and visually impaired people navigate safely. It combines object detection, traffic signal recognition, OCR text reading, Google Maps walking navigation, and multi-modal feedback (voice, beeps, haptics) into a single, accessible dashboard.

---

## ✨ Feature Overview

| Feature | Description |
|---|---|
| **Real-Time Object Detection** | YOLOv8 detects 30+ object classes (vehicles, furniture, obstacles, people) from a live camera feed |
| **Priority-Based Alerts** | Only the single most dangerous object triggers an alert — avoids sensory overload |
| **Directional Audio Guidance** | Spoken alerts include left / center / right positioning |
| **Traffic Signal Recognition** | HSV color analysis identifies red vs. green traffic lights in real time |
| **Zebra Crossing Detection** | Hough line analysis detects crosswalk markings near traffic signals |
| **Approach Speed Estimation** | Frame-to-frame tracking estimates whether objects are approaching or receding |
| **Person Intent Awareness** | Approaching people are prioritized; those moving away are deprioritized |
| **OCR Text Reading** | EasyOCR extracts text from signs, documents, and boards, reads it aloud, and shows scanned text in Real-time Detections |
| **Offline Voice Commands** | Vosk listens for "scan" / "read" to trigger OCR and "stop" to stop speech |
| **Google Maps Navigation** | Walking directions with turn-by-turn voice guidance and live route tracking |
| **SOS Emergency System** | Long-press (2 sec) emergency button with haptic + audio confirmation |
| **Dual Alert Modes** | English voice alerts or beep/siren sound patterns |
| **Environment Modes** | Indoor, Outdoor, and Auto modes adjust detection priority logic |
| **Haptic Feedback** | Vibration API patterns for alerts, confirmations, and errors |
| **Premium Dark UI** | Glassmorphic dashboard with mesh gradients, micro-animations, and responsive layout |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Frontend)                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────────┐  │
│  │ Camera   │  │ Google   │  │ Detection │  │ Audio / TTS   │  │
│  │ Feed     │  │ Maps     │  │ Panel     │  │ (Speech API)  │  │
│  │ (MJPEG)  │  │ (API)    │  │           │  │               │  │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └──────┬────────┘  │
│       │              │              │               │           │
│  Web Speech API · Web Audio API · Vibration API · Fetch API    │
└───────┼──────────────┼──────────────┼───────────────┼───────────┘
        │              │              │               │
   /video_feed    Google APIs    /get_alert       /scan
        │                             │           /latest_scan
        ▼                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Flask Backend (Python)                       │
│                                                                 │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │ Camera Thread  │  │ Detection       │  │ Audio Worker     │  │
│  │ (capture loop) │→ │ Thread (YOLO)   │→ │ (local TTS)     │  │
│  └────────────────┘  └─────────────────┘  └──────────────────┘  │
│                                                                 │
│  ┌──────────┐  ┌───────────────────────┐                        │
│  │ EasyOCR  │  │ Priority Engine       │                        │
│  │ Reader   │  │ (multi-factor scoring)│                        │
│  └──────────┘  └───────────────────────┘                        │
│                                                                 │
│  YOLOv8n.pt (COCO)  ·  Vosk model (speech-to-text, bundled)   │
└─────────────────────────────────────────────────────────────────┘
```

### Threading Model

GuideX runs background workers to keep camera capture, detection, speech, and voice commands responsive:

| Thread | Purpose | Rate |
|---|---|---|
| **Camera Capture** | Continuously grabs frames from the webcam | ~24 FPS |
| **Detection Worker** | Runs YOLOv8 inference on the latest frame | Every ~180 ms |
| **Audio Worker** | Dequeues alert/OCR messages and speaks them through `espeak-ng`, `espeak`, or pyttsx3 fallback | On demand |
| **Voice Command Worker** | Uses Vosk + microphone input for offline "scan/read" and "stop" commands | On demand |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- A **webcam** (USB or built-in)
- **Google Maps API key** (for navigation features — [get one here](https://console.cloud.google.com/apis/credentials))
- Optional but recommended: **espeak-ng** or **espeak** for reliable backend speech output

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/koushik59/guidex.git
cd guidex/Guidex

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The `yolov8n.pt` model file is included in the repo. If missing, Ultralytics will auto-download it on first run.

### Running the Application

```bash
python web_app.py
```

Then open **http://localhost:5000** in your browser and allow camera access when prompted.

---

## 📖 Usage Guide

### Core Workflow

1. **Open the dashboard** → Camera feed starts streaming automatically
2. **Click "Start Navigation"** → YOLOv8 detection activates
3. **Receive audio alerts** → The system announces the most dangerous nearby object
4. **Read nearby text** → Click **Read Text**, press **T**, or say **"scan" / "read"**
5. **Use the sidebar** → Switch alert modes, environment modes, locate yourself, toggle map
6. **Stop anytime** → Click **Stop** to pause detection (camera feed stays live)

### OCR Text Scanning

Text scanning uses the same backend flow whether it is triggered from the UI or by voice:

- The **Read Text** button and **T** keyboard shortcut call `/scan`
- Voice commands **"scan"** or **"read"** call the same OCR function in `web_app.py`
- Detected text is printed in the terminal as `[SCAN] Detected text: ...`
- The latest scanned text is exposed through `/latest_scan`
- The browser polls `/latest_scan` and displays new scanned text in **Real-time Detections**
- The legacy `/read_text` endpoint still returns OCR text without speech for compatibility

### SOS Emergency

Press and **hold the SOS button for 2 seconds** to activate. The system provides:
- Haptic vibration pattern (200-100-200-100-200 ms)
- Audio confirmation via speech synthesis
- Server-side acknowledgment (extensible for SMS/call integration)

### Alert System Deep Dive

#### Object Categories & Distance Thresholds

| Category | Objects | HIGH Danger | MEDIUM Danger | LOW |
|---|---|---|---|---|
| **Large Vehicle** | Car, Bus, Truck | < 25 m | < 45 m | > 45 m |
| **Medium Vehicle** | Motorcycle | < 15 m | < 30 m | > 30 m |
| **Small Object** | Person, Bicycle, Furniture, etc. | < 1.5 m | < 3 m | > 3 m |

#### Priority Scoring Formula

Each detected object gets a priority score computed as:

```
priority = level_factor × type_factor × speed_factor × environment_factor
```

| Factor | Values |
|---|---|
| **Level** | LOW = 1.0, MEDIUM = 2.0, HIGH = 3.0 |
| **Type** | large_vehicle = 3.0, medium_vehicle = 2.0, small_object = 1.0 |
| **Speed** | 1.0 + min(abs(speed), 10.0) / 5.0 — approaching people get 3× boost |
| **Environment** | Outdoor: vehicles get 1.3×, Indoor: small objects get 1.3× |

Only the **single highest-priority** object triggers an alert per cycle.

#### Per-Category Alert Cooldowns

| Category | Cooldown |
|---|---|
| Large Vehicle | 2.0 seconds |
| Medium Vehicle | 3.0 seconds |
| Small Object | 5.0 seconds |

---

## 🗺️ Google Maps Integration

The map panel provides:

- **Live GPS tracking** with a styled marker on a dark-themed map
- **Places Autocomplete** for destination search
- **Walking directions** with route rendering on the map
- **Turn-by-turn voice guidance** that auto-advances as you walk
- **Route info** showing distance and estimated walking time
- **Periodic reminders** (every 30 seconds if no step change)

> **Required APIs**: Enable **Maps JavaScript API**, **Places API**, and **Directions API** in your Google Cloud Console project.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the main dashboard HTML |
| `GET` | `/video_feed` | MJPEG video stream with detection overlays |
| `POST` | `/start` | Start object detection |
| `POST` | `/stop` | Stop object detection |
| `GET` | `/status` | Returns `{ "running": true/false }` |
| `GET` | `/get_alert` | Dequeues the next alert message |
| `POST` | `/set_alert_mode` | Set alert mode: `"english"` or `"sound"` |
| `GET` | `/get_alert_mode` | Get current alert mode |
| `POST` | `/set_environment` | Set environment: `"indoor"`, `"outdoor"`, or `"auto"` |
| `GET` | `/get_environment` | Get current environment mode |
| `GET` | `/scene_description` | Legacy natural-language summary of current detections |
| `GET` | `/read_text` | Legacy OCR endpoint that extracts text without speech |
| `GET`/`POST` | `/scan` | Runs OCR on the latest frame, speaks the result, stores it as the latest scan, and returns `{ "text", "timestamp" }` |
| `GET` | `/latest_scan` | Returns the most recent OCR text and timestamp for the Real-time Detections panel |
| `GET`/`POST` | `/stop_speech` | Stops local backend speech playback |
| `GET` | `/voice_status` | Returns whether optional voice command support is available/running |
| `POST` | `/sos` | Receives emergency SOS requests |

---

## 📁 Project Structure

```
Guidex_1/
├── .gitignore
└── Guidex/
    ├── web_app.py                     # Flask backend — detection, alerts, API routes
    ├── requirements.txt               # Python dependencies
    ├── README.md                      # Project README
    ├── yolov8n.pt                     # YOLOv8 Nano model weights (COCO)
    ├── vosk-model-small-en-us-0.15.zip # Vosk speech recognition model (bundled)
    ├── model/                         # Extracted Vosk model files
    │   ├── README
    │   ├── am/                        # Acoustic model
    │   ├── conf/                      # Configuration
    │   ├── graph/                     # Language model graph
    │   └── ivector/                   # i-vector extractor
    ├── templates/
    │   └── index.html                 # Main dashboard template (Jinja2)
    └── static/
        ├── style.css                  # Premium dark UI with glassmorphism
        ├── script.js                  # Frontend logic — alerts, TTS, SOS, haptics
        └── maps.js                    # Google Maps — location, directions, guidance
```

---

## ⚙️ Configuration

All configurable values are in `web_app.py`:

| Parameter | Default | Description |
|---|---|---|
| `CAMERA_WIDTH` | 640 | Camera capture width |
| `CAMERA_HEIGHT` | 480 | Camera capture height |
| `DETECTION_WIDTH` | 416 | Frame resize for YOLO inference |
| `DETECTION_INTERVAL` | 0.18s | Seconds between detection cycles |
| `STREAM_FPS` | 24 | MJPEG streaming frame rate |
| `JPEG_QUALITY` | 70 | JPEG compression quality (0–100) |
| `DANGEROUS_CLASSES` | 30+ classes | Object classes to detect and alert on |
| `COOLDOWN_MAP` | varies | Per-category alert cooldown timers |
| `environment_mode` | `outdoor` | Default environment mode |
| `alert_mode` | `english` | Default alert mode |

---

## ♿ Accessibility

GuideX is designed with accessibility as a first-class concern:

- **Keyboard Navigation** — All controls are keyboard-accessible with Space/Enter
- **ARIA Labels** — Every interactive element has descriptive `aria-label` attributes
- **Live Regions** — Detection panel uses `aria-live="assertive"` for screen readers
- **Reduced Motion** — CSS respects `prefers-reduced-motion` media query
- **High Contrast** — Dark theme with carefully chosen contrast ratios
- **Multiple Feedback Channels** — Voice, beeps/sirens, haptic vibration, and visual indicators
- **Responsive Design** — Works on desktop and mobile; sidebar collapses to horizontal on small screens
- **Screen Reader Friendly** — Hidden `.sr-only` utility class for screen-reader-only text

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | Flask 3.0, Flask-CORS |
| **Object Detection** | YOLOv8 Nano (Ultralytics), OpenCV 4.8 |
| **OCR** | EasyOCR 1.7 |
| **Text-to-Speech** | espeak-ng/espeak or pyttsx3 fallback (backend), Web Speech API (frontend) |
| **Speech Recognition** | Vosk (offline, bundled model) |
| **Maps & Navigation** | Google Maps JavaScript API, Places API, Directions API |
| **Frontend** | HTML5, CSS3 (custom properties, grid, glassmorphism), vanilla JavaScript |
| **Audio Feedback** | Web Audio API (oscillator-based beeps/sirens) |
| **Haptic Feedback** | Vibration API |
| **Deep Learning** | PyTorch 2.0, torchvision |

---

## 🐛 Troubleshooting

### Camera Not Working
- Ensure no other application is using the camera
- Check browser permissions for camera access
- The app tries camera indices 0, 1, 2, and 700 automatically

### Audio Not Working
- **Backend TTS**: Prefers `espeak-ng`, then `espeak`, then pyttsx3; falls back to a silent logger if none are available
- **Frontend TTS**: Requires a browser that supports the Web Speech API (Chrome, Edge, Firefox)
- Check system audio volume and output device

### Voice Commands Not Working
- Voice commands require `sounddevice`, PortAudio microphone access, and the bundled/extracted Vosk `model/` folder
- Check the terminal for `[VOICE]` messages; the app disables voice commands gracefully if audio input is unavailable
- The supported commands are **"scan" / "read"** for OCR and **"stop"** for speech stop

### Model Not Loading
- Ensure `yolov8n.pt` is present in the `Guidex/` directory
- Internet is required on first run if the model needs to be downloaded

### Maps Not Loading
- Replace the Google Maps API key in `templates/index.html` with your own key
- Enable **Maps JavaScript API**, **Places API**, and **Directions API** in Google Cloud Console
- Ensure billing is active on your Google Cloud project

---

## 🔮 Roadmap

- [ ] GPS-integrated outdoor navigation
- [ ] Obstacle pathfinding suggestions
- [ ] Multiple camera support
- [ ] Mobile app version (React Native / Flutter)
- [ ] Offline mode with on-device models
- [ ] Caregiver notification integration (SMS/push)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is created for assistive technology purposes.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection model
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — Optical character recognition
- [Vosk](https://alphacephei.com/vosk/) — Offline speech recognition
- [Google Maps Platform](https://developers.google.com/maps) — Navigation and geocoding APIs
