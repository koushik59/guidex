# GuideX – AI Navigation Assistant for Visually Impaired People

## Overview

GuideX is an AI-powered navigation assistant designed to help visually impaired individuals navigate their surroundings safely and independently.

The system combines real-time object detection, OCR (text reading), facial recognition, offline voice commands, and speech feedback into a single application running on the NVIDIA Jetson platform.

The application provides spoken alerts about nearby obstacles, reads text from the environment, recognizes familiar faces, and allows complete hands-free interaction using offline voice commands.

---

# Features

## Real-Time Object Detection

- YOLOv8 Nano model
- Detects pedestrians
- Cars
- Trucks
- Buses
- Motorcycles
- Traffic lights
- Chairs
- Tables
- Benches
- Fire hydrants
- Stop signs
- Potted plants
- Bags
- Laptops
- Books
- Cell phones
- and many more.

Each detected object includes:

- Distance estimation
- Direction (Left / Center / Right)
- Danger level
- Priority scoring

---

## Smart Voice Alerts

GuideX intelligently decides when to announce objects.

Examples:

- Large vehicle approaching from your left.
- Person approaching fast on your right.
- Red signal. Do not cross.
- Green signal. Safe to cross.

Alert cooldown logic prevents repeated announcements.

---

## OCR Text Reading

EasyOCR is used to read text from the camera.

Examples:

- Shop names
- Street signs
- Product labels
- Documents
- Notices

The detected text is automatically spoken using Text-to-Speech.

---

## Facial Recognition

Recognizes previously registered people.

Features:

- Register new faces
- Reload face database
- Identify known people
- Detect unknown persons
- Announces their relative position

Example:

"I can see John on your left."

---

## Offline Voice Commands

Uses Vosk Offline Speech Recognition.

Supported commands:

- Scan
- Read
- Who
- Face
- Recognize
- Stop

Examples:

User:

Scan

GuideX:

Reads nearby text aloud.

User:

Who

GuideX:

Recognizes nearby faces.

User:

Stop

GuideX:

Immediately stops speech.

---

## Local Text-to-Speech

Speech output works completely offline using:

- espeak-ng
- pyttsx3 (fallback)

No internet connection is required.

---

## Flask Web Dashboard

The web interface provides:

- Live camera feed
- Start detection
- Stop detection
- OCR Scan button
- Face Recognition button
- Environment selection
- Alert mode selection
- Emergency SOS

---

# Technologies Used

## Backend

- Python 3
- Flask
- OpenCV
- NumPy
- Threading

---

## AI Models

- YOLOv8 Nano (Ultralytics)
- EasyOCR
- face_recognition
- Vosk Speech Recognition

---

## Speech

- espeak-ng
- pyttsx3

---

## Frontend

- HTML
- CSS
- JavaScript
- Flask Templates

---

# Project Structure

```
GuideX/

│
├── app.py
├── yolov8n.pt
├── model/
│
├── known_faces/
│      John.jpg
│      Alice.jpg
│
├── templates/
│      index.html
│
├── static/
│      css/
│      js/
│      images/
│
├── requirements.txt
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/GuideX.git

cd GuideX
```

---

## Create Virtual Environment

Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Speech Engine

Ubuntu / Jetson

```bash
sudo apt install espeak-ng

sudo apt install alsa-utils
```

---

## Download YOLO Model

Place

```
yolov8n.pt
```

inside the project directory.

---

## Download Vosk Model

Download an English Vosk model and place it as

```
model/
```

inside the project.

---

# Running the Application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

in your browser.

---

# Voice Commands

| Command | Action |
|----------|--------|
| Scan | Read nearby text |
| Read | Read nearby text |
| Who | Recognize faces |
| Face | Recognize faces |
| Recognize | Recognize faces |
| Stop | Stop speech |

---

# API Endpoints

## Camera

```
GET /video_feed
```

Live video stream.

---

## Detection

```
POST /start
```

Start detection.

```
POST /stop
```

Stop detection.

```
GET /status
```

Returns detection status.

---

## OCR

```
POST /scan
```

Runs OCR and speaks the detected text.

```
GET /latest_scan
```

Returns the most recent OCR result.

---

## Facial Recognition

```
POST /recognize_face
```

Recognize faces.

```
POST /register_face
```

Register a new face.

```
GET /list_faces
```

List all registered faces.

```
POST /reload_faces
```

Reload the face database.

```
GET /latest_face
```

Returns latest recognition result.

---

## Speech

```
POST /stop_speech
```

Immediately stops audio playback.

---

## Environment

```
POST /set_environment
```

Switch between:

- Indoor
- Outdoor
- Auto

---

## Alert Mode

```
POST /set_alert_mode
```

Switch between:

- English
- Sound

---

# Key Algorithms

- Object Detection using YOLOv8
- Distance Estimation using Bounding Box Height
- Priority-Based Threat Assessment
- Dynamic Voice Alert Scheduling
- OCR using EasyOCR
- Face Encoding using face_recognition
- Offline Speech Recognition using Vosk
- Local Speech Synthesis using espeak-ng

---

# Performance Optimizations

- Multi-threaded architecture
- Separate camera and detection threads
- OCR locking for thread safety
- Face recognition runs only on demand
- Reduced detection image size
- OCR warm-up during startup
- JPEG compression for streaming
- Alert cooldown system
- Priority-based announcements

---

# Future Improvements

- GPS Navigation Integration
- Google Maps Navigation
- Obstacle Path Planning
- Depth Estimation
- Traffic Sign Classification
- Currency Recognition
- Scene Captioning
- Indoor Localization
- Emergency Contact Calling
- Cloud-based Face Synchronization

---

# Authors

GuideX Development Team

Developed as an AI-powered assistive technology project to improve mobility and independence for visually impaired individuals.

---

# License

This project is intended for educational and research purposes.
