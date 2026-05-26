from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import time
import pyttsx3
from ultralytics import YOLO
import threading
import queue
import base64
import numpy as np
from io import BytesIO
import os
import atexit
import signal
import easyocr
import json
import urllib.error
import urllib.request

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Change to script directory to ensure relative paths work
os.chdir(BASE_DIR)

# Debug: Print paths to verify they're correct
print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] TEMPLATE_DIR: {TEMPLATE_DIR}")
print(f"[DEBUG] STATIC_DIR: {STATIC_DIR}")
print(f"[DEBUG] Current working directory: {os.getcwd()}")
print(f"[DEBUG] Templates exist: {os.path.exists(TEMPLATE_DIR)}")
print(f"[DEBUG] index.html exists: {os.path.exists(os.path.join(TEMPLATE_DIR, 'index.html'))}")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DETECTION_WIDTH = 416
DETECTION_INTERVAL = 0.18
STREAM_FPS = 24
JPEG_QUALITY = 70
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# ------------------ Setup ------------------
# Expanded object detection including furniture and obstacles
# Note: YOLO COCO dataset includes these classes. "Tree" is not in COCO, but "potted plant" can represent small trees/plants
DANGEROUS_CLASSES = [
    "person", "car", "bus", "truck", "motorcycle", "bicycle",
    "chair", "couch", "potted plant", "bed", "bench", "dining table",
    "tv", "laptop", "sink", "refrigerator", "toilet", "umbrella",
    "backpack", "handbag", "suitcase", "fire hydrant", "stop sign",
    "traffic light", "pothole", "stairs", "water puddle", "construction zone",
    "book", "cell phone"
]

# Object categories for different alert thresholds
LARGE_VEHICLES = ["car", "bus", "truck"]  # Need alerts from far away
MEDIUM_VEHICLES = ["motorcycle"]  # Moderate distance alerts
OBSTACLES = ["chair", "couch", "bed", "bench", "dining table", "refrigerator", "toilet", 
             "fire hydrant", "stop sign", "pothole", "stairs", "water puddle", "construction zone"]  # Furniture/obstacles
SMALL_OBJECTS = ["person", "bicycle", "potted plant", "tv", "laptop", "sink", 
                 "umbrella", "backpack", "handbag", "suitcase", "traffic light", "book", "cell phone"]  # Close range alerts

model = YOLO("yolov8n.pt")

class SilentSpeechEngine:
    def say(self, message):
        print(f"[TTS disabled] {message}")

    def runAndWait(self):
        return None

    def stop(self):
        return None

try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 0.9)
except Exception as e:
    print(f"[WARNING] Local pyttsx3 speech disabled: {e}")
    engine = SilentSpeechEngine()

# Initialize EasyOCR
print("Initializing EasyOCR (this may take a moment to download models on first run)...")
reader = easyocr.Reader(['en'])
print("EasyOCR initialized.")

# Global state
alert_queue = queue.Queue()
audio_alert_queue = queue.Queue()
camera = None
is_running = False
latest_detections = []
latest_frame = None
latest_annotated_frame = None
latest_frame_lock = threading.Lock()
camera_thread = None
detection_thread = None
camera_stop_event = threading.Event()
detection_stop_event = threading.Event()

# Alert / environment configuration
alert_mode = "english"  # or "sound" (used by frontend for persistence only)
environment_mode = "outdoor"  # "indoor", "outdoor", or "auto"

# Per-object-type cooldown
COOLDOWN_MAP = {
    "large_vehicle": 2.0,   # alert faster
    "medium_vehicle": 3.0,
    "small_object": 5.0,    # slower alerts for furniture / small obstacles
}

# Track last alert time per object category
last_alert_times = {
    "large_vehicle": 0.0,
    "medium_vehicle": 0.0,
    "small_object": 0.0,
}

# Simple tracking state to estimate approach speed
# key: (label, direction) -> {"distance": float, "time": float}
object_track_state = {}

# ------------------ Helper Functions ------------------

def estimate_distance(box_height, frame_height):
    """Estimate distance based on bounding box height"""
    if box_height == 0:
        return 999
    return (frame_height / box_height) * 0.5

def get_object_category(label):
    """Categorize object by size/danger level"""
    if label in LARGE_VEHICLES:
        return "large_vehicle"
    elif label in MEDIUM_VEHICLES:
        return "medium_vehicle"
    elif label in SMALL_OBJECTS:
        return "small_object"
    else:
        return "small_object"  # Default to small object

def danger_level(distance, object_category):
    """
    Determine danger level based on distance and object category.
    Larger vehicles need alerts from much farther away.
    
    Thresholds:
    - Large vehicles (car, bus, truck): HIGH < 25m, MEDIUM < 45m
    - Medium vehicles (motorcycle): HIGH < 15m, MEDIUM < 30m
    - Small objects (person, bicycle): HIGH < 1.5m, MEDIUM < 3m
    """
    if object_category == "large_vehicle":
        # Large vehicles: alert from far away (they're fast and dangerous)
        if distance < 25:
            return "HIGH"
        elif distance < 45:
            return "MEDIUM"
        else:
            return "LOW"
    elif object_category == "medium_vehicle":
        # Medium vehicles: moderate distance alerts
        if distance < 15:
            return "HIGH"
        elif distance < 30:
            return "MEDIUM"
        else:
            return "LOW"
    else:  # small_object
        # Small objects: close range alerts (current behavior)
        if distance < 1.5:
            return "HIGH"
        elif distance < 3:
            return "MEDIUM"
        else:
            return "LOW"

def get_direction(x1, x2, frame_width):
    """Get direction of object relative to frame"""
    center_x = (x1 + x2) / 2
    if center_x < frame_width / 3:
        return "left"
    elif center_x < 2 * frame_width / 3:
        return "center"
    else:
        return "right"

def compute_priority(level, object_category, speed_mps, label=""):
    """
    Compute a numeric priority score for a detected object.
    Higher score = higher priority for alerts.
    Priority factors in:
    - Danger level (HIGH/MEDIUM/LOW)
    - Object type (vehicles > small objects)
    - Approach speed (faster objects are more dangerous)
    - Environment context (outdoor = prioritize vehicles, indoor = prioritize obstacles)
    """
    level_factor = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0}.get(level, 1.0)
    type_factor = {
        "large_vehicle": 3.0,
        "medium_vehicle": 2.0,
        "small_object": 1.0,
    }.get(object_category, 1.0)

    # Speed logic
    speed = float(speed_mps or 0.0)
    speed_factor = 1.0 + min(abs(speed), 10.0) / 5.0

    # Human intent prioritization
    if label == "person":
        if speed > 0.3: # Approaching
            speed_factor *= 3.0 # Highly prioritize moving threats
        elif speed < -0.3: # Going away
            speed_factor *= 0.2 # Deprioritize people walking away

    # Environment context adjustment
    env_factor = 1.0
    if environment_mode == "outdoor":
        # Outdoors: prioritize vehicles more
        if object_category in ("large_vehicle", "medium_vehicle"):
            env_factor = 1.3
    elif environment_mode == "indoor":
        # Indoors: slightly prioritize small/obstacle-type objects
        if object_category == "small_object":
            env_factor = 1.3

    return level_factor * type_factor * speed_factor * env_factor

def queue_alert(message):
    """Send alerts to both browser clients and the local TTS worker."""
    alert_queue.put(message)
    audio_alert_queue.put(message)

def get_latest_camera_frame():
    """Return a copy of the latest raw camera frame."""
    with latest_frame_lock:
        if latest_frame is not None:
            return latest_frame.copy()
    return None

def encode_frame_for_vision(frame, max_width=768):
    """Encode a frame as a compact base64 JPEG for VLM requests."""
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, int(height * scale)))

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")

def call_gemini_vision(frame, user_prompt=""):
    """Ask Gemini to describe the latest scene for blind navigation."""
    if not GEMINI_API_KEY:
        return None, "Gemini API key is not set. Add GEMINI_API_KEY before using Smart Look."

    image_base64 = encode_frame_for_vision(frame)
    if not image_base64:
        return None, "Could not prepare the camera image for Gemini."

    detections_context = latest_detections[:8] if latest_detections else []
    prompt = (
        "You are Mickey, an assistive vision guide for a blind person. "
        "Analyze the camera image and give concise, practical guidance. "
        "Mention immediate hazards first, then useful navigation cues, then any readable signs or text. "
        "Use short spoken sentences. Do not overclaim. If uncertain, say so. "
        "Do not replace the user's cane, guide dog, or human judgment.\n\n"
        f"User question: {user_prompt or 'What is around me and what should I be careful about?'}\n"
        f"Fast detector context: {json.dumps(detections_context)}"
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64,
                }},
            ],
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 220,
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    request_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=request_data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[Gemini] HTTP error {e.code}: {body}")
        return None, f"Gemini request failed with HTTP {e.code}."
    except Exception as e:
        print(f"[Gemini] Request error: {e}")
        return None, "Gemini request failed. Check internet connection and API key."

    try:
        parts = result["candidates"][0]["content"]["parts"]
        text = " ".join(part.get("text", "") for part in parts).strip()
        if text:
            return text, None
    except Exception:
        pass

    return None, "Gemini did not return a scene description."


def process_frame(frame):
    """Process a single frame and return detection results"""
    results = model(frame, imgsz=DETECTION_WIDTH, verbose=False)
    frame_height, frame_width, _ = frame.shape
    
    detections = []
    current_time = time.time()
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])
        
        if label in DANGEROUS_CLASSES and confidence > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_height = y2 - y1
            distance = estimate_distance(box_height, frame_height)
            
            # Get object category and determine danger level
            object_category = get_object_category(label)
            level = danger_level(distance, object_category)
            direction = get_direction(x1, x2, frame_width)

            # Traffic Signal logic
            if label == "traffic light":
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                    mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
                    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
                    
                    if cv2.countNonZero(mask_green) > cv2.countNonZero(mask_red) and cv2.countNonZero(mask_green) > 10:
                        label = "green traffic light"
                    elif cv2.countNonZero(mask_red) > 10:
                        label = "red traffic light"
                        
                    # Zebra crossing heuristic (look for horizontal white lines below the traffic light)
                    if y2 < frame_height - 50:
                        roi_bottom = frame[y2:, max(0, x1-50):min(frame_width, x2+50)]
                        gray_bottom = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray_bottom, 50, 150, apertureSize=3)
                        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
                        if lines is not None and len(lines) > 2:
                            label += " with zebra crossing"

            # Simple tracking-based speed estimation using previous distance
            track_key = (label.split(" ")[-1], direction) # Use base label for tracking
            prev_state = object_track_state.get(track_key)
            speed_mps = 0.0
            if prev_state:
                dt = current_time - prev_state["time"]
                if dt > 0:
                    # Positive speed means object is getting closer, negative means going away
                    speed_mps = (prev_state["distance"] - distance) / dt
            # Update track state
            object_track_state[track_key] = {"distance": distance, "time": current_time}

            # Compute priority score for this detection
            priority_score = compute_priority(level, object_category, speed_mps, label)

            detections.append({
                "label": label,
                "distance": round(distance, 2),
                "level": level,
                "direction": direction,
                "confidence": round(confidence, 2),
                "bbox": [x1, y1, x2, y2],
                "category": object_category,
                "speed": round(speed_mps, 2),
                "priority": round(priority_score, 2),
            })

    # ---------------- Priority-based alert selection ----------------
    # Only raise alert for the single highest-priority object at a time
    if detections:
        # Pick detection with maximum priority score
        best_detection = max(detections, key=lambda d: d.get("priority", 0.0))
        category = best_detection["category"]
        level = best_detection["level"]
        direction = best_detection["direction"]
        label = best_detection["label"]

        # Should we alert for this detection?
        should_alert = False
        if category == "large_vehicle":
            # For large vehicles, alert on HIGH and MEDIUM
            should_alert = level in ("HIGH", "MEDIUM")
        else:
            # For other objects, alert only on HIGH
            should_alert = level == "HIGH"

        if should_alert:
            last_time_for_category = last_alert_times.get(category, 0.0)
            cooldown = COOLDOWN_MAP.get(category, 4.0)

            if current_time - last_time_for_category > cooldown:
                # Create appropriate alert message based on object type and danger level.
                # Keep English messages compatible with the existing frontend parsing.
                if "green traffic light" in label:
                    alert_message = "Signal is green, safe to cross."
                elif "red traffic light" in label:
                    alert_message = "Signal is red, do not cross."
                elif category == "large_vehicle":
                    if level == "HIGH":
                        # High danger, very close / approaching fast
                        alert_message = f"Large vehicle very close on your {direction}. Please stop."
                    else:  # MEDIUM
                        alert_message = f"Large vehicle approaching from your {direction}. Be cautious."
                elif category == "medium_vehicle":
                    alert_message = f"Motorcycle very close on your {direction}. Please stop."
                elif label == "person":
                    if best_detection.get("speed", 0.0) > 0.3:
                        alert_message = f"Person approaching fast on your {direction}."
                    elif best_detection.get("speed", 0.0) < -0.3:
                        alert_message = f"Person moving away on your {direction}."
                    else:
                        alert_message = f"Stationary person on your {direction}."
                else:  # small_object and furniture / hazards
                    alert_message = f"{label} very close on your {direction}. Please stop."

                queue_alert(alert_message)
                last_alert_times[category] = current_time

    global latest_detections
    latest_detections = detections
    return detections

def draw_detections(frame, detections):
    """Draw current detections on a frame for the live preview."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        level = det["level"]
        color = (0, 0, 255) if level == "HIGH" else (0, 255, 255) if level == "MEDIUM" else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{det['label']} | {det['level']} | {det['direction']}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return frame

def open_camera():
    """Open a camera with low-latency settings."""
    global camera

    print("[DEBUG] Attempting to open camera...")
    for index in [0, 1, 2, 700]:
        try:
            candidate = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(index)
            candidate.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            candidate.set(cv2.CAP_PROP_FPS, STREAM_FPS)
            candidate.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if candidate.isOpened():
                ret, _ = candidate.read()
                if ret:
                    camera = candidate
                    print(f"[SUCCESS] Camera opened on index {index}")
                    return True
                print(f"[WARNING] Camera opened on index {index} but failed to read frame.")
            candidate.release()
        except Exception as e:
            print(f"[ERROR] Failed to open camera index {index}: {e}")

    camera = None
    return False

def camera_capture_worker():
    """Continuously capture fresh frames so streaming never waits for YOLO."""
    global camera, latest_frame, latest_annotated_frame

    while not camera_stop_event.is_set():
        if camera is None or not camera.isOpened():
            if not open_camera():
                print("[ERROR] Could not open any camera. Retrying in 2s...")
                time.sleep(2)
                continue

        ret, frame = camera.read()
        if not ret:
            print("[WARNING] Failed to read frame from camera. Releasing and retrying...")
            try:
                if camera:
                    camera.release()
            except Exception:
                pass
            camera = None
            time.sleep(0.2)
            continue

        with latest_frame_lock:
            latest_frame = frame.copy()
            if latest_annotated_frame is None:
                latest_annotated_frame = frame.copy()

        time.sleep(1 / STREAM_FPS)

def detection_worker():
    """Run detection in the background at a controlled rate."""
    global latest_annotated_frame

    while not detection_stop_event.is_set():
        frame_to_process = None
        with latest_frame_lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()

        if frame_to_process is None:
            time.sleep(0.05)
            continue

        if is_running:
            height, width = frame_to_process.shape[:2]
            scale = 1.0
            if width > DETECTION_WIDTH:
                scale = DETECTION_WIDTH / width
                resized = cv2.resize(frame_to_process, (DETECTION_WIDTH, int(height * scale)))
            else:
                resized = frame_to_process

            detections = process_frame(resized)
            if scale != 1.0:
                for det in detections:
                    det["bbox"] = [int(coord / scale) for coord in det["bbox"]]

            annotated = draw_detections(frame_to_process, detections)
        else:
            annotated = frame_to_process
            cv2.putText(
                annotated,
                "Ready - Press Start to begin detection",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 100),
                2,
            )

        with latest_frame_lock:
            latest_annotated_frame = annotated

        time.sleep(DETECTION_INTERVAL)

def ensure_background_workers():
    """Start camera and detection workers once."""
    global camera_thread, detection_thread

    if camera_thread is None or not camera_thread.is_alive():
        camera_stop_event.clear()
        camera_thread = threading.Thread(target=camera_capture_worker, daemon=True)
        camera_thread.start()

    if detection_thread is None or not detection_thread.is_alive():
        detection_stop_event.clear()
        detection_thread = threading.Thread(target=detection_worker, daemon=True)
        detection_thread.start()

def generate_frames():
    """
    Generate video frames with object detection.

    Important:
    - The generator keeps running and always yields frames to keep the stream alive.
    - When `is_running` is False, we still capture and send frames (without processing),
      so the camera feed is always visible.
    """
    global latest_annotated_frame

    ensure_background_workers()
    while True:
        with latest_frame_lock:
            frame = latest_annotated_frame.copy() if latest_annotated_frame is not None else None

        if frame is None:
            frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Opening camera...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                2,
            )
            
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ret:
            time.sleep(0.02)
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(1 / STREAM_FPS)

def audio_worker():
    """Background worker for text-to-speech"""
    while True:
        try:
            message = audio_alert_queue.get(timeout=1)
            if message:
                engine.say(message)
                engine.runAndWait()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Audio error: {e}")

# Start audio worker thread
audio_thread = threading.Thread(target=audio_worker, daemon=True)
audio_thread.start()

# ------------------ Routes ------------------

@app.route('/')
def index():
    """Main page"""
    # Additional debug check
    template_path = os.path.join(TEMPLATE_DIR, 'index.html')
    if not os.path.exists(template_path):
        print(f"[ERROR] Template not found at: {template_path}")
        print(f"[ERROR] Current working directory: {os.getcwd()}")
        print(f"[ERROR] __file__ location: {__file__}")
        return f"Template not found. Looking for: {template_path}", 500
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_alert_mode', methods=['POST'])
def set_alert_mode():
    """
    Persist the user's preferred alert mode.
    Frontend is responsible for actually playing voice / beeps.
    """
    global alert_mode
    try:
        data = request.get_json(force=True, silent=True) or {}
        mode = data.get("mode", "english")
    except Exception:
        mode = "english"

    if mode not in ("english", "sound"):
        mode = "english"

    alert_mode = mode
    return jsonify({"mode": alert_mode})


@app.route('/get_alert_mode', methods=['GET'])
def get_alert_mode():
    """Return the current alert mode for frontend initialization."""
    return jsonify({"mode": alert_mode})


@app.route('/set_environment', methods=['POST'])
def set_environment():
    """
    Set the current environment context.
    Options: "indoor", "outdoor", "auto".
    """
    global environment_mode
    try:
        data = request.get_json(force=True, silent=True) or {}
        mode = data.get("mode", "outdoor")
    except Exception:
        mode = "outdoor"

    if mode not in ("indoor", "outdoor", "auto"):
        mode = "outdoor"

    environment_mode = mode
    return jsonify({"mode": environment_mode})


@app.route('/get_environment', methods=['GET'])
def get_environment():
    """Return the current environment context."""
    return jsonify({"mode": environment_mode})


@app.route('/sos', methods=['POST'])
def sos():
    """
    Emergency SOS endpoint.
    In a real deployment this could trigger calls, SMS, or notify caregivers.
    For now we simply acknowledge the request on the server.
    """
    print("[SOS] Emergency assistance requested from client.")
    return jsonify({"status": "received"})

@app.route('/start', methods=['POST'])
def start_detection():
    """Start object detection"""
    global is_running
    ensure_background_workers()
    is_running = True
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop_detection():
    """Stop object detection"""
    global is_running, camera
    is_running = False
    # We keep the video stream alive in generate_frames(), so we do not
    # release the camera here. It will be reused on next start.
    return jsonify({"status": "stopped"})

@app.route('/status', methods=['GET'])
def get_status():
    """Get current detection status"""
    return jsonify({"running": is_running})

@app.route('/get_alert', methods=['GET'])
def get_alert():
    """Get next alert message (for web audio)"""
    try:
        message = alert_queue.get_nowait()
        return jsonify({"alert": message})
    except queue.Empty:
        return jsonify({"alert": None})

@app.route('/scene_description', methods=['GET'])
def scene_description():
    """Generate a scene description for the voice assistant."""
    global latest_detections
    if not latest_detections:
        return jsonify({"description": "I don't see anything around you right now."})
    
    # Count objects
    counts = {}
    for det in latest_detections:
        base_label = det["label"].replace("red ", "").replace("green ", "").replace(" with zebra crossing", "")
        counts[base_label] = counts.get(base_label, 0) + 1
    
    parts = []
    
    # Check context/surroundings
    if "bus" in counts or "bench" in counts or "stop sign" in counts:
        parts.append("You appear to be near a bus stop or crosswalk.")
    elif "chair" in counts or "dining table" in counts or "tv" in counts:
        parts.append("You appear to be indoors.")
        
    # Describe counts
    items = []
    for label, count in counts.items():
        if count == 1:
            items.append(f"1 {label}")
        else:
            if label.endswith("s"):
                items.append(f"{count} {label}es")
            else:
                items.append(f"{count} {label}s")
                
    if items:
        if len(items) == 1:
            parts.append(f"I see {items[0]} in front of you.")
        else:
            parts.append(f"I see {', '.join(items[:-1])}, and {items[-1]} in front of you.")
            
    description = " ".join(parts)
    return jsonify({"description": description})

@app.route('/read_text', methods=['GET'])
def read_text():
    """Extract and return text from the latest camera frame using EasyOCR."""
    global latest_frame
    
    frame_to_process = None
    with latest_frame_lock:
        if latest_frame is not None:
            frame_to_process = latest_frame.copy()
            
    if frame_to_process is None:
        return jsonify({"text": ""})
        
    try:
        # reader.readtext returns a list of tuples: (bounding box, text, confidence)
        results = reader.readtext(frame_to_process)
        
        # Join all detected text pieces
        extracted_texts = [text for (bbox, text, prob) in results if prob > 0.3]
        final_text = " ".join(extracted_texts).strip()
        
        if not final_text:
            return jsonify({"text": ""})
            
        return jsonify({"text": final_text})
    except Exception as e:
        print(f"OCR Error: {e}")
        return jsonify({"text": ""})

@app.route('/mickey_vision', methods=['POST'])
def mickey_vision():
    """Use Gemini/VLM to understand the current camera frame."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = str(data.get("prompt", "")).strip()
    except Exception:
        prompt = ""

    frame = get_latest_camera_frame()
    if frame is None:
        return jsonify({
            "description": "",
            "error": "Camera is not ready yet. Start the camera feed and try again.",
        }), 503

    description, error = call_gemini_vision(frame, prompt)
    if error:
        return jsonify({"description": "", "error": error}), 503

    return jsonify({"description": description, "error": None})

@app.route('/mickey', methods=['POST'])
def mickey_assistant():
    """Small personal assistant brain for voice commands."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        message = str(data.get("message", "")).strip().lower()
    except Exception:
        message = ""

    if not message:
        return jsonify({
            "reply": "Hi, I am Mickey. Ask me to navigate somewhere, use smart look, read text, start detection, stop detection, repeat directions, or call SOS.",
            "action": None,
        })

    destination = None
    navigation_phrases = [
        "navigate to", "take me to", "go to", "directions to", "route to", "guide me to"
    ]
    for phrase in navigation_phrases:
        if phrase in message:
            destination = message.split(phrase, 1)[1].strip(" .")
            break

    if destination:
        return jsonify({
            "reply": f"Finding a walking route to {destination}.",
            "action": "navigate",
            "destination": destination,
        })

    if any(phrase in message for phrase in ["repeat direction", "repeat directions", "next direction", "where do i go", "what is next"]):
        return jsonify({"reply": "Repeating your current direction.", "action": "repeat_navigation"})

    if any(phrase in message for phrase in ["where am i", "my location", "current location"]):
        return jsonify({"reply": "Checking your current location.", "action": "location"})

    if "clear" in message and ("route" in message or "navigation" in message or "directions" in message):
        return jsonify({"reply": "Clearing the current route.", "action": "clear_route"})

    if "start" in message and ("detection" in message or "detect" in message or "camera" in message):
        return jsonify({"reply": "Starting obstacle detection now.", "action": "start"})

    if "stop" in message and ("detection" in message or "detect" in message or "camera" in message):
        return jsonify({"reply": "Stopping obstacle detection now.", "action": "stop"})

    if any(phrase in message for phrase in ["smart look", "look carefully", "can i cross", "is it safe", "guide me through", "what is around me"]):
        return jsonify({
            "reply": "Using Smart Look for a deeper scene check.",
            "action": "vision",
            "prompt": message,
        })

    if any(word in message for word in ["front", "scene", "around", "describe", "see"]):
        return jsonify({"reply": "Let me describe what is in front of you.", "action": "vision", "prompt": message})

    if any(word in message for word in ["read", "text", "written", "paper", "document", "board"]):
        return jsonify({"reply": "I will scan the camera view and read any text I can find.", "action": "read"})

    if "indoor" in message:
        return jsonify({"reply": "Switching to indoor mode.", "action": "indoor"})

    if "outdoor" in message:
        return jsonify({"reply": "Switching to outdoor mode.", "action": "outdoor"})

    if any(word in message for word in ["emergency", "sos", "help", "call"]):
        return jsonify({"reply": "Activating SOS.", "action": "sos"})

    return jsonify({
        "reply": "I am Mickey. I can guide you to a place, repeat walking directions, use Smart Look to understand the camera view, read text, change indoor or outdoor mode, and trigger SOS.",
        "action": None,
    })

# ------------------ Cleanup on exit ------------------
def cleanup():
    """Release camera and TTS engine when the application exits."""
    global camera, engine
    camera_stop_event.set()
    detection_stop_event.set()
    print("[CLEANUP] Shutting down — releasing camera...")
    try:
        if camera is not None and camera.isOpened():
            camera.release()
            print("[CLEANUP] Camera released successfully.")
    except Exception as e:
        print(f"[CLEANUP] Error releasing camera: {e}")
    try:
        engine.stop()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[CLEANUP] OpenCV window cleanup skipped: {e}")
    print("[CLEANUP] Done.")

# Register cleanup for normal exit
atexit.register(cleanup)

# Register cleanup for Ctrl+C / termination signals
def signal_handler(sig, frame):
    print(f"\n[SIGNAL] Received signal {sig}, cleaning up...")
    cleanup()
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=5000)
    finally:
        cleanup()
