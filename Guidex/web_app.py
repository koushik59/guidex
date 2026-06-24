from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import time
import torch
from ultralytics import YOLO
import threading
import queue
import numpy as np
import os
import atexit
import signal
import easyocr
import json
import subprocess
import tempfile
import shutil

# ------------------ Optional dependencies (fail-safe) ------------------
# sounddevice (PortAudio) is needed for the microphone used by the Vosk voice listener.
try:
    import sounddevice as sd
    VOICE_AUDIO_AVAILABLE = True
except Exception as e:
    sd = None
    VOICE_AUDIO_AVAILABLE = False
    print(f"[VOICE] sounddevice unavailable -> voice commands disabled: {e}")

# pyttsx3 is only a FALLBACK. On Jetson/Linux espeak-ng is the reliable path.
try:
    import pyttsx3
    _pyttsx3_engine = pyttsx3.init()
    _pyttsx3_engine.setProperty("rate", 160)
    _pyttsx3_engine.setProperty("volume", 1.0)
except Exception as e:
    _pyttsx3_engine = None
    print(f"[TTS] pyttsx3 unavailable (will use espeak-ng): {e}")

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.chdir(BASE_DIR)

print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] TEMPLATE_DIR: {TEMPLATE_DIR}")
print(f"[DEBUG] STATIC_DIR: {STATIC_DIR}")
print(f"[DEBUG] Templates exist: {os.path.exists(TEMPLATE_DIR)}")
print(f"[DEBUG] index.html exists: {os.path.exists(os.path.join(TEMPLATE_DIR, 'index.html'))}")

# Compatibility for older Ultralytics with PyTorch 2.6+ (weights_only default change)
_original_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DETECTION_WIDTH = 416
DETECTION_INTERVAL = 0.18
STREAM_FPS = 24
JPEG_QUALITY = 70

# ------------------ Detection setup ------------------
DANGEROUS_CLASSES = [
    "person", "car", "bus", "truck", "motorcycle", "bicycle",
    "chair", "couch", "potted plant", "bed", "bench", "dining table",
    "tv", "laptop", "sink", "refrigerator", "toilet", "umbrella",
    "backpack", "handbag", "suitcase", "fire hydrant", "stop sign",
    "traffic light", "pothole", "stairs", "water puddle", "construction zone",
    "book", "cell phone"
]

LARGE_VEHICLES = ["car", "bus", "truck"]
MEDIUM_VEHICLES = ["motorcycle"]
OBSTACLES = ["chair", "couch", "bed", "bench", "dining table", "refrigerator", "toilet",
             "fire hydrant", "stop sign", "pothole", "stairs", "water puddle", "construction zone"]
SMALL_OBJECTS = ["person", "bicycle", "potted plant", "tv", "laptop", "sink",
                 "umbrella", "backpack", "handbag", "suitcase", "traffic light", "book", "cell phone"]

model = YOLO("yolov8n.pt")

# EasyOCR (shared by /read_text, /scan and the voice listener). Guarded by a lock
# because readtext is not safe to call from multiple threads at once.
print("Initializing EasyOCR (first run may download models)...")
reader = easyocr.Reader(['en'])
ocr_lock = threading.Lock()
print("EasyOCR initialized.")

# ------------------ Global state ------------------
alert_queue = queue.Queue()        # consumed by the browser via /get_alert
camera = None
is_running = False
latest_detections = []
latest_frame = None
latest_annotated_frame = None
latest_frame_lock = threading.Lock()
latest_scan_text = ""
latest_scan_timestamp = 0.0
latest_scan_lock = threading.Lock()
camera_thread = None
detection_thread = None
camera_stop_event = threading.Event()
detection_stop_event = threading.Event()

# Voice listener state
voice_thread = None
voice_stop_event = threading.Event()

# Alert / environment configuration
alert_mode = "english"
environment_mode = "outdoor"

COOLDOWN_MAP = {"large_vehicle": 2.0, "medium_vehicle": 3.0, "small_object": 5.0}
last_alert_times = {"large_vehicle": 0.0, "medium_vehicle": 0.0, "small_object": 0.0}
object_track_state = {}

# ==================================================================
#                       LOCAL TEXT-TO-SPEECH
#  espeak-ng -> wav, then play through pw-play / paplay / aplay.
#  This is the path that actually produces sound on Jetson/Linux.
# ==================================================================
speech_queue = queue.Queue()
audio_process = None                 # currently running synth/playback subprocess
audio_process_lock = threading.Lock()
tts_stop_event = threading.Event()   # shuts the worker down on exit
interrupt_event = threading.Event()  # set by stop_playback() to abort current speech

def _run_audio_command(cmd, timeout=30):
    """Run a single audio subprocess and return True on success."""
    global audio_process
    try:
        with audio_process_lock:
            audio_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        out, err = audio_process.communicate(timeout=timeout)
        rc = audio_process.returncode
    except FileNotFoundError:
        return False  # this binary is not installed, try the next one
    except subprocess.TimeoutExpired:
        with audio_process_lock:
            if audio_process:
                audio_process.kill()
        return False
    except Exception as e:
        print(f"[TTS] command error ({cmd[0]}): {e}")
        return False
    finally:
        with audio_process_lock:
            audio_process = None
    return rc == 0

def _synthesize_wav(text, path):
    """Render text to a wav file using espeak-ng (or espeak)."""
    for tts_bin in ("espeak-ng", "espeak"):
        if shutil.which(tts_bin):
            if _run_audio_command([tts_bin, "-s", "160", "-w", path, text]):
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return True
    return False

def _play_wav(path):
    """Play a wav file through whichever audio backend is available."""
    players = []
    if shutil.which("pw-play"):
        players.append(["pw-play", path])
    if shutil.which("paplay"):
        players.append(["paplay", path])
    players.append(["aplay", "-q", "-D", "default", path])
    players.append(["aplay", "-q", path])

    for p in players:
        if interrupt_event.is_set():
            return False
        if _run_audio_command(p):
            return True
        if interrupt_event.is_set():
            return False
    return False

def tts_worker():
    """Background worker that speaks every message placed on speech_queue."""
    print("[TTS] Speech worker online.")
    have_espeak = bool(shutil.which("espeak-ng") or shutil.which("espeak"))
    if not have_espeak and _pyttsx3_engine is None:
        print("[TTS] WARNING: no espeak-ng/espeak and no pyttsx3 -> no audio will play. "
              "Install with: sudo apt install espeak-ng alsa-utils")

    while not tts_stop_event.is_set():
        try:
            text = speech_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if not text:
            continue

        interrupt_event.clear()
        spoke = False

        if have_espeak:
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                if _synthesize_wav(text, tmp):
                    spoke = _play_wav(tmp)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        if not spoke and _pyttsx3_engine is not None and not interrupt_event.is_set():
            try:
                _pyttsx3_engine.say(text)
                _pyttsx3_engine.runAndWait()
                spoke = True
            except Exception as e:
                print(f"[TTS] pyttsx3 fallback error: {e}")

        if not spoke and not interrupt_event.is_set():
            print(f"[TTS] (silent) wanted to say: {text}")

def stop_playback():
    """Flush pending speech and kill any audio that is currently playing."""
    print("[TTS] Stop requested -> flushing speech pipeline.")
    interrupt_event.set()

    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break

    with audio_process_lock:
        if audio_process:
            try:
                audio_process.terminate()
                audio_process.wait(timeout=0.5)
            except Exception:
                try:
                    audio_process.kill()
                except Exception:
                    pass
    try:
        if _pyttsx3_engine is not None:
            _pyttsx3_engine.stop()
    except Exception:
        pass

def speak(text):
    """Queue a message for local speech output."""
    if text:
        speech_queue.put(text)

# ------------------ Helper functions ------------------
def estimate_distance(box_height, frame_height):
    if box_height == 0:
        return 999
    return (frame_height / box_height) * 0.5

def get_object_category(label):
    if label in LARGE_VEHICLES:
        return "large_vehicle"
    elif label in MEDIUM_VEHICLES:
        return "medium_vehicle"
    elif label in SMALL_OBJECTS:
        return "small_object"
    return "small_object"

def danger_level(distance, object_category):
    if object_category == "large_vehicle":
        if distance < 25:
            return "HIGH"
        elif distance < 45:
            return "MEDIUM"
        return "LOW"
    elif object_category == "medium_vehicle":
        if distance < 15:
            return "HIGH"
        elif distance < 30:
            return "MEDIUM"
        return "LOW"
    else:
        if distance < 1.5:
            return "HIGH"
        elif distance < 3:
            return "MEDIUM"
        return "LOW"

def get_direction(x1, x2, frame_width):
    center_x = (x1 + x2) / 2
    if center_x < frame_width / 3:
        return "left"
    elif center_x < 2 * frame_width / 3:
        return "center"
    return "right"

def compute_priority(level, object_category, speed_mps, label=""):
    level_factor = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0}.get(level, 1.0)
    type_factor = {"large_vehicle": 3.0, "medium_vehicle": 2.0, "small_object": 1.0}.get(object_category, 1.0)

    speed = float(speed_mps or 0.0)
    speed_factor = 1.0 + min(abs(speed), 10.0) / 5.0

    if label == "person":
        if speed > 0.3:
            speed_factor *= 3.0
        elif speed < -0.3:
            speed_factor *= 0.2

    env_factor = 1.0
    if environment_mode == "outdoor":
        if object_category in ("large_vehicle", "medium_vehicle"):
            env_factor = 1.3
    elif environment_mode == "indoor":
        if object_category == "small_object":
            env_factor = 1.3

    return level_factor * type_factor * speed_factor * env_factor

def queue_alert(message):
    """Send a detection alert to the browser AND speak it locally."""
    alert_queue.put(message)   # browser polls this via /get_alert
    speak(message)             # local espeak-ng output

def process_frame(frame):
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

            object_category = get_object_category(label)
            level = danger_level(distance, object_category)
            direction = get_direction(x1, x2, frame_width)

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

                    if y2 < frame_height - 50:
                        roi_bottom = frame[y2:, max(0, x1 - 50):min(frame_width, x2 + 50)]
                        gray_bottom = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray_bottom, 50, 150, apertureSize=3)
                        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
                        if lines is not None and len(lines) > 2:
                            label += " with zebra crossing"

            track_key = (label.split(" ")[-1], direction)
            prev_state = object_track_state.get(track_key)
            speed_mps = 0.0
            if prev_state:
                dt = current_time - prev_state["time"]
                if dt > 0:
                    speed_mps = (prev_state["distance"] - distance) / dt
            object_track_state[track_key] = {"distance": distance, "time": current_time}

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

    if detections:
        best_detection = max(detections, key=lambda d: d.get("priority", 0.0))
        category = best_detection["category"]
        level = best_detection["level"]
        direction = best_detection["direction"]
        label = best_detection["label"]

        should_alert = False
        if category == "large_vehicle":
            should_alert = level in ("HIGH", "MEDIUM")
        else:
            should_alert = level == "HIGH"

        if should_alert:
            last_time_for_category = last_alert_times.get(category, 0.0)
            cooldown = COOLDOWN_MAP.get(category, 4.0)

            if current_time - last_time_for_category > cooldown:
                if "green traffic light" in label:
                    alert_message = "Signal is green, safe to cross."
                elif "red traffic light" in label:
                    alert_message = "Signal is red, do not cross."
                elif category == "large_vehicle":
                    if level == "HIGH":
                        alert_message = f"Large vehicle very close on your {direction}. Please stop."
                    else:
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
                else:
                    alert_message = f"{label} very close on your {direction}. Please stop."

                queue_alert(alert_message)
                last_alert_times[category] = current_time

    global latest_detections
    latest_detections = detections
    return detections

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        level = det["level"]
        color = (0, 0, 255) if level == "HIGH" else (0, 255, 255) if level == "MEDIUM" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{det['label']} | {det['level']} | {det['direction']}",
                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame

def open_camera():
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
    global camera, latest_frame, latest_annotated_frame
    while not camera_stop_event.is_set():
        if camera is None or not camera.isOpened():
            if not open_camera():
                print("[ERROR] Could not open any camera. Retrying in 2s...")
                time.sleep(2)
                continue
        ret, frame = camera.read()
        if not ret:
            print("[WARNING] Failed to read frame. Releasing and retrying...")
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
            cv2.putText(annotated, "Ready - Press Start to begin detection",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        with latest_frame_lock:
            latest_annotated_frame = annotated
        time.sleep(DETECTION_INTERVAL)

def ensure_background_workers():
    """Start camera + detection workers once (voice/tts started at module load)."""
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
    global latest_annotated_frame
    ensure_background_workers()
    while True:
        with latest_frame_lock:
            frame = latest_annotated_frame.copy() if latest_annotated_frame is not None else None
        if frame is None:
            frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, "Opening camera...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ret:
            time.sleep(0.02)
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(1 / STREAM_FPS)

# ==================================================================
#                   OCR SCAN  (read text -> speak)
# ==================================================================
def run_scan_and_speak():
    """Grab the current frame, OCR it, and speak the detected text. Returns the text."""
    global latest_scan_text, latest_scan_timestamp
    ensure_background_workers()

    # Wait briefly in case the camera just started.
    frame = None
    for _ in range(20):
        with latest_frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
        if frame is not None:
            break
        time.sleep(0.1)

    if frame is None:
        print("[SCAN] No camera frame available yet.")
        speak("Camera is not ready yet.")
        return ""

    print("[SCAN] Running OCR on current frame...")
    try:
        h, w = frame.shape[:2]
        if w > 800:
            scale = 800.0 / w
            frame = cv2.resize(frame, (800, int(h * scale)))

        with ocr_lock:
            results = reader.readtext(frame)

        texts = [t.strip() for (bbox, t, prob) in results if prob > 0.3 and len(t.strip()) > 1]
        final_text = " ".join(texts).strip()

        if final_text:
            print(f"[SCAN] Detected text: {final_text}")
            with latest_scan_lock:
                latest_scan_text = final_text
                latest_scan_timestamp = time.time()
            speak(final_text)
        else:
            print("[SCAN] No clear text detected.")
            speak("No clear text detected.")
        return final_text
    except Exception as e:
        print(f"[SCAN] OCR error: {e}")
        speak("Sorry, I could not read the text.")
        return ""

# ==================================================================
#                 VOICE COMMAND LISTENER (Vosk, offline)
#   say "scan" / "read"  -> OCR + speak the text
#   say "stop"           -> stop the speech
# ==================================================================
def find_working_microphone():
    devices = sd.query_devices()
    fallback_index = sd.default.device[0]
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            name = dev['name'].lower()
            if 'pipewire' in name or 'pulse' in name or 'default' in name:
                return idx
    return fallback_index

def voice_command_thread():
    if not VOICE_AUDIO_AVAILABLE:
        print("[VOICE] sounddevice missing -> voice listener disabled.")
        return
    try:
        from vosk import Model, KaldiRecognizer
    except ImportError:
        print("[VOICE] Vosk not installed -> voice listener disabled. "
              "Install with: pip install vosk")
        return

    print("[VOICE] Loading Vosk model...")
    model_path = os.path.join(BASE_DIR, "model")
    if not os.path.exists(model_path):
        model_path = "model"
    if not os.path.exists(model_path):
        print("[VOICE] No 'model' folder found. Download a Vosk model "
              "(e.g. vosk-model-small-en-us) and unzip it into a folder named 'model' "
              "next to this script. Voice listener disabled.")
        return
    try:
        vosk_model = Model(model_path)
        print("[VOICE] Vosk model loaded.")
    except Exception as e:
        print(f"[VOICE] Vosk model error: {e}")
        return

    recognizer = KaldiRecognizer(vosk_model, 16000)
    audio_queue = queue.Queue()

    try:
        mic_index = find_working_microphone()
        devices = sd.query_devices()
        print(f"[VOICE] Microphone -> index {mic_index}: {devices[mic_index]['name']}")
    except Exception as e:
        print(f"[VOICE] Could not query microphone: {e}")
        return

    def audio_callback(indata, frames, time_info, status):
        audio_queue.put(bytes(indata))

    try:
        stream = sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16',
                                   channels=1, device=mic_index, callback=audio_callback)
    except Exception as e:
        print(f"[VOICE] Microphone stream failure: {e}")
        return

    with stream:
        print("[VOICE] --- MICROPHONE LIVE. Say 'scan' to read text, 'stop' to stop. ---")
        speak("Voice system ready.")
        while not voice_stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.5)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    command = result.get("text", "").lower().strip()
                    if command:
                        print(f"[VOICE] Heard: '{command}'")
                    if "stop" in command:
                        stop_playback()
                    elif "read" in command or "scan" in command:
                        stop_playback()      # cut off anything already speaking
                        speak("Scanning.")
                        run_scan_and_speak()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VOICE] Listener error: {e}")
                continue

# ------------------ Routes ------------------
@app.route('/')
def index():
    template_path = os.path.join(TEMPLATE_DIR, 'index.html')
    if not os.path.exists(template_path):
        print(f"[ERROR] Template not found at: {template_path}")
        return f"Template not found. Looking for: {template_path}", 500
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_alert_mode', methods=['POST'])
def set_alert_mode():
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
    return jsonify({"mode": alert_mode})

@app.route('/set_environment', methods=['POST'])
def set_environment():
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
    return jsonify({"mode": environment_mode})

@app.route('/sos', methods=['POST'])
def sos():
    print("[SOS] Emergency assistance requested from client.")
    return jsonify({"status": "received"})

@app.route('/start', methods=['POST'])
def start_detection():
    global is_running
    ensure_background_workers()
    is_running = True
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop_detection():
    global is_running
    is_running = False
    return jsonify({"status": "stopped"})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({"running": is_running})

@app.route('/get_alert', methods=['GET'])
def get_alert():
    try:
        message = alert_queue.get_nowait()
        return jsonify({"alert": message})
    except queue.Empty:
        return jsonify({"alert": None})

@app.route('/scene_description', methods=['GET'])
def scene_description():
    global latest_detections
    if not latest_detections:
        return jsonify({"description": "I don't see anything around you right now."})
    counts = {}
    for det in latest_detections:
        base_label = det["label"].replace("red ", "").replace("green ", "").replace(" with zebra crossing", "")
        counts[base_label] = counts.get(base_label, 0) + 1
    parts = []
    if "bus" in counts or "bench" in counts or "stop sign" in counts:
        parts.append("You appear to be near a bus stop or crosswalk.")
    elif "chair" in counts or "dining table" in counts or "tv" in counts:
        parts.append("You appear to be indoors.")
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
    return jsonify({"description": " ".join(parts)})

@app.route('/read_text', methods=['GET'])
def read_text():
    """Return text from the latest frame (no speech). Kept for backward compatibility."""
    frame_to_process = None
    with latest_frame_lock:
        if latest_frame is not None:
            frame_to_process = latest_frame.copy()
    if frame_to_process is None:
        return jsonify({"text": ""})
    try:
        with ocr_lock:
            results = reader.readtext(frame_to_process)
        extracted_texts = [text for (bbox, text, prob) in results if prob > 0.3]
        final_text = " ".join(extracted_texts).strip()
        return jsonify({"text": final_text})
    except Exception as e:
        print(f"OCR Error: {e}")
        return jsonify({"text": ""})

@app.route('/scan', methods=['POST', 'GET'])
def scan_route():
    """Trigger OCR + speech from the UI (button). Same action as the 'scan' voice command."""
    stop_playback()
    text = run_scan_and_speak()
    with latest_scan_lock:
        timestamp = latest_scan_timestamp if text else 0.0
    return jsonify({"text": text, "timestamp": timestamp})

@app.route('/latest_scan', methods=['GET'])
def latest_scan():
    """Return the latest OCR text detected by button or voice scan."""
    with latest_scan_lock:
        return jsonify({
            "text": latest_scan_text,
            "timestamp": latest_scan_timestamp
        })

@app.route('/stop_speech', methods=['POST', 'GET'])
def stop_speech_route():
    """Stop any ongoing speech. Call this from the Navigation button."""
    stop_playback()
    return jsonify({"status": "stopped"})

@app.route('/voice_status', methods=['GET'])
def voice_status():
    active = bool(voice_thread is not None and voice_thread.is_alive())
    return jsonify({"voice_available": VOICE_AUDIO_AVAILABLE, "voice_running": active})

# ------------------ Cleanup ------------------
def cleanup():
    global camera
    camera_stop_event.set()
    detection_stop_event.set()
    voice_stop_event.set()
    tts_stop_event.set()
    stop_playback()
    print("[CLEANUP] Shutting down — releasing camera...")
    try:
        if camera is not None and camera.isOpened():
            camera.release()
            print("[CLEANUP] Camera released successfully.")
    except Exception as e:
        print(f"[CLEANUP] Error releasing camera: {e}")
    try:
        if _pyttsx3_engine is not None:
            _pyttsx3_engine.stop()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[CLEANUP] OpenCV window cleanup skipped: {e}")
    print("[CLEANUP] Done.")

atexit.register(cleanup)

def signal_handler(sig, frame):
    print(f"\n[SIGNAL] Received signal {sig}, cleaning up...")
    cleanup()
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ------------------ Start background workers at import ------------------
# Started here (not lazily) so you can immediately see in the terminal whether
# voice/TTS are working.
threading.Thread(target=tts_worker, daemon=True).start()

if VOICE_AUDIO_AVAILABLE:
    voice_stop_event.clear()
    voice_thread = threading.Thread(target=voice_command_thread, daemon=True)
    voice_thread.start()
else:
    print("[VOICE] Voice listener not started (sounddevice unavailable).")

if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=5000)
    finally:
        cleanup()
