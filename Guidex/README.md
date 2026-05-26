# GuideX - Navigation Assistant for Visually Impaired

A web-based AI navigation assistant that helps blind and visually impaired people navigate safely in open areas by detecting dangerous objects and providing real-time audio & sound feedback.

## Features

- **Real-time Object Detection**: Uses YOLOv8 to detect dangerous objects (people, cars, buses, trucks, motorcycles, bicycles)
- **Priority-Based Alert System**: Alerts only the most dangerous object at a time to avoid confusion
- **Directional Alerts**: Provides left/center/right direction information
- **Audio & Haptic Feedback**: Text-to-speech and also siren/beep alerts when objects are detected
- **Environment Modes**: Indoor, Outdoor, and Auto modes for smarter alert prioritization
- **Live Camera Feed**: Real-time video streaming with object detection overlays
- **SOS Feature**: Long-press SOS button with audio and vibration confirmation

## Technology Stack

- **Backend**: Flask (Python)
- **AI Model**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Text-to-Speech**: pyttsx3 (backend) + Web Speech API (frontend)
- **Frontend**: HTML5, CSS3, JavaScript web Audio API, Vibration API

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Ensure YOLO model is present**:
   - The `yolov8n.pt` model file should be in the project root
   - It will be automatically downloaded on first run if missing

## Usage

### Running the Web Application

Optional Smart Look setup:

Smart Look uses Gemini image understanding for richer scene guidance. Create a Gemini API key in Google AI Studio, then set it before starting the app:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
python web_app.py
```

Without this key, GuideX still runs with YOLO, OCR, Maps, and typed Mickey commands. Smart Look will simply tell you the key is missing.

1. **Start the Flask server**:
   ```bash
   python web_app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Allow camera access** when prompted by your browser

4. **Click "Start Navigation"** to begin object detection

5. **The system will provide audio alerts** when dangerous objects are detected:
   
   **Large Vehicles (Car, Bus, Truck):**
   - High danger: Object is very close (< 15m) - "Stop immediately!"
   - Medium danger: Object is approaching (15-30m) - "Be cautious"
   - Low danger: Object is detected but at safe distance (> 30m)
   
   **Medium Vehicles (Motorcycle):**
   - High danger: Object is very close (< 8m) - "Please stop"
   - Medium danger: Object is moderately close (8-20m)
   - Low danger: Object is detected but at safe distance (> 20m)
   
   **Small Objects (Person, Bicycle):**
   - High danger: Object is very close (< 1.5m) - "Please stop"
   - Medium danger: Object is moderately close (1.5-3m)
   - Low danger: Object is detected but at safe distance (> 3m)
   
   **Note:** Large vehicles trigger alerts at much greater distances because they move faster and require more reaction time.

### Alert Priority Logic
Alerts are generated only for the highest-priority object and priority depends on;
- Distance
- Environment mode
- Object Type
- Speed of Approach

## Project Structure

```
GuideX/
├── web_app.py         # Main File
├── requirements.txt    # Python dependencies
├── yolov8n.pt         # YOLO model weights
└── static/            # Static files
    ├── style.css      # Stylesheet
    └── script.js      # Frontend JavaScript
```

## How It Works

1. **Camera Capture**: The application captures frames from your webcam
2. **Object Detection**: YOLOv8 processes each frame to detect objects
3. **Object Categorization**: Objects are categorized by size/danger:
   - Large vehicles (car, bus, truck)
   - Medium vehicles (motorcycle)
   - Small objects (person, bicycle)
4. **Distance Estimation**: Based on bounding box height, estimates distance
5. **Danger Assessment**: Uses category-specific thresholds to determine danger level:
   - Large vehicles: Alert from 15-30m away (they're fast and dangerous)
   - Medium vehicles: Alert from 8-20m away
   - Small objects: Alert from 1.5-3m away (close range)
6. **Direction Calculation**: Determines object position (left/center/right)
8. **Priority calculation**: Each object is assigned a priority score
7. **Audio Alert**: Only the highest-priority object triggers alerts
8. **Visual Display**: Shows detected objects with bounding boxes and labels

## Accessibility Features

- **Keyboard Navigation**: Full keyboard support for all controls
- **Screen Reader Support**: ARIA labels and live regions
- **Audio Feedback**: Multiple audio feedback methods
- **High Contrast**: Dark theme with clear visual indicators
- **Responsive Design**: Works on desktop and mobile devices

## Configuration

You can modify the following in `app_web.py`:

- **Environment_Mode**: environment_mode(indoor / outdoor / auto)
- **DANGEROUS_CLASSES**: List of object types to detect
- **Distance thresholds**: Modify `danger_level()` function
- **Alert frequency**: Change the 3-second cooldown in `process_frame()`
- **Speech rate**: Adjust `engine.setProperty("rate", 160)`

## Troubleshooting

### Camera not working
- Ensure your camera is connected and not being used by another application
- Check browser permissions for camera access
- Try a different browser (Chrome, Firefox, Edge)

### Model not loading
- Ensure internet connection for first-time model download
- Check that `yolov8n.pt` file exists in project root

### Audio not working
- Check system audio settings
- For web audio, ensure browser supports Web Speech API
- Backend audio uses system TTS (pyttsx3)

## Future Enhancements

- [ ] GPS integration for outdoor navigation
- [ ] Obstacle pathfinding suggestions
- [ ] Multiple camera support
- [ ] Mobile app version
- [ ] Offline mode support
- [ ] Caregiver notification integration

## License

This project is created for assistive technology purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For issues or questions, please open an issue on the project repository.
