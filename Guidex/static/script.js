// GuideX - Frontend JavaScript
// Accessibility-first navigation assistant

let isRunning = false;
let alertCheckInterval = null;
let statusCheckInterval = null;
let scanCheckInterval = null;
let lastScanTimestamp = 0;

// Initialize Web Speech API for audio feedback
const synth = window.speechSynthesis;

function speak(text) {
    if (synth.speaking) {
        synth.cancel();
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    synth.speak(utterance);
}

// Haptic feedback helper (vibration API)
function vibrate(pattern) {
    if (navigator.vibrate) {
        navigator.vibrate(pattern);
    }
}

// DOM elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const sosBtn = document.getElementById('sosBtn');
const statusIndicator = document.getElementById('status');
const statusDot = statusIndicator.querySelector('.status-dot');
const statusText = statusIndicator.querySelector('.status-text');
const detectionsDiv = document.getElementById('detections');
const videoStream = document.getElementById('videoStream');
const alertModeSelect = document.getElementById('alertMode');
const environmentModeSelect = document.getElementById('environmentMode');
const readTextBtn = document.getElementById('readTextBtn');

// SOS long-press handling
let sosPressTimer = null;
let sosPressed = false;
const SOS_PRESS_DURATION = 2000; // 2 seconds

function startSOSPress() {
    sosPressed = true;
    vibrate([100]); // Initial haptic feedback
    speak('SOS button pressed. Hold for two seconds to activate emergency.');

    sosPressTimer = setTimeout(() => {
        activateSOS();
    }, SOS_PRESS_DURATION);
}

function cancelSOSPress() {
    if (sosPressTimer) {
        clearTimeout(sosPressTimer);
        sosPressTimer = null;
    }
    if (sosPressed) {
        sosPressed = false;
        vibrate([50, 50, 50]); // Cancel feedback
    }
}

async function activateSOS() {
    sosPressed = false;
    vibrate([200, 100, 200, 100, 200]); // Strong SOS pattern
    speak('Emergency SOS activated. Help is being notified.');

    try {
        const response = await fetch('/sos', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            speak('Emergency SOS request sent successfully.');
        } else {
            speak('SOS request failed. Please try again or contact emergency services directly.');
        }
    } catch (error) {
        console.error('SOS error:', error);
        speak('SOS request failed. Please contact emergency services directly.');
    }
}

// SOS button event handlers
sosBtn.addEventListener('mousedown', startSOSPress);
sosBtn.addEventListener('mouseup', cancelSOSPress);
sosBtn.addEventListener('mouseleave', cancelSOSPress);
sosBtn.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startSOSPress();
});
sosBtn.addEventListener('touchend', (e) => {
    e.preventDefault();
    cancelSOSPress();
});
sosBtn.addEventListener('touchcancel', (e) => {
    e.preventDefault();
    cancelSOSPress();
});

// Audio context for beep/siren sounds
let audioContext = null;
function getAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioContext;
}

// Generate beep sound
function generateBeep(frequency = 800, duration = 0.3) {
    const ctx = getAudioContext();
    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);

    oscillator.frequency.value = frequency;
    oscillator.type = 'sine';

    gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration);

    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + duration);
}

// Generate siren sound
function generateSiren() {
    const ctx = getAudioContext();
    const frequencies = [600, 800, 1000, 800, 600];
    let currentTime = ctx.currentTime;

    frequencies.forEach((freq, index) => {
        const oscillator = ctx.createOscillator();
        const gainNode = ctx.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(ctx.destination);

        oscillator.frequency.value = freq;
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0.3, currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, currentTime + 0.1);

        oscillator.start(currentTime);
        oscillator.stop(currentTime + 0.1);

        currentTime += 0.1;
    });
}

// Provide directional guidance through beep patterns
function provideDirectionalBeeps(direction) {
    if (direction === 'left') {
        generateBeep(600, 0.2);
        setTimeout(() => generateBeep(600, 0.2), 300);
    } else if (direction === 'right') {
        generateBeep(800, 0.15);
        setTimeout(() => generateBeep(800, 0.15), 250);
        setTimeout(() => generateBeep(800, 0.15), 500);
    } else if (direction === 'backward') {
        generateBeep(500, 0.5);
    }
}

// Event listeners
startBtn.addEventListener('click', startNavigation);
stopBtn.addEventListener('click', stopNavigation);
alertModeSelect.addEventListener('change', handleAlertModeChange);
environmentModeSelect.addEventListener('change', handleEnvironmentChange);

if (readTextBtn) {
    readTextBtn.addEventListener('click', () => {
        vibrate([50]);
        readTextFromCamera();
    });
}

// Keyboard accessibility
document.addEventListener('keydown', (e) => {
    // Prevent shortcut activation when typing in input/select fields
    if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') {
        return;
    }

    const key = e.key.toLowerCase();

    if (e.key === ' ' || e.key === 'Enter') {
        if (document.activeElement === startBtn && !isRunning) {
            startNavigation();
        } else if (document.activeElement === stopBtn && isRunning) {
            stopNavigation();
        }
    } else if (key === 't') {
        e.preventDefault();
        vibrate([50]);
        readTextFromCamera();
    }
});

async function handleAlertModeChange() {
    const mode = alertModeSelect.value;
    vibrate([50]); // Haptic feedback
    try {
        const response = await fetch('/set_alert_mode', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ mode: mode })
        });

        if (response.ok) {
            const data = await response.json();
            console.log('Alert mode changed to:', data.mode);
            if (mode === 'english') {
                speak('Alert mode set to English voice.');
            } else {
                generateBeep(600, 0.2); // Confirm with beep
                speak('Alert mode set to sound.');
            }
        }
    } catch (error) {
        console.error('Error setting alert mode:', error);
        speak('Error changing alert mode.');
    }
}

async function handleEnvironmentChange() {
    const mode = environmentModeSelect.value;
    vibrate([50]); // Haptic feedback
    try {
        const response = await fetch('/set_environment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ mode: mode })
        });

        if (response.ok) {
            const data = await response.json();
            console.log('Environment mode changed to:', data.mode);
            const modeNames = {
                'outdoor': 'outdoor',
                'indoor': 'indoor',
                'auto': 'auto'
            };
            speak(`Environment set to ${modeNames[mode]}.`);
        }
    } catch (error) {
        console.error('Error setting environment:', error);
        speak('Error changing environment mode.');
    }
}

async function startNavigation() {
    vibrate([100, 50, 100]); // Haptic feedback
    try {
        const response = await fetch('/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            isRunning = true;
            updateUI();
            const currentMode = alertModeSelect.value;
            if (currentMode === 'english') {
                speak('Navigation started. Object detection is now active.');
            } else {
                generateBeep(600, 0.3);
                vibrate([100]);
            }

            // Start checking for alerts
            startAlertCheck();
            startStatusCheck();
        } else {
            throw new Error('Failed to start navigation');
        }
    } catch (error) {
        console.error('Error starting navigation:', error);
        speak('Error starting navigation. Please try again.');
        vibrate([200, 100, 200]); // Error pattern
    }
}

async function stopNavigation() {
    vibrate([100, 50, 100]); // Haptic feedback
    try {
        const response = await fetch('/stop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            isRunning = false;
            updateUI();
            speak('Navigation stopped.');

            // Stop checking for alerts
            stopAlertCheck();
            stopStatusCheck();

            // Clear detections
            detectionsDiv.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🔍</div><p>No objects detected yet</p><p class="empty-state-hint">Press <strong>Start Navigation</strong> to begin AI detection</p></div>';
        } else {
            throw new Error('Failed to stop navigation');
        }
    } catch (error) {
        console.error('Error stopping navigation:', error);
        speak('Error stopping navigation.');
        vibrate([200, 100, 200]); // Error pattern
    }
}

function updateUI() {
    if (isRunning) {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusDot.className = 'status-dot running';
        statusText.textContent = 'Navigation Active';
        videoStream.style.borderColor = 'rgba(129,140,248,0.3)';
    } else {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        statusDot.className = 'status-dot';
        statusText.textContent = 'Ready';
        videoStream.style.borderColor = 'rgba(255,255,255,0.03)';
    }
}

function startAlertCheck() {
    alertCheckInterval = setInterval(async () => {
        try {
            const response = await fetch('/get_alert');
            const data = await response.json();

            if (data.alert) {
                const currentMode = alertModeSelect.value;

                if (currentMode === 'english') {
                    // English mode: use TTS
                    speak(data.alert);
                    vibrate([150, 100, 150]); // Alert haptic feedback
                    updateDetectionsDisplay(data.alert);
                } else {
                    // Sound mode: parse alert messages
                    // Check if it's a standard alert message
                    if (data.alert.includes('very close') || data.alert.includes('approaching')) {
                        // Determine danger level from message
                        const isHighDanger = data.alert.includes('very close') || data.alert.includes('Stop');

                        if (isHighDanger) {
                            generateSiren(); // Siren for high danger
                            vibrate([200, 100, 200, 100, 200]); // Strong vibration
                        } else {
                            generateBeep(600, 0.4); // Beep for medium danger
                            vibrate([150, 100, 150]); // Medium vibration
                        }

                        updateDetectionsDisplay(data.alert);
                    } else {
                        // Fallback beep
                        generateBeep();
                        vibrate([100]);
                    }
                }
            }
        } catch (error) {
            console.error('Error checking alerts:', error);
        }
    }, 500); // Check every 500ms
}

function stopAlertCheck() {
    if (alertCheckInterval) {
        clearInterval(alertCheckInterval);
        alertCheckInterval = null;
    }
}

function startStatusCheck() {
    statusCheckInterval = setInterval(async () => {
        try {
            const response = await fetch('/status');
            const data = await response.json();

            if (!data.running && isRunning) {
                // Server stopped, update UI
                isRunning = false;
                updateUI();
                stopAlertCheck();
                stopStatusCheck();
            }
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }, 2000); // Check every 2 seconds
}

function stopStatusCheck() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

function clearDetectionEmptyState() {
    const emptyState = detectionsDiv.querySelector('.empty-state');
    const noDetections = detectionsDiv.querySelector('.no-detections');
    if (emptyState) {
        emptyState.remove();
    }
    if (noDetections) {
        noDetections.remove();
    }
}

function trimDetectionItems() {
    const items = detectionsDiv.querySelectorAll('.detection-item');
    if (items.length > 5) {
        items[items.length - 1].remove();
    }
}

function addDetectionItem({ label, level = 'medium', badges = [], autoRemove = true }) {
    clearDetectionEmptyState();

    const detectionItem = document.createElement('div');
    detectionItem.className = `detection-item ${level}`;
    detectionItem.setAttribute('role', 'alert');
    detectionItem.setAttribute('aria-live', 'assertive');

    const labelWrap = document.createElement('div');
    const labelEl = document.createElement('span');
    labelEl.className = 'detection-label';
    labelEl.textContent = label;
    labelWrap.appendChild(labelEl);

    const infoEl = document.createElement('div');
    infoEl.className = 'detection-info';
    badges.forEach((badge) => {
        const badgeEl = document.createElement('span');
        badgeEl.className = `detection-badge ${badge.className || ''}`.trim();
        badgeEl.textContent = badge.text;
        infoEl.appendChild(badgeEl);
    });

    detectionItem.appendChild(labelWrap);
    detectionItem.appendChild(infoEl);
    detectionsDiv.insertBefore(detectionItem, detectionsDiv.firstChild);
    trimDetectionItems();

    if (autoRemove) {
        setTimeout(() => {
            if (detectionItem.parentNode) {
                detectionItem.remove();

                if (detectionsDiv.children.length === 0) {
                    detectionsDiv.innerHTML = '<p class="no-detections">No objects detected.</p>';
                }
            }
        }, 10000);
    }
}

function updateDetectionsDisplay(alertMessage) {
    // Parse alert message to extract information
    // Format: "{label} very close on your {direction}. Please stop."
    // or: "Large vehicle approaching from your {direction}. Be cautious."
    let label, direction, level;

    if (alertMessage.includes('very close')) {
        const parts = alertMessage.split(' very close on your ');
        if (parts.length === 2) {
            label = parts[0];
            direction = parts[1].replace('. Please stop.', '').replace('.', '').trim();
            level = 'high';
        }
    } else if (alertMessage.includes('approaching')) {
        const parts = alertMessage.split(' approaching from your ');
        if (parts.length === 2) {
            label = parts[0];
            direction = parts[1].replace('. Be cautious.', '').replace('.', '').trim();
            level = 'medium';
        }
    }

    if (label && direction) {
        addDetectionItem({
            label,
            level,
            badges: [
                { text: `${level.toUpperCase()} DANGER`, className: `badge-level ${level}` },
                { text: direction.toUpperCase(), className: 'badge-direction' }
            ]
        });
    }
}

function addScannedTextDetection(text) {
    addDetectionItem({
        label: text.trim(),
        level: 'medium',
        badges: [
            { text: 'SCANNED TEXT', className: 'badge-level medium' },
            { text: 'OCR', className: 'badge-direction' }
        ],
        autoRemove: false
    });
}

function startScanCheck() {
    if (scanCheckInterval) {
        return;
    }

    scanCheckInterval = setInterval(async () => {
        try {
            const response = await fetch('/latest_scan');
            const data = await response.json();
            const timestamp = Number(data.timestamp || 0);
            const text = (data.text || '').trim();

            if (text && timestamp > lastScanTimestamp) {
                lastScanTimestamp = timestamp;
                addScannedTextDetection(text);
            }
        } catch (error) {
            console.error('Error checking latest scan:', error);
        }
    }, 1000);
}

// Handle video stream errors
videoStream.addEventListener('error', () => {
    console.error('Video stream error');
    if (isRunning) {
        speak('Camera error. Please check your camera connection.');
        vibrate([200, 100, 200, 100, 200]); // Error pattern
    }
});

// Ensure video stream loads
videoStream.addEventListener('load', () => {
    console.log('Video stream loaded');
});

videoStream.addEventListener('loadstart', () => {
    console.log('Video stream started loading');
});

// Initialize UI
updateUI();
startScanCheck();

// Load current settings on page load
Promise.all([
    fetch('/get_alert_mode').then(r => r.json()),
    fetch('/get_environment').then(r => r.json()),
    fetch('/status').then(r => r.json())
]).then(([alertData, envData, statusData]) => {
    if (alertData.mode) {
        alertModeSelect.value = alertData.mode;
    }
    if (envData.mode) {
        environmentModeSelect.value = envData.mode;
    }
    if (statusData.running) {
        isRunning = true;
        updateUI();
        startAlertCheck();
        startStatusCheck();
    }
}).catch(error => {
    console.error('Error loading initial settings:', error);
});

async function readTextFromCamera() {
    speak("Scanning for text. This may take a moment.");
    try {
        const response = await fetch('/scan');
        const data = await response.json();

        if (data.text && data.text.trim() !== '') {
            lastScanTimestamp = Number(data.timestamp || Date.now() / 1000);
            addScannedTextDetection(data.text);
        } else {
            speak("I couldn't detect any text.");
        }
    } catch (error) {
        console.error('Error reading text:', error);
        speak('Error trying to read text.');
    }
}
