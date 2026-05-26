// GuideX - Frontend JavaScript
// Accessibility-first navigation assistant

let isRunning = false;
let alertCheckInterval = null;
let statusCheckInterval = null;

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
const voiceBtn = document.getElementById('voiceBtn');
const mickeyLiveBtn = document.getElementById('mickeyLiveBtn');
const smartLookBtn = document.getElementById('smartLookBtn');
const mickeyTextInput = document.getElementById('mickeyTextInput');
const mickeySendBtn = document.getElementById('mickeySendBtn');
const mickeyStatus = document.getElementById('mickeyStatus');
let mickeyLiveInterval = null;

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

// Keyboard accessibility
document.addEventListener('keydown', (e) => {
    if (e.key === ' ' || e.key === 'Enter') {
        if (document.activeElement === startBtn && !isRunning) {
            startNavigation();
        } else if (document.activeElement === stopBtn && isRunning) {
            stopNavigation();
        }
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
        // Create detection item
        const detectionItem = document.createElement('div');
        detectionItem.className = `detection-item ${level}`;
        detectionItem.setAttribute('role', 'alert');
        detectionItem.setAttribute('aria-live', 'assertive');
        
        detectionItem.innerHTML = `
            <div>
                <span class="detection-label">${label}</span>
            </div>
            <div class="detection-info">
                <span class="detection-badge badge-level ${level}">${level.toUpperCase()} DANGER</span>
                <span class="detection-badge badge-direction">${direction.toUpperCase()}</span>
            </div>
        `;
        
        // Remove "no detections" message if present
        const noDetections = detectionsDiv.querySelector('.no-detections');
        if (noDetections) {
            noDetections.remove();
        }
        
        // Add new detection at the top
        detectionsDiv.insertBefore(detectionItem, detectionsDiv.firstChild);
        
        // Keep only last 5 detections
        const items = detectionsDiv.querySelectorAll('.detection-item');
        if (items.length > 5) {
            items[items.length - 1].remove();
        }
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (detectionItem.parentNode) {
                detectionItem.remove();
                
                // Show "no detections" if list is empty
                if (detectionsDiv.children.length === 0) {
                    detectionsDiv.innerHTML = '<p class="no-detections">No objects detected.</p>';
                }
            }
        }, 10000);
    }
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

// Voice Assistant Integration
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let isListeningForMickey = false;
let mickeyRecognitionAttempt = 0;
let mickeyRetryPending = false;
let mickeyGotResult = false;
const mickeyRecognitionLanguages = ['en-IN', 'en-US', 'en-GB'];

function setMickeyStatus(message) {
    if (mickeyStatus) {
        mickeyStatus.textContent = message;
    }
}

function sendTypedMickeyCommand() {
    if (!mickeyTextInput) {
        return;
    }

    const command = mickeyTextInput.value.trim();
    if (!command) {
        speak("Type a command for Mickey first.");
        return;
    }

    mickeyTextInput.value = '';
    setMickeyStatus('Mickey is thinking...');
    askMickey(command);
}

if (mickeySendBtn) {
    mickeySendBtn.addEventListener('click', sendTypedMickeyCommand);
}

if (mickeyTextInput) {
    mickeyTextInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendTypedMickeyCommand();
        }
    });
}

if (mickeyLiveBtn) {
    mickeyLiveBtn.addEventListener('click', toggleMickeyLive);
}

if (smartLookBtn) {
    smartLookBtn.addEventListener('click', () => getMickeyVision('What is around me and what should I be careful about?'));
}

function toggleMickeyLive() {
    if (mickeyLiveInterval) {
        clearInterval(mickeyLiveInterval);
        mickeyLiveInterval = null;
        mickeyLiveBtn.classList.remove('listening');
        setMickeyStatus('Mickey Live is off.');
        speak('Mickey Live stopped.');
        return;
    }

    mickeyLiveBtn.classList.add('listening');
    setMickeyStatus('Mickey Live is on. I will announce route and scene updates.');
    speak('Mickey Live started. I will help with route and scene updates.');
    runMickeyLiveTick();
    mickeyLiveInterval = setInterval(runMickeyLiveTick, 15000);
}

async function runMickeyLiveTick() {
    if (window.guideXMaps && window.guideXMaps.repeatGuidance) {
        window.guideXMaps.repeatGuidance();
    }

    try {
        const response = await fetch('/scene_description');
        const data = await response.json();
        if (data.description && !data.description.includes("don't see anything")) {
            setTimeout(() => speak(data.description), 3500);
        }
    } catch (error) {
        console.error('Mickey Live scene error:', error);
    }
}

if (SpeechRecognition && voiceBtn) {
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 3;

    async function startMickeyListening(languageIndex = 0) {
        if (isListeningForMickey) {
            recognition.stop();
            return;
        }

        mickeyRecognitionAttempt = languageIndex;
        mickeyRetryPending = false;
        mickeyGotResult = false;
        recognition.lang = mickeyRecognitionLanguages[mickeyRecognitionAttempt];
        setMickeyStatus(`Mickey is listening (${recognition.lang})...`);

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                stream.getTracks().forEach(track => track.stop());
            } catch (error) {
                console.error('Microphone permission error:', error);
                setMickeyStatus('Microphone permission is blocked or unavailable.');
                speak('Microphone permission is blocked. Please allow microphone access in the browser.');
                return;
            }
        }

        vibrate([50]);
        generateBeep(700, 0.12);
        if (synth.speaking) {
            synth.cancel();
        }

        setTimeout(() => {
            try {
                recognition.start();
            } catch (e) {
                console.error("Speech recognition error:", e);
                setMickeyStatus('Mickey could not open the microphone.');
                speak("Mickey could not open the microphone. You can type your command below.");
            }
        }, 150); 
    }

    voiceBtn.addEventListener('click', () => {
        startMickeyListening();
    });

    recognition.onstart = () => {
        isListeningForMickey = true;
        setMickeyStatus(`Listening now (${recognition.lang}). Speak after the beep.`);
        voiceBtn.classList.add('listening');
        voiceBtn.setAttribute('aria-label', 'Mickey is listening');
        const label = voiceBtn.querySelector('span:last-child');
        if (label) {
            label.textContent = 'Listening...';
        }
    };

    recognition.onend = () => {
        isListeningForMickey = false;
        voiceBtn.classList.remove('listening');
        voiceBtn.setAttribute('aria-label', 'Voice Assistant');
        const label = voiceBtn.querySelector('span:last-child');
        if (label) {
            label.textContent = 'Ask Mickey';
        }

        if (mickeyRetryPending && !mickeyGotResult) {
            const nextAttempt = mickeyRecognitionAttempt + 1;
            mickeyRetryPending = false;
            if (nextAttempt < mickeyRecognitionLanguages.length) {
                setMickeyStatus(`Trying again with ${mickeyRecognitionLanguages[nextAttempt]}...`);
                setTimeout(() => startMickeyListening(nextAttempt), 450);
            }
        }
    };

    recognition.onresult = (event) => {
        mickeyGotResult = true;
        const alternatives = Array.from(event.results[0]);
        const bestMatch = alternatives.sort((a, b) => b.confidence - a.confidence)[0];
        const command = bestMatch.transcript.toLowerCase();
        console.log('Voice command received:', command);
        setMickeyStatus(`Heard: "${command}"`);
        askMickey(command);
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        if (event.error === 'no-speech') {
            setMickeyStatus('Mickey could not hear speech. Try again or type below.');
            speak("Mickey could not hear you. Move closer to the mic or type the command below.");
        } else if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
            setMickeyStatus('Microphone permission is blocked.');
            speak("Microphone permission is blocked. Please allow microphone access in the browser.");
        } else if (event.error === 'audio-capture') {
            setMickeyStatus('No microphone was found by the browser.');
            speak("I cannot access the microphone. Please check your mic connection.");
        } else if ((event.error === 'network' || event.error === 'language-not-supported') && mickeyRecognitionAttempt < mickeyRecognitionLanguages.length - 1) {
            mickeyRetryPending = true;
            setMickeyStatus(`Voice error: ${event.error}. Trying another language...`);
        } else if (event.error === 'aborted') {
            setMickeyStatus('Voice listening stopped.');
        } else if (event.error === 'network') {
            setMickeyStatus('Voice error: network. Chrome speech needs internet access. Type below for now.');
            speak("Voice error network. Chrome speech needs internet access. Type your command below for now.");
        } else if (event.error === 'language-not-supported') {
            setMickeyStatus('Voice error: language not supported. Type below for now.');
            speak("Voice language is not supported on this browser. Type your command below for now.");
        } else {
            const errorName = event.error || 'unknown';
            setMickeyStatus(`Voice error: ${errorName}. Type below or try Chrome with mic permission.`);
            speak(`Voice error ${errorName}. Type your command below or check microphone permission.`);
        }
    };
} else if (voiceBtn) {
    voiceBtn.disabled = true;
    voiceBtn.title = 'Speech recognition is not supported in this browser. Use the text command box.';
    setMickeyStatus('Speech recognition is not supported here. Type your command below.');
    console.log("Speech Recognition not supported in this browser.");
}

async function askMickey(command) {
    try {
        const response = await fetch('/mickey', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: command })
        });
        const data = await response.json();

        if (data.reply) {
            speak(data.reply);
        }

        setTimeout(() => {
            executeMickeyAction(data);
        }, data.reply ? 900 : 0);
    } catch (error) {
        console.error('Mickey error:', error);
        speak("Mickey had trouble understanding that.");
    }
}

function executeMickeyAction(data) {
    const action = data ? data.action : null;

    if (action === 'scene') {
        getSceneDescription();
    } else if (action === 'read') {
        readTextFromCamera();
    } else if (action === 'vision') {
        getMickeyVision(data.prompt || 'What is around me and what should I be careful about?');
    } else if (action === 'navigate') {
        if (window.guideXMaps && window.guideXMaps.navigateTo) {
            window.guideXMaps.navigateTo(data.destination);
        } else {
            speak('Map navigation is not ready yet. Please wait and try again.');
        }
    } else if (action === 'repeat_navigation') {
        if (window.guideXMaps && window.guideXMaps.repeatGuidance) {
            window.guideXMaps.repeatGuidance();
        } else {
            speak('No route guidance is available yet.');
        }
    } else if (action === 'location') {
        if (window.guideXMaps && window.guideXMaps.whereAmI) {
            window.guideXMaps.whereAmI();
        } else {
            speak('Location is not ready yet.');
        }
    } else if (action === 'clear_route') {
        if (window.guideXMaps && window.guideXMaps.clearRoute) {
            window.guideXMaps.clearRoute();
            speak('Route cleared.');
        }
    } else if (action === 'start') {
        startNavigation();
    } else if (action === 'stop') {
        stopNavigation();
    } else if (action === 'indoor') {
        environmentModeSelect.value = 'indoor';
        handleEnvironmentChange();
    } else if (action === 'outdoor') {
        environmentModeSelect.value = 'outdoor';
        handleEnvironmentChange();
    } else if (action === 'sos') {
        activateSOS();
    }
}

async function getSceneDescription() {
    try {
        const response = await fetch('/scene_description');
        const data = await response.json();
        if (data.description) {
            speak(data.description);
        }
    } catch (error) {
        console.error('Error getting scene description:', error);
        speak('Error retrieving scene description.');
    }
}

async function getMickeyVision(prompt) {
    setMickeyStatus('Smart Look is checking the camera...');
    speak('Smart Look is checking the camera.');

    try {
        const response = await fetch('/mickey_vision', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: prompt || '' })
        });
        const data = await response.json();

        if (data.description) {
            setMickeyStatus('Smart Look ready.');
            speak(data.description);
        } else {
            const errorMessage = data.error || 'Smart Look could not understand the scene.';
            setMickeyStatus(errorMessage);
            speak(errorMessage);
        }
    } catch (error) {
        console.error('Smart Look error:', error);
        setMickeyStatus('Smart Look failed. Check server and internet.');
        speak('Smart Look failed. Check server and internet.');
    }
}

async function readTextFromCamera() {
    speak("Scanning for text. This may take a moment.");
    try {
        const response = await fetch('/read_text');
        const data = await response.json();
        
        if (data.text && data.text.trim() !== '') {
            speak("I read: " + data.text);
        } else {
            speak("I couldn't detect any text.");
        }
    } catch (error) {
        console.error('Error reading text:', error);
        speak('Error trying to read text.');
    }
}
