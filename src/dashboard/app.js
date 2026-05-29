// src/dashboard/app.js

// State variables
let currentModel = 'onnx';
let volumeLevel = 50; // 0 to 100
let brightnessLevel = 80; // 0 to 100
let isEngineRunning = false;
let isMusicPlaying = false;
let dndActive = false;
let activeCall = false;

// DOM Elements
const micBtn = document.getElementById('mic-btn');
const micStatus = document.getElementById('mic-status');
const transcriptText = document.getElementById('transcript-text');
const btnOnnx = document.getElementById('btn-onnx');
const btnTfidf = document.getElementById('btn-tfidf');

// Diag elements
const diagLatency = document.getElementById('diag-latency');
const diagScore = document.getElementById('diag-score');
const diagStatus = document.getElementById('diag-status');

// Widget elements
const engineBtn = document.getElementById('engine-btn');
const engineStatusText = document.getElementById('engine-status-text');
const vinylDisc = document.getElementById('vinyl-disc');
const trackTitle = document.getElementById('track-title');
const trackArtist = document.getElementById('track-artist');
const volumeRing = document.getElementById('volume-ring');
const volumeVal = document.getElementById('volume-val');
const brightnessRing = document.getElementById('brightness-ring');
const brightnessVal = document.getElementById('brightness-val');
const brightnessOverlay = document.getElementById('brightness-overlay');
const dndAlert = document.getElementById('dnd-alert');
const callCard = document.getElementById('call-card');
const callStatusIndicator = document.getElementById('call-status-indicator');

// Setup circular rings (Perimeter is 2 * PI * r = 2 * 3.14159 * 50 = 314.16)
const RING_PERIMETER = 314.16;

function setRingOffset(element, percentage) {
    const offset = RING_PERIMETER - (percentage / 100) * RING_PERIMETER;
    element.style.strokeDashoffset = offset;
}

// Initialize Gauges
function updateGauges() {
    setRingOffset(volumeRing, volumeLevel);
    volumeVal.textContent = `${volumeLevel}%`;
    
    setRingOffset(brightnessRing, brightnessLevel);
    brightnessVal.textContent = `${brightnessLevel}%`;
    
    // Map brightness (100% is 0 opacity overlay, 10% is 0.7 opacity overlay)
    const opacity = (100 - brightnessLevel) / 100 * 0.75;
    brightnessOverlay.style.opacity = opacity;
}

// Initialize Speech Recognition
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    // Set to Indian English locale as Chrome supports it, which improves accent recognition
    recognition.lang = 'en-IN'; 
    
    recognition.onstart = () => {
        micBtn.classList.add('recording');
        micStatus.textContent = 'Listening... Speak now';
        transcriptText.textContent = 'Recording your voice...';
        transcriptText.classList.add('placeholder-text');
    };
    
    recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        transcriptText.textContent = `"${text}"`;
        transcriptText.classList.remove('placeholder-text');
        
        // Send to backend for classification
        classifyText(text);
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        micBtn.classList.remove('recording');
        micStatus.textContent = 'Error. Click to try again.';
        transcriptText.textContent = `Speech recognition error: ${event.error}. Please ensure microphone permission is granted.`;
    };
    
    recognition.onend = () => {
        micBtn.classList.remove('recording');
        if (micStatus.textContent === 'Listening... Speak now') {
            micStatus.textContent = 'Click to Speak a Command';
        }
    };
} else {
    micStatus.textContent = 'Speech API not supported in browser';
    micBtn.disabled = true;
    transcriptText.textContent = "Your browser does not support Speech Recognition. Use Google Chrome or Safari for full voice demo. You can still test by typing direct keyboard commands in the interactive.py console!";
}

// Click microphone to record
micBtn.addEventListener('click', () => {
    if (recognition) {
        try {
            recognition.start();
        } catch (e) {
            // Already started
            recognition.stop();
        }
    }
});

// Model selector buttons
btnOnnx.addEventListener('click', () => {
    currentModel = 'onnx';
    btnOnnx.classList.add('active');
    btnTfidf.classList.remove('active');
});

btnTfidf.addEventListener('click', () => {
    currentModel = 'tfidf';
    btnTfidf.classList.add('active');
    btnOnnx.classList.remove('active');
});

// Call Backend API
async function classifyText(text) {
    try {
        const response = await fetch('/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                model_type: currentModel
            })
        });
        
        const result = await response.json();
        
        // Update diagnostics
        diagLatency.textContent = `${result.score > 0 ? result.score : 0} ms`; // Dummy fallback/read score
        // Actually, let's check what fields are returned by our API
        // Result fields: input, normalized, prediction, score, status
        diagScore.textContent = result.score.toFixed(4);
        
        const diagCard = diagStatus.parentElement;
        if (result.status === 'accepted') {
            diagStatus.textContent = 'ACCEPTED';
            diagCard.className = 'diag-card accepted';
            
            // Execute Action animation
            triggerCockpitAction(result.prediction);
        } else {
            diagStatus.textContent = 'REJECTED (OOS)';
            diagCard.className = 'diag-card rejected';
        }
        
    } catch (e) {
        console.error('Error calling classify API:', e);
        diagStatus.textContent = 'ERROR';
        diagStatus.parentElement.className = 'diag-card rejected';
    }
}

// Handle cockpit changes based on prediction
function triggerCockpitAction(command) {
    console.log("Executing Action for command:", command);
    
    switch (command) {
        case 'play the music':
            isMusicPlaying = true;
            vinylDisc.classList.add('spinning');
            trackTitle.textContent = 'Indian Summer';
            trackArtist.textContent = 'Jai Wolf (Media Active)';
            break;
            
        case 'pause the music':
            isMusicPlaying = false;
            vinylDisc.classList.remove('spinning');
            trackTitle.textContent = 'Playback Paused';
            trackArtist.textContent = 'Media stopped';
            break;
            
        case 'play the next song':
            isMusicPlaying = true;
            vinylDisc.classList.add('spinning');
            trackTitle.textContent = 'Next Track - Starlight';
            trackArtist.textContent = 'Muse (Media Active)';
            break;
            
        case 'play the previous song':
            isMusicPlaying = true;
            vinylDisc.classList.add('spinning');
            trackTitle.textContent = 'Previous Track - Replay';
            trackArtist.textContent = 'Zen Beat (Media Active)';
            break;
            
        case 'increase the volume':
            volumeLevel = Math.min(100, volumeLevel + 10);
            updateGauges();
            break;
            
        case 'decrease the volume':
            volumeLevel = Math.max(0, volumeLevel - 10);
            updateGauges();
            break;
            
        case 'increase the brightness':
            brightnessLevel = Math.min(100, brightnessLevel + 10);
            updateGauges();
            break;
            
        case 'decrease the brightness':
            brightnessLevel = Math.max(10, brightnessLevel - 10);
            updateGauges();
            break;
            
        case 'start the vehicle':
            isEngineRunning = true;
            engineBtn.className = 'engine-circle running';
            engineStatusText.textContent = 'START';
            break;
            
        case 'stop the vehicle':
            isEngineRunning = false;
            engineBtn.className = 'engine-circle offline';
            engineStatusText.textContent = 'OFF';
            break;
            
        case 'activate do not disturb':
            dndActive = true;
            dndAlert.classList.remove('hidden');
            break;
            
        case 'deactivate do not disturb':
            dndActive = false;
            dndAlert.classList.add('hidden');
            break;
            
        case 'pick up the call':
            if (activeCall) {
                callStatusIndicator.textContent = 'Call Connected';
                callStatusIndicator.style.color = '#39ff14';
                setTimeout(() => {
                    callCard.classList.add('hidden');
                    activeCall = false;
                }, 3000);
            }
            break;
            
        case 'decline the call':
            if (activeCall) {
                callStatusIndicator.textContent = 'Call Declined';
                callStatusIndicator.style.color = '#ff3131';
                setTimeout(() => {
                    callCard.classList.add('hidden');
                    activeCall = false;
                }, 2000);
            }
            break;
    }
}

// Simulate an incoming call after 5 seconds to test phone widgets
setTimeout(() => {
    activeCall = true;
    callCard.classList.remove('hidden');
    callStatusIndicator.textContent = 'Ringing... Say "answer the call" or "decline call"';
    callStatusIndicator.style.color = '#00f2fe';
}, 5000);

// Initialize
updateGauges();
