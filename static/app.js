/* Mirage Real-time AI Avatar Client */

// Globals
let audioWs = null;
let videoWs = null;
let audioContext = null;
let processorNode = null;
let playerNode = null;
let lastVideoSentTs = 0;
let remoteImageURL = null;
let isRunning = false;
let pipelineInitialized = false;
let referenceSet = false;
let virtualCameraStream = null;
let metricsInterval = null;

// Configuration
const videoMaxFps = 20; // Increased for real-time avatar
const videoFrameIntervalMs = 1000 / videoMaxFps;

// DOM elements
const LOG_EL = document.getElementById('log');
const INIT_BTN = document.getElementById('initBtn');
const START_BTN = document.getElementById('startBtn');
const STOP_BTN = document.getElementById('stopBtn');
const LOCAL_VID = document.getElementById('localVid');
const REMOTE_VID_IMG = document.getElementById('remoteVid');
const REMOTE_AUDIO = document.getElementById('remoteAudio');
const STATUS_DIV = document.getElementById('statusDiv');
const REFERENCE_INPUT = document.getElementById('referenceInput');
const VIRTUAL_CAM_BTN = document.getElementById('virtualCamBtn');
const VIRTUAL_CANVAS = document.getElementById('virtualCanvas');

function log(msg) {
  const ts = new Date().toISOString().split('T')[1].replace('Z','');
  LOG_EL.textContent += `[${ts}] ${msg}\n`;
  LOG_EL.scrollTop = LOG_EL.scrollHeight;
}

function showStatus(message, type = 'info') {
  STATUS_DIV.innerHTML = `<div class="status ${type}">${message}</div>`;
  setTimeout(() => STATUS_DIV.innerHTML = '', 5000);
}

function wsURL(path) {
  const proto = (location.protocol === 'https:') ? 'wss:' : 'ws:';
  return `${proto}//${location.host}${path}`;
}

// Initialize AI Pipeline
async function initializePipeline() {
  INIT_BTN.disabled = true;
  INIT_BTN.textContent = 'Initializing...';
  
  try {
    log('Initializing AI pipeline...');
    const response = await fetch('/initialize', { method: 'POST' });
    const result = await response.json();
    
    if (result.status === 'success' || result.status === 'already_initialized') {
      pipelineInitialized = true;
      showStatus('AI pipeline initialized successfully!', 'success');
      log('AI pipeline ready');
      
      // Enable controls
      START_BTN.disabled = false;
      REFERENCE_INPUT.disabled = false;
      
      // Start metrics updates
      startMetricsUpdates();
    } else {
      showStatus(`Initialization failed: ${result.message}`, 'error');
      log(`Pipeline init failed: ${result.message}`);
    }
  } catch (error) {
    showStatus(`Initialization error: ${error.message}`, 'error');
    log(`Init error: ${error}`);
  } finally {
    INIT_BTN.disabled = false;
    INIT_BTN.textContent = 'Initialize AI Pipeline';
  }
}

// Handle reference image upload
async function handleReferenceUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  
  log('Uploading reference image...');
  
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/set_reference', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    
    if (result.status === 'success') {
      referenceSet = true;
      showStatus('Reference image set successfully!', 'success');
      log('Reference image configured');
      VIRTUAL_CAM_BTN.disabled = false;
    } else {
      showStatus(`Reference setup failed: ${result.message}`, 'error');
      log(`Reference error: ${result.message}`);
    }
  } catch (error) {
    showStatus(`Upload error: ${error.message}`, 'error');
    log(`Reference upload error: ${error}`);
  }
}

async function setupAudio(stream) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  if (audioContext.state === 'suspended') {
    try { await audioContext.resume(); } catch (e) { log('AudioContext resume failed'); }
  }

  // Worklet loading
  try {
    await audioContext.audioWorklet.addModule('/static/worklet.js');
  } catch (e) {
    log('Failed to load worklet.js - audio processing disabled.');
    console.error(e);
    return;
  }

  // Enhanced chunk configuration for real-time processing
  const chunkMs = 160; // Keep at 160ms for balance between latency and quality
  const samplesPerChunk = Math.round(audioContext.sampleRate * (chunkMs / 1000));
  
  log(`Audio chunk config: sampleRate=${audioContext.sampleRate}Hz chunkMs=${chunkMs}ms samplesPerChunk=${samplesPerChunk}`);
  
  processorNode = new AudioWorkletNode(audioContext, 'pcm-chunker', {
    processorOptions: { samplesPerChunk }
  });
  playerNode = new AudioWorkletNode(audioContext, 'pcm-player');

  // Capture mic
  const source = audioContext.createMediaStreamSource(stream);
  source.connect(processorNode);
  
  // Keep worklet active
  const gain = audioContext.createGain();
  gain.gain.value = 0;
  processorNode.connect(gain).connect(audioContext.destination);

  processorNode.port.onmessage = (event) => {
    if (!audioWs || audioWs.readyState !== WebSocket.OPEN) return;
    const ab = event.data;
    if (ab instanceof ArrayBuffer) audioWs.send(ab);
  };

  // Connect playback node
  playerNode.connect(audioContext.destination);
  log('Audio nodes ready (enhanced for AI processing)');
}

let _rxChunks = 0;
function setupAudioWebSocket() {
  audioWs = new WebSocket(wsURL('/audio'));
  audioWs.binaryType = 'arraybuffer';
  audioWs.onopen = () => log('Audio WebSocket connected');
  audioWs.onclose = () => log('Audio WebSocket disconnected');
  audioWs.onerror = (e) => log('Audio WebSocket error');
  audioWs.onmessage = (evt) => {
    if (!(evt.data instanceof ArrayBuffer)) return;
    
    const src = evt.data;
    const copyBuf = src.slice(0);
    
    // Amplitude analysis for voice activity detection
    const view = new Int16Array(src);
    let min = 32767, max = -32768;
    for (let i = 0; i < view.length; i++) { 
      const v = view[i]; 
      if (v < min) min = v; 
      if (v > max) max = v; 
    }
    
    // Forward to player
    if (playerNode) playerNode.port.postMessage(copyBuf, [copyBuf]);
    
    _rxChunks++;
    if ((_rxChunks % 30) === 0) { // Reduced logging frequency
      log(`Audio processed: ${_rxChunks} chunks, amp:[${min},${max}]`);
    }
  };
}

async function setupVideo(stream) {
  const track = stream.getVideoTracks()[0];
  if (!track) {
    log('No video track found');
    return;
  }
  
  const processor = new MediaStreamTrackProcessor({ track });
  const reader = processor.readable.getReader();

  const canvas = document.createElement('canvas');
  canvas.width = 512;  // Increased resolution for AI processing
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  async function readLoop() {
    try {
      const { value: frame, done } = await reader.read();
      if (done) return;

      const now = performance.now();
      const elapsed = now - lastVideoSentTs;
      const needSend = elapsed >= videoFrameIntervalMs;

      if (needSend && frame) {
        try {
          // Draw frame with improved quality
          if ('displayWidth' in frame && 'displayHeight' in frame) {
            ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);
          } else {
            const bmp = await createImageBitmap(frame);
            ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
            bmp.close && bmp.close();
          }

          // Send to AI pipeline with higher quality
          await new Promise((res, rej) => {
            canvas.toBlob((blob) => {
              if (!blob) return res();
              blob.arrayBuffer().then((ab) => {
                if (videoWs && videoWs.readyState === WebSocket.OPEN) {
                  videoWs.send(ab);
                }
                res();
              }).catch(rej);
            }, 'image/jpeg', 0.8); // Higher quality for AI processing
          });
          
          lastVideoSentTs = now;
        } catch (err) {
          log('Video frame processing error');
          console.error(err);
        }
      }

      frame.close && frame.close();
      readLoop();
    } catch (err) {
      log('Video read loop error');
      console.error(err);
    }
  }
  readLoop();
}

function setupVideoWebSocket() {
  videoWs = new WebSocket(wsURL('/video'));
  videoWs.binaryType = 'arraybuffer';
  videoWs.onopen = () => log('Video WebSocket connected');
  videoWs.onclose = () => log('Video WebSocket disconnected');
  videoWs.onerror = () => log('Video WebSocket error');
  videoWs.onmessage = (evt) => {
    if (!(evt.data instanceof ArrayBuffer)) return;
    
    // Display AI-processed video
    const blob = new Blob([evt.data], { type: 'image/jpeg' });
    if (remoteImageURL) URL.revokeObjectURL(remoteImageURL);
    remoteImageURL = URL.createObjectURL(blob);
    REMOTE_VID_IMG.src = remoteImageURL;
    
    // Update virtual camera if enabled
    updateVirtualCamera(evt.data);
  };
}

// Virtual Camera Support
function updateVirtualCamera(imageData) {
  if (!virtualCameraStream) return;
  
  try {
    // Create image from received data
    const blob = new Blob([imageData], { type: 'image/jpeg' });
    const img = new Image();
    
    img.onload = () => {
      // Draw to virtual canvas
      const ctx = VIRTUAL_CANVAS.getContext('2d');
      VIRTUAL_CANVAS.width = 512;
      VIRTUAL_CANVAS.height = 512;
      ctx.drawImage(img, 0, 0, 512, 512);
    };
    
    img.src = URL.createObjectURL(blob);
  } catch (error) {
    console.error('Virtual camera update error:', error);
  }
}

async function enableVirtualCamera() {
  try {
    if (!VIRTUAL_CANVAS.captureStream) {
      showStatus('Virtual camera not supported in this browser', 'error');
      return;
    }
    
    // Create virtual camera stream from canvas
    virtualCameraStream = VIRTUAL_CANVAS.captureStream(30);
    
    // Try to create a virtual camera device (browser-dependent)
    if (navigator.mediaDevices.getDisplayMedia) {
      log('Virtual camera enabled - canvas stream ready');
      showStatus('Virtual camera enabled! Use canvas stream in video apps.', 'success');
      VIRTUAL_CAM_BTN.textContent = 'Virtual Camera Active';
      VIRTUAL_CAM_BTN.disabled = true;
    } else {
      showStatus('Virtual camera API not available', 'error');
    }
  } catch (error) {
    showStatus(`Virtual camera error: ${error.message}`, 'error');
    log(`Virtual camera error: ${error}`);
  }
}

// Metrics and Performance Monitoring
function startMetricsUpdates() {
  if (metricsInterval) clearInterval(metricsInterval);
  
  metricsInterval = setInterval(async () => {
    try {
      const response = await fetch('/pipeline_status');
      const data = await response.json();
      
      if (data.initialized && data.stats) {
        const stats = data.stats;
        
        document.getElementById('fpsValue').textContent = stats.video_fps?.toFixed(1) || '0';
        document.getElementById('latencyValue').textContent = 
          Math.round(stats.avg_video_latency_ms || 0) + 'ms';
        document.getElementById('gpuValue').textContent = 
          stats.gpu_memory_used?.toFixed(1) + 'GB' || 'N/A';
        document.getElementById('statusValue').textContent = 
          stats.models_loaded ? 'Active' : 'Loading';
      }
    } catch (error) {
      console.error('Metrics update error:', error);
    }
  }, 2000); // Update every 2 seconds
}

async function start() {
  if (!pipelineInitialized) {
    showStatus('Please initialize the AI pipeline first', 'error');
    return;
  }
  
  START_BTN.disabled = true;
  START_BTN.textContent = 'Starting...';
  
  log('Requesting media access...');
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: true, 
      video: { 
        width: 640, 
        height: 480, 
        frameRate: 30 
      } 
    });
    
    LOCAL_VID.srcObject = stream;
    log('Media access granted');
    
    // Setup WebSocket connections
    setupAudioWebSocket();
    setupVideoWebSocket();
    
    // Setup audio and video processing
    await setupAudio(stream);
    await setupVideo(stream);
    
    isRunning = true;
    START_BTN.style.display = 'none';
    STOP_BTN.disabled = false;
    STOP_BTN.style.display = 'inline-block';
    
    log(`Real-time AI avatar started: ${videoMaxFps} fps, 160ms audio chunks`);
    showStatus('AI Avatar system is now running!', 'success');
    
  } catch (error) {
    showStatus(`Media access failed: ${error.message}`, 'error');
    log(`getUserMedia failed: ${error}`);
    START_BTN.disabled = false;
    START_BTN.textContent = 'Start Capture';
  }
}

function stop() {
  log('Stopping AI avatar system...');
  
  // Close WebSocket connections
  if (audioWs) {
    audioWs.close();
    audioWs = null;
  }
  if (videoWs) {
    videoWs.close();
    videoWs = null;
  }
  
  // Stop media tracks
  if (LOCAL_VID.srcObject) {
    LOCAL_VID.srcObject.getTracks().forEach(track => track.stop());
    LOCAL_VID.srcObject = null;
  }
  
  // Reset audio context
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  
  // Reset UI
  isRunning = false;
  START_BTN.disabled = false;
  START_BTN.textContent = 'Start Capture';
  START_BTN.style.display = 'inline-block';
  STOP_BTN.disabled = true;
  STOP_BTN.style.display = 'none';
  
  log('System stopped');
  showStatus('AI Avatar system stopped', 'info');
}

// Event Listeners
INIT_BTN.addEventListener('click', initializePipeline);
START_BTN.addEventListener('click', start);
STOP_BTN.addEventListener('click', stop);
REFERENCE_INPUT.addEventListener('change', handleReferenceUpload);
VIRTUAL_CAM_BTN.addEventListener('click', enableVirtualCamera);

// Debug functions
function testTone(seconds = 1, freq = 440) {
  if (!audioContext || !playerNode) { 
    log('testTone: audio not ready'); 
    return; 
  }
  
  const sampleRate = audioContext.sampleRate;
  const total = Math.floor(sampleRate * seconds);
  const int16 = new Int16Array(total);
  
  for (let i = 0; i < total; i++) {
    const s = Math.sin(2 * Math.PI * freq * (i / sampleRate));
    int16[i] = s * 32767;
  }
  
  const chunk = Math.floor(sampleRate * 0.25);
  for (let off = 0; off < int16.length; off += chunk) {
    const view = int16.subarray(off, Math.min(off + chunk, int16.length));
    const copy = new Int16Array(view.length);
    copy.set(view);
    playerNode.port.postMessage(copy.buffer, [copy.buffer]);
  }
  
  log(`Test tone ${freq}Hz for ${seconds}s injected`);
}

// Global API for debugging
window.__mirage = { 
  start, 
  stop, 
  initializePipeline,
  audioWs: () => audioWs, 
  videoWs: () => videoWs, 
  testTone,
  pipelineInitialized: () => pipelineInitialized,
  referenceSet: () => referenceSet
};

// Auto-initialize on load for development
log('Mirage Real-time AI Avatar System loaded');
log('Click "Initialize AI Pipeline" to begin setup');
