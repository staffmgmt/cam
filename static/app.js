/* Mirage Echo Baseline Client */

// Globals (scoped to this module)
let audioWs = null;
let videoWs = null;
let audioContext = null;
let processorNode = null; // AudioWorkletNode for capturing (pcm-chunker)
let playerNode = null; // AudioWorkletNode for playback (pcm-player)
let lastVideoSentTs = 0;
let remoteImageURL = null;

const LOG_EL = document.getElementById('log');
const START_BTN = document.getElementById('startBtn');
const LOCAL_VID = document.getElementById('localVid');
const REMOTE_VID_IMG = document.getElementById('remoteVid');
const REMOTE_AUDIO = document.getElementById('remoteAudio');

function log(msg) {
  const ts = new Date().toISOString().split('T')[1].replace('Z','');
  LOG_EL.textContent += `[${ts}] ${msg}\n`;
  LOG_EL.scrollTop = LOG_EL.scrollHeight;
}

function wsURL(path) {
  const proto = (location.protocol === 'https:') ? 'wss:' : 'ws:';
  return `${proto}//${location.host}${path}`;
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
    log('Failed to load worklet.js (pcm-chunker) - audio sending disabled.');
    console.error(e);
    return;
  }

  const samplesPerChunk = Math.round(audioContext.sampleRate * 0.25); // 250 ms -> spec said 0.25 (192ms earlier spec but instruction uses 0.25 here)
  log(`AudioContext sampleRate=${audioContext.sampleRate}, samplesPerChunk=${samplesPerChunk}`);
  processorNode = new AudioWorkletNode(audioContext, 'pcm-chunker', {
    processorOptions: { samplesPerChunk }
  });
  playerNode = new AudioWorkletNode(audioContext, 'pcm-player');

  // Capture mic
  const source = audioContext.createMediaStreamSource(stream);
  source.connect(processorNode);
  // Keep worklet active via silent gain path (0 gain) to destination (some browsers optimize away otherwise)
  const gain = audioContext.createGain();
  gain.gain.value = 0;
  processorNode.connect(gain).connect(audioContext.destination);
  // Do NOT connect processorNode to destination to avoid local direct monitor; playback handled by pcm-player.

  processorNode.port.onmessage = (event) => {
    if (!audioWs || audioWs.readyState !== WebSocket.OPEN) return;
    const ab = event.data;
    if (ab instanceof ArrayBuffer) audioWs.send(ab);
  };

  // Connect playback node
  playerNode.connect(audioContext.destination);
  log('Audio nodes ready (pcm-chunker + pcm-player)');
}

let _rxChunks = 0;
let _loopback = false;
function setupAudioWebSocket() {
  audioWs = new WebSocket(wsURL('/audio'));
  audioWs.binaryType = 'arraybuffer';
  audioWs.onopen = () => log('Audio WS open');
  audioWs.onclose = () => log('Audio WS closed');
  audioWs.onerror = (e) => log('Audio WS error');
  audioWs.onmessage = (evt) => {
    if (!(evt.data instanceof ArrayBuffer)) return;
    // Clone buffer BEFORE transferring to avoid ArrayBuffer detachment errors when reusing
    const src = evt.data;
    const copyBuf = src.slice(0); // shallow copy; original remains intact for stats
    // Amplitude stats (compute on copy or original before transfer)
    const view = new Int16Array(src);
    let min = 32767, max = -32768;
    for (let i=0;i<view.length;i++) { const v=view[i]; if (v<min) min=v; if (v>max) max=v; }
    // Forward copy to player (transfer copy to avoid overhead next GC cycle)
    if (playerNode) playerNode.port.postMessage(copyBuf, [copyBuf]);
    _rxChunks++;
    if ((_rxChunks % 20) === 0) {
      log(`Audio chunks received: ${_rxChunks} amp:[${min},${max}]`);
    }
    if (_loopback && audioWs && audioWs.readyState === WebSocket.OPEN) {
      // echo back again (will double) purely for test; guard to prevent infinite recursion (already from server)
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
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');

  async function readLoop() {
    try {
      const { value: frame, done } = await reader.read();
      if (done) return;

      const now = performance.now();
      const needSend = (now - lastVideoSentTs) >= 100; // ~10 fps

      if (needSend && frame) {
        try {
          // Draw frame
          if ('displayWidth' in frame && 'displayHeight' in frame) {
            ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);
          } else {
            // Fallback path: createImageBitmap then draw
            const bmp = await createImageBitmap(frame);
            ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
            bmp.close && bmp.close();
          }

          await new Promise((res, rej) => {
            canvas.toBlob((blob) => {
              if (!blob) return res();
              blob.arrayBuffer().then((ab) => {
                if (videoWs && videoWs.readyState === WebSocket.OPEN) {
                  videoWs.send(ab);
                }
                res();
              }).catch(rej);
            }, 'image/jpeg', 0.65);
          });
          lastVideoSentTs = now;
        } catch (err) {
          log('Video frame send error');
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
  videoWs.onopen = () => log('Video WS open');
  videoWs.onclose = () => log('Video WS closed');
  videoWs.onerror = () => log('Video WS error');
  videoWs.onmessage = (evt) => {
    if (!(evt.data instanceof ArrayBuffer)) return;
    const blob = new Blob([evt.data], { type: 'image/jpeg' });
    if (remoteImageURL) URL.revokeObjectURL(remoteImageURL);
    remoteImageURL = URL.createObjectURL(blob);
    REMOTE_VID_IMG.src = remoteImageURL;
  };
}

async function start() {
  START_BTN.disabled = true;
  log('Requesting media...');
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
  } catch (e) {
    log('getUserMedia failed');
    console.error(e);
    START_BTN.disabled = false;
    return;
  }
  LOCAL_VID.srcObject = stream;
  log('Media acquired');

  setupAudioWebSocket();
  setupVideoWebSocket();
  await setupAudio(stream);
  await setupVideo(stream);
}

START_BTN.addEventListener('click', start);

// Expose for debugging
function testTone(seconds = 1, freq = 440) {
  if (!audioContext || !playerNode) { log('testTone: audio not ready'); return; }
  const sampleRate = audioContext.sampleRate;
  const total = Math.floor(sampleRate * seconds);
  const int16 = new Int16Array(total);
  for (let i=0;i<total;i++) {
    const s = Math.sin(2 * Math.PI * freq * (i / sampleRate));
    int16[i] = s * 32767;
  }
  // slice into chunk-sized buffers similar to inbound network flow
  const chunk = Math.floor(sampleRate * 0.25);
  for (let off = 0; off < int16.length; off += chunk) {
    const view = int16.subarray(off, Math.min(off + chunk, int16.length));
    // copy to standalone buffer for transfer
    const copy = new Int16Array(view.length);
    copy.set(view);
    playerNode.port.postMessage(copy.buffer, [copy.buffer]);
  }
  log(`Injected test tone ${freq}Hz for ${seconds}s`);
}

window.__mirage = { start, audioWs: () => audioWs, videoWs: () => videoWs, testTone };
// Diagnostics helpers
window.__mirage.toggleLoopback = function(on){ _loopback = on !== undefined ? !!on : !_loopback; log('Local loopback=' + _loopback); };
