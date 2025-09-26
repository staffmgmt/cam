/* Enterprise-grade WebRTC client for Mirage AI Avatar Studio */
(function(){
  'use strict';

  // Application state
  const state = {
    pc: null,
    control: null,
    localStream: null,
    metricsTimer: null,
    referenceImage: null,
    connected: false,
    authToken: null,
    connecting: false,
    cancelled: false,
    initialized: false
  };
  
  // URL parameters
  const params = new URLSearchParams(location.search);
  const FORCE_RELAY_URL = params.get('relay') === '1';
  const VERBOSE = (window.MIRAGE_WEBRTC_VERBOSE === true) || (params.get('wv') === '1');
  const STATS_INTERVAL_MS = window.MIRAGE_WEBRTC_STATS_INTERVAL_MS || 5000;
  
  let statsTimer = null;

  // DOM Elements
  const els = {
    // File upload
    ref: document.getElementById('referenceInput'),
    uploadButton: document.getElementById('uploadButton'),
    uploadText: document.getElementById('uploadText'),
    
    // Control buttons
    init: document.getElementById('initBtn'),
    debug: document.getElementById('debugBtn'),
    connect: document.getElementById('connectBtn'),
    disconnect: document.getElementById('disconnectBtn'),
    
    // Video elements
    localVideo: document.getElementById('localVideo'),
    remoteVideo: document.getElementById('remoteVideo'),
    localWrapper: document.getElementById('localWrapper'),
    avatarWrapper: document.getElementById('avatarWrapper'),
    localOverlay: document.getElementById('localOverlay'),
    avatarOverlay: document.getElementById('avatarOverlay'),
    
    // Status indicators
    systemStatus: document.getElementById('systemStatus'),
    localStatus: document.getElementById('localStatus'),
    avatarStatus: document.getElementById('avatarStatus'),
    statusText: document.getElementById('statusText'),
    localStatusText: document.getElementById('localStatusText'),
    avatarStatusText: document.getElementById('avatarStatusText'),
    
    // Metrics
    latencyValue: document.getElementById('latencyValue'),
    fpsValue: document.getElementById('fpsValue'),
    gpuValue: document.getElementById('gpuValue'),
    qualityValue: document.getElementById('qualityValue'),
    
    // Toast
    toast: document.getElementById('toast'),
    toastContent: document.getElementById('toastContent')
  };

  // Utility Functions
  function log(...args) { 
    console.log('[MIRAGE]', ...args); 
  }
  
  function vlog(...args) { 
    if(VERBOSE) console.log('[MIRAGE][VERBOSE]', ...args); 
  }

  // UI Helper Functions
  function setSystemStatus(status, text) {
    if (els.systemStatus) {
      els.systemStatus.className = `status-indicator status-${status}`;
    }
    if (els.statusText) {
      els.statusText.textContent = text;
    }
  }

  function setLocalStatus(status, text) {
    if (els.localStatus) {
      els.localStatus.className = `status-indicator status-${status}`;
    }
    if (els.localStatusText) {
      els.localStatusText.textContent = text;
    }
  }

  function setAvatarStatus(status, text) {
    if (els.avatarStatus) {
      els.avatarStatus.className = `status-indicator status-${status}`;
    }
    if (els.avatarStatusText) {
      els.avatarStatusText.textContent = text;
    }
  }

  function showToast(message, type = 'info') {
    if (els.toastContent && els.toast) {
      els.toastContent.textContent = message;
      els.toast.className = `toast toast-${type} show`;
      setTimeout(() => {
        els.toast.classList.remove('show');
      }, 4000);
    }
  }

  function updateMetrics(latency, fps, gpu, quality = 'HD') {
    if (els.latencyValue) els.latencyValue.textContent = latency || '--';
    if (els.fpsValue) els.fpsValue.textContent = fps || '--';
    if (els.gpuValue) els.gpuValue.textContent = gpu || '--';
    if (els.qualityValue) els.qualityValue.textContent = quality || '--';
  }

  function setButtonLoading(button, loading = true) {
    if (!button) return;
    if (loading) {
      button.disabled = true;
      const originalText = button.innerHTML;
      button.dataset.originalText = originalText;
      button.innerHTML = '<span class="loading-spinner"></span> Processing...';
    } else {
      button.disabled = false;
      if (button.dataset.originalText) {
        button.innerHTML = button.dataset.originalText;
        delete button.dataset.originalText;
      }
    }
  }

  function setStatus(txt) { 
    setSystemStatus('idle', txt);
  }

  // WebRTC Diagnostics
  function attachPcDiagnostics(pc) {
    if (!pc) return;
    const evMap = ['signalingstatechange','iceconnectionstatechange','icegatheringstatechange','connectionstatechange','negotiationneeded'];
    evMap.forEach(ev => {
      pc.addEventListener(ev, () => { vlog('pc event', ev, diagSnapshot()); });
    });
    pc.addEventListener('track', ev => { vlog('pc track event', ev.track && ev.track.kind, ev.streams && ev.streams.length); });
    pc.onicecandidate = (e) => { 
      if (e.candidate) { 
        vlog('ice candidate', e.candidate.type, e.candidate.protocol, e.candidate.address, e.candidate.relatedAddress||null); 
      } else { 
        vlog('ice candidate gathering complete'); 
      } 
    };
  }

  function diagSnapshot() {
    if (!state.pc) return {};
    return {
      signaling: state.pc.signalingState,
      iceConnection: state.pc.iceConnectionState,
      iceGathering: state.pc.iceGatheringState,
      connection: state.pc.connectionState,
      localTracks: state.pc.getSenders().map(s => ({kind: s.track && s.track.kind, ready: s.track && s.track.readyState})),
      remoteTracks: state.pc.getReceivers().map(r => ({kind: r.track && r.track.kind, ready: r.track && r.track.readyState}))
    };
  }

  async function collectStats() {
    if (!state.pc) return;
    try {
      const stats = await state.pc.getStats();
      const summary = { ts: Date.now(), outbound: {}, inbound: {}, candidatePairs: [] };
      stats.forEach(report => {
        if (report.type === 'outbound-rtp' && !report.isRemote) {
          summary.outbound[report.kind || report.mediaType || 'unknown'] = {
            bitrateKbps: report.bytesSent && report.timestamp ? undefined : undefined,
            frames: report.framesEncoded,
            q: report.qualityLimitationReason,
            packetsSent: report.packetsSent
          };
        } else if (report.type === 'inbound-rtp' && !report.isRemote) {
          summary.inbound[report.kind || report.mediaType || 'unknown'] = {
            jitter: report.jitter,
            packetsLost: report.packetsLost,
            frames: report.framesDecoded,
            packetsReceived: report.packetsReceived
          };
        } else if (report.type === 'candidate-pair' && report.state === 'succeeded') {
          summary.candidatePairs.push({
            current: report.nominated,
            bytesSent: report.bytesSent,
            bytesReceived: report.bytesReceived,
            rtt: report.currentRoundTripTime,
            availableOutgoingBitrate: report.availableOutgoingBitrate
          });
        }
      });
      vlog('webrtc stats', summary);
    } catch(e) { 
      vlog('stats error', e); 
    }
  }

  // Reference Image Handler
  async function handleReference(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) {
      if (els.uploadButton) els.uploadButton.classList.remove('has-file');
      if (els.uploadText) els.uploadText.textContent = 'Choose Reference Image';
      return;
    }
    
    // Update UI immediately
    if (els.uploadButton) els.uploadButton.classList.add('has-file');
    if (els.uploadText) els.uploadText.textContent = `✓ ${file.name}`;
    showToast(`Reference image selected: ${file.name}`, 'success');
    
    // Cache base64 for datachannel use
    const buf = await file.arrayBuffer();
    const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
    state.referenceImage = b64;
    
    // Upload to server
    try {
      setSystemStatus('connecting', 'Uploading reference image...');
      const fd = new FormData();
      fd.append('file', new Blob([buf], {type: file.type || 'application/octet-stream'}), file.name || 'reference');
      const resp = await fetch('/set_reference', {method: 'POST', body: fd});
      const jr = await resp.json().catch(() => ({}));
      
      if (resp.ok && jr && (jr.status === 'success' || jr.status === 'ok')) {
        setSystemStatus('connected', 'Reference image uploaded successfully');
        showToast('Reference image set successfully', 'success');
      } else {
        setSystemStatus('error', 'Reference upload failed');
        showToast('Failed to upload reference image', 'error');
        console.warn('set_reference response', resp.status, jr);
      }
    } catch(err) {
      console.warn('set_reference error', err);
      setSystemStatus('error', 'Reference upload error');
      showToast('Error uploading reference image', 'error');
    }
    
    // Send via data channel if already connected
    try {
      if (state.connected && state.control && state.control.readyState === 'open') {
        state.control.send(JSON.stringify({type: 'set_reference', image_base64: state.referenceImage}));
        showToast('Reference updated in live session', 'success');
      }
    } catch(_) {}
  }

  // WebRTC Connection
  async function connect(options) {
    const overrideRelay = options && options.forceRelay === true;
    if (state.connected || state.connecting) return;
    
    try {
      setSystemStatus('connecting', 'Requesting camera access...');
      setLocalStatus('connecting', 'Initializing');
      setButtonLoading(els.connect, true);
      els.disconnect.disabled = false;
      state.cancelled = false; 
      state.connecting = true;

      // Ping WebRTC router
      try {
        const ping = await fetch('/webrtc/ping');
        if (ping.ok) { 
          const j = await ping.json(); 
          log('webrtc ping', j); 
        }
      } catch(_) {}

      // Get auth token
      let authToken = state.authToken;
      try {
        const t = await fetch('/webrtc/token');
        if (t.ok) {
          const j = await t.json();
          authToken = j.token; 
          state.authToken = authToken;
        } else if (t.status === 404) {
          console.warn('Token endpoint 404 - proceeding without token');
        }
      } catch(_) {}

      // Get user media
      state.localStream = await navigator.mediaDevices.getUserMedia({video: true, audio: true});
      els.localVideo.srcObject = state.localStream;
      if (els.localWrapper) els.localWrapper.classList.add('active');
      setLocalStatus('connected', 'Camera Active');
      
      try { 
        els.localVideo.play && els.localVideo.play(); 
      } catch(_) {}

      setSystemStatus('connecting', 'Establishing connection...');

      // Get ICE configuration
      let iceCfg = {iceServers: [{urls: ['stun:stun.l.google.com:19302']}]};
      try {
        const ic = await fetch('/webrtc/ice_config');
        if (ic.ok) { iceCfg = await ic.json(); }
      } catch(_) {}
      
      state._lastIceCfg = iceCfg;
      if (overrideRelay || FORCE_RELAY_URL || iceCfg.forceRelay === true) { 
        iceCfg.iceTransportPolicy = 'relay'; 
      }
      log('ice config', iceCfg);

      // Create peer connection
      state.pc = new RTCPeerConnection(iceCfg);
      attachPcDiagnostics(state.pc);
      
      state._usedRelay = !!iceCfg.iceTransportPolicy && iceCfg.iceTransportPolicy === 'relay';
      state._relayFallbackTried = !!overrideRelay || !!FORCE_RELAY_URL;

      // Connection state handlers
      state.pc.oniceconnectionstatechange = () => {
        log('ice state', state.pc.iceConnectionState);
        if (['failed', 'closed'].includes(state.pc.iceConnectionState)) {
          if (!state.cancelled) disconnect();
        }
        if (state.pc.iceConnectionState === 'disconnected') {
          vlog('ICE disconnected snapshot', diagSnapshot());
        }
      };

      state.pc.onconnectionstatechange = () => {
        const st = state.pc.connectionState;
        log('pc state', st);
        
        if (st === 'connected' && !statsTimer) {
          statsTimer = setInterval(collectStats, STATS_INTERVAL_MS);
        }
        
        if (st === 'disconnected') {
          vlog('PC disconnected snapshot', diagSnapshot());
          try { 
            state.pc.restartIce && state.pc.restartIce(); 
            vlog('Attempted ICE restart'); 
          } catch(_) {}
        }
        
        if (['failed', 'closed'].includes(st)) {
          if (statsTimer) { 
            clearInterval(statsTimer); 
            statsTimer = null; 
          }
          
          const hasTurn = state._lastIceCfg && (state._lastIceCfg.turnCount || 0) > 0;
          const tryRelay = hasTurn && !state._usedRelay && !state._relayFallbackTried;
          const snapshot = diagSnapshot();
          vlog('Final failure snapshot', snapshot, {hasTurn, usedRelay: state._usedRelay, relayTried: state._relayFallbackTried});
          
          disconnect().then(() => {
            if (tryRelay) {
              state._relayFallbackTried = true;
              log('retrying with relay-only');
              setSystemStatus('connecting', 'Retrying with TURN relay...');
              connect({forceRelay: true});
            } else if (!hasTurn && !state._usedRelay && !state._relayFallbackTried) {
              log('skipping relay-only retry: no TURN servers available');
              setSystemStatus('error', 'No TURN servers available');
              showToast('Connection failed - no TURN servers available', 'error');
            }
          });
        }
      };

      // Track handler
      state.pc.ontrack = ev => {
        try {
          const tr = ev.track;
          log('ontrack', tr && tr.kind, tr && tr.readyState, ev.streams && ev.streams.length);
          
          if (tr && tr.kind === 'video') {
            setSystemStatus('connected', 'Avatar stream received');
            setAvatarStatus('connected', 'Active');
            
            let stream;
            if (ev.streams && ev.streams[0]) {
              stream = ev.streams[0];
              log('Using provided stream:', stream.id, 'tracks:', stream.getTracks().length);
            } else {
              stream = new MediaStream([ev.track]);
              log('Created new MediaStream:', stream.id);
            }
            
            // Set video source
            log('Setting srcObject on video element');
            els.remoteVideo.srcObject = null;
            els.remoteVideo.srcObject = stream;
            if (els.avatarWrapper) els.avatarWrapper.classList.add('active');
            
            // Video event handlers
            els.remoteVideo.onloadeddata = () => {
              log('video: loadeddata, attempting play()');
              els.remoteVideo.play().catch(e => {
                log('play error', e.name, e.message);
                setTimeout(() => {
                  log('Retry play() after error...');
                  els.remoteVideo.play().catch(e2 => log('Retry play failed:', e2.name));
                }, 100);
              });
            };
            
            els.remoteVideo.onplaying = () => {
              log('video: playing');
              showToast('Avatar stream connected successfully', 'success');
            };
            
            els.remoteVideo.onerror = (e) => {
              log('video error:', e);
              setAvatarStatus('error', 'Stream Error');
            };
            
            // Track state handlers
            tr.onended = () => {
              log('video track ended');
              setAvatarStatus('idle', 'Disconnected');
              if (els.avatarWrapper) els.avatarWrapper.classList.remove('active');
            };
            
            tr.onmute = () => {
              log('video track muted');
              setAvatarStatus('warning', 'Muted');
            };
            
            tr.onunmute = () => {
              log('video track unmuted');
              setAvatarStatus('connected', 'Active');
            };
            
          } else if (tr && tr.kind === 'audio') {
            setSystemStatus('connected', 'Audio stream received');
          }
        } catch(e) { 
          log('ontrack error', e);
          setAvatarStatus('error', 'Connection Error');
        }
      };

      // Data channel setup
      state.control = state.pc.createDataChannel('control');
      
      state.control.onopen = () => {
        setSystemStatus('connected', 'WebRTC connection established');
        state.connected = true;
        state.connecting = false;
        setButtonLoading(els.connect, false);
        els.connect.disabled = true;
        els.disconnect.disabled = false;
        showToast('WebRTC connection established', 'success');
        
        // Send reference image if available
        if (state.referenceImage) {
          try { 
            state.control.send(JSON.stringify({type: 'set_reference', image_base64: state.referenceImage}));
            showToast('Reference image sent to avatar', 'success');
          } catch(e) {
            showToast('Failed to send reference image', 'error');
          }
        }
        
        // Start metrics polling
        state.metricsTimer = setInterval(() => {
          try { 
            state.control.send(JSON.stringify({type: 'metrics_request'})); 
          } catch(_) {}
        }, 4000);
      };

      state.control.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.type === 'metrics' && data.payload) {
            updatePerf(data.payload);
          } else if (data.type === 'reference_ack') {
            setSystemStatus('connected', 'Reference acknowledged');
          } else if (data.type === 'error' && data.message) {
            setSystemStatus('error', 'Error: ' + data.message);
            showToast('Avatar error: ' + data.message, 'error');
          }
        } catch(_) {}
      };

      // Add local tracks
      state.localStream.getTracks().forEach(t => state.pc.addTrack(t, state.localStream));

      // Create offer
      const offer = await state.pc.createOffer({offerToReceiveAudio: true, offerToReceiveVideo: true});
      await state.pc.setLocalDescription(offer);

      // Wait for ICE gathering
      setSystemStatus('connecting', 'Gathering ICE candidates...');
      await new Promise((resolve) => {
        if (state.pc.iceGatheringState === 'complete') return resolve();
        const timeout = setTimeout(() => { resolve(); }, 7000);
        state.pc.onicegatheringstatechange = () => {
          if (state.pc.iceGatheringState === 'complete') {
            clearTimeout(timeout); 
            resolve();
          }
        };
      });

      // Send offer to server
      setSystemStatus('connecting', 'Negotiating connection...');
      const headers = {'Content-Type': 'application/json'};
      if (authToken) headers['X-Auth-Token'] = authToken;

      const ld = state.pc.localDescription;
      const r = await fetch('/webrtc/offer', {
        method: 'POST', 
        headers, 
        body: JSON.stringify({sdp: ld.sdp, type: ld.type})
      });

      if (!r.ok) {
        let bodyText = '';
        try { bodyText = await r.text(); } catch(_) {}
        if (r.status === 401 || r.status === 403) {
          setSystemStatus('error', 'Unauthorized (check API key/token)');
          showToast('Authentication failed', 'error');
        } else {
          setSystemStatus('error', `Server error: ${r.status}`);
          showToast(`Connection failed: ${r.status}`, 'error');
        }
        throw new Error(`Server returned ${r.status}: ${bodyText}`);
      }

      const answer = await r.json();
      await state.pc.setRemoteDescription(new RTCSessionDescription(answer));
      log('WebRTC negotiation complete');

    } catch(e) {
      log('connect error', e);
      setSystemStatus('error', 'Connection failed');
      showToast('Failed to establish connection', 'error');
      state.connecting = false;
      setButtonLoading(els.connect, false);
      throw e;
    }
  }

  // Disconnect
  async function disconnect() {
    if (state.cancelled) return;
    state.cancelled = true;
    log('disconnecting...');
    
    // Clear timers
    if (statsTimer) { 
      clearInterval(statsTimer); 
      statsTimer = null; 
    }
    if (state.metricsTimer) { 
      clearInterval(state.metricsTimer); 
      state.metricsTimer = null; 
    }

    // Close connections
    if (state.control) { 
      try { 
        state.control.onmessage = null; 
        state.control.close(); 
      } catch(_) {} 
    }
    
    if (state.pc) { 
      try { 
        state.pc.ontrack = null; 
        state.pc.onconnectionstatechange = null; 
        state.pc.oniceconnectionstatechange = null; 
        state.pc.onicegatheringstatechange = null; 
        state.pc.close(); 
      } catch(_) {} 
    }
    
    if (state.localStream) { 
      try { 
        state.localStream.getTracks().forEach(t => t.stop()); 
      } catch(_) {} 
    }
    
    // Clear media elements and UI state
    try { 
      els.localVideo.srcObject = null;
      if (els.localWrapper) els.localWrapper.classList.remove('active');
      setLocalStatus('idle', 'Inactive');
    } catch(_) {}
    
    try { 
      if (els.remoteVideo.srcObject) {
        els.remoteVideo.pause();
        els.remoteVideo.srcObject = null; 
      }
      if (els.avatarWrapper) els.avatarWrapper.classList.remove('active');
      setAvatarStatus('idle', 'Inactive');
    } catch(_) {}
    
    // Reset metrics
    updateMetrics('--', '--', '--', '--');
    
    // Server cleanup
    try {
      const hdrs = {};
      if (state.authToken) hdrs['X-Auth-Token'] = state.authToken;
      await fetch('/webrtc/cleanup', {method: 'POST', headers: hdrs});
    } catch(_) {}
    
    // Reset state
    state.pc = null; 
    state.control = null; 
    state.localStream = null; 
    state.connected = false; 
    state.connecting = false;
    
    setButtonLoading(els.connect, false);
    els.connect.disabled = false; 
    els.disconnect.disabled = true; 
    setSystemStatus('idle', 'Disconnected');
    showToast('Connection terminated', 'warning');
  }

  // Performance metrics update
  function updatePerf(metrics) {
    try {
      const latency = metrics.latency_ms ? `${Math.round(metrics.latency_ms)}` : '--';
      const fps = metrics.fps ? `${Math.round(metrics.fps)}` : '--';
      const gpu = metrics.gpu_memory_used_mb ? `${Math.round(metrics.gpu_memory_used_mb)}MB` : '--';
      const quality = metrics.quality || (fps > 25 ? 'HD' : fps > 15 ? 'SD' : 'Low');
      
      updateMetrics(latency, fps, gpu, quality);
      
      // Update connection quality
      if (metrics.latency_ms) {
        if (metrics.latency_ms < 100) {
          setAvatarStatus('connected', 'Excellent');
        } else if (metrics.latency_ms < 250) {
          setAvatarStatus('connected', 'Good');
        } else {
          setAvatarStatus('warning', 'High Latency');
        }
      }
    } catch(e) {
      console.warn('updatePerf error', e);
    }
  }

  // Event Handlers
  function initializeEventListeners() {
    // Reference image upload
    if (els.ref) {
      els.ref.addEventListener('change', handleReference);
    }

    // Initialize pipeline
    if (els.init) {
      els.init.addEventListener('click', async () => {
        try {
          setSystemStatus('connecting', 'Initializing AI pipeline...');
          setButtonLoading(els.init, true);
          
          const r = await fetch('/initialize', {method: 'POST'});
          const j = await r.json().catch(() => ({}));
          
          if (r.ok && j && (j.status === 'success' || j.status === 'already_initialized')) {
            state.initialized = true;
            setSystemStatus('connected', j.message || 'Pipeline initialized');
            showToast('AI pipeline initialized successfully', 'success');
            
            els.init.classList.remove('btn-secondary');
            els.init.classList.add('btn-success');
            els.init.innerHTML = `
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                <polyline points="22,4 12,14.01 9,11.01"/>
              </svg>
              Pipeline Ready
            `;
          } else {
            setSystemStatus('error', 'Pipeline initialization failed');
            showToast('Failed to initialize AI pipeline', 'error');
            console.warn('initialize response', r.status, j);
          }
        } catch(e) {
          setSystemStatus('error', 'Pipeline initialization error');
          showToast('Error initializing AI pipeline', 'error');
        } finally {
          setButtonLoading(els.init, false);
        }
      });
    }

    // Debug
    if (els.debug) {
      els.debug.addEventListener('click', async () => {
        try {
          setSystemStatus('connecting', 'Fetching debug information...');
          setButtonLoading(els.debug, true);
          
          const r = await fetch('/debug/models');
          const j = await r.json();
          console.log('[DEBUG] /debug/models', j);
          
          const modelCount = Object.keys(j.files || {}).length;
          const existingModels = Object.values(j.files || {}).filter(f => f.exists).length;
          showToast(`Debug: ${existingModels}/${modelCount} models loaded`, 'info');
          
          const inswapper = j.files?.['inswapper_128_fp16.onnx'] || j.files?.['inswapper_128.onnx'];
          const codeformer = j.files?.['codeformer.pth'];
          
          const statusText = `Models: InSwapper=${inswapper?.exists?'✓':'✗'}, CodeFormer=${codeformer?.exists?'✓':'✗'}`;
          setSystemStatus(inswapper?.exists ? 'connected' : 'warning', statusText);
          
          if (!inswapper?.exists) {
            setSystemStatus('connecting', 'Downloading models...');
            try {
              const d = await fetch('/debug/download_models', {method: 'POST'});
              const dj = await d.json().catch(() => ({}));
              console.log('[DEBUG] /debug/download_models', dj);
              showToast('Model download initiated', 'info');
              
              setTimeout(async () => {
                const r2 = await fetch('/debug/models');
                const j2 = await r2.json();
                const inswapper2 = j2.files?.['inswapper_128_fp16.onnx'] || j2.files?.['inswapper_128.onnx'];
                const newStatus = `Models refreshed: InSwapper=${inswapper2?.exists?'✓':'✗'}`;
                setSystemStatus(inswapper2?.exists ? 'connected' : 'warning', newStatus);
              }, 2000);
            } catch(e) {
              showToast('Model download failed', 'error');
              console.warn('download_models failed', e);
            }
          }
        } catch(e) {
          setSystemStatus('error', 'Debug fetch failed');
          showToast('Failed to fetch debug information', 'error');
        } finally {
          setButtonLoading(els.debug, false);
        }
      });
    }

    // Connect/Disconnect
    if (els.connect) {
      els.connect.addEventListener('click', () => connect());
    }
    if (els.disconnect) {
      els.disconnect.addEventListener('click', () => disconnect());
    }
  }

  // Auto-initialization
  async function autoInitialize() {
    try {
      setSystemStatus('connecting', 'Auto-initializing system...');
      
      const r = await fetch('/initialize', {method: 'POST'});
      const j = await r.json().catch(() => ({}));
      
      if (r.ok && j && (j.status === 'success' || j.status === 'already_initialized')) {
        state.initialized = true;
        setSystemStatus('connected', j.message || 'System ready');
        
        if (els.init) {
          els.init.classList.remove('btn-secondary');
          els.init.classList.add('btn-success');
          els.init.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22,4 12,14.01 9,11.01"/>
            </svg>
            Pipeline Ready
          `;
        }
      } else {
        console.warn('auto-initialize response', r.status, j);
        setSystemStatus('idle', 'Click Initialize to start');
      }
    } catch(e) {
      setSystemStatus('idle', 'Click Initialize to start');
    }
  }

  // Initialize application
  function init() {
    log('Initializing Mirage AI Avatar Studio');
    initializeEventListeners();
    autoInitialize();
  }

  // Start the application when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();