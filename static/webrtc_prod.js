/* Enterprise-grade WebRTC client with premium UI integration */
(function(){
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
  
  const params = new URLSearchParams(location.search);
  const FORCE_RELAY_URL = params.get('relay') === '1';
  
  const els = {
    // File upload
    ref: document.getElementById('referenceInput'),
    uploadButton: document.getElementById('uploadButton'),
    uploadText: document.getElementById('u  // Auto-initialize on page load (idempotent)
  (async ()=>{
    try {
      setSystemStatus('connecting', 'Auto-initializing pipeline...');
      const r = await fetch('/initialize', {method:'POST'});
      const j = await r.json().catch(()=>({}));
      if (r.ok && j && (j.status==='success' || j.status==='already_initialized')){
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
    } catch(e){
      setSystemStatus('idle', 'Click Initialize to start');
    }
  })();    // Control buttons
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

  // UI Helper Functions
  function setSystemStatus(status, text) {
    els.systemStatus.className = `status-indicator status-${status}`;
    els.statusText.textContent = text;
  }

  function setLocalStatus(status, text) {
    els.localStatus.className = `status-indicator status-${status}`;
    els.localStatusText.textContent = text;
  }

  function setAvatarStatus(status, text) {
    els.avatarStatus.className = `status-indicator status-${status}`;
    els.avatarStatusText.textContent = text;
  }

  function showToast(message, type = 'info') {
    els.toastContent.textContent = message;
    els.toast.className = `toast toast-${type} show`;
    setTimeout(() => {
      els.toast.classList.remove('show');
    }, 4000);
  }

  function updateMetrics(latency, fps, gpu, quality = 'HD') {
    els.latencyValue.textContent = latency || '--';
    els.fpsValue.textContent = fps || '--';
    els.gpuValue.textContent = gpu || '--';
    els.qualityValue.textContent = quality || '--';
  }

  function setButtonLoading(button, loading = true) {
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
  
  function log(...a) { 
    console.log('[MIRAGE]', ...a); 
  }
  // Verbose toggle (can be overridden by backend-provided global or URL param ?wv=1)
  const VERBOSE = (window.MIRAGE_WEBRTC_VERBOSE === true) || (new URLSearchParams(location.search).get('wv')==='1');
  const STATS_INTERVAL_MS = (window.MIRAGE_WEBRTC_STATS_INTERVAL_MS) || 5000;
  let statsTimer = null;

  function vlog(...args){ if(VERBOSE) console.log('[PROD][VERBOSE]', ...args); }

  function attachPcDiagnostics(pc){
    if(!pc) return;
    const evMap = ['signalingstatechange','iceconnectionstatechange','icegatheringstatechange','connectionstatechange','negotiationneeded'];
    evMap.forEach(ev => {
      pc.addEventListener(ev, ()=>{ vlog('pc event', ev, diagSnapshot()); });
    });
    pc.addEventListener('track', ev => { vlog('pc track event', ev.track && ev.track.kind, ev.streams && ev.streams.length); });
    pc.onicecandidate = (e)=>{ if(e.candidate){ vlog('ice candidate', e.candidate.type, e.candidate.protocol, e.candidate.address, e.candidate.relatedAddress||null); } else { vlog('ice candidate gathering complete'); } };
  }

  function diagSnapshot(){
    if(!state.pc) return {};
    return {
      signaling: state.pc.signalingState,
      iceConnection: state.pc.iceConnectionState,
      iceGathering: state.pc.iceGatheringState,
      connection: state.pc.connectionState,
      localTracks: state.pc.getSenders().map(s=>({kind:s.track && s.track.kind, ready:s.track && s.track.readyState})),
      remoteTracks: state.pc.getReceivers().map(r=>({kind:r.track && r.track.kind, ready:r.track && r.track.readyState}))
    };
  }

  async function collectStats(){
    if(!state.pc) return;
    try {
      const stats = await state.pc.getStats();
      const summary = { ts: Date.now(), outbound: {}, inbound: {}, candidatePairs: [] };
      stats.forEach(report => {
        if(report.type === 'outbound-rtp' && !report.isRemote){
          summary.outbound[report.kind||report.mediaType||'unknown'] = {
            bitrateKbps: report.bytesSent && report.timestamp ? undefined : undefined,
            frames: report.framesEncoded,
            q: report.qualityLimitationReason,
            packetsSent: report.packetsSent
          };
        } else if(report.type === 'inbound-rtp' && !report.isRemote){
          summary.inbound[report.kind||report.mediaType||'unknown'] = {
            jitter: report.jitter,
            packetsLost: report.packetsLost,
            frames: report.framesDecoded,
            packetsReceived: report.packetsReceived
          };
        } else if(report.type === 'candidate-pair' && report.state === 'succeeded'){
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
    } catch(e){ vlog('stats error', e); }
  }

  async function handleReference(e){
    const file = e.target.files && e.target.files[0];
    if(!file) {
      els.uploadButton.classList.remove('has-file');
      els.uploadText.textContent = 'Choose Reference Image';
      return;
    }
    
    // Update UI immediately
    els.uploadButton.classList.add('has-file');
    els.uploadText.textContent = `✓ ${file.name}`;
    showToast(`Reference image selected: ${file.name}`, 'success');
    
    // Cache base64 for datachannel use
    const buf = await file.arrayBuffer();
    const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
    state.referenceImage = b64;
    
    // Also POST to HTTP endpoint so the pipeline has the reference even before WebRTC connects
    try {
      setSystemStatus('connecting', 'Uploading reference image...');
      const fd = new FormData();
      fd.append('file', new Blob([buf], {type: file.type||'application/octet-stream'}), file.name||'reference');
      const resp = await fetch('/set_reference', {method:'POST', body: fd});
      const jr = await resp.json().catch(()=>({}));
      if (resp.ok && jr && (jr.status==='success' || jr.status==='ok')){
        setSystemStatus('connected', 'Reference image uploaded successfully');
        showToast('Reference image set successfully', 'success');
      } else {
        setSystemStatus('error', 'Reference upload failed');
        showToast('Failed to upload reference image', 'error');
        console.warn('set_reference response', resp.status, jr);
      }
    } catch(err){
      console.warn('set_reference error', err);
      setSystemStatus('error', 'Reference upload error');
      showToast('Error uploading reference image', 'error');
    }
    
    // If already connected, also send via data channel for immediate in-session update
    try {
      if (state.connected && state.control && state.control.readyState === 'open') {
        state.control.send(JSON.stringify({type:'set_reference', image_base64: state.referenceImage}));
      }
    } catch(_) {}
  }

  async function connect(options){
    const overrideRelay = options && options.forceRelay === true;
    if(state.connected) return;
    if(state.connecting) return;
    try {
      setSystemStatus('connecting', 'Requesting camera access...');
      setLocalStatus('connecting', 'Initializing');
      setButtonLoading(els.connect, true);
      els.disconnect.disabled = false; // allow cancel during negotiation
      state.cancelled = false; state.connecting = true;
      // Quick ping to verify router is mounted
      try {
        const ping = await fetch('/webrtc/ping');
        if(ping.ok){ const j = await ping.json(); log('webrtc ping', j); }
      } catch(_){ }
      // Fetch short-lived auth token (if server requires)
      let authToken = state.authToken;
      try {
        const t = await fetch('/webrtc/token');
        if (t.ok) {
          const j = await t.json();
          authToken = j.token; state.authToken = authToken;
        } else if (t.status === 404) {
          // Likely router not mounted; keep null and let server decide
          console.warn('Token endpoint 404 - proceeding without token');
        }
      } catch(_){}
      state.localStream = await navigator.mediaDevices.getUserMedia({video:true,audio:true});
      els.localVideo.srcObject = state.localStream;
      els.localWrapper.classList.add('active');
      setLocalStatus('connected', 'Camera Active');
      try { els.localVideo.play && els.localVideo.play(); } catch(_) {}
      setSystemStatus('connecting', 'Establishing connection...');
      let iceCfg = {iceServers:[{urls:['stun:stun.l.google.com:19302']}]};
      try {
        const ic = await fetch('/webrtc/ice_config');
        if (ic.ok) { iceCfg = await ic.json(); }
      } catch(_){ }
      state._lastIceCfg = iceCfg; // keep for potential retry decision
      if (overrideRelay || FORCE_RELAY_URL || iceCfg.forceRelay === true) { iceCfg.iceTransportPolicy = 'relay'; }
      log('ice config', iceCfg);
  state.pc = new RTCPeerConnection(iceCfg);
  attachPcDiagnostics(state.pc);
      state._usedRelay = !!iceCfg.iceTransportPolicy && iceCfg.iceTransportPolicy === 'relay';
      state._relayFallbackTried = !!overrideRelay || !!FORCE_RELAY_URL; // if already forcing relay, don't fallback again
      state.pc.oniceconnectionstatechange = ()=>{
        log('ice state', state.pc.iceConnectionState);
        if(['failed','closed'].includes(state.pc.iceConnectionState)){
          if(!state.cancelled) disconnect();
        }
        if(state.pc.iceConnectionState === 'disconnected'){
          vlog('ICE disconnected snapshot', diagSnapshot());
        }
      };
      state.pc.onconnectionstatechange = ()=>{
        const st = state.pc.connectionState;
        log('pc state', st);
        if(st === 'connected' && !statsTimer){
          statsTimer = setInterval(collectStats, STATS_INTERVAL_MS);
        }
        if(st === 'disconnected'){
          vlog('PC disconnected snapshot', diagSnapshot());
          try { state.pc.restartIce && state.pc.restartIce(); vlog('Attempted ICE restart'); } catch(_){ }
        }
        if(['failed','closed'].includes(st)){
          if(statsTimer){ clearInterval(statsTimer); statsTimer=null; }
          const hasTurn = state._lastIceCfg && (state._lastIceCfg.turnCount||0) > 0;
          const tryRelay = hasTurn && !state._usedRelay && !state._relayFallbackTried;
          const snapshot = diagSnapshot();
          vlog('Final failure snapshot', snapshot, {hasTurn, usedRelay: state._usedRelay, relayTried: state._relayFallbackTried});
          disconnect().then(()=>{
            if (tryRelay) {
              state._relayFallbackTried = true;
              log('retrying with relay-only');
              setStatus('Retrying with TURN relay');
              connect({forceRelay:true});
            } else if (!hasTurn && !state._usedRelay && !state._relayFallbackTried) {
              log('skipping relay-only retry: no TURN servers available');
              setStatus('No TURN servers; cannot retry relay-only');
            }
          });
        }
      };
      state.pc.ontrack = ev => {
        try {
          const tr = ev.track;
          log('ontrack', tr && tr.kind, tr && tr.readyState, ev.streams && ev.streams.length);
          if (tr && tr.kind === 'video') {
            setSystemStatus('connected', 'Avatar stream received');
            setAvatarStatus('connected', 'Active');
            let stream;
            if(ev.streams && ev.streams[0]){
              stream = ev.streams[0];
              log('Using provided stream:', stream.id, 'tracks:', stream.getTracks().length);
            } else {
              stream = new MediaStream([ev.track]);
              log('Created new MediaStream:', stream.id);
            }
            
            // Force new srcObject every time to avoid conflicts
            log('Setting srcObject on video element, current value:', els.remoteVideo.srcObject);
            els.remoteVideo.srcObject = null;
            els.remoteVideo.srcObject = stream;
            els.avatarWrapper.classList.add('active');
            log('srcObject set, waiting for loadeddata...');
            
            // Add more event listeners for debugging
            els.remoteVideo.onloadstart = () => log('video: loadstart');
            els.remoteVideo.onloadedmetadata = () => log('video: loadedmetadata', els.remoteVideo.videoWidth, 'x', els.remoteVideo.videoHeight);
            els.remoteVideo.onloadeddata = () => {
              log('video: loadeddata, attempting play()');
              els.remoteVideo.play().catch(e => {
                log('play error', e.name, e.message);
                // Try playing again after a short delay
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
            els.remoteVideo.onstalled = () => log('video: stalled');
            
            // Monitor video track state changes
            tr.onended = () => {
              log('video track ended');
              setAvatarStatus('idle', 'Disconnected');
              els.avatarWrapper.classList.remove('active');
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
        } catch(e){ 
          log('ontrack error', e);
          setAvatarStatus('error', 'Connection Error');
        }
      };
      state.control = state.pc.createDataChannel('control');
      state.control.onopen = ()=>{
        setSystemStatus('connected', 'WebRTC connection established');
        state.connected = true;
        state.connecting = false;
        setButtonLoading(els.connect, false);
        els.connect.disabled = true;
        els.disconnect.disabled = false;
        showToast('WebRTC connection established', 'success');
        
        if(state.referenceImage){
          try { 
            state.control.send(JSON.stringify({type:'set_reference', image_base64: state.referenceImage}));
            showToast('Reference image sent to avatar', 'success');
          } catch(e) {
            showToast('Failed to send reference image', 'error');
          }
        }
        // Metrics polling
        state.metricsTimer = setInterval(()=>{
          try { state.control.send(JSON.stringify({type:'metrics_request'})); }catch(_){ }
        }, 4000);
      };
      state.control.onmessage = (e)=>{
        try {
          const data = JSON.parse(e.data);
          if(data.type==='metrics' && data.payload){
            updatePerf(data.payload);
          } else if (data.type === 'reference_ack'){
            setStatus('Reference set');
          } else if (data.type === 'error' && data.message){
            setStatus('Error: '+data.message);
          }
        } catch(_){ }
      };
      state.localStream.getTracks().forEach(t=> state.pc.addTrack(t, state.localStream));
      const offer = await state.pc.createOffer({offerToReceiveAudio:true,offerToReceiveVideo:true});
      await state.pc.setLocalDescription(offer);
      // Wait for ICE gathering to complete (non-trickle) to avoid connectivity issues
      setStatus('Gathering ICE');
      await new Promise((resolve)=>{
        if(state.pc.iceGatheringState === 'complete') return resolve();
        const to = setTimeout(()=>{ resolve(); }, 7000);
        state.pc.onicegatheringstatechange = ()=>{
          if(state.pc.iceGatheringState === 'complete'){
            clearTimeout(to); resolve();
          }
        };
      });
      setStatus('Negotiating');
  const headers = {'Content-Type':'application/json'};
  if (authToken) headers['X-Auth-Token'] = authToken;
      // Use the possibly-updated localDescription (with ICE candidates)
      const ld = state.pc.localDescription;
      const r = await fetch('/webrtc/offer',{method:'POST', headers, body: JSON.stringify({sdp:ld.sdp, type:ld.type})});
      if(!r.ok){
        let bodyText = '';
        try { bodyText = await r.text(); } catch(_){ }
        if(r.status===401 || r.status===403){
          setStatus('Unauthorized (check API key/token)');
        } else if (r.status===404){
          setStatus('Offer endpoint not found (server not exposing /webrtc)');
        } else if (r.status===503){
          try { const txt = bodyText || ''; setStatus('Offer failed 503'); console.warn('503 body', txt); }
          catch(_){ setStatus('Offer failed 503'); }
        } else {
          setStatus('Offer failed '+r.status);
          console.warn('offer error body', bodyText);
        }
        els.connect.disabled=false; els.disconnect.disabled=true; state.connecting=false; return;
      }
      const answer = await r.json();
      await state.pc.setRemoteDescription(answer);
      setStatus('Finalizing');
    } catch(e){
      log('connect error', e);
      setStatus('Error');
      els.connect.disabled = false; els.disconnect.disabled=true;
    }
    finally {
      state.connecting = false;
    }
  }

  function updatePerf(p){
    try {
      const fps = (p.video_fps || 0).toFixed(1);
      const lat = Math.round(p.avg_video_latency_ms || 0);
      const gpu = (p.gpu_memory_used !== undefined) ? (p.gpu_memory_used.toFixed(2)+'GB') : '--';
      const method = p.last_method ? String(p.last_method) : '--';
      els.perf.textContent = `Latency: ${lat} ms · FPS: ${fps} · GPU: ${gpu} · Mode: ${method}`;
    } catch(_){}
  }

  async function disconnect(){
    state.cancelled = true;
    if(statsTimer){ clearInterval(statsTimer); statsTimer=null; }
    if(state.metricsTimer){ clearInterval(state.metricsTimer); state.metricsTimer=null; }
    if(state.control){ try { state.control.onmessage=null; state.control.close(); }catch(_){} }
    if(state.pc){ try { state.pc.ontrack=null; state.pc.onconnectionstatechange=null; state.pc.oniceconnectionstatechange=null; state.pc.onicegatheringstatechange=null; state.pc.close(); }catch(_){} }
    if(state.localStream){ try { state.localStream.getTracks().forEach(t=>t.stop()); } catch(_){} }
    
    // Clear media elements and UI state
    try { 
      els.localVideo.srcObject = null;
      els.localWrapper.classList.remove('active');
      setLocalStatus('idle', 'Inactive');
    } catch(_){}
    try { 
      if (els.remoteVideo.srcObject) {
        els.remoteVideo.pause();
        els.remoteVideo.srcObject = null; 
      }
      els.avatarWrapper.classList.remove('active');
      setAvatarStatus('idle', 'Inactive');
    } catch(_){}
    
    // Reset metrics
    updateMetrics('--', '--', '--', '--');
    
    // Best-effort server cleanup
    try {
      const hdrs = {};
      if (state.authToken) hdrs['X-Auth-Token'] = state.authToken;
      await fetch('/webrtc/cleanup', {method:'POST', headers: hdrs});
    } catch(_){ }
    
    state.pc=null; state.control=null; state.localStream=null; state.connected=false; state.connecting=false;
    setButtonLoading(els.connect, false);
    els.connect.disabled=false; els.disconnect.disabled=true; 
    setSystemStatus('idle', 'Disconnected');
    showToast('Connection terminated', 'warning');
  }

  els.ref.addEventListener('change', handleReference);
  if (els.init) {
    els.init.addEventListener('click', async ()=>{
      try {
        setSystemStatus('connecting', 'Initializing AI pipeline...');
        setButtonLoading(els.init, true);
        const r = await fetch('/initialize', {method:'POST'});
        const j = await r.json().catch(()=>({}));
        if (r.ok && j && (j.status==='success' || j.status==='already_initialized')){
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
      } catch(e){
        setSystemStatus('error', 'Pipeline initialization error');
        showToast('Error initializing AI pipeline', 'error');
      } finally {
        setButtonLoading(els.init, false);
      }
    });
  }
  if (els.debug) {
    els.debug.addEventListener('click', async ()=>{
      try {
        setSystemStatus('connecting', 'Fetching debug information...');
        setButtonLoading(els.debug, true);
        const r = await fetch('/debug/models');
        const j = await r.json();
        console.log('[DEBUG] /debug/models', j);
        
        // Show debug info in toast
        const modelCount = Object.keys(j.files || {}).length;
        const existingModels = Object.values(j.files || {}).filter(f => f.exists).length;
        showToast(`Debug: ${existingModels}/${modelCount} models loaded`, 'info');
        
        const app = j.files?.['appearance_feature_extractor.onnx'];
        const motion = j.files?.['motion_extractor.onnx'];
        const inswapper = j.files?.['inswapper_128_fp16.onnx'] || j.files?.['inswapper_128.onnx'];
        
        let statusText = `Models: InSwapper=${inswapper?.exists?'✓':'✗'}, App=${app?.exists?'✓':'✗'}, Motion=${motion?.exists?'✓':'✗'}`;
        setSystemStatus(inswapper?.exists ? 'connected' : 'warning', statusText);
        
        // If missing critical models, try to download
        if (!inswapper?.exists) {
          setSystemStatus('connecting', 'Downloading models...');
          try {
            const d = await fetch('/debug/download_models', {method:'POST'});
            const dj = await d.json().catch(()=>({}));
            console.log('[DEBUG] /debug/download_models', dj);
            showToast('Model download initiated', 'info');
            
            // Refresh model status
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
      } catch(e){
        setSystemStatus('error', 'Debug fetch failed');
        showToast('Failed to fetch debug information', 'error');
      } finally {
        setButtonLoading(els.debug, false);
      }
    });
  }

  // Update performance metrics display
  function updatePerf(metrics) {
    try {
      const latency = metrics.latency_ms ? `${Math.round(metrics.latency_ms)}` : '--';
      const fps = metrics.fps ? `${Math.round(metrics.fps)}` : '--';
      const gpu = metrics.gpu_memory_used_mb ? `${Math.round(metrics.gpu_memory_used_mb)}MB` : '--';
      const quality = metrics.quality || (fps > 25 ? 'HD' : fps > 15 ? 'SD' : 'Low');
      
      updateMetrics(latency, fps, gpu, quality);
      
      // Update connection quality indicator
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

  // Event listeners
  els.connect.addEventListener('click', connect);
  els.disconnect.addEventListener('click', disconnect);

  // Auto-initialize on page load (idempotent). Helps when HTML cache hides the button.
  (async ()=>{
    try {
      setStatus('Initializing pipeline...');
      const r = await fetch('/initialize', {method:'POST'});
      const j = await r.json().catch(()=>({}));
      if (r.ok && j && (j.status==='success' || j.status==='already_initialized')){
        setStatus(j.message || 'Initialized');
      } else {
        // Don’t spam status if it fails; user can still proceed to Connect
        console.warn('auto-initialize response', r.status, j);
        setStatus('Idle');
      }
    } catch(e){
      // Silent fail; keep UI responsive
      setStatus('Idle');
    }
  })();
})();
