/* Production-focused WebRTC client (replaces dev UI). */
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
    cancelled: false
  };
  const params = new URLSearchParams(location.search);
  const FORCE_RELAY = params.get('relay') === '1';
  const els = {
    ref: document.getElementById('referenceInput'),
    init: document.getElementById('initBtn'),
    debug: document.getElementById('debugBtn'),
    connect: document.getElementById('connectBtn'),
    disconnect: document.getElementById('disconnectBtn'),
    localVideo: document.getElementById('localVideo'),
    remoteVideo: document.getElementById('remoteVideo'),
    status: document.getElementById('statusText'),
    perf: document.getElementById('perfBar')
  };
  function setStatus(txt){ els.status.textContent = txt; }
  function log(...a){ console.log('[PROD]', ...a); }

  async function handleReference(e){
    const file = e.target.files && e.target.files[0];
    if(!file) return;
    // Cache base64 for datachannel use
    const buf = await file.arrayBuffer();
    const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
    state.referenceImage = b64;
    // Also POST to HTTP endpoint so the pipeline has the reference even before WebRTC connects
    try {
      setStatus('Uploading reference...');
      const fd = new FormData();
      fd.append('file', new Blob([buf], {type: file.type||'application/octet-stream'}), file.name||'reference');
      const resp = await fetch('/set_reference', {method:'POST', body: fd});
      const jr = await resp.json().catch(()=>({}));
      if (resp.ok && jr && (jr.status==='success' || jr.status==='ok')){
        setStatus('Reference set');
      } else {
        setStatus('Reference upload failed');
        console.warn('set_reference response', resp.status, jr);
      }
    } catch(err){
      console.warn('set_reference error', err);
      setStatus('Reference upload error');
    }
    // If already connected, also send via data channel for immediate in-session update
    try {
      if (state.connected && state.control && state.control.readyState === 'open') {
        state.control.send(JSON.stringify({type:'set_reference', image_base64: state.referenceImage}));
      }
    } catch(_) {}
  }

  async function connect(){
    if(state.connected) return;
    if(state.connecting) return;
    try {
      setStatus('Requesting media');
      els.connect.disabled = true;
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
      setStatus('Creating peer');
      let iceCfg = {iceServers:[{urls:['stun:stun.l.google.com:19302']}]};
      try {
        const ic = await fetch('/webrtc/ice_config');
        if (ic.ok) { iceCfg = await ic.json(); }
      } catch(_){}
      if (FORCE_RELAY) { iceCfg.iceTransportPolicy = 'relay'; }
      log('ice config', iceCfg);
      state.pc = new RTCPeerConnection(iceCfg);
      state.pc.oniceconnectionstatechange = ()=>{
        log('ice state', state.pc.iceConnectionState);
        if(['failed','closed'].includes(state.pc.iceConnectionState)){
          if(!state.cancelled) disconnect();
        }
      };
      state.pc.onconnectionstatechange = ()=>{
        const st = state.pc.connectionState;
        log('pc state', st);
        // Allow transient 'disconnected' to recover; try ICE restart once
        if(st === 'disconnected'){
          try { state.pc.restartIce && state.pc.restartIce(); } catch(_){ }
        }
        if(['failed','closed'].includes(st)){
          disconnect();
        }
      };
      state.pc.ontrack = ev => {
        if(ev.streams && ev.streams[0]){
          els.remoteVideo.srcObject = ev.streams[0];
        } else if (ev.track) {
          const ms = new MediaStream([ev.track]);
          els.remoteVideo.srcObject = ms;
        }
      };
      state.control = state.pc.createDataChannel('control');
      state.control.onopen = ()=>{
        setStatus('Connected');
        state.connected = true;
        els.disconnect.disabled = false;
        if(state.referenceImage){
          try { state.control.send(JSON.stringify({type:'set_reference', image_base64: state.referenceImage})); } catch(e) {}
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
        if(r.status===401 || r.status===403){
          setStatus('Unauthorized (check API key/token)');
        } else if (r.status===404){
          setStatus('Offer endpoint not found (server not exposing /webrtc)');
        } else if (r.status===503){
          try { const txt = await r.text(); setStatus('Offer failed 503'); console.warn('503 body', txt); }
          catch(_){ setStatus('Offer failed 503'); }
        } else {
          setStatus('Offer failed '+r.status);
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
      els.perf.textContent = `Latency: ${lat} ms · FPS: ${fps} · GPU: ${gpu}`;
    } catch(_){}
  }

  async function disconnect(){
    state.cancelled = true;
    if(state.metricsTimer){ clearInterval(state.metricsTimer); state.metricsTimer=null; }
    if(state.control){ try { state.control.onmessage=null; state.control.close(); }catch(_){} }
    if(state.pc){ try { state.pc.ontrack=null; state.pc.onconnectionstatechange=null; state.pc.oniceconnectionstatechange=null; state.pc.onicegatheringstatechange=null; state.pc.close(); }catch(_){} }
    if(state.localStream){ try { state.localStream.getTracks().forEach(t=>t.stop()); } catch(_){} }
    // Clear media elements
    try { els.localVideo.srcObject = null; } catch(_){}
    try { els.remoteVideo.srcObject = null; } catch(_){}
    // Best-effort server cleanup
    try {
      const hdrs = {};
      if (state.authToken) hdrs['X-Auth-Token'] = state.authToken;
      await fetch('/webrtc/cleanup', {method:'POST', headers: hdrs});
    } catch(_){ }
    state.pc=null; state.control=null; state.localStream=null; state.connected=false; state.connecting=false;
    els.connect.disabled=false; els.disconnect.disabled=true; setStatus('Idle');
  }

  els.ref.addEventListener('change', handleReference);
  if (els.init) {
    els.init.addEventListener('click', async ()=>{
      try {
        setStatus('Initializing pipeline...');
        els.init.disabled = true;
        const r = await fetch('/initialize', {method:'POST'});
        const j = await r.json().catch(()=>({}));
        if (r.ok && j && (j.status==='success' || j.status==='already_initialized')){
          setStatus(j.message || 'Initialized');
        } else {
          setStatus('Init failed');
          console.warn('initialize response', r.status, j);
        }
      } catch(e){
        setStatus('Init error');
      } finally {
        els.init.disabled = false;
      }
    });
  }
  if (els.debug) {
    els.debug.addEventListener('click', async ()=>{
      try {
        setStatus('Fetching debug info...');
        const r = await fetch('/debug/models');
        const j = await r.json();
        console.log('[DEBUG] /debug/models', j);
        const app = j.files?.['appearance_feature_extractor.onnx'];
        const motion = j.files?.['motion_extractor.onnx'];
        let statusText = `ONNX: app=${app?.exists?'✔':'✖'}(${app?.size_bytes||0}), motion=${motion?.exists?'✔':'✖'}(${motion?.size_bytes||0})`;
        setStatus(statusText);
        // If missing, try to force a download now
        if (!app?.exists) {
          setStatus('Downloading models...');
          try {
            const d = await fetch('/debug/download_models', {method:'POST'});
            const dj = await d.json().catch(()=>({}));
            console.log('[DEBUG] /debug/download_models', dj);
            // Refresh presence
            const r2 = await fetch('/debug/models');
            const j2 = await r2.json();
            const app2 = j2.files?.['appearance_feature_extractor.onnx'];
            const motion2 = j2.files?.['motion_extractor.onnx'];
            statusText = `ONNX: app=${app2?.exists?'✔':'✖'}(${app2?.size_bytes||0}), motion=${motion2?.exists?'✔':'✖'}(${motion2?.size_bytes||0})`;
            setStatus(statusText);
          } catch(e){
            console.warn('download_models failed', e);
            setStatus('Download failed');
          }
        }
      } catch(e){
        setStatus('Debug fetch failed');
      }
    });
  }
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
