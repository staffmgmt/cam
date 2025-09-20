/* Production-focused WebRTC client (replaces dev UI). */
(function(){
  const state = {
    pc: null,
    control: null,
    localStream: null,
    metricsTimer: null,
    referenceImage: null,
    connected: false,
    authToken: null
  };
  const els = {
    ref: document.getElementById('referenceInput'),
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
    const buf = await file.arrayBuffer();
    const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
    state.referenceImage = b64; // cache; send now if connected
    try {
      if (state.connected && state.control && state.control.readyState === 'open') {
        state.control.send(JSON.stringify({type:'set_reference', image_base64: state.referenceImage}));
      }
    } catch(_) {}
  }

  async function connect(){
    if(state.connected) return;
    try {
      setStatus('Requesting media');
      els.connect.disabled = true;
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
      state.pc = new RTCPeerConnection({iceServers:[{urls:['stun:stun.l.google.com:19302']}]});
      state.pc.onconnectionstatechange = ()=>{
        const st = state.pc.connectionState;
        log('pc state', st);
        // Allow transient 'disconnected' to recover; only hard close on failed/closed
        if(['failed','closed'].includes(st)){
          disconnect();
        }
      };
      state.pc.ontrack = ev => {
        if(ev.streams && ev.streams[0]){
          els.remoteVideo.srcObject = ev.streams[0];
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
        try { const data = JSON.parse(e.data); if(data.type==='metrics' && data.payload){ updatePerf(data.payload); } } catch(_){ }
      };
      state.localStream.getTracks().forEach(t=> state.pc.addTrack(t, state.localStream));
      const offer = await state.pc.createOffer();
      await state.pc.setLocalDescription(offer);
      setStatus('Negotiating');
  const headers = {'Content-Type':'application/json'};
  if (authToken) headers['X-Auth-Token'] = authToken;
      const r = await fetch('/webrtc/offer',{method:'POST', headers, body: JSON.stringify({sdp:offer.sdp, type:offer.type})});
      if(!r.ok){
        if(r.status===401 || r.status===403){
          setStatus('Unauthorized (check API key/token)');
        } else if (r.status===404){
          setStatus('Offer endpoint not found (server not exposing /webrtc)');
        } else if (r.status===503){
          try { const txt = await r.text(); setStatus('Offer failed 503'); console.warn('503 body', txt); } catch(_){ setStatus('Offer failed 503'); }
        } else {
          setStatus('Offer failed '+r.status);
        }
        els.connect.disabled=false; return;
      }
      const answer = await r.json();
      await state.pc.setRemoteDescription(answer);
      setStatus('Finalizing');
    } catch(e){
      log('connect error', e);
      setStatus('Error');
      els.connect.disabled = false;
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
    if(state.metricsTimer){ clearInterval(state.metricsTimer); state.metricsTimer=null; }
    if(state.control){ try { state.control.onmessage=null; state.control.close(); }catch(_){} }
    if(state.pc){ try { state.pc.ontrack=null; state.pc.onconnectionstatechange=null; state.pc.close(); }catch(_){} }
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
    state.pc=null; state.control=null; state.localStream=null; state.connected=false;
    els.connect.disabled=false; els.disconnect.disabled=true; setStatus('Idle');
  }

  els.ref.addEventListener('change', handleReference);
  els.connect.addEventListener('click', connect);
  els.disconnect.addEventListener('click', disconnect);
})();
