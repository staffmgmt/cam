/* Production-focused WebRTC client (replaces dev UI). */
(function(){
  const state = {
    pc: null,
    control: null,
    localStream: null,
    metricsTimer: null,
    referenceImage: null,
    connected: false
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
    state.referenceImage = b64; // send after control channel open
  }

  async function connect(){
    if(state.connected) return;
    try {
      setStatus('Requesting media');
      els.connect.disabled = true;
      // Fetch short-lived auth token (if server requires)
      let authToken = null;
      try {
        const t = await fetch('/webrtc/token');
        if (t.ok) {
          const j = await t.json();
          authToken = j.token;
        }
      } catch(_){}
      state.localStream = await navigator.mediaDevices.getUserMedia({video:true,audio:true});
      els.localVideo.srcObject = state.localStream;
      setStatus('Creating peer');
      state.pc = new RTCPeerConnection({iceServers:[{urls:['stun:stun.l.google.com:19302']}]});
      state.pc.onconnectionstatechange = ()=>{ log('pc state', state.pc.connectionState); if(['failed','disconnected','closed'].includes(state.pc.connectionState)){ disconnect(); } };
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
          try { state.control.send(JSON.stringify({type:'set_reference', image_jpeg_base64: state.referenceImage})); } catch(e) {}
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
    if(state.control){ try { state.control.close(); }catch(_){} }
    if(state.pc){ try { state.pc.close(); }catch(_){} }
    if(state.localStream){ state.localStream.getTracks().forEach(t=>t.stop()); }
    state.pc=null; state.control=null; state.localStream=null; state.connected=false;
    els.connect.disabled=false; els.disconnect.disabled=true; setStatus('Idle');
  }

  els.ref.addEventListener('change', handleReference);
  els.connect.addEventListener('click', connect);
  els.disconnect.addEventListener('click', disconnect);
})();
