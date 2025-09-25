"""WebRTC integration using aiortc for low-latency bi-directional media.

This module exposes:
- POST /webrtc/offer : Accepts an SDP offer from browser, returns SDP answer.
- GET  /webrtc/ice   : (Optional) polling ICE candidates (simplified; trickle or full offer/answer)

Media Flow (Phase 1):
Browser camera/mic -> WebRTC -> aiortc PeerConnection ->
  Video track -> frame hook -> pipeline.process_video_frame -> return video track to client
  Audio track -> chunk hook  -> pipeline.process_audio_chunk -> return audio track to client

Control/Data channel: "control" used for lightweight JSON messages:
  {"type":"metrics_request"} -> server replies {"type":"metrics","payload":...}
  {"type":"set_reference","image_jpeg_base64":...}

Fallback: If aiortc not supported in environment or import fails, endpoint returns 503.

Security: (basic) Optional shared secret via X-API-Key header (env MIRAGE_API_KEY).

NOTE: This is a minimal, production-ready skeleton focusing on structure, error handling,
resource cleanup and integration points. Actual model inference now resides exclusively in swap_pipeline.FaceSwapPipeline.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass
import hashlib
import hmac
import secrets as pysecrets
import base64 as pybase64
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Header

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer, VideoStreamTrack
    from aiortc.contrib.media import MediaBlackhole
    import av  # noqa: F401 (required by aiortc for codecs)
    AIORTC_AVAILABLE = True
except Exception as e:  # pragma: no cover
    AIORTC_IMPORT_ERROR = str(e)
    AIORTC_AVAILABLE = False

import numpy as np
import cv2
try:
    from webrtc_connection_monitoring import add_connection_monitoring  # optional diagnostics
except Exception:
    add_connection_monitoring = None

logger = logging.getLogger(__name__)

# Lazy pipeline getter with safe pass-through fallback to ensure router mounts
_pipeline_singleton = None

class _PassThroughPipeline:
    def __init__(self):
        # Mark as loaded so initialization is skipped
        self.loaded = True

    def initialize(self):
        return True

    def set_source_image(self, img):
        return False

    def process_video_frame(self, img, frame_idx=None):
        # Pass-through video
        return img

    def process_audio_chunk(self, pcm):
        # Pass-through audio (bytes or np array)
        return pcm

    def get_performance_stats(self):
        return {}

def get_pipeline():  # type: ignore
    global _pipeline_singleton
    if _pipeline_singleton is not None:
        return _pipeline_singleton
    try:
        from swap_pipeline import get_pipeline as _real_get_pipeline
        _pipeline_singleton = _real_get_pipeline()
    except Exception as e:
        logger.error(f"swap_pipeline unavailable, using pass-through: {e}")
        _pipeline_singleton = _PassThroughPipeline()
    return _pipeline_singleton

# Router mounted by app with prefix "/webrtc"; declare here without its own prefix
router = APIRouter(tags=["webrtc"])

API_KEY = os.getenv("MIRAGE_API_KEY")
REQUIRE_API_KEY = os.getenv("MIRAGE_REQUIRE_API_KEY", "0").strip().lower() in {"1","true","yes","on"}
TOKEN_TTL_SECONDS = int(os.getenv("MIRAGE_TOKEN_TTL", "300"))  # 5 minutes default
STUN_URLS = os.getenv(
    "MIRAGE_STUN_URLS",
    "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302,stun:stun2.l.google.com:19302,stun:stun3.l.google.com:19302,stun:stun4.l.google.com:19302,stun:stun.stunprotocol.org:3478"
)
TURN_URL = os.getenv("MIRAGE_TURN_URL")
TURN_USER = os.getenv("MIRAGE_TURN_USER")
TURN_PASS = os.getenv("MIRAGE_TURN_PASS")
METERED_API_KEY = os.getenv("MIRAGE_METERED_API_KEY")
TURN_TLS_ONLY = os.getenv("MIRAGE_TURN_TLS_ONLY", "1").strip().lower() in {"1","true","yes","on"}
PREFER_H264 = os.getenv("MIRAGE_PREFER_H264", "0").strip().lower() in {"1","true","yes","on"}
FORCE_RELAY = os.getenv("MIRAGE_FORCE_RELAY", "0").strip().lower() in {"1","true","yes","on"}


def _b64u(data: bytes) -> str:
    return pybase64.urlsafe_b64encode(data).decode('ascii').rstrip('=')


def _b64u_decode(data: str) -> bytes:
    pad = '=' * (-len(data) % 4)
    return pybase64.urlsafe_b64decode(data + pad)


async def _test_turn_connectivity(ice_servers):
    """Test TURN server connectivity to diagnose NAT traversal issues."""
    import socket
    from urllib.parse import urlparse
    
    turn_tests = []
    for server in ice_servers:
        urls = server.urls if isinstance(server.urls, list) else [server.urls]
        for url in urls:
            if url.startswith('turn'):
                try:
                    parsed = urlparse(url.replace('turn:', 'http://').replace('turns:', 'https://'))
                    host = parsed.hostname
                    port = parsed.port or (443 if url.startswith('turns:') else 3478)
                    
                    # Basic TCP connectivity test
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result == 0:
                        logger.info(f"TURN server reachable: {host}:{port}")
                        turn_tests.append(True)
                    else:
                        logger.warning(f"TURN server unreachable: {host}:{port} (error {result})")
                        turn_tests.append(False)
                        
                except Exception as e:
                    logger.warning(f"TURN connectivity test failed for {url}: {e}")
                    turn_tests.append(False)
    
    if turn_tests:
        reachable = sum(turn_tests)
        total = len(turn_tests)
        if reachable == 0:
            logger.error(f"All {total} TURN servers are unreachable - NAT traversal will likely fail")
        elif reachable < total:
            logger.warning(f"Only {reachable}/{total} TURN servers are reachable")
        else:
            logger.info(f"All {total} TURN servers are reachable")


def _mint_token() -> str:
    """Stateless signed token: base64url(ts:nonce:mac)."""
    ts = str(int(time.time()))
    nonce = _b64u(pysecrets.token_bytes(12))
    msg = f"{ts}:{nonce}".encode('utf-8')
    mac = hmac.new(API_KEY.encode('utf-8'), msg, hashlib.sha256).digest()
    return _b64u(msg) + '.' + _b64u(mac)


@router.get("/ping")
async def webrtc_ping():
    """Lightweight check indicating the WebRTC router is mounted.
    Returns aiortc availability and import error (if any)."""
    return {
        "router": True,
        "aiortc_available": AIORTC_AVAILABLE,
        "aiortc_error": None if AIORTC_AVAILABLE else AIORTC_IMPORT_ERROR,
        "turn_configured": bool(TURN_URL and TURN_USER and TURN_PASS),
        "metered_configured": bool(METERED_API_KEY)
    }

# Minimal root to confirm router is mounted
@router.get("")
async def webrtc_root():
    return {"webrtc": True, "aiortc_available": AIORTC_AVAILABLE}

@router.get("/ice_config")
async def webrtc_ice_config():
    """Expose ICE server configuration so the client can include TURN if configured.
    Returns a structure compatible with RTCPeerConnection's configuration.
    """
    try:
        cfg = _ice_configuration()
        servers = []
        for s in cfg.iceServers:
            entry = {"urls": s.urls}
            if getattr(s, 'username', None):
                entry["username"] = s.username
            if getattr(s, 'credential', None):
                entry["credential"] = s.credential
            servers.append(entry)
        payload = {"iceServers": servers}
        if FORCE_RELAY:
            payload["forceRelay"] = True
        return payload
    except Exception as e:
        # Fallback to public STUN
        return {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}], "error": str(e)}


@router.get("/debug_state")
async def webrtc_debug_state():
    """Return simplified current peer connection debug info."""
    try:
        st = _peer_state
        if st is None:
            return {
                "active": False,
                "last_connection_state": None,
                "last_ice_state": None,
            }
        pc = st.pc
        senders = getattr(pc, 'getSenders', lambda: [])()
        def _sender_info(s):
            try:
                tr = s.track
                info = {
                    "kind": getattr(tr, 'kind', None) if tr else None,
                    "readyState": getattr(tr, 'readyState', None) if tr else None,
                    "exists": tr is not None
                }
                # Include frame emission counter if available
                try:
                    if tr and getattr(tr, 'kind', None) == 'video' and hasattr(tr, '_debug_emitted'):
                        info["frames_emitted"] = getattr(tr, '_debug_emitted')
                        # Outbound diagnostics (VideoStreamTrack subclass fields)
                        diag_pairs = [
                            ('_placeholder_active','placeholder_active'),
                            ('_placeholder_sent','placeholder_frames'),
                            ('_relay_failures','relay_failures'),
                            ('_relay_last_error','relay_last_error'),
                            ('_relay_last_error_ts','relay_last_error_ts'),
                            ('_first_relay_ts','first_relay_ts'),
                            ('_placeholder_initial_ts','placeholder_initial_ts'),
                            ('_placeholder_deactivated_ts','placeholder_deactivated_ts'),
                            ('_raw_frames_in','raw_frames_in'),
                        ]
                        for attr, key in diag_pairs:
                            if hasattr(tr, attr):
                                info[key] = getattr(tr, attr)
                except Exception:
                    pass
                return info
            except Exception:
                return {"kind": None, "exists": False}
        return {
            "active": True,
            "connectionState": getattr(pc, 'connectionState', None),
            "iceConnectionState": getattr(pc, 'iceConnectionState', None),
            "senders": [_sender_info(s) for s in senders],
            "control_channel_ready": st.control_channel_ready,
            "last_connection_state": st.last_connection_state,
            "last_ice_state": st.last_ice_state,
        }
    except Exception as e:
        return {"active": False, "error": str(e)}


def _verify_token(token: str) -> bool:
    try:
        parts = token.split('.')
        if len(parts) != 2:
            return False
        msg_b64, mac_b64 = parts
        msg = _b64u_decode(msg_b64)
        mac = _b64u_decode(mac_b64)
        ts_str, nonce = msg.decode('utf-8').split(':', 1)
        ts = int(ts_str)
        if time.time() - ts > TOKEN_TTL_SECONDS:
            return False
        expected = hmac.new(API_KEY.encode('utf-8'), msg, hashlib.sha256).digest()
        return hmac.compare_digest(expected, mac)
    except Exception:
        return False


def _check_api_key(header_val: Optional[str], token_val: Optional[str] = None):
    # If no API key configured, allow
    if not API_KEY:
        return
    # If enforcement disabled, allow
    if not REQUIRE_API_KEY:
        return
    # Accept raw key or signed token
    if header_val and header_val == API_KEY:
        return
    if token_val and _verify_token(token_val):
        return
    raise HTTPException(status_code=401, detail="Unauthorized")


def _ice_configuration() -> RTCConfiguration:
    servers = []
    turn_server_count = 0
    stun_server_count = 0
    
    # STUN servers (comma-separated)
    for url in [u.strip() for u in STUN_URLS.split(',') if u.strip()]:
        servers.append(RTCIceServer(urls=[url]))
        stun_server_count += 1
        
    # Optional TURN (static)
    if TURN_URL and TURN_USER and TURN_PASS:
        for tur in [u.strip() for u in str(TURN_URL).split(',') if u.strip()]:
            servers.append(RTCIceServer(urls=[tur], username=TURN_USER, credential=TURN_PASS))
            turn_server_count += 1
            
    # Optional Metered.ca ephemeral TURN using API key
    if METERED_API_KEY:
        try:
            from urllib.request import urlopen
            from urllib.parse import urlencode
            import json as _json
            # Global endpoint that returns iceServers list
            url = f"https://global.relay.metered.ca/turn?{urlencode({'apiKey': METERED_API_KEY})}"
            with urlopen(url, timeout=5) as resp:  # nosec - fixed provider URL
                data = _json.loads(resp.read().decode('utf-8'))
            ice_list = data.get('iceServers') or []
            for s in ice_list:
                urls = s.get('urls')
                if not urls:
                    continue
                if isinstance(urls, str):
                    urls = [urls]
                username = s.get('username')
                credential = s.get('credential')
                servers.append(RTCIceServer(urls=urls, username=username, credential=credential))
                turn_server_count += len([u for u in urls if u.startswith('turn')])
        except Exception as e:
            logger.warning(f"Metered ICE fetch failed: {e} - This may cause connection failures in restricted networks")
            
    # Optionally filter to TLS/TCP-only TURN to succeed behind strict firewalls
    def _is_tls_tcp(url: str) -> bool:
        u = url.lower()
        return u.startswith('turns:') or 'transport=tcp' in u or ':443' in u

    if TURN_TLS_ONLY:
        filtered = []
        original_turn_count = turn_server_count
        turn_server_count = 0
        for s in servers:
            urls = s.urls if isinstance(s.urls, list) else [s.urls]
            keep_urls = [u for u in urls if _is_tls_tcp(u)]
            if keep_urls:
                filtered.append(RTCIceServer(urls=keep_urls, username=getattr(s,'username',None), credential=getattr(s,'credential',None)))
                turn_server_count += len([u for u in keep_urls if u.startswith('turn')])
        if filtered:
            servers = filtered
            logger.info(f"TLS_ONLY filter: kept {turn_server_count}/{original_turn_count} TURN servers")
        else:
            # As a safety, if nothing matched, keep originals
            logger.warning("TLS_ONLY filter removed all TURN servers - keeping originals")
            turn_server_count = original_turn_count
    
    # Log server availability for connection failure diagnostics
    if turn_server_count == 0:
        logger.warning("No TURN servers configured - NAT traversal will likely fail in restricted networks")
    else:
        logger.info(f"ICE configuration: {stun_server_count} STUN, {turn_server_count} TURN servers")

    # Configure ICE transport policy
    if FORCE_RELAY:
        # Force all connections through TURN servers only
        config = RTCConfiguration(iceServers=servers, iceTransportPolicy="relay")
        logger.info("ICE transport policy: relay-only (FORCE_RELAY enabled)")
        if turn_server_count == 0:
            logger.error("FORCE_RELAY enabled but no TURN servers available - connections will fail")
    else:
        config = RTCConfiguration(iceServers=servers)
        logger.info("ICE transport policy: all (direct + relay)")
    
    return config


def _prefer_codec(sdp: str, kind: str, codec: str) -> str:
    """Move payload types for the given codec to the front of the m-line.
    Minimal SDP munging for preferring codecs (e.g., H264 or VP8).
    """
    try:
        lines = sdp.splitlines()
        # Map pt -> codec
        pt_to_codec = {}
        for ln in lines:
            if ln.startswith('a=rtpmap:'):
                try:
                    rest = ln[len('a=rtpmap:'):]
                    pt, enc = rest.split(' ', 1)
                    codec_name = enc.split('/')[0].upper()
                    pt_to_codec[pt] = codec_name
                except Exception:
                    pass
        # Find m-line for kind
        for i, ln in enumerate(lines):
            if ln.startswith('m=') and kind in ln:
                parts = ln.split(' ')
                header = parts[:3]
                pts = parts[3:]
                preferred = [pt for pt in pts if pt_to_codec.get(pt, '') == codec.upper()]
                others = [pt for pt in pts if pt not in preferred]
                lines[i] = ' '.join(header + preferred + others)
                break
        return '\r\n'.join(lines) + '\r\n'
    except Exception:
        return sdp


# Pipeline initialization lock to prevent concurrent init attempts
_init_lock = asyncio.Lock()

async def _ensure_pipeline_initialized():
    """Initialize the pipeline if not already loaded."""
    pipeline = get_pipeline()
    try:
        if not getattr(pipeline, "loaded", False):
            async with _init_lock:
                # Double-check after acquiring lock
                if not getattr(pipeline, "loaded", False):
                    init = getattr(pipeline, "initialize", None)
                    if callable(init):
                        result = init()
                        if asyncio.iscoroutine(result):
                            await result
    except Exception as e:
        logger.error(f"Pipeline init failed: {e}")


@dataclass
class PeerState:
    pc: RTCPeerConnection
    created: float
    control_channel_ready: bool = False
    last_connection_state: Optional[str] = None
    last_ice_state: Optional[str] = None
    cleanup_task: Optional[asyncio.Task] = None
    outbound_video: Optional['OutboundVideoTrack'] = None


# In-memory single peer (extend to dict for multi-user)
_peer_state: Optional[PeerState] = None
_peer_lock = asyncio.Lock()


class IncomingVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track: MediaStreamTrack):
        super().__init__()  # base init
        self.track = track
        self.pipeline = get_pipeline()
        self.frame_id = 0
        self._last_processed: Optional[np.ndarray] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        # Latency / timing metrics
        self._capture_ts: Optional[float] = None
        self._last_latency_ms: Optional[float] = None
        self._avg_latency_ms: Optional[float] = None
        self._lat_hist: list[float] = []
        self._queue_wait_last_ms: Optional[float] = None
        self._queue_wait_hist: list[float] = []
        self._frames_passthrough = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._placeholder_active = True
        self._sync_if_idle = os.getenv('MIRAGE_SYNC_IF_IDLE','1').lower() in ('1','true','yes','on')
        self._pts_origin: Optional[float] = None  # monotonic origin
        self._last_sent_pts: Optional[int] = None
        self._time_base = (1, 90000)  # 90kHz typical video clock
        self._raw_frames_in = 0

    async def recv(self):  # type: ignore[override]
        frame = await self.track.recv()
        self._raw_frames_in += 1
        if self._raw_frames_in == 1:
            try:
                logger.info("IncomingVideoTrack first frame received size=%sx%s" % (getattr(frame, 'width', '?'), getattr(frame, 'height', '?')))
            except Exception:
                pass
        self.frame_id += 1
        capture_t = time.time()
        if self._pts_origin is None:
            self._pts_origin = capture_t
        # Convert to numpy BGR for pipeline
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        proc_input = img
        # Optional downscale (same as prior)
        try:
            max_dim_cfg = int(os.getenv('MIRAGE_PROC_MAX_DIM', '512') or '512')
            if max_dim_cfg < 64:
                max_dim_cfg = 64
            if max(h, w) > max_dim_cfg:
                if w >= h:
                    scale_w = max_dim_cfg
                    scale_h = int(h * (max_dim_cfg / w))
                else:
                    scale_h = max_dim_cfg
                    scale_w = int(w * (max_dim_cfg / h))
                proc_input = cv2.resize(img, (max(1, scale_w), max(1, scale_h)))
        except Exception as e:
            logger.debug(f"Video downscale skip: {e}")

        expected_size = (w, h)
        processed: Optional[np.ndarray] = None

        # Hybrid processing: inline if no background task running OR sync flag set; else schedule
        if self._sync_if_idle and (self._processing_task is None):
            t_q_start = time.time()
            try:
                out_small = self.pipeline.process_video_frame(proc_input, self.frame_id)
                if out_small is not None and (out_small.shape[1], out_small.shape[0]) != expected_size:
                    processed = cv2.resize(out_small, expected_size)
                else:
                    processed = out_small if out_small is not None else img
                self._queue_wait_last_ms = (time.time() - t_q_start) * 1000.0  # inclusive (no wait, pure proc)
                self._queue_wait_hist.append(self._queue_wait_last_ms)
                if len(self._queue_wait_hist) > 300:
                    self._queue_wait_hist.pop(0)
                self._frames_processed += 1
            except Exception as ex:
                logger.debug(f"inline processing error: {ex}")
                processed = img
        else:
            # Background path
            if self._processing_task is None:
                async def _process_async(inp: np.ndarray, expected_size: tuple[int,int], fid: int, enqueue_t: float):
                    try:
                        out_small = self.pipeline.process_video_frame(inp, fid)
                        out = out_small
                        if out_small is not None and (out_small.shape[1], out_small.shape[0]) != expected_size:
                            out = cv2.resize(out_small, expected_size)
                        elif out is None:
                            out = inp  # fallback
                        async with self._lock:
                            self._last_processed = out
                        q_wait = (time.time() - enqueue_t) * 1000.0
                        self._queue_wait_last_ms = q_wait
                        self._queue_wait_hist.append(q_wait)
                        if len(self._queue_wait_hist) > 300:
                            self._queue_wait_hist.pop(0)
                        self._frames_processed += 1
                    except Exception as ex:
                        logger.debug(f"video processing error(bg): {ex}")
                    finally:
                        self._processing_task = None
                self._processing_task = asyncio.create_task(_process_async(proc_input, expected_size, self.frame_id, time.time()))
            # Use last processed snapshot; count passthrough if not yet available
            async with self._lock:
                if self._last_processed is not None:
                    processed = self._last_processed
                else:
                    processed = img
                    self._frames_passthrough += 1
                    # We'll consider this frame 'dropped' re: processing freshness if a task already running
                    if self._processing_task is not None:
                        self._frames_dropped += 1

        # Metrics update
        proc_latency_ms = (time.time() - capture_t) * 1000.0
        self._last_latency_ms = proc_latency_ms
        self._lat_hist.append(proc_latency_ms)
        if len(self._lat_hist) > 300:
            self._lat_hist.pop(0)
        self._avg_latency_ms = float(np.mean(self._lat_hist)) if self._lat_hist else None

        # Placeholder becomes inactive as soon as we emit a frame post-first capture
        if self._placeholder_active:
            self._placeholder_active = False

        # Timestamp handling: derive pts from capture time relative to origin on a 90kHz clock
        try:
            clock_rate = 90000
            rel_sec = capture_t - (self._pts_origin or capture_t)
            pts = int(rel_sec * clock_rate)
            # Guard against monotonic regressions
            if self._last_sent_pts is not None and pts <= self._last_sent_pts:
                pts = self._last_sent_pts + int(clock_rate / 30)  # assume ~30fps minimal increment
            self._last_sent_pts = pts
        except Exception:
            pts = frame.pts if frame.pts is not None else 0

        import av as _av
        vframe = _av.VideoFrame.from_ndarray(processed, format="bgr24")
        vframe.pts = pts
        vframe.time_base = _av.time_base.TimeBase(num=1, den=90000) if hasattr(_av, 'time_base') else frame.time_base
        if (self.frame_id % 120) == 0:
            logger.debug(
                f"vid frame={self.frame_id} inline={self._sync_if_idle and self._processing_task is None} "
                f"proc_ms={proc_latency_ms:.1f} avg_ms={self._avg_latency_ms:.1f if self._avg_latency_ms else None} "
                f"queue_wait_last={self._queue_wait_last_ms} passthrough={self._frames_passthrough} dropped={self._frames_dropped}"
            )
        return vframe


class OutboundVideoTrack(VideoStreamTrack):
    """Outbound track that sends black frames until a real source is attached.
    Once set_source is called with a MediaStreamTrack, it relays frames from that track.
    """
    kind = "video"

    def __init__(self, width: int = 320, height: int = 240, fps: int = 15):
        super().__init__()
        self._source: Optional[MediaStreamTrack] = None
        self._width = width
        self._height = height
        self._frame_interval = 1.0 / max(1, fps)
        self._last_ts = time.time()
        self._frame_count = 0
        self._debug_emitted = 0
        self._placeholder_sent = 0
        # Placeholder mode active until first successful frame relay OR timeout
        self._placeholder_active = True
        self._placeholder_timeout = time.time() + 5.0
        # Diagnostics
        self._relay_failures = 0
        self._relay_last_error: Optional[str] = None
        self._relay_last_error_ts: Optional[float] = None
        self._first_relay_ts: Optional[float] = None
        self._placeholder_deactivated_ts: Optional[float] = None
        self._placeholder_initial_ts: float = time.time()

    def set_source(self, track: MediaStreamTrack):
        self._source = track
        
    def clear_source(self):
        """Clear the source to prevent hanging on failed connections"""
        self._source = None
        self._placeholder_active = True  # Revert to placeholder mode
        self._placeholder_timeout = time.time() + 5.0  # Reset timeout

    async def recv(self):  # type: ignore[override]
        src = self._source
        if src is not None:
            try:
                f = await src.recv()
                # Detect if frame is still a raw passthrough sized frame (heuristic: placeholder period or frame_count==0)
                self._frame_count += 1
                self._debug_emitted += 1
                if self._placeholder_active:
                    self._placeholder_active = False
                    self._placeholder_deactivated_ts = time.time()
                if (self._frame_count % 30) == 0:
                    try:
                        logger.info(f"OutboundVideoTrack relayed frame {self._frame_count} size={getattr(f, 'width', '?')}x{getattr(f, 'height', '?')}")
                    except Exception:
                        pass
                if self._first_relay_ts is None:
                    self._first_relay_ts = time.time()
                return f
            except Exception as e:
                self._relay_failures += 1
                if self._relay_failures <= 5 or (self._relay_failures % 50) == 0:
                    logger.warning(f"OutboundVideoTrack relay failure count={self._relay_failures} err={e}")
                self._relay_last_error = str(e)
                self._relay_last_error_ts = time.time()
                # fall back to placeholder pattern
        # generate black/diagnostic placeholder frame at target fps (early stage before processed frames ready)
        now = time.time()
        delay = self._frame_interval - (now - self._last_ts)
        if delay > 0:
            await asyncio.sleep(delay)
        self._last_ts = time.time()
        # Diagnostic test pattern only while placeholder active and before timeout
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        if self._placeholder_active and time.time() > self._placeholder_timeout:
            self._placeholder_active = False
        placeholder_active = self._placeholder_active
        if placeholder_active and (self._frame_count % 15) == 0:
            try:
                logger.warning(f"[video-placeholder] still active fc={self._frame_count} src={'set' if self._source else 'none'} timeout_in={int(self._placeholder_timeout-time.time())}")
            except Exception:
                pass
        try:
            # Color bars
            num_bars = 6
            bar_w = max(1, self._width // num_bars)
            colors = [
                (0,0,255),    # Red
                (0,255,0),    # Green
                (255,0,0),    # Blue
                (0,255,255),  # Yellow
                (255,0,255),  # Magenta
                (255,255,0),  # Cyan
            ]
            for i in range(num_bars):
                x0 = i*bar_w
                x1 = self._width if i==num_bars-1 else (i+1)*bar_w
                frame[:, x0:x1] = colors[i]
            # Moving square / indicator
            t = int(time.time()*2)
            sq = max(10, min(self._height, self._width)//10)
            x = (t*25) % max(1, (self._width - sq))
            y = (t*18) % max(1, (self._height - sq))
            color = (255,255,255) if placeholder_active else (0,0,0)
            cv2.rectangle(frame, (x,y), (x+sq,y+sq), color, thickness=-1)
            # Text with frame count
            text = f"OUT {self._frame_count}{' P' if placeholder_active else ''}"
            cv2.putText(frame, text, (10, self._height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (10, self._height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        except Exception:
            # If OpenCV drawing fails for any reason, keep plain black
            pass
        import av as _av
        vframe = _av.VideoFrame.from_ndarray(frame, format="bgr24")
        # Provide monotonically increasing timestamps for encoder
        pts, time_base = await self.next_timestamp()
        vframe.pts = pts
        vframe.time_base = time_base
        self._frame_count += 1
        self._debug_emitted += 1
        if placeholder_active:
            self._placeholder_sent += 1
            if self._placeholder_sent in (1, 10):
                try:
                    logger.info(f"OutboundVideoTrack placeholder frame emitted (count={self._placeholder_sent})")
                except Exception:
                    pass
        if (self._frame_count % 30) == 0 and not placeholder_active:
            try:
                logger.info(f"OutboundVideoTrack generated pattern frame {self._frame_count} {self._width}x{self._height}")
            except Exception:
                pass
        return vframe


class IncomingAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track: MediaStreamTrack):
        super().__init__()
        self.track = track
        self.pipeline = get_pipeline()
        self._resample_to_16k = None
        self._resample_from_16k = None

    async def recv(self):  # type: ignore[override]
        frame = await self.track.recv()
        # frame is an AudioFrame (PCM)
        try:
            import av
            from av.audio.resampler import AudioResampler
            # Initialize resamplers once using input characteristics
            if self._resample_to_16k is None:
                self._resample_to_16k = AudioResampler(format='s16', layout='mono', rate=16000)
            if self._resample_from_16k is None:
                # Back to original sample rate and layout; keep s16 for low overhead
                target_layout = frame.layout.name if frame.layout else 'mono'
                target_rate = frame.sample_rate or 48000
                self._resample_from_16k = AudioResampler(format='s16', layout=target_layout, rate=target_rate)

            # 1) To mono s16 @16k for pipeline
            f_16k_list = self._resample_to_16k.resample(frame)
            if isinstance(f_16k_list, list):
                f_16k = f_16k_list[0]
            else:
                f_16k = f_16k_list
            pcm16k = f_16k.to_ndarray()  # (channels, samples), dtype=int16
            if pcm16k.ndim == 2:
                # convert to mono if needed
                if pcm16k.shape[0] > 1:
                    pcm16k = np.mean(pcm16k, axis=0, keepdims=True).astype(np.int16)
                # drop channel dim -> (samples,)
                pcm16k = pcm16k.reshape(-1)

            # 2) Pipeline processing (mono 16k int16 ndarray)
            processed_arr = self.pipeline.process_audio_chunk(pcm16k)
            if isinstance(processed_arr, bytes):
                processed_bytes = processed_arr
            else:
                processed_bytes = np.asarray(processed_arr, dtype=np.int16).tobytes()

            # 3) Wrap processed back into an av frame @16k mono s16
            samples = len(processed_bytes) // 2
            f_proc_16k = av.AudioFrame(format='s16', layout='mono', samples=samples)
            f_proc_16k.sample_rate = 16000
            f_proc_16k.planes[0].update(processed_bytes)

            # 4) Resample back to original sample rate/layout
            f_out_list = self._resample_from_16k.resample(f_proc_16k)
            if isinstance(f_out_list, list) and len(f_out_list) > 0:
                f_out = f_out_list[0]
            else:
                f_out = f_proc_16k  # fallback

            # Preserve timing as best-effort
            f_out.pts = frame.pts
            f_out.time_base = frame.time_base
            return f_out
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return frame


@router.post("/offer")
async def webrtc_offer(offer: Dict[str, Any], x_api_key: Optional[str] = Header(default=None), x_auth_token: Optional[str] = Header(default=None)):
    """Accept SDP offer and return SDP answer."""
    global _peer_state  # declare once at top to avoid 'used prior to global declaration'
    # If enforcement enabled, require a valid signed token; otherwise allow
    if REQUIRE_API_KEY:
        if not (x_auth_token and _verify_token(x_auth_token)):
            raise HTTPException(status_code=401, detail="Unauthorized")
    if not AIORTC_AVAILABLE:
        raise HTTPException(status_code=503, detail=f"aiortc not available: {AIORTC_IMPORT_ERROR}")

    async with _peer_lock:
        # Ensure pipeline is ready before wiring tracks
        await _ensure_pipeline_initialized()
        # Cleanup existing peer if present - critical for retry scenarios
        if _peer_state is not None:
            try:
                await _peer_state.pc.close()
                logger.info("Closed existing peer connection for new offer")
            except Exception:
                pass
            # Clear outbound video source from previous connection
            if _peer_state.outbound_video:
                _peer_state.outbound_video.clear_source()
                logger.info("Cleared outbound video source from previous connection")
            # Reset pipeline state for clean reconnection
            try:
                from swap_pipeline import reset_pipeline
                reset_pipeline()
                logger.info("Pipeline reset for new offer")
            except Exception as e:
                logger.warning(f"Pipeline reset failed during new offer: {e}")
            _peer_state = None

    ice_config = _ice_configuration()
    # Log ICE configuration for diagnostics
    server_summary = []
    for server in ice_config.iceServers:
        urls = server.urls if isinstance(server.urls, list) else [server.urls]
        has_auth = bool(getattr(server, 'username', None))
        for url in urls:
            server_type = 'TURN' if url.startswith('turn') else 'STUN'
            server_summary.append(f"{server_type}:{url}{'(auth)' if has_auth else ''}")
    logger.info(f"ICE servers configured: {', '.join(server_summary)}")
    
    # Test TURN server connectivity (async, don't block connection)
    if ice_config.iceServers:
        asyncio.create_task(_test_turn_connectivity(ice_config.iceServers))
    
    pc = RTCPeerConnection(configuration=ice_config)
    blackhole = MediaBlackhole()  # optional sink

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info("Data channel received: %s", channel.label)
        if channel.label == "control":
            # Mark control channel readiness on open/close
            @channel.on("open")
            def _on_open():
                try:
                    global _peer_state
                    if _peer_state is not None:
                        _peer_state.control_channel_ready = True
                except Exception:
                    pass

            @channel.on("close")
            def _on_close():
                try:
                    global _peer_state
                    if _peer_state is not None:
                        _peer_state.control_channel_ready = False
                except Exception:
                    pass
            def send_metrics():
                pipeline = get_pipeline()
                stats = pipeline.get_performance_stats() if pipeline.loaded else {}
                payload = json.dumps({"type": "metrics", "payload": stats})
                try:
                    channel.send(payload)
                except Exception:
                    logger.debug("Failed sending metrics")

            @channel.on("message")
            def on_message(message):
                    try:
                        if isinstance(message, bytes):
                            return
                        data = json.loads(message)
                        mtype = data.get("type")
                        if mtype == "ping":
                            channel.send(json.dumps({"type": "pong", "t": time.time()}))
                        elif mtype == "metrics_request":
                            send_metrics()
                        elif mtype == "set_reference":
                            b64 = data.get("image_jpeg_base64") or data.get("image_base64")
                            if b64:
                                async def _set_ref_async(b64data: str):
                                    try:
                                        # Allow moderately large images; cap to ~6MB base64 length
                                        if len(b64data) > 6_000_000:
                                            channel.send(json.dumps({"type": "error", "message": "reference too large"}))
                                            return
                                        raw = base64.b64decode(b64data)
                                        arr = np.frombuffer(raw, np.uint8)
                                        # cv2.imdecode handles JPEG, PNG, WebP, etc. automatically
                                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                        if img is None:
                                            channel.send(json.dumps({"type": "error", "message": "decode failed (unsupported image or corrupt)"}))
                                            return
                                        # Downscale to max 512 for stability
                                        h, w = img.shape[:2]
                                        scale = max(h, w)
                                        if scale > 512:
                                            if w >= h:
                                                new_w = 512
                                                new_h = max(1, int(h * (512 / w)))
                                            else:
                                                new_h = 512
                                                new_w = max(1, int(w * (512 / h)))
                                            img = cv2.resize(img, (new_w, new_h))
                                        pipeline = get_pipeline()
                                        loop = asyncio.get_running_loop()
                                        def _set_ref_blocking():
                                            return pipeline.set_reference_frame(img)
                                        ok = await loop.run_in_executor(None, _set_ref_blocking)
                                        if ok:
                                            channel.send(json.dumps({"type": "reference_ack"}))
                                        else:
                                            channel.send(json.dumps({"type": "error", "message": "no suitable face found"}))
                                    except Exception as e:
                                        logger.error(f"set_reference error: {e}")
                                        try:
                                            channel.send(json.dumps({"type": "error", "message": str(e)}))
                                        except Exception:
                                            pass
                                asyncio.create_task(_set_ref_async(b64))
                    except Exception as e:
                        logger.error(f"Data channel message error: {e}")

    @pc.on("connectionstatechange")
    async def on_state_change():
        global _peer_state  # single global declaration for entire handler
        logger.info("Peer connection state: %s", pc.connectionState)
        try:
            if _peer_state is not None:
                _peer_state.last_connection_state = pc.connectionState
        except Exception:
            pass
        # Immediately close failed connections to prevent resource leaks
        if pc.connectionState in ("failed", "disconnected", "closed"):
            try:
                # Clean pipeline resources on connection failure/close
                from swap_pipeline import reset_pipeline
                reset_pipeline()
                logger.info(f"Pipeline reset due to connection state: {pc.connectionState}")
            except Exception as e:
                logger.warning(f"Pipeline reset failed on state change: {e}")
            
            # Clear global peer state to allow clean retry
            async with _peer_lock:
                if _peer_state is not None and _peer_state.pc == pc:
                    logger.info("Clearing global peer state due to connection failure")
                    # Clear outbound video source to prevent hanging on retry
                    if _peer_state.outbound_video:
                        _peer_state.outbound_video.clear_source()
                        logger.info("Cleared outbound video source")
                    _peer_state = None
            
            if pc.connectionState == "failed":
                try:
                    await pc.close()
                    logger.info("Closed failed peer connection")
                except Exception:
                    pass

    @pc.on("iceconnectionstatechange")
    async def on_ice_state_change():
        global _peer_state
        logger.info("ICE connection state: %s", pc.iceConnectionState)
        try:
            if _peer_state is not None:
                _peer_state.last_ice_state = pc.iceConnectionState
        except Exception:
            pass
        
        # Log detailed ICE failure diagnostics
        if pc.iceConnectionState in ("disconnected", "failed"):
            try:
                # Log ICE transport stats if available
                stats = await pc.getStats()
                ice_candidates = []
                ice_pairs = []
                for stat_id, stat in stats.items():
                    if hasattr(stat, 'type'):
                        if stat.type == "local-candidate":
                            ice_candidates.append(f"local:{getattr(stat, 'candidateType', '?')}:{getattr(stat, 'protocol', '?')}:{getattr(stat, 'address', '?')}:{getattr(stat, 'port', '?')}")
                        elif stat.type == "remote-candidate": 
                            ice_candidates.append(f"remote:{getattr(stat, 'candidateType', '?')}:{getattr(stat, 'protocol', '?')}:{getattr(stat, 'address', '?')}:{getattr(stat, 'port', '?')}")
                        elif stat.type == "candidate-pair":
                            state = getattr(stat, 'state', '?')
                            ice_pairs.append(f"pair:{state}:{getattr(stat, 'priority', '?')}")
                
                logger.warning(f"ICE {pc.iceConnectionState} - candidates: {len(ice_candidates)} pairs: {len(ice_pairs)}")
                if ice_candidates:
                    logger.info(f"ICE candidates: {', '.join(ice_candidates[:10])}")  # Limit to first 10
                if ice_pairs:
                    logger.info(f"ICE pairs: {', '.join(ice_pairs[:5])}")  # Limit to first 5
                    
            except Exception as e:
                logger.debug(f"ICE stats collection failed: {e}")
    
    @pc.on("icegatheringstatechange") 
    async def on_ice_gathering_change():
        logger.info("ICE gathering state: %s", pc.iceGatheringState)
        
    @pc.on("icecandidate")
    async def on_ice_candidate(candidate):
        if candidate:
            logger.debug(f"ICE candidate: {candidate.candidate}")
        else:
            logger.info("ICE gathering complete (end-of-candidates)")
    

    # Prepare outbound video first and register track handler before remote description,
    # so we don't miss the initial 'track' events fired during setRemoteDescription.
    try:
        outbound_video = OutboundVideoTrack()
    except Exception as e:
        logger.error(f"Failed to construct outbound video: {e}")
        raise HTTPException(status_code=500, detail=f"outbound_video_setup: {e}")

    @pc.on("track")
    def on_track(track):
        logger.info("Track received: %s", track.kind)
        if track.kind == "video":
            local = IncomingVideoTrack(track)
            try:
                outbound_video.set_source(local)
                logger.info("Outbound video source bound to incoming video")
            except Exception as e:
                logger.error(f"video source assign error: {e}")
        elif track.kind == "audio":
            local_a = IncomingAudioTrack(track)
            try:
                pc.addTrack(local_a)
                logger.info("Loopback processed audio track added")
            except Exception as e:
                logger.error(f"audio addTrack error: {e}")

    # Add outbound video to ensure the answer includes a send m-line
    try:
        sender = pc.addTrack(outbound_video)
        try:
            params = sender.getParameters()
            if params and hasattr(params, 'encodings'):
                if not params.encodings:
                    params.encodings = [{}]
                for enc in params.encodings:
                    enc['maxBitrate'] = min(enc.get('maxBitrate', 300_000), 300_000)
                    enc.setdefault('scaleResolutionDownBy', 2.0)
                    enc.setdefault('degradationPreference', 'maintain-resolution')
            sender.setParameters(params)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Failed to set up outbound video: {e}")
        raise HTTPException(status_code=500, detail=f"outbound_video_setup: {e}")

    # Now apply the remote description (offer)
    try:
        desc = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
        await pc.setRemoteDescription(desc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SDP offer: {e}")

    # Create answer with error surfacing
    try:
        answer = await pc.createAnswer()
    except Exception as e:
        logger.error(f"createAnswer error: {e}")
        raise HTTPException(status_code=500, detail=f"createAnswer: {e}")
    # Avoid SDP munging to reduce negotiation fragility
    try:
        # Optionally prefer H264 for broader compatibility
        if PREFER_H264 and isinstance(answer.sdp, str):
            try:
                answer = RTCSessionDescription(sdp=_prefer_codec(answer.sdp, 'video', 'H264'), type=answer.type)
            except Exception:
                pass
        await pc.setLocalDescription(answer)
    except Exception as e:
        logger.error(f"setLocalDescription error: {e}")
        raise HTTPException(status_code=500, detail=f"setLocalDescription: {e}")

    _peer_state = PeerState(pc=pc, created=time.time(), outbound_video=outbound_video)

    logger.info("WebRTC answer created")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@router.get("/token")
async def mint_token():
    """Return a short-lived signed token that can be used as X-Auth-Token.
    Public endpoint; signature uses server-held API key, if configured.
    """
    if not API_KEY:
        raise HTTPException(status_code=400, detail="API key not configured")
    return {"token": _mint_token(), "ttl": TOKEN_TTL_SECONDS}


@router.post("/cleanup")
async def cleanup_peer(x_api_key: Optional[str] = Header(default=None), x_auth_token: Optional[str] = Header(default=None)):
    # Accept either raw API key or signed auth token when enforcement enabled
    _check_api_key(x_api_key, token_val=x_auth_token)
    async with _peer_lock:
        global _peer_state
        if _peer_state is None:
            return {"status": "no_peer"}
        try:
            await _peer_state.pc.close()
        except Exception:
            pass
        # Clean up pipeline resources to prevent hang on reconnection
        try:
            from swap_pipeline import reset_pipeline
            reset_pipeline()
        except Exception as e:
            logger.warning(f"Pipeline reset failed: {e}")
        _peer_state = None
        return {"status": "closed"}

@router.get("/frame_counter")
async def frame_counter():
    try:
        st = _peer_state
        if st is None:
            return {"active": False}
        pc = st.pc
        count = None
        try:
            for s in pc.getSenders():
                tr = getattr(s, 'track', None)
                if tr and getattr(tr, 'kind', None) == 'video' and hasattr(tr, '_debug_emitted'):
                    count = getattr(tr, '_debug_emitted')
                    break
        except Exception:
            pass
        return {"active": True, "frames_emitted": count}
    except Exception as e:
        return {"active": False, "error": str(e)}

@router.get("/pipeline_stats")
async def pipeline_stats():
    """Return merged swap pipeline stats and live video track latency metrics."""
    try:
        pipeline = get_pipeline()
        base_stats = pipeline.get_performance_stats() if getattr(pipeline, 'loaded', False) else {}
        # Attempt to locate the active IncomingVideoTrack via peer senders
        track_stats = {}
        try:
            st = _peer_state
            if st is not None:
                pc = st.pc
                for sender in pc.getSenders():
                    tr = getattr(sender, 'track', None)
                    if tr and isinstance(tr, MediaStreamTrack) and getattr(tr, 'kind', None) == 'video':
                        # Heuristic: if it has our added attributes
                        for attr in [
                            '_last_latency_ms','_avg_latency_ms','_queue_wait_last_ms','_frames_passthrough',
                            '_frames_processed','_frames_dropped','_placeholder_active','_raw_frames_in'
                        ]:
                            if hasattr(tr, attr):
                                track_stats[attr.lstrip('_')] = getattr(tr, attr)
                        # Outbound diagnostics (if this is the outbound track)
                        for oattr, key in [
                            ('_relay_failures','relay_failures'),
                            ('_relay_last_error','relay_last_error'),
                            ('_relay_last_error_ts','relay_last_error_ts'),
                            ('_placeholder_sent','placeholder_frames'),
                            ('_placeholder_initial_ts','placeholder_initial_ts'),
                            ('_placeholder_deactivated_ts','placeholder_deactivated_ts'),
                        ]:
                            if hasattr(tr, oattr):
                                track_stats[key] = getattr(tr, oattr)
                        break
        except Exception as e:
            track_stats['error'] = f"track_stats: {e}" 
        return {"pipeline": base_stats, "video_track": track_stats}
    except Exception as e:
        return {"error": str(e)}

# Optional: connection monitoring endpoint for diagnostics
if add_connection_monitoring is not None:
    try:
        # Provide a getter to reflect live _peer_state rather than a stale snapshot
        def _get_peer_state():
            return _peer_state
        add_connection_monitoring(router, _get_peer_state)
    except Exception:
        pass
# Force rebuild Thu Sep 25 13:03:20 EDT 2025
