import os
import io
import time
import logging
from typing import Optional, Dict, Any, List

import numpy as np
import cv2
from PIL import Image

try:
    # Defer heavy imports; may fail if onnxruntime/torch missing
    import insightface  # type: ignore
    from insightface.app import FaceAnalysis  # type: ignore
except Exception:
    insightface = None  # type: ignore
    FaceAnalysis = None  # type: ignore

logger = logging.getLogger(__name__)

INSWAPPER_ONNX_PATH = os.path.join('models', 'inswapper', 'inswapper_128_fp16.onnx')
ALT_INSWAPPER_PATH = os.path.join('models', 'inswapper', 'inswapper_128.onnx')
CODEFORMER_PATH = os.path.join('models', 'codeformer', 'codeformer.pth')

class FaceSwapPipeline:
    """Direct face swap + optional enhancement pipeline.

        Lifecycle:
            1. initialize() -> loads detector/recognizer (buffalo_l) and inswapper onnx
            2. set_source_image(image_bytes|np.array) -> extracts source identity face object
            3. process_frame(frame) -> swap all or top-N faces using source face
            4. (optional) CodeFormer enhancement (always attempted if model present)
    """
    def __init__(self):
        self.initialized = False
        self.source_face = None
        self.source_img_meta = {}
        # Legacy compatibility flags expected by old WebRTC data channel handlers
        # 'loaded' previously indicated full reenactment stack ready; here it maps to self.initialized
        self.loaded = False
        # Single enhancer path: CodeFormer (optional)
        self.max_faces = int(os.getenv('MIRAGE_MAX_FACES', '1'))
        # CodeFormer config / controls
        self.codeformer_enabled = os.getenv('MIRAGE_CODEFORMER_ENABLED', '1').lower() in ('1','true','yes','on')
        self.codeformer_frame_stride = int(os.getenv('MIRAGE_CODEFORMER_FRAME_STRIDE', '1') or '1')
        if self.codeformer_frame_stride < 1:
            self.codeformer_frame_stride = 1
        self.codeformer_face_only = os.getenv('MIRAGE_CODEFORMER_FACE_ONLY', '0').lower() in ('1','true','yes','on')
        self.codeformer_face_margin = float(os.getenv('MIRAGE_CODEFORMER_MARGIN', '0.15'))
        self._stats = {
            'frames': 0,
            'last_latency_ms': None,
            'avg_latency_ms': None,
            'swap_faces_last': 0,
            'enhanced_frames': 0,
            # New diagnostic counters
            'early_no_source': 0,
            'early_uninitialized': 0,
            'frames_no_faces': 0,
            'total_faces_detected': 0,
            'total_faces_swapped': 0,
            # Brightness / quality tracking
            'frames_low_brightness': 0,
            'brightness_last': None,
            # Similarity metrics
            'last_primary_similarity': None,
        }
        self._lat_hist: List[float] = []
        self._codeformer_lat_hist: List[float] = []
        self._frame_index = 0
        self._last_faces_cache: List[Any] | None = None
        self.app: Optional[FaceAnalysis] = None
        self.swapper = None
        self.codeformer = None
        self.codeformer_fidelity = float(os.getenv('MIRAGE_CODEFORMER_FIDELITY', '0.75'))
        self.codeformer_loaded = False
        self.codeformer_error: str | None = None
        # Debug verbosity for swap decisions
        self.swap_debug = os.getenv('MIRAGE_SWAP_DEBUG', '0').lower() in ('1','true','yes','on')
        # Brightness compensation configuration
        self.enable_brightness_comp = os.getenv('MIRAGE_BRIGHTNESS_COMP', '1').lower() in ('1','true','yes','on')
        self.target_brightness = float(os.getenv('MIRAGE_TARGET_BRIGHTNESS', '90'))  # mean luminance target (0-255)
        self.low_brightness_threshold = float(os.getenv('MIRAGE_LOW_BRIGHTNESS_THRESH', '40'))
        # Similarity threshold for logging (cosine similarity typical range [-1,1])
        self.similarity_warn_threshold = float(os.getenv('MIRAGE_SIMILARITY_WARN', '0.15'))
        # Temporal reuse configuration
        self.face_cache_ttl = int(os.getenv('MIRAGE_FACE_CACHE_TTL', '5') or '5')  # frames
        self._cached_face = None
        self._cached_face_age = 0
        # Aggressive blend toggle for visibility
        self.aggressive_blend = os.getenv('MIRAGE_AGGRESSIVE_BLEND', '0').lower() in ('1','true','yes','on')
        # Optional face ROI upscaling for tiny faces
        self.face_min_size = int(os.getenv('MIRAGE_FACE_MIN_SIZE', '80') or '80')
        self.face_upscale_factor = float(os.getenv('MIRAGE_FACE_UPSCALE', '1.6'))
        # Toggle for (currently disabled) naive small-face upscale path that caused artifacts
        # Default OFF to prevent black rectangle artifacts observed when re-pasting scaled ROI
        self.enable_face_upscale = os.getenv('MIRAGE_FACE_UPSCALE_ENABLE', '0').lower() in ('1','true','yes','on')
        # Detector preprocessing (CLAHE) low light
        self.det_clahe = os.getenv('MIRAGE_DET_CLAHE', '1').lower() in ('1','true','yes','on')
        # End-to-end latency markers
        self._last_e2e_ms = None
        self._e2e_hist: List[float] = []
        # Track model file actually loaded for diagnostics
        self.inswapper_model_path: str | None = None

    def initialize(self):
        if self.initialized:
            return True
        providers = self._select_providers()
        if insightface is None or FaceAnalysis is None:
            raise ImportError("insightface (and its deps like onnxruntime) not available. Ensure onnxruntime, onnx, torch installed.")
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640,640))
        # Capture active providers after prepare (best effort)
        try:
            self._active_providers = getattr(self.app, 'providers', providers)
        except Exception:
            self._active_providers = providers
        # Load swapper
        model_path = INSWAPPER_ONNX_PATH
        if not os.path.isfile(model_path):
            if os.path.isfile(ALT_INSWAPPER_PATH):
                model_path = ALT_INSWAPPER_PATH
            else:
                raise FileNotFoundError(f"Missing InSwapper model (checked {INSWAPPER_ONNX_PATH} and {ALT_INSWAPPER_PATH})")
        self.swapper = insightface.model_zoo.get_model(model_path, providers=providers)
        self.inswapper_model_path = model_path
        logger.info(f"Loaded InSwapper model: {model_path}")
        # Optional CodeFormer enhancer
        if self.codeformer_enabled:
            self._try_load_codeformer()
        self.initialized = True
        self.loaded = True  # legacy attribute for external checks
        logger.info('FaceSwapPipeline initialized')
        return True

    def _select_providers(self) -> List[str] | None:
        """Decide ONNX Runtime providers with GPU preference and diagnostics.

        Env controls:
          MIRAGE_FORCE_CPU=1        -> force CPU only
          MIRAGE_CUDA_ONLY=1        -> request CUDA + CPU fallback (legacy name)
          MIRAGE_REQUIRE_GPU=1      -> raise if CUDA provider not available
        """
        force_cpu = os.getenv('MIRAGE_FORCE_CPU', '0').lower() in ('1','true','yes','on')
        require_gpu = os.getenv('MIRAGE_REQUIRE_GPU', '0').lower() in ('1','true','yes','on')
        cuda_only_flag = os.getenv('MIRAGE_CUDA_ONLY', '0').lower() in ('1','true','yes','on')
        try:
            import onnxruntime as ort  # type: ignore
            avail = ort.get_available_providers()
            try:
                ver = ort.__version__  # type: ignore
            except Exception:
                ver = 'unknown'
            logger.info(f"[providers] onnxruntime {ver} available={avail}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ONNX Runtime not importable ({e}); letting insightface choose default providers")
            return None
        if force_cpu:
            logger.info('[providers] MIRAGE_FORCE_CPU=1 -> using CPUExecutionProvider only')
            return ['CPUExecutionProvider'] if 'CPUExecutionProvider' in avail else None
        providers: List[str] = []
        if 'CUDAExecutionProvider' in avail:
            providers.append('CUDAExecutionProvider')
        if 'CPUExecutionProvider' in avail:
            providers.append('CPUExecutionProvider')
        if 'CUDAExecutionProvider' not in providers:
            msg = '[providers] CUDAExecutionProvider NOT available; running on CPU'
            if require_gpu or cuda_only_flag:
                # escalate to warning / potential exception
                logger.warning(msg + ' (require_gpu flag set)')
                if require_gpu:
                    raise RuntimeError('GPU required but CUDAExecutionProvider unavailable')
            else:
                logger.info(msg)
        else:
            logger.info(f"[providers] Using providers order: {providers}")
        return providers or None

    def _ensure_repo_clone(self, target_dir: str) -> bool:
        """Clone CodeFormer repo shallowly if missing. Returns True if directory exists after call."""
        try:
            if os.path.isdir(target_dir) and os.path.isdir(os.path.join(target_dir, '.git')):
                return True
            import subprocess, shlex
            os.makedirs(target_dir, exist_ok=True)
            # If directory empty, clone
            if not any(os.scandir(target_dir)):
                logger.info('Cloning CodeFormer repository (shallow)...')
                cmd = f"git clone --depth 1 https://github.com/sczhou/CodeFormer.git {shlex.quote(target_dir)}"
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"CodeFormer auto-clone failed: {e}")
            return False

    def _try_load_codeformer(self):  # pragma: no cover (runtime GPU path)
        if not os.path.isfile(CODEFORMER_PATH):
            logger.warning(f"CodeFormer weight missing; skipping: {CODEFORMER_PATH}")
            return
        try:
            import torch  # type: ignore
        except Exception:
            logger.warning('Torch missing; cannot enable CodeFormer')
            self.codeformer_error = 'torch_missing'
            return
        # Direct import path used by upstream project (packaged when installed)
        # Primary expected path: basicsr.archs.* (weights independent). Some forks use codeformer.archs
        CodeFormer = None  # type: ignore
        try:
            from basicsr.archs.codeformer_arch import CodeFormer as _CF  # type: ignore
            CodeFormer = _CF  # type: ignore
        except Exception as e:
            # Second import path attempt; if both fail capture reason
            try:
                from codeformer.archs.codeformer_arch import CodeFormer as _CF  # type: ignore
                CodeFormer = _CF  # type: ignore
            except Exception as e2:
                self.codeformer_error = f"import_failed:{e2}"
                logger.warning(f"CodeFormer import failed (basicsr & codeformer paths). Skipping enhancement. Root error: {e2}")
                return
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
        except Exception:
            # Basicsr usually present (in requirements); if not, can't proceed
            logger.warning('basicsr not available; skipping CodeFormer')
            return
        # facexlib is required for some preprocessing utilities; warn if absent (not fatal for direct arch usage)
        try:  # pragma: no cover
            import facexlib  # type: ignore  # noqa: F401
        except Exception:
            logger.info('facexlib not installed; continuing (may reduce CodeFormer effectiveness)')
        try:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                             connect_list=['32','64','128','256']).to(device)
            ckpt = torch.load(CODEFORMER_PATH, map_location='cpu')
            if 'params_ema' in ckpt:
                net.load_state_dict(ckpt['params_ema'], strict=False)
            else:
                # Some weights store under 'state_dict'
                net.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
            net.eval()
            fidelity = min(max(self.codeformer_fidelity, 0.0), 1.0)
            class _CFWrap:
                def __init__(self, net, device, fidelity):
                    self.net = net
                    self.device = device
                    self.fidelity = fidelity
                @torch.no_grad()
                def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
                    import torch, torch.nn.functional as F  # type: ignore
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(img).float().to(self.device) / 255.0
                    tensor = tensor.permute(2,0,1).unsqueeze(0)
                    try:
                        out = self.net(tensor, w=self.fidelity, adain=True)[0]
                    except Exception:
                        out = self.net(tensor, w=self.fidelity)[0]
                    out = (out.clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy()
                    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            self.codeformer = _CFWrap(net, device, fidelity)
            self.codeformer_loaded = True
            logger.info('CodeFormer fully loaded')
        except Exception as e:
            self.codeformer_error = f"init_failed:{e}"
            logger.warning(f"CodeFormer final init failed: {e}")
            self.codeformer = None

    def _decode_image(self, data) -> np.ndarray:
        if isinstance(data, bytes):
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        if isinstance(data, np.ndarray):
            return data
        if hasattr(data, 'read'):
            buff = data.read()
            arr = np.frombuffer(buff, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        raise TypeError('Unsupported image input type')

    def set_source_image(self, image_input) -> bool:
        if not self.initialized:
            self.initialize()
        img = self._decode_image(image_input)
        if img is None:
            logger.error('Failed to decode source image')
            return False
        faces = self.app.get(img)
        if not faces:
            logger.error('No face detected in source image')
            return False
        # Choose the largest face by bbox area
        def _area(face):
            x1,y1,x2,y2 = face.bbox.astype(int)
            return (x2-x1)*(y2-y1)
        faces.sort(key=_area, reverse=True)
        self.source_face = faces[0]
        self.source_img_meta = {'resolution': img.shape[:2], 'num_faces': len(faces)}
        logger.info('Source face set')
        return True

    # Legacy method name alias used by some data channel messages
    def set_reference_frame(self, image_input) -> bool:  # pragma: no cover - thin shim
        return self.set_source_image(image_input)

    # Audio processing stubs (voice conversion not yet integrated in new simplified pipeline)
    def process_audio_chunk(self, pcm_bytes: bytes) -> bytes:  # pragma: no cover
        """Pass-through audio to satisfy legacy interface expectations.
        Future: integrate voice conversion here. For now: return original audio data.
        """
        return pcm_bytes

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_in_ts = time.time()
        if not self.initialized or self.swapper is None or self.app is None:
            self._stats['early_uninitialized'] += 1
            if self.swap_debug:
                logger.debug('process_frame: pipeline not fully initialized yet')
            return frame
        if self.source_face is None:
            self._stats['early_no_source'] += 1
            if self.swap_debug:
                logger.debug('process_frame: no source_face set yet')
            return frame
        t0 = time.time()
        # Brightness analysis (grayscale mean) to understand low-light degradation
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            self._stats['brightness_last'] = brightness
            if brightness < self.low_brightness_threshold:
                self._stats['frames_low_brightness'] += 1
                if self.enable_brightness_comp:
                    # Simple gamma / gain compensation (scale then clip)
                    gain = self.target_brightness / max(1.0, brightness)
                    gain = min(gain, 3.0)  # clamp to avoid noise amplification
                    frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
                    if self.swap_debug:
                        logger.debug(f'Applied brightness compensation gain={gain:.2f} (brightness={brightness:.1f})')
        except Exception:
            pass
        # Detector preprocessing path for improved low-light detect
        det_input = frame
        if self.det_clahe:
            try:
                gray_det = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if float(np.mean(gray_det)) < (self.low_brightness_threshold + 15):
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    eq = clahe.apply(gray_det)
                    det_input = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass
        faces = self.app.get(det_input)
        self._last_faces_cache = faces
        if not faces:
            # Attempt temporal reuse of last successful face if within ttl
            if self._cached_face is not None and self._cached_face_age < self.face_cache_ttl:
                faces = [self._cached_face]
                self._cached_face_age += 1
            else:
                self._cached_face = None
                self._cached_face_age = 0
            if self.swap_debug:
                logger.debug('process_frame: no faces detected in incoming frame')
            self._record_latency(time.time() - t0)
            self._stats['swap_faces_last'] = 0
            self._stats['frames_no_faces'] += 1
            # Count processed frame even if no faces detected
            self._stats['frames'] += 1
            self._frame_index += 1
            return frame
        # Accumulate total faces detected (pre top-N filter)
        self._stats['total_faces_detected'] += len(faces)
        # Sort faces by area and keep top-N
        def _area(face):
            x1,y1,x2,y2 = face.bbox.astype(int)
            return (x2-x1)*(y2-y1)
        faces.sort(key=_area, reverse=True)
        out = frame
        count = 0
        similarities: List[float] = []
        for idx, f in enumerate(faces[:self.max_faces]):
            try:
                # Compute similarity to source embedding (cosine) for diagnostics
                try:
                    src_emb = getattr(self.source_face, 'normed_embedding', None)
                    tgt_emb = getattr(f, 'normed_embedding', None)
                    sim = None
                    if src_emb is not None and tgt_emb is not None:
                        # Both are numpy arrays
                        a = src_emb.astype(np.float32)
                        b = tgt_emb.astype(np.float32)
                        denom = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)
                        sim = float(np.dot(a, b) / denom)
                        similarities.append(sim)
                        if idx == 0:
                            self._stats['last_primary_similarity'] = sim
                        if self.swap_debug and sim is not None and sim < self.similarity_warn_threshold:
                            logger.debug(f'Low similarity primary face sim={sim:.3f}')
                except Exception:
                    pass
                # NOTE: Previously attempted naive small-face ROI upscale introduced moving black box artifacts
                # because face landmark coordinates remained in original frame space. We disable that path by default.
                # If re-enabled in future, we must recompute detection landmarks on the upscaled ROI.
                try:
                    out = self.swapper.get(out, f, self.source_face, paste_back=True)
                except Exception:
                    # Fallback: attempt again (rarely helps, mostly for logging symmetry)
                    out = self.swapper.get(out, f, self.source_face, paste_back=True)
                count += 1
            except Exception as e:
                logger.debug(f"Swap failed for face: {e}")
        self._stats['total_faces_swapped'] += count
        # Cache first face for reuse
        if faces:
            self._cached_face = faces[0]
            self._cached_face_age = 0
        # Optional debug overlay for visual confirmation
        if count > 0 and os.getenv('MIRAGE_DEBUG_OVERLAY', '0').lower() in ('1','true','yes','on'):
            try:
                for i, f in enumerate(faces[:self.max_faces]):
                    x1,y1,x2,y2 = f.bbox.astype(int)
                    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
                    label = 'SWAP'
                    if i < len(similarities) and similarities[i] is not None:
                        label += f' {similarities[i]:.2f}'
                    cv2.putText(out, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            except Exception:
                pass
        if self.swap_debug:
            logger.debug(f'process_frame: detected={len(faces)} swapped={count} stride={self.codeformer_frame_stride} apply_cf={count>0 and (self._frame_index % self.codeformer_frame_stride == 0)}')
        # CodeFormer stride / face-region logic
        apply_cf = (
            self.codeformer is not None and
            self.codeformer_enabled and
            self.codeformer_frame_stride > 0 and
            (self._frame_index % self.codeformer_frame_stride == 0)
        )
        if count > 0 and apply_cf:
            cf_t0 = time.time()
            try:
                if self.codeformer_face_only and faces:
                    # Use largest face bbox
                    f0 = faces[0]
                    x1,y1,x2,y2 = f0.bbox.astype(int)
                    h, w = out.shape[:2]
                    mx = int((x2 - x1) * self.codeformer_face_margin)
                    my = int((y2 - y1) * self.codeformer_face_margin)
                    x1c = max(0, x1 - mx); y1c = max(0, y1 - my)
                    x2c = min(w, x2 + mx); y2c = min(h, y2 + my)
                    region = out[y1c:y2c, x1c:x2c]
                    if region.size > 0:
                        enhanced = self.codeformer.enhance(region)
                        out[y1c:y2c, x1c:x2c] = enhanced
                else:
                    out = self.codeformer.enhance(out)
                self._stats['enhanced_frames'] += 1
                cf_dt = (time.time() - cf_t0)*1000.0
                self._codeformer_lat_hist.append(cf_dt)
                if len(self._codeformer_lat_hist) > 200:
                    self._codeformer_lat_hist.pop(0)
            except Exception as e:
                logger.debug(f"CodeFormer enhancement failed: {e}")
        self._record_latency(time.time() - t0)
        self._stats['swap_faces_last'] = count
        self._stats['frames'] += 1
        self._frame_index += 1
        # End-to-end latency including pre-detection + swap path
        self._last_e2e_ms = (time.time() - frame_in_ts) * 1000.0
        self._e2e_hist.append(self._last_e2e_ms)
        if len(self._e2e_hist) > 200:
            self._e2e_hist.pop(0)
        return out

    def _record_latency(self, dt: float):
        ms = dt * 1000.0
        self._stats['last_latency_ms'] = ms
        self._lat_hist.append(ms)
        if len(self._lat_hist) > 200:
            self._lat_hist.pop(0)
        self._stats['avg_latency_ms'] = float(np.mean(self._lat_hist)) if self._lat_hist else None

    def get_stats(self) -> Dict[str, Any]:
        cf_avg = float(np.mean(self._codeformer_lat_hist)) if self._codeformer_lat_hist else None
        info: Dict[str, Any] = dict(
            self._stats,
            initialized=self.initialized,
            codeformer_fidelity=self.codeformer_fidelity if self.codeformer is not None else None,
            codeformer_loaded=self.codeformer_loaded,
            codeformer_frame_stride=self.codeformer_frame_stride,
            codeformer_face_only=self.codeformer_face_only,
            codeformer_avg_latency_ms=cf_avg,
            max_faces=self.max_faces,
            debug_overlay=os.getenv('MIRAGE_DEBUG_OVERLAY', '0'),
            e2e_latency_ms=self._last_e2e_ms,
            e2e_latency_avg_ms=(float(np.mean(self._e2e_hist)) if self._e2e_hist else None),
            inswapper_model_path=self.inswapper_model_path,
            face_upscale_enabled=self.enable_face_upscale,
            codeformer_error=self.codeformer_error,
        )
        # Provider diagnostics (best-effort)
        try:  # pragma: no cover
            import onnxruntime as ort  # type: ignore
            info['available_providers'] = ort.get_available_providers()
            info['active_providers'] = getattr(self, '_active_providers', None)
        except Exception:
            pass
        return info

    # Legacy interface used by webrtc_server data channel
    def get_performance_stats(self) -> Dict[str, Any]:  # pragma: no cover simple delegate
        stats = self.get_stats()
        # Provide alias field names expected historically (if any)
        stats['frames_processed'] = stats.get('frames')
        return stats

    # Backwards compatibility for earlier server expecting process_video_frame
    def process_video_frame(self, frame: np.ndarray, frame_idx: int | None = None) -> np.ndarray:
        return self.process_frame(frame)

# Singleton access similar to previous pattern
_pipeline_instance: Optional[FaceSwapPipeline] = None

def get_pipeline() -> FaceSwapPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = FaceSwapPipeline()
        _pipeline_instance.initialize()
    return _pipeline_instance
