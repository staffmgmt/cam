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
            'enhanced_frames': 0
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

    def initialize(self):
        if self.initialized:
            return True
        providers = None
        try:
            # Let insightface choose; can restrict with env MIRAGE_CUDA_ONLY
            if os.getenv('MIRAGE_CUDA_ONLY'):
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        except Exception:
            providers = None
        if insightface is None or FaceAnalysis is None:
            raise ImportError("insightface (and its deps like onnxruntime) not available. Ensure onnxruntime, onnx, torch installed.")
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640,640))
        # Load swapper
        model_path = INSWAPPER_ONNX_PATH
        if not os.path.isfile(model_path):
            if os.path.isfile(ALT_INSWAPPER_PATH):
                model_path = ALT_INSWAPPER_PATH
            else:
                raise FileNotFoundError(f"Missing InSwapper model (checked {INSWAPPER_ONNX_PATH} and {ALT_INSWAPPER_PATH})")
        self.swapper = insightface.model_zoo.get_model(model_path, providers=providers)
        # Optional CodeFormer enhancer
        if self.codeformer_enabled:
            self._try_load_codeformer()
        self.initialized = True
        self.loaded = True  # legacy attribute for external checks
        logger.info('FaceSwapPipeline initialized')
        return True

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
            return
        # Try import; if fails attempt auto-clone
        need_clone = False
        try:
            from codeformer.archs.codeformer_arch import CodeFormer  # type: ignore
        except Exception:
            need_clone = True
        if need_clone and os.getenv('MIRAGE_CODEFORMER_AUTOCLONE', '1').lower() in ('1','true','yes','on'):
            # Clone into a writable path inside models (repo root may be read-only in some deploy envs)
            repo_dir = os.getenv('MIRAGE_CODEFORMER_REPO_DIR', os.path.join('models', '_codeformer_repo'))
            if self._ensure_repo_clone(repo_dir):
                import sys as _sys
                if repo_dir not in _sys.path:
                    _sys.path.append(repo_dir)
            else:
                logger.warning('CodeFormer repo clone failed; enhancement disabled')
                return
            try:
                from codeformer.archs.codeformer_arch import CodeFormer  # type: ignore
            except ModuleNotFoundError as e:
                # Likely missing dependencies such as facexlib / basicsr
                logger.warning(f"CodeFormer module import still failing after clone (dependency missing?): {e}")
                return
            except Exception as e:  # still failing
                logger.warning(f"CodeFormer module import failed after clone attempt: {e}")
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
        if not self.initialized or self.swapper is None or self.app is None or self.source_face is None:
            return frame
        t0 = time.time()
        faces = self.app.get(frame)
        self._last_faces_cache = faces
        if not faces:
            self._record_latency(time.time() - t0)
            self._stats['swap_faces_last'] = 0
            return frame
        # Sort faces by area and keep top-N
        def _area(face):
            x1,y1,x2,y2 = face.bbox.astype(int)
            return (x2-x1)*(y2-y1)
        faces.sort(key=_area, reverse=True)
        out = frame
        count = 0
        for f in faces[:self.max_faces]:
            try:
                out = self.swapper.get(out, f, self.source_face, paste_back=True)
                count += 1
            except Exception as e:
                logger.debug(f"Swap failed for face: {e}")
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
        return dict(
            self._stats,
            initialized=self.initialized,
            codeformer_fidelity=self.codeformer_fidelity if self.codeformer is not None else None,
            codeformer_loaded=self.codeformer_loaded,
            codeformer_frame_stride=self.codeformer_frame_stride,
            codeformer_face_only=self.codeformer_face_only,
            codeformer_avg_latency_ms=cf_avg,
        )

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
