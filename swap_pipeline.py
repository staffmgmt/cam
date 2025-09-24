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
        self._stats = {
            'frames': 0,
            'last_latency_ms': None,
            'avg_latency_ms': None,
            'swap_faces_last': 0,
            'enhanced_frames': 0
        }
        self._lat_hist: List[float] = []
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
        try:
            # CodeFormer dependencies
            from basicsr.utils import imwrite  # noqa: F401
            from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
            import torch
            from torchvision import transforms  # noqa: F401
            from collections import OrderedDict
            # Lazy import codeformer util packaged structure (user expected to mount model)
            if not os.path.isfile(CODEFORMER_PATH):
                logger.warning(f"CodeFormer selected but model file missing: {CODEFORMER_PATH}")
            else:
                # Minimal inline loader (avoid full repo clone)
                from torch import nn
                class CodeFormerWrapper:
                    def __init__(self, model_path: str, fidelity: float):
                        from codeformer.archs.codeformer_arch import CodeFormer  # type: ignore
                        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        self.net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                              connect_list=['32','64','128','256']).to(self.device)
                        ckpt = torch.load(model_path, map_location='cpu')
                        if 'params_ema' in ckpt:
                            self.net.load_state_dict(ckpt['params_ema'], strict=False)
                        else:
                            self.net.load_state_dict(ckpt['state_dict'], strict=False)
                        self.net.eval()
                        self.fidelity = min(max(fidelity, 0.0), 1.0)
                    @torch.no_grad()
                    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
                        import torch.nn.functional as F
                        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        tensor = torch.from_numpy(img).float().to(self.device) / 255.0
                        tensor = tensor.permute(2,0,1).unsqueeze(0)
                        # CodeFormer forward expects (B,C,H,W)
                        try:
                            out = self.net(tensor, w=self.fidelity, adain=True)[0]
                        except Exception:
                            # Some variants return tuple
                            out = self.net(tensor, w=self.fidelity)[0]
                        out = (out.clamp(0,1) * 255.0).byte().permute(1,2,0).cpu().numpy()
                        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                self.codeformer = CodeFormerWrapper(CODEFORMER_PATH, self.codeformer_fidelity)
                self.codeformer_loaded = True
                logger.info('CodeFormer loaded')
        except Exception as e:
            logger.warning(f"CodeFormer init failed, disabling: {e}")
            self.codeformer = None
        self.initialized = True
    self.loaded = True  # legacy attribute for external checks
        logger.info('FaceSwapPipeline initialized')
        return True

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
        if count > 0 and self.codeformer is not None:
            try:
                out = self.codeformer.enhance(out)
                self._stats['enhanced_frames'] += 1
            except Exception as e:
                logger.debug(f"CodeFormer enhancement failed: {e}")
        self._record_latency(time.time() - t0)
        self._stats['swap_faces_last'] = count
        self._stats['frames'] += 1
        return out

    def _record_latency(self, dt: float):
        ms = dt * 1000.0
        self._stats['last_latency_ms'] = ms
        self._lat_hist.append(ms)
        if len(self._lat_hist) > 200:
            self._lat_hist.pop(0)
        self._stats['avg_latency_ms'] = float(np.mean(self._lat_hist)) if self._lat_hist else None

    def get_stats(self) -> Dict[str, Any]:
        return dict(
            self._stats,
            initialized=self.initialized,
            codeformer_fidelity=self.codeformer_fidelity if self.codeformer is not None else None,
            codeformer_loaded=self.codeformer_loaded,
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
