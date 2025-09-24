"""Model downloader for face swap stack (InSwapper + CodeFormer).

Environment:
    MIRAGE_DOWNLOAD_MODELS=1|0
    MIRAGE_INSWAPPER_URL  (default HF inswapper 128)
    MIRAGE_CODEFORMER_URL (default CodeFormer official release)

Models are stored under:
    models/inswapper/inswapper_128_fp16.onnx
    models/codeformer/codeformer.pth

Download priority: requests -> huggingface_hub heuristic. Safe across parallel processes via file locks.
"""
import os
import sys
import shutil
import json
from pathlib import Path
import time
from typing import Optional, List
import os
import errno

try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    import onnx  # type: ignore
except Exception:
    onnx = None
try:
    # Optional: version converter for opset downgrade
    from onnx import version_converter  # type: ignore
except Exception:
    version_converter = None  # type: ignore

try:
        from huggingface_hub import hf_hub_download  # type: ignore
except Exception:
        hf_hub_download = None

INSWAPPER_DIR = Path(__file__).parent / 'models' / 'inswapper'
CODEFORMER_DIR = Path(__file__).parent / 'models' / 'codeformer'
# Ensure base directories exist early to avoid lock file creation errors
for _d in (INSWAPPER_DIR, CODEFORMER_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
HF_HOME = Path(os.getenv('HF_HOME', Path(__file__).parent / '.cache' / 'huggingface'))
HF_HOME.mkdir(parents=True, exist_ok=True)


def _download_requests(url: str, dest: Path, timeout: float = 30.0, retries: int = 3) -> bool:
    if requests is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Use a unique temporary filename to avoid races across concurrent downloads
    tmp = dest.with_suffix(dest.suffix + f'.part.{os.getpid()}.{int(time.time()*1000)}')
    for attempt in range(1, retries + 1):
        try:
            headers = {}
            # Hugging Face token support for gated/private repos
            hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
            if hf_token and 'huggingface.co' in url:
                headers['Authorization'] = f'Bearer {hf_token}'
            print(f"[downloader] (requests) GET {url} -> {dest} (attempt {attempt}/{retries})")
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()
                with open(tmp, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            # Atomic replace
            os.replace(tmp, dest)
            return True
        except Exception as e:
            print(f"[downloader] requests error: {e}")
            time.sleep(min(2 * attempt, 6))
        finally:
            # Clean up any stray temp file
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
    return False

def _download_hf(url: str, dest: Path) -> bool:
    if hf_hub_download is None:
        return False
    # Try to parse repo_id and filename from a typical Hugging Face URL
    # Supports:
    #   https://huggingface.co/<user>/<repo>/resolve/main/<filename>
    #   https://huggingface.co/datasets/<user>/<repo>/resolve/main/<filename>
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        parts = [s for s in p.path.split('/') if s]
        repo_id = None
        filename = None
        # dataset pattern
        if len(parts) >= 6 and parts[0] == 'datasets' and parts[3] == 'resolve':
            # datasets/<user>/<repo>/resolve/main/<filename>
            repo_id = f"{parts[1]}/{parts[2]}"
            filename = '/'.join(parts[5:])
            repo_type = 'dataset'
        # model pattern
        elif len(parts) >= 5 and parts[2] == 'resolve':
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = '/'.join(parts[4:])
            repo_type = None
        if repo_id and filename:
            print(f"[downloader] (hf_hub) repo={repo_id} file={filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Direct huggingface cache to writable location
            os.environ.setdefault('HF_HOME', str(HF_HOME))
            os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(HF_HOME / 'hub'))
            kwargs = { 'repo_id': repo_id, 'filename': filename, 'local_dir': str(HF_HOME / 'hub') }
            if repo_type == 'dataset':
                kwargs['repo_type'] = 'dataset'
            tmp_path = hf_hub_download(**kwargs)
            shutil.copyfile(tmp_path, dest)
            return True
    except Exception as e:
        print(f"[downloader] hf_hub_download error: {e}")
    return False

def _is_valid_onnx(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 262144:  # 256KB minimum sanity size
            return False
        if onnx is None:
            return True
        onnx.load(str(path), load_external_data=True)
        return True
    except Exception:
        return False

def _maybe_convert_opset_to_19(path: Path) -> Path:
    """If ONNX opset > 19, attempt to convert to opset 19 for ORT 1.16.3 compatibility.
    Returns the path to a converted file (sibling with _op19 suffix) or the original path on failure/no-op.
    """
    if onnx is None or version_converter is None or path.suffix != ".onnx":
        return path
    try:
        model = onnx.load(str(path), load_external_data=True)
        max_opset = max((imp.version for imp in model.opset_import), default=0)
        if max_opset and max_opset > 19:
            print(f"[downloader] Downgrading opset from {max_opset} to 19 for {path.name}")
            converted = version_converter.convert_version(model, 19)
            out_path = path.with_name(path.stem + "_op19.onnx")
            onnx.save(converted, str(out_path))
            return out_path
    except Exception as e:
        print(f"[downloader] Opset conversion skipped for {path.name}: {e}")
    return path

def _attempt_urls(urls: List[str], dest: Path) -> bool:
    errors = []
    for u in urls:
        try:
            ok = _download_requests(u, dest)
            if ok:
                return True
            ok = _download_hf(u, dest)
            if ok:
                return True
            errors.append(f"no handler success {u}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{u}: {e}")
    if errors:
        print('[downloader] errors: ' + ' | '.join(errors))
    return False

# Simple cross-process file lock using exclusive create of a .lock file
class _FileLock:
    def __init__(self, target: Path, timeout: float = 60.0, poll: float = 0.2):
        self.lock_path = target.with_suffix(target.suffix + '.lock')
        self.timeout = timeout
        self.poll = poll
        self.acquired = False

    def __enter__(self):
        start = time.time()
        while True:
            try:
                # O_CREAT|O_EXCL ensures exclusive creation
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self.acquired = True
                return self
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if (time.time() - start) > self.timeout:
                    raise TimeoutError(f"Timeout acquiring lock {self.lock_path}")
                time.sleep(self.poll)

    def __exit__(self, exc_type, exc, tb):
        if self.acquired:
            try:
                os.unlink(str(self.lock_path))
            except Exception:
                pass


def _audit(event: str, **extra):
    try:
        audit_dir = Path(__file__).parent / 'models' / '_logs'
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / 'download_audit.jsonl'
        payload = {
            'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'event': event,
            'tag': os.getenv('MIRAGE_DL_TAG', 'downloader'),
        }
        payload.update(extra)
        with audit_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload) + '\n')
    except Exception:
        pass


def maybe_download() -> bool:
    if os.getenv('MIRAGE_DOWNLOAD_MODELS', '1').lower() not in ('1','true','yes','on'):
        print('[downloader] MIRAGE_DOWNLOAD_MODELS disabled')
        _audit('disabled')
        return False
    _audit('start')
    success = True

    inswapper_primary = os.getenv('MIRAGE_INSWAPPER_URL', '').strip()
    codeformer_primary = os.getenv('MIRAGE_CODEFORMER_URL', '').strip()

    inswapper_urls: List[str] = []
    if inswapper_primary:
        inswapper_urls.append(inswapper_primary)
    # Known public mirrors / variants (fp16 and standard)
    # User-requested primary mirror (persistent storage dataset)
    inswapper_urls.extend([
        'https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128_fp16.onnx',
        'https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128_fp16.onnx',
        'https://huggingface.co/damo-vilab/model-zoo/resolve/main/inswapper_128_fp16.onnx',
        'https://huggingface.co/damo-vilab/model-zoo/resolve/main/inswapper_128.onnx',
    ])
    # Deduplicate preserving order
    seen = set()
    inswapper_urls = [u for u in inswapper_urls if not (u in seen or seen.add(u))]

    codeformer_urls: List[str] = []
    if codeformer_primary:
        codeformer_urls.append(codeformer_primary)
    # Official release (redirect sometimes), plus fallback community mirrors (replace if license requires)
    codeformer_urls.extend([
        # GitHub release asset (preferred explicit version pin)
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        'https://huggingface.co/sczhou/CodeFormer/resolve/main/codeformer.pth',
        'https://huggingface.co/lllyasviel/CodeFormer/resolve/main/codeformer.pth',
    ])
    seen2 = set()
    codeformer_urls = [u for u in codeformer_urls if not (u in seen2 or seen2.add(u))]

    # InSwapper
    inswapper_dest = INSWAPPER_DIR / 'inswapper_128_fp16.onnx'
    if not inswapper_dest.exists():
        try:
            print('[downloader] Downloading InSwapper model...')
            inswapper_dest.parent.mkdir(parents=True, exist_ok=True)
            with _FileLock(inswapper_dest):
                if not inswapper_dest.exists():
                    if not _attempt_urls(inswapper_urls, inswapper_dest):
                        raise RuntimeError('all download methods failed')
            print(f'[downloader] ✅ InSwapper ready: {inswapper_dest}')
            _audit('download_ok', model='inswapper', path=str(inswapper_dest))
        except Exception as e:
            print(f'[downloader] ❌ InSwapper download failed: {e}')
            _audit('download_error', model='inswapper', error=str(e))
            success = False
    else:
        print(f'[downloader] ✅ InSwapper exists: {inswapper_dest}')
        _audit('exists', model='inswapper', path=str(inswapper_dest))

    # CodeFormer (optional)
    codef_dest = CODEFORMER_DIR / 'codeformer.pth'
    if not codef_dest.exists():
        try:
            print('[downloader] Downloading CodeFormer model...')
            codef_dest.parent.mkdir(parents=True, exist_ok=True)
            with _FileLock(codef_dest):
                if not codef_dest.exists():
                    _attempt_urls(codeformer_urls, codef_dest)
            print(f'[downloader] ✅ CodeFormer ready: {codef_dest}')
            _audit('download_ok', model='codeformer', path=str(codef_dest))
        except Exception as e:
            print(f'[downloader] ⚠️ CodeFormer download failed (continuing): {e}')
            _audit('download_error', model='codeformer', error=str(e))
    else:
        print(f'[downloader] ✅ CodeFormer exists: {codef_dest}')
        _audit('exists', model='codeformer', path=str(codef_dest))

    _audit('complete', success=success)
    return success


if __name__ == '__main__':
    print("=== Model Downloader (InSwapper + CodeFormer) ===")
    ok = maybe_download()
    if ok:
        print("✅ All required models downloaded successfully (some optional)")
    else:
        print("❌ Some required model downloads failed")
        sys.exit(1)
