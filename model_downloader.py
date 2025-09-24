"""
Optional model downloader for deterministic builds.
- Controlled with env MIRAGE_DOWNLOAD_MODELS=1
- LivePortrait ONNX URLs controlled via env:
    * MIRAGE_LP_APPEARANCE_URL
    * MIRAGE_LP_MOTION_URL (required for motion)
    * MIRAGE_LP_GENERATOR_URL (optional; enables full neural synthesis)
    * MIRAGE_LP_STITCHING_URL (optional; some pipelines include extra stitching stage)
- InsightFace models will still use the package cache; SCRFD will populate on first run.

More robust with retries and alternative download methods (requests, huggingface_hub).
"""
import os
import sys
import shutil
from pathlib import Path
import time
from typing import Optional
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

LP_DIR = Path(__file__).parent / 'models' / 'liveportrait'
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
            print(f"[downloader] (requests) GET {url} -> {dest} (attempt {attempt}/{retries})")
            with requests.get(url, stream=True, timeout=timeout) as r:
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
    # e.g. https://huggingface.co/<repo_id>/resolve/main/<filename>
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        parts = [s for s in p.path.split('/') if s]
        # Expect parts like ['<repo_user_or_org>', '<repo_name>', 'resolve', 'main', '<filename>']
        if len(parts) >= 5 and parts[2] == 'resolve':
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = '/'.join(parts[4:])
            print(f"[downloader] (hf_hub) repo={repo_id} file={filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Direct huggingface cache to writable location
            os.environ.setdefault('HF_HOME', str(HF_HOME))
            os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(HF_HOME / 'hub'))
            tmp_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(HF_HOME / 'hub'))
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

def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Ensure parent dir is writable
        dest.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # Try requests first, then huggingface_hub
    ok = _download_requests(url, dest)
    if ok:
        return
    ok = _download_hf(url, dest)
    if ok:
        return
    raise RuntimeError("all download methods failed")

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
        lp_dir = LP_DIR
        lp_dir.mkdir(parents=True, exist_ok=True)
        audit_path = lp_dir / '_download_audit.jsonl'
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
    if os.getenv('MIRAGE_DOWNLOAD_MODELS', '1').lower() not in ('1', 'true', 'yes', 'on'):
        print('[downloader] MIRAGE_DOWNLOAD_MODELS disabled')
        _audit('disabled')
        return False

    app_url = os.getenv('MIRAGE_LP_APPEARANCE_URL')
    motion_url = os.getenv('MIRAGE_LP_MOTION_URL')
    success = True
    _audit('start')
    
    # Download LivePortrait appearance extractor
    if app_url:
        dest = LP_DIR / 'appearance_feature_extractor.onnx'
        if not dest.exists():
            try:
                print(f'[downloader] Downloading appearance extractor...')
                with _FileLock(dest):
                    if not dest.exists():
                        _download(app_url, dest)
                converted = _maybe_convert_opset_to_19(dest)
                if converted != dest:
                    try:
                        shutil.copyfile(converted, dest)
                        print(f"[downloader] Replaced appearance with opset19: {converted.name}")
                    except Exception:
                        pass
                print(f'[downloader] ✅ Downloaded: {dest}')
                _audit('download_ok', model='appearance', path=str(dest))
            except Exception as e:
                print(f'[downloader] ❌ Failed to download appearance extractor: {e}')
                _audit('download_error', model='appearance', error=str(e))
                success = False
        else:
            converted = _maybe_convert_opset_to_19(dest)
            if converted != dest:
                try:
                    shutil.copyfile(converted, dest)
                    print(f"[downloader] Updated cached appearance to opset19")
                except Exception:
                    pass
            print(f'[downloader] ✅ Appearance extractor already exists: {dest}')
            _audit('exists', model='appearance', path=str(dest))
    
    # Download LivePortrait motion extractor
    if motion_url:
        dest = LP_DIR / 'motion_extractor.onnx'
        if not dest.exists():
            try:
                print(f'[downloader] Downloading motion extractor...')
                with _FileLock(dest):
                    if not dest.exists():
                        _download(motion_url, dest)
                converted = _maybe_convert_opset_to_19(dest)
                if converted != dest:
                    try:
                        shutil.copyfile(converted, dest)
                        print(f"[downloader] Replaced motion with opset19: {converted.name}")
                    except Exception:
                        pass
                print(f'[downloader] ✅ Downloaded: {dest}')
                _audit('download_ok', model='motion', path=str(dest))
            except Exception as e:
                print(f'[downloader] ❌ Failed to download motion extractor: {e}')
                _audit('download_error', model='motion', error=str(e))
                success = False
        else:
            converted = _maybe_convert_opset_to_19(dest)
            if converted != dest:
                try:
                    shutil.copyfile(converted, dest)
                    print(f"[downloader] Updated cached motion to opset19")
                except Exception:
                    pass
            print(f'[downloader] ✅ Motion extractor already exists: {dest}')
            _audit('exists', model='motion', path=str(dest))
    
    # Download additional models (generator required in neural-only mode)
    generator_url = os.getenv('MIRAGE_LP_GENERATOR_URL')
    if generator_url:
        dest = LP_DIR / 'generator.onnx'
        if not dest.exists():
            try:
                print(f'[downloader] Downloading generator model...')
                with _FileLock(dest):
                    if not dest.exists():
                        _download(generator_url, dest)
                if not _is_valid_onnx(dest):
                    print(f"[downloader] ❌ Generator ONNX validation failed for {generator_url}")
                    try:
                        dest.unlink()
                    except Exception:
                        pass
                    raise RuntimeError('generator download invalid')
                print(f'[downloader] ✅ Downloaded: {dest}')
                _audit('download_ok', model='generator', path=str(dest))
            except Exception as e:
                print(f'[downloader] ❌ Failed to download generator (required): {e}')
                _audit('download_error', model='generator', error=str(e))
                success = False
        else:
            if not _is_valid_onnx(dest):
                try:
                    print(f"[downloader] Existing generator is invalid, removing and retrying download")
                    dest.unlink()
                except Exception:
                    pass
                try:
                    print(f'[downloader] Downloading generator model...')
                    with _FileLock(dest):
                        if not dest.exists():
                            _download(generator_url, dest)
                    if not _is_valid_onnx(dest):
                        raise RuntimeError(f'generator invalid after re-download: {generator_url}')
                    print(f'[downloader] ✅ Downloaded: {dest}')
                    _audit('download_ok', model='generator', path=str(dest), refreshed=True)
                except Exception as e2:
                    print(f'[downloader] ❌ Failed to refresh invalid generator: {e2}')
                    _audit('download_error', model='generator', error=str(e2), refreshed=True)
                    success = False
            else:
                print(f'[downloader] ✅ Generator already exists: {dest}')
                _audit('exists', model='generator', path=str(dest))
    # Optional stitching model
    stitching_url = os.getenv('MIRAGE_LP_STITCHING_URL')
    if stitching_url:
        dest = LP_DIR / 'stitching.onnx'
        if not dest.exists():
            try:
                print(f'[downloader] Downloading stitching model...')
                _download(stitching_url, dest)
                print(f'[downloader] ✅ Downloaded: {dest}')
                _audit('download_ok', model='stitching', path=str(dest))
            except Exception as e:
                print(f'[downloader] ⚠️ Failed to download stitching (optional): {e}')
                _audit('download_error', model='stitching', error=str(e))
    
    # Optional custom ops plugin for GridSample 3D used by some generator variants
    grid_plugin_url = os.getenv('MIRAGE_LP_GRID_PLUGIN_URL')
    if grid_plugin_url:
        dest = LP_DIR / 'libgrid_sample_3d_plugin.so'
        if not dest.exists():
            try:
                print(f'[downloader] Downloading grid sample plugin...')
                _download(grid_plugin_url, dest)
                print(f'[downloader] ✅ Downloaded: {dest}')
                _audit('download_ok', model='grid_plugin', path=str(dest))
            except Exception as e:
                print(f'[downloader] ⚠️ Failed to download grid plugin (optional): {e}')
                _audit('download_error', model='grid_plugin', error=str(e))
    
    _audit('complete', success=success)
    return success


if __name__ == '__main__':
    """Direct execution for debugging"""
    print("=== LivePortrait Model Downloader ===")
    success = maybe_download()
    if success:
        print("✅ All required models downloaded successfully")
    else:
        print("❌ Some model downloads failed")
        sys.exit(1)
