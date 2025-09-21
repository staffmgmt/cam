"""
Optional model downloader for deterministic builds.
- Controlled with env MIRAGE_DOWNLOAD_MODELS=1
- LivePortrait ONNX URLs controlled via env:
    * MIRAGE_LP_APPEARANCE_URL
    * MIRAGE_LP_MOTION_URL (optional)
- InsightFace models will still use the package cache; SCRFD will populate on first run.

More robust with retries and alternative download methods (requests, huggingface_hub).
"""
import os
import sys
import shutil
from pathlib import Path
import time
from typing import Optional

try:
        import requests  # type: ignore
except Exception:
        requests = None

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
    tmp = dest.with_suffix(dest.suffix + '.part')
    for attempt in range(1, retries + 1):
        try:
            print(f"[downloader] (requests) GET {url} -> {dest} (attempt {attempt}/{retries})")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            tmp.replace(dest)
            return True
        except Exception as e:
            print(f"[downloader] requests error: {e}")
            time.sleep(min(2 * attempt, 6))
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


def maybe_download() -> bool:
    if os.getenv('MIRAGE_DOWNLOAD_MODELS', '0').lower() not in ('1', 'true', 'yes', 'on'):
        print('[downloader] MIRAGE_DOWNLOAD_MODELS disabled')
        return False
    app_url = os.getenv('MIRAGE_LP_APPEARANCE_URL')
    motion_url = os.getenv('MIRAGE_LP_MOTION_URL')
    success = True
    if app_url:
        try:
            _download(app_url, LP_DIR / 'appearance_feature_extractor.onnx')
        except Exception as e:
            print(f"[downloader] appearance download failed: {e}")
            success = False
    if motion_url:
        try:
            _download(motion_url, LP_DIR / 'motion_extractor.onnx')
        except Exception as e:
            print(f"[downloader] motion download failed: {e}")
            # motion is optional; don't flip success
    return success


if __name__ == '__main__':
    ok = maybe_download()
    print(f'[downloader] Done (ok={ok})')
