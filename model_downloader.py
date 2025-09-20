"""
Optional model downloader for deterministic builds.
- Controlled with env MIRAGE_DOWNLOAD_MODELS=1
- LivePortrait ONNX URLs controlled via env:
  * MIRAGE_LP_APPEARANCE_URL
  * MIRAGE_LP_MOTION_URL (optional)
- InsightFace models will still use the package cache; SCRFD will populate on first run.
"""
import os
import sys
import shutil
from pathlib import Path
import urllib.request

LP_DIR = Path(__file__).parent / 'models' / 'liveportrait'


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    print(f"[downloader] Fetching {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def maybe_download():
    if os.getenv('MIRAGE_DOWNLOAD_MODELS', '0').lower() not in ('1', 'true', 'yes', 'on'):
        return
    app_url = os.getenv('MIRAGE_LP_APPEARANCE_URL')
    motion_url = os.getenv('MIRAGE_LP_MOTION_URL')
    if app_url:
        _download(app_url, LP_DIR / 'appearance_feature_extractor.onnx')
    if motion_url:
        _download(motion_url, LP_DIR / 'motion_extractor.onnx')


if __name__ == '__main__':
    maybe_download()
    print('[downloader] Done')
