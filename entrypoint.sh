#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Starting Mirage container..."
echo "[entrypoint] Python: $(python3 --version 2>&1)"

MODEL_DIR="/app/models/liveportrait"
REQ_FILES=("appearance_feature_extractor.onnx" "motion_extractor.onnx" "generator.onnx")

echo "[entrypoint] Ensuring model directory exists: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

if [[ "${MIRAGE_DOWNLOAD_MODELS:-1}" =~ ^(1|true|TRUE|yes|on)$ ]]; then
  echo "[entrypoint] Running model downloader (env enabled)"
  python3 /app/model_downloader.py || echo "[entrypoint] Downloader reported issues (continuing)"
else
  echo "[entrypoint] Skipping model download (MIRAGE_DOWNLOAD_MODELS=${MIRAGE_DOWNLOAD_MODELS:-unset})"
fi

echo "[entrypoint] Model directory contents after download attempt:"
ls -lh "${MODEL_DIR}" || true

# Minimal SHA256 function (skip if file huge and hashing disabled)
hash_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then return 0; fi
  if [[ "${MIRAGE_HASH_MODELS:-1}" =~ ^(0|false|no)$ ]]; then
    echo "(disabled)"
  else
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "$f" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
      shasum -a 256 "$f" | awk '{print $1}'
    else
      echo "(sha256 tool missing)"
    fi
  fi
}

MISSING=0
for f in "${REQ_FILES[@]}"; do
  path="${MODEL_DIR}/$f"
  if [[ -f "$path" ]]; then
    size=$(stat -c%s "$path" 2>/dev/null || wc -c <"$path")
    hash=$(hash_file "$path")
    echo "[entrypoint] ✅ $f size=${size}B sha256=${hash}"
  else
    echo "[entrypoint] ❌ Missing $f"
    MISSING=1
  fi
done

if [[ $MISSING -eq 1 ]]; then
  if [[ ! "${MIRAGE_ALLOW_WARP_FALLBACK:-0}" =~ ^(1|true|yes|on)$ ]]; then
    echo "[entrypoint] FATAL: Required model(s) missing and fallback not permitted. Set MIRAGE_ALLOW_WARP_FALLBACK=1 to bypass." >&2
    exit 1
  else
    echo "[entrypoint] Proceeding with warp fallback (generator may be absent)." >&2
  fi
fi

echo "[entrypoint] Launching uvicorn..."
exec uvicorn original_fastapi_app:app --host 0.0.0.0 --port "${PORT:-7860}" --no-server-header
