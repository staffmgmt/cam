#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Starting Mirage container..."
echo "[entrypoint] Python: $(python3 --version 2>&1)"

# Updated provisioning for face swap pipeline (InSwapper + optional CodeFormer)
MODEL_ROOT="/app/models"
SENTINEL="${MODEL_ROOT}/.provisioned"
AUDIT_FILE="${MODEL_ROOT}/_download_audit.jsonl"
REQ_FILES=("inswapper/inswapper_128_fp16.onnx")
DL_TAG="${MIRAGE_DL_TAG:-startup}"

echo "[entrypoint] Ensuring model directory exists: ${MODEL_ROOT}"
mkdir -p "${MODEL_ROOT}"

should_download=0
if [[ ! -f "${SENTINEL}" ]]; then
  should_download=1
  echo "[entrypoint] Sentinel missing; provisioning required"
else
  if [[ "${MIRAGE_PROVISION_FRESH:-0}" =~ ^(1|true|yes|on)$ ]]; then
    echo "[entrypoint] MIRAGE_PROVISION_FRESH set; forcing fresh provisioning"
    rm -f "${SENTINEL}" || true
    should_download=1
  else
    echo "[entrypoint] Sentinel present; skipping download (idempotent)"
    # Audit skip
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"ts\":\"${ts}\",\"event\":\"skip_provision\",\"tag\":\"${DL_TAG}\"}" >> "${AUDIT_FILE}" 2>/dev/null || true
  fi
fi

if [[ $should_download -eq 1 ]]; then
  if [[ "${MIRAGE_DOWNLOAD_MODELS:-1}" =~ ^(1|true|TRUE|yes|on)$ ]]; then
    echo "[entrypoint] Running model downloader (provisioning)"
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"ts\":\"${ts}\",\"event\":\"start_provision\",\"tag\":\"${DL_TAG}\"}" >> "${AUDIT_FILE}" 2>/dev/null || true
    if python3 /app/model_downloader.py; then
      ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
      echo "{\"ts\":\"${ts}\",\"event\":\"provision_success\",\"tag\":\"${DL_TAG}\"}" >> "${AUDIT_FILE}" 2>/dev/null || true
      touch "${SENTINEL}" || true
    else
      ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
      echo "{\"ts\":\"${ts}\",\"event\":\"provision_error\",\"tag\":\"${DL_TAG}\"}" >> "${AUDIT_FILE}" 2>/dev/null || true
      echo "[entrypoint] Downloader reported issues (continuing)"
    fi
  else
    echo "[entrypoint] Skipping model download (MIRAGE_DOWNLOAD_MODELS=${MIRAGE_DOWNLOAD_MODELS:-unset})"
  fi
fi

echo "[entrypoint] Model directory contents after download attempt:"
ls -lh "${MODEL_ROOT}" || true
echo "[entrypoint] InSwapper dir contents:"
ls -lh "${MODEL_ROOT}/inswapper" 2>/dev/null || true
echo "[entrypoint] CodeFormer dir contents:"
ls -lh "${MODEL_ROOT}/codeformer" 2>/dev/null || true

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
  path="${MODEL_ROOT}/$f"
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
  echo "[entrypoint] FATAL: Required model(s) missing (InSwapper). Cannot continue." >&2
  exit 1
fi

echo "[entrypoint] Launching uvicorn..."
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-7860}" --no-server-header
