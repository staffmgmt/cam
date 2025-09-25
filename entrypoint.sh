#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Starting Mirage container..."
echo "[entrypoint] Python: $(python3 --version 2>&1)"

#############################################
# Model persistence & provisioning strategy  #
#############################################
# We support persistent storage on Hugging Face Spaces by symlinking /app/models
# to /data/mirage_models (HF persists /data across restarts). Controlled by
# MIRAGE_PERSIST_MODELS (default=1). If disabled, we keep ephemeral /app/models.

PERSIST_DEFAULT=1
if [[ "${MIRAGE_PERSIST_MODELS:-}" =~ ^(0|false|no|off)$ ]]; then
  PERSIST_DEFAULT=0
fi

if [[ $PERSIST_DEFAULT -eq 1 ]]; then
  PERSIST_ROOT="/data/mirage_models"
  mkdir -p "$PERSIST_ROOT" || true
  # If /app/models exists and is not a symlink, migrate its contents (first run)
  if [[ -d /app/models && ! -L /app/models ]]; then
    # Only migrate if persistent dir is empty (avoid overwriting existing cache)
    if [[ -z "$(ls -A "$PERSIST_ROOT" 2>/dev/null)" ]]; then
      echo "[entrypoint] Migrating existing /app/models/* -> $PERSIST_ROOT (first persistent run)"
      shopt -s dotglob nullglob
      for f in /app/models/*; do
        mv "$f" "$PERSIST_ROOT/" 2>/dev/null || true
      done
      shopt -u dotglob nullglob
    fi
    rm -rf /app/models
    ln -s "$PERSIST_ROOT" /app/models
    echo "[entrypoint] Symlinked /app/models -> $PERSIST_ROOT"
  elif [[ -L /app/models ]]; then
    echo "[entrypoint] /app/models already symlinked"
  else
    # Create then symlink if missing
    rm -rf /app/models 2>/dev/null || true
    ln -s "$PERSIST_ROOT" /app/models
    echo "[entrypoint] Initialized persistent symlink /app/models -> $PERSIST_ROOT"
  fi
  STORAGE_MODE="persistent"
  STORAGE_PATH="$PERSIST_ROOT"
else
  mkdir -p /app/models || true
  STORAGE_MODE="ephemeral"
  STORAGE_PATH="/app/models"
fi

# Updated provisioning for face swap pipeline (InSwapper + optional CodeFormer)
MODEL_ROOT="/app/models"
SENTINEL="${MODEL_ROOT}/.provisioned"
SENTINEL_META="${MODEL_ROOT}/.provisioned_meta.json"
AUDIT_FILE="${MODEL_ROOT}/_download_audit.jsonl"
REQ_FILES=("inswapper/inswapper_128_fp16.onnx")
DL_TAG="${MIRAGE_DL_TAG:-startup}"

echo "[entrypoint] Storage mode: ${STORAGE_MODE} (root=${STORAGE_PATH})"
echo "[entrypoint] Ensuring model directory exists: ${MODEL_ROOT}"
mkdir -p "${MODEL_ROOT}" || true

# Function to validate required model presence
validate_models() {
  local missing=0
  for f in "${REQ_FILES[@]}"; do
    if [[ ! -f "${MODEL_ROOT}/$f" ]]; then
      missing=1
    fi
  done
  return $missing
}

# Decide whether to (re)download. Conditions forcing provisioning:
#  - Sentinel missing
#  - MIRAGE_PROVISION_FRESH set
#  - Sentinel exists but required files missing (stale sentinel)
#  - Sentinel meta hash mismatch (future extension; placeholder now)
should_download=0

SENTINEL_REASON="skip"
if [[ ! -f "${SENTINEL}" ]]; then
  should_download=1
  SENTINEL_REASON="no_sentinel"
elif [[ "${MIRAGE_PROVISION_FRESH:-0}" =~ ^(1|true|yes|on)$ ]]; then
  echo "[entrypoint] MIRAGE_PROVISION_FRESH set; forcing fresh provisioning"
  rm -f "${SENTINEL}" "${SENTINEL_META}" || true
  should_download=1
  SENTINEL_REASON="forced_fresh"
else
  if ! validate_models; then
    echo "[entrypoint] Sentinel present but required model(s) missing; will re-provision"
    should_download=1
    SENTINEL_REASON="stale_sentinel"
  fi
fi
if [[ $should_download -eq 0 ]]; then
  echo "[entrypoint] Sentinel OK; skipping download (reason=${SENTINEL_REASON})"
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  echo "{\"ts\":\"${ts}\",\"event\":\"skip_provision\",\"reason\":\"${SENTINEL_REASON}\",\"tag\":\"${DL_TAG}\"}" >> "${AUDIT_FILE}" 2>/dev/null || true
fi

if [[ $should_download -eq 1 ]]; then
  if [[ "${MIRAGE_DOWNLOAD_MODELS:-1}" =~ ^(1|true|TRUE|yes|on)$ ]]; then
    echo "[entrypoint] Running model downloader (provisioning)"
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"ts\":\"${ts}\",\"event\":\"start_provision\",\"tag\":\"${DL_TAG}\"}" >> "${AUDIT_FILE}" 2>/dev/null || true
    if python3 /app/model_downloader.py; then
      ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
      echo "{\"ts\":\"${ts}\",\"event\":\"provision_success\",\"tag\":\"${DL_TAG}\"}" >> "${AUDIT_FILE}" 2>/dev/null || true
      # Write sentinel + meta summary (sizes of required models)
      touch "${SENTINEL}" || true
      {
        echo '{'
        echo '  "ts": "'"${ts}"'",'
        echo '  "storage_mode": "'"${STORAGE_MODE}"'",'
        echo '  "required_models": {'
        comma=0
        for f in "${REQ_FILES[@]}"; do
          size=0
          if [[ -f "${MODEL_ROOT}/$f" ]]; then
            size=$(stat -c%s "${MODEL_ROOT}/$f" 2>/dev/null || wc -c <"${MODEL_ROOT}/$f")
          fi
          if [[ $comma -eq 1 ]]; then echo ','; fi
          printf '    "%s": {"size": %s}' "$f" "$size"
          comma=1
        done
        echo ''
        echo '  }'
        echo '}'
      } > "${SENTINEL_META}" 2>/dev/null || true
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

echo "[entrypoint] InsightFace analysis model cache (buffalo_l if present):"
ls -lh /app/.insightface/models/buffalo_l 2>/dev/null || echo "(buffalo_l directory not yet present)"

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
