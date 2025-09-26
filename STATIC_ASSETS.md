# Static Asset Inventory

This document enumerates all files under `static/` after removal of the legacy `webrtc_prod.js` and clarifies their current status.

| File | Purpose | Loaded By | Active Use | Notes |
|------|---------|-----------|------------|-------|
| `index.html` | Main application UI (enterprise-grade layout) | FastAPI root (`/`) serves file contents | Yes | References only `webrtc_enterprise.js` | 
| `webrtc_enterprise.js` | Primary WebRTC + UI control client (camera, datachannel, metrics, TURN retry) | `<script src="/static/webrtc_enterprise.js">` in `index.html` | Yes | Canonical implementation; supersedes older scripts |
| `worklet.js` | AudioWorklet processors (`pcm-chunker`, `pcm-player`) for deprecated WebSocket pipeline | Historically loaded by `app.js` | Legacy / Potentially removable | Not referenced by `index.html`; keep temporarily in case of fallback testing |
| `app.js` | Deprecated WebSocket streaming client (pre-WebRTC) | Not referenced | Legacy / Safe to remove later | Contains only legacy code; flagged as deprecated in header comment |
| `webrtc_client.js` | Placeholder legacy bootstrap (empty IIFE) | Not referenced | Legacy / Remove | Safe to delete in a future cleanup PR |

## Summary

Only two files are required for the production UI: `index.html` and `webrtc_enterprise.js`.

Recommended cleanup (next steps):
1. Remove `app.js` and `webrtc_client.js` after confirming no external bookmarks / embeddings rely on them.
2. If no audio worklet fallback path is desired, remove `worklet.js` and excise related dead code from `app.js`.
3. Add a minimal regression test ensuring `/` returns HTML containing `webrtc_enterprise.js` to prevent accidental script reference regressions.

## Rationale for Keeping Some Legacy Files (Temporarily)

`worklet.js` is retained for a short deprecation window in case future experimentation with pure WebSocket audio is required. Once WebRTC audio processing is fully stable and no fallbacks are needed, it can be removed.

## Enforcement Suggestion

Introduce a CI lint step that fails if new unreferenced large JS bundles appear in `static/` without being documented here. This keeps the static surface lean and auditable.
