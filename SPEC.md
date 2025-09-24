## Architectural Reassessment (September 2025)

The initial implementation adopted a motion-driven portrait reenactment stack (LivePortrait ONNX models + custom alignment & smoothing) which is misaligned with the updated product goal: low-latency real-time face swapping with optional enhancement.

### Misalignment Summary

| Target Need | LivePortrait Path | Impact |
|-------------|-------------------|--------|
| Direct identity substitution | Motion reenactment of a canonicalized reference | Unnecessary motion keypoint pipeline |
| Minimal per-frame latency (<80ms) | ~500–600ms generator stages logged | Fails real-time threshold |
| Simple detector→swap flow | Multi-stage appearance + motion + generator | Complexity & fragile compositing |
| Artifact cleanup (optional) | No enhancement stage | Lower visual fidelity |
| Multi-face capability | Single-face canonical reenactment focus | Limits scalability |

### New Model Stack
1. Detector / embeddings: insightface FaceAnalysis (buffalo_l pack → SCRFD_10G_KPS + recognition)
2. Swapper: inswapper_128_fp16.onnx
3. Enhancement (optional):
  - CodeFormer (codeformer.pth) for fidelity‑controllable restoration

### New Processing Loop
1. Capture frame
2. Detect faces (FaceAnalysis)
3. For each target face (top-N): apply InSwapper with pre-extracted source identity
4. (Optional) Run CodeFormer enhancer on final composited frame (if weights present)
5. Emit frame to WebRTC

### Environment Variables (Video / Enhancer)
| Variable | Values | Description |
|----------|--------|-------------|
| MIRAGE_MAX_FACES | int (default 1) | Swap up to N largest faces |
| MIRAGE_CODEFORMER_FIDELITY | 0.0–1.0 (default 0.75) | Balance identity (1.0) vs reconstruction sharpness |
| MIRAGE_INSWAPPER_URL | URL | Override InSwapper model source |
| MIRAGE_CODEFORMER_URL | URL | Override CodeFormer model source |

### Deprecated / To Remove
liveportrait_engine.py, avatar_pipeline.py, alignment.py, smoothing.py, realtime_optimizer.py, virtual_camera.py (current unused), enhanced_metrics.py, landmark_reenactor.py, safe_model_integration.py, debug_mediapipe.py

These abstractions are reenactment-specific (appearance feature caching, keypoint smoothing, inverse warp compositing) and will be replaced by a concise `swap_pipeline.py`.

---
## Goals
- End-to-end audio latency < 250 ms (capture -> inference -> playback)
- Video pipeline: 512x512 @ ≥20 FPS target under load

## Hardware Target
- Phase 1–2: CPU basic (development + echo scaffolds)
- Later phases: Single NVIDIA A10G (24GB) for combined audio + video low-latency inference

## Voice Pipeline
| Item | Planned |
|------|---------|
| Framework | TODO |
| Content Encoder | TODO |
| F0 Extractor | RMVPE |
| Chunk Size | 192 ms |
| Sample Rate | 16 kHz |
| Precision | FP16 mixed |
| Overlap-Add | disabled |
| Accept Threshold | < 0.65 * chunk_ms ratio runtime/real |
| Fail Condition | > 0.80 * chunk_ms for 40 consecutive chunks |

## Video Pipeline
| Item | Planned |
|------|---------|
| Model | TODO |
| Detector | SCRFD |
| Detect Interval | 5 frames |
| Resolution | 512x512 |
| FPS Target | 20 |
| Confidence Threshold (stable) | ≥0.85 |
| Re-detect Threshold | <0.70 confidence triggers re-detect next frame |
| Quality Degrade Order | quality → fps → resolution |

## Transport
- WebSockets (bi-directional control + media chunks)
- Audio: PCM16, 16 kHz, mono frames (chunked ~192 ms) 
- Video: JPEG compressed frames (progressive baseline) initially

## Sync Strategy
- Audio clock is master timeline
- Drop late video frames if video timestamp >150 ms behind audio head
- Never let audio lead video by more than 150 ms (else request video degrade)
- If sustained drift (>120 frames) re-align by soft skip (video) not audio stretch

## Metrics
Planned collection (no aggregation service yet):
- Audio chunks processed (count, accept ratio)
- Per-stage timings: encode / f0 / convert / post
- Video frames processed & dropped
- Detector invoke interval stats
- GPU memory (allocated / reserved / fragmentation indicator)
- Queue backlog lengths (audio, video, outbound)

## Adaptation Rules
- Increase audio chunk size if runtime/real ratio >0.75 for last 40 chunks
- Decrease audio chunk size if runtime/real ratio <0.55 for last 120 chunks
- Video degrade order:
  1. Lower quality (JPEG quality step down)
  2. Reduce FPS toward 15
  3. Reduce resolution (512 → 384 → 256)
- Restore in reverse order after 300 stable frames (<0.60 ratio)

## Security
- Planned: JWT token required for WebSocket upgrade
- Rate limiting (connection + message frequency)
- Frame size guard (reject > configured max bytes)
- Basic anomaly detection: abandon session if >30% frames invalid over 200 frame window

## Licensing
- Current: MIT
- Future: Add `LICENSES.md` enumerating third-party components and model licenses

## Phases
| Phase | Status |
|-------|--------|
| 1 | Completed (Echo scaffold, static client) |
| 2 | Completed (Metrics + config + voice stub + GPU info) |
| 3 | In Progress (governance + groundwork for adaptation) |
| 4 | Pending |
| 5 | Pending |
| 6 | Pending |
| 7 | Pending |
| 8 | Pending |
| 9 | Pending |
| 10 | Pending |

## Open Questions
- RVC fork URL to adopt / baseline? (Need candidate repo)
- Reenact / face animation model repo selection?
- Alignment expectation: do we need phoneme-level alignment or chunk-level only?

## Face Detector Standardization
Adopt SCRFD as unified face detector. Run detection every 5 frames; reuse boxes otherwise. Force immediate re-detect if last confidence <0.70. Consider face track stable when confidence ≥0.85 for 3 consecutive detections. This reduces detector load while maintaining robustness against drift and occlusion.
