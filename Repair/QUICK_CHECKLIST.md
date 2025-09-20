# 🚀 QUICK ACTION CHECKLIST

## Immediate Actions (Today - 30 minutes)

### ✅ Step 1: Add Enhanced Metrics (5 minutes)
```python
# Copy enhanced_metrics.py to your project
# In your existing avatar_pipeline.py get_performance_stats():
from enhanced_metrics import enhance_existing_stats
return enhance_existing_stats(base_stats)
```

### ✅ Step 2: Add Safe Model Integration (10 minutes)  
```python
# Copy safe_model_integration.py to your project
# In your existing avatar_pipeline.py __init__():
from safe_model_integration import get_safe_model_loader
self.safe_loader = get_safe_model_loader()

# In your existing initialize():
await self.safe_loader.safe_load_scrfd()
await self.safe_loader.safe_load_liveportrait()

# In your process_video_frame():
bbox = self.safe_loader.safe_detect_face(frame)
if self.reference_frame is not None:
    result = self.safe_loader.safe_animate_face(self.reference_frame, frame)
else:
    result = frame
```

### ✅ Step 3: Test with Features Disabled (5 minutes)
```bash
export MIRAGE_ENABLE_SCRFD=0
export MIRAGE_ENABLE_LIVEPORTRAIT=0
# Verify system works exactly as before
curl /health && curl /metrics
```

### ✅ Step 4: Enable SCRFD Gradually (5 minutes)
```bash
export MIRAGE_ENABLE_SCRFD=1
# Test face detection
curl -X POST /initialize
curl /metrics  # Check for face detection timing
```

### ✅ Step 5: Enable LivePortrait (5 minutes)
```bash  
export MIRAGE_ENABLE_LIVEPORTRAIT=1
# Test animation
curl /metrics  # Check for animation timing
```

## Success Indicators

- [ ] Enhanced metrics show P50/P95 latency percentiles
- [ ] SCRFD=1 enables face detection, fallback works on errors
- [ ] LIVEPORTRAIT=1 enables animation, fallback works on errors
- [ ] System maintains existing pass-through behavior
- [ ] /health endpoint shows models_loaded status
- [ ] Token auth and message schemas unchanged

## Instant Rollback
```bash
export MIRAGE_ENABLE_SCRFD=0
export MIRAGE_ENABLE_LIVEPORTRAIT=0
# Returns to exact previous behavior
```

## Files to Copy
- [x] `safe_model_integration.py` → Your project root
- [x] `enhanced_metrics.py` → Your project root  
- [x] `scripts/optional_download_models.py` → Your scripts/ folder
- [x] `webrtc_connection_monitoring.py` → Optional for /webrtc/connections

**🎯 Total Time: 30 minutes for safe AI integration**
