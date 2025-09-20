# Incremental Model Integration Guide

## Respecting Your Existing Architecture

Your team made excellent decisions to avoid wholesale replacement. Here's how to safely integrate AI models:

## Phase 1: Add Feature Flags (Zero Risk)

Add to your environment or startup:
```bash
# Start with models disabled
export MIRAGE_ENABLE_SCRFD=0
export MIRAGE_ENABLE_LIVEPORTRAIT=0

# Enable gradually
export MIRAGE_ENABLE_SCRFD=1        # Enable face detection first
export MIRAGE_ENABLE_LIVEPORTRAIT=1 # Enable animation second
```

## Phase 2: Integrate Safe Model Loader

In your existing `avatar_pipeline.py`, add:
```python
# At the top
from safe_model_integration import get_safe_model_loader

class RealTimeAvatarPipeline:
    def __init__(self):
        # Your existing code...

        # Add safe model loader
        self.safe_loader = get_safe_model_loader()

    async def initialize(self):
        # Your existing initialization...

        # Add safe model loading
        await self.safe_loader.safe_load_scrfd()
        await self.safe_loader.safe_load_liveportrait()

    def process_video_frame(self, frame, frame_idx):
        # Your existing code...

        # Enhanced face detection (graceful fallback)
        bbox = self.safe_loader.safe_detect_face(frame)

        # Enhanced animation (graceful fallback to pass-through)
        if self.reference_frame is not None:
            result = self.safe_loader.safe_animate_face(self.reference_frame, frame)
        else:
            result = frame  # Keep existing pass-through logic

        return result
```

## Phase 3: Enhanced Metrics (Drop-in)

In your existing `get_performance_stats()`:
```python
from enhanced_metrics import enhance_existing_stats

def get_performance_stats(self):
    # Your existing stats collection...
    base_stats = {
        "models_loaded": self.loaded,
        # ... your existing metrics
    }

    # Enhance with percentiles
    return enhance_existing_stats(base_stats)
```

## Phase 4: Optional Model Download

When you want models:
```bash
# Check what's needed
python3 scripts/optional_download_models.py --status

# Download only when features are enabled
MIRAGE_ENABLE_SCRFD=1 python3 scripts/optional_download_models.py --download-needed
```

## Phase 5: WebRTC Monitoring (Optional)

In your existing `webrtc_server.py`:
```python
from webrtc_connection_monitoring import add_connection_monitoring

# After creating your router
add_connection_monitoring(router, _peer_state)
```

## Validation Steps

1. **Feature Flags Off**: System works exactly as before
2. **SCRFD Enabled**: Face detection works, falls back gracefully
3. **LivePortrait Enabled**: Animation works, falls back to pass-through
4. **Metrics Enhanced**: More detailed latency tracking
5. **Models Optional**: Download only when needed

## Rollback Strategy

At any point:
```bash
# Disable all features
export MIRAGE_ENABLE_SCRFD=0
export MIRAGE_ENABLE_LIVEPORTRAIT=0

# System returns to existing pass-through behavior
```

This approach:
- ✅ Keeps your token auth intact
- ✅ Preserves existing WebRTC message schema  
- ✅ Maintains Docker compatibility
- ✅ Allows gradual rollout with instant rollback
- ✅ No background tasks at import time
- ✅ Compatible with your A10G + CUDA 12.1 setup
