#!/usr/bin/env python3
"""
Debug MediaPipe installation and avatar pipeline status
"""

def check_mediapipe():
    """Check if MediaPipe is available and working"""
    print("=== MediaPipe Availability Check ===")
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
        print(f"   Version: {mp.__version__}")
        
        # Try to initialize face landmarks
        try:
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True
            )
            print("✅ FaceMesh initialized successfully")
            face_mesh.close()
        except Exception as e:
            print(f"❌ FaceMesh initialization failed: {e}")
        
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        print("   → Run: pip install mediapipe==0.10.7")
        return False
    
    return True

def check_avatar_pipeline():
    """Check avatar pipeline components"""
    print("\n=== Avatar Pipeline Check ===")
    
    try:
        from avatar_pipeline import get_pipeline
        pipeline = get_pipeline()
        print("✅ Avatar pipeline imported")
        
        # Check if landmark reenactor is available
        if hasattr(pipeline, 'landmark_reenactor') and pipeline.landmark_reenactor is not None:
            print("✅ Landmark reenactor available")
        else:
            print("⚠️  Landmark reenactor not available (MediaPipe not installed?)")
        
        # Check LivePortrait components
        if hasattr(pipeline, 'liveportrait') and pipeline.liveportrait is not None:
            print("✅ LivePortrait components available") 
        else:
            print("⚠️  LivePortrait components not loaded yet")
            
    except Exception as e:
        print(f"❌ Avatar pipeline error: {e}")
        return False
    
    return True

def check_webrtc_status():
    """Check WebRTC server status"""
    print("\n=== WebRTC Status Check ===")
    
    try:
        import requests
        import os
        
        # Check if running locally
        port = os.environ.get('PORT', '7860')
        base_url = f"http://localhost:{port}"
        
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            health = resp.json()
            print("✅ Server is running")
            print(f"   Pipeline loaded: {health.get('pipeline_loaded', False)}")
            print(f"   GPU available: {health.get('gpu_available', False)}")
            print(f"   WebRTC loaded: {health.get('webrtc_router_loaded', False)}")
            
            if health.get('webrtc_import_error'):
                print(f"   WebRTC error: {health['webrtc_import_error']}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Server not running locally")
            print(f"   Expected at {base_url}")
            
    except Exception as e:
        print(f"❌ Status check error: {e}")

if __name__ == "__main__":
    print("🔍 Mirage Debug Tool\n")
    
    mp_ok = check_mediapipe()
    pipeline_ok = check_avatar_pipeline() 
    check_webrtc_status()
    
    print("\n=== Summary ===")
    if mp_ok and pipeline_ok:
        print("✅ All components should work")
        print("   → Try uploading a reference image to see avatar effect")
    else:
        print("⚠️  Some components missing")
        print("   → Simple alpha-blend fallback should still work")
        print("   → For full features, rebuild Space after adding MediaPipe to requirements.txt")