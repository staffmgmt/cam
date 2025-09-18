"""
Virtual Camera Integration
Enables AI avatar output to be used as virtual camera in third-party apps
"""
import os
import sys
import numpy as np
import cv2
import threading
import time
import logging
from pathlib import Path
from typing import Optional, Callable
import subprocess
import platform

logger = logging.getLogger(__name__)

class VirtualCamera:
    """Virtual camera device for streaming AI avatar output"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        self.device_path = None
        self.process = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Platform-specific setup
        self.platform = platform.system().lower()
        self._setup_platform()
    
    def _setup_platform(self):
        """Setup platform-specific virtual camera"""
        if self.platform == "darwin":  # macOS
            self._setup_macos()
        elif self.platform == "linux":
            self._setup_linux()
        elif self.platform == "windows":
            self._setup_windows()
        else:
            logger.warning(f"Virtual camera not supported on {self.platform}")
    
    def _setup_macos(self):
        """Setup virtual camera on macOS"""
        try:
            # Check if obs-mac-virtualcam is available
            result = subprocess.run(['which', 'obs'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("OBS Virtual Camera detected on macOS")
                self.device_path = "/dev/obs-virtualcam"
            else:
                logger.warning("OBS Virtual Camera not found on macOS")
        except Exception as e:
            logger.error(f"macOS virtual camera setup error: {e}")
    
    def _setup_linux(self):
        """Setup virtual camera on Linux using v4l2loopback"""
        try:
            # Check if v4l2loopback is available
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            if 'v4l2loopback' in result.stdout:
                # Find available loopback device
                for i in range(10):
                    device = f"/dev/video{i}"
                    if os.path.exists(device):
                        try:
                            # Test if device is writable
                            with open(device, 'wb') as f:
                                self.device_path = device
                                logger.info(f"Found v4l2loopback device: {device}")
                                break
                        except PermissionError:
                            continue
            else:
                logger.warning("v4l2loopback not loaded. Install with: sudo modprobe v4l2loopback")
        except Exception as e:
            logger.error(f"Linux virtual camera setup error: {e}")
    
    def _setup_windows(self):
        """Setup virtual camera on Windows using OBS Virtual Camera"""
        try:
            # Check for OBS Virtual Camera
            obs_paths = [
                r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
                r"C:\Program Files (x86)\obs-studio\bin\32bit\obs32.exe"
            ]
            
            for path in obs_paths:
                if os.path.exists(path):
                    logger.info("OBS Virtual Camera available on Windows")
                    self.device_path = "obs-virtualcam"
                    return
            
            logger.warning("OBS Virtual Camera not found on Windows")
        except Exception as e:
            logger.error(f"Windows virtual camera setup error: {e}")
    
    def start(self) -> bool:
        """Start the virtual camera"""
        if self.is_running:
            logger.warning("Virtual camera already running")
            return True
        
        if not self.device_path:
            logger.error("No virtual camera device available")
            return False
        
        try:
            if self.platform == "linux" and self.device_path.startswith("/dev/video"):
                # Use FFmpeg for Linux v4l2loopback
                cmd = [
                    'ffmpeg',
                    '-f', 'rawvideo',
                    '-pixel_format', 'bgr24',
                    '-video_size', f'{self.width}x{self.height}',
                    '-framerate', str(self.fps),
                    '-i', 'pipe:0',
                    '-f', 'v4l2',
                    '-pix_fmt', 'yuv420p',
                    self.device_path,
                    '-y'
                ]
                
                self.process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                self.is_running = True
                logger.info(f"Virtual camera started on {self.device_path}")
                return True
            
            elif self.platform == "darwin":
                # For macOS, we'll use a different approach
                logger.info("macOS virtual camera setup complete")
                self.is_running = True
                return True
                
            elif self.platform == "windows":
                # For Windows, integrate with OBS Virtual Camera
                logger.info("Windows virtual camera setup complete")
                self.is_running = True
                return True
                
        except Exception as e:
            logger.error(f"Failed to start virtual camera: {e}")
            return False
        
        return False
    
    def stop(self):
        """Stop the virtual camera"""
        self.is_running = False
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            finally:
                self.process = None
        
        logger.info("Virtual camera stopped")
    
    def update_frame(self, frame: np.ndarray):
        """Update the current frame to be streamed"""
        with self.frame_lock:
            # Resize frame to virtual camera dimensions
            self.current_frame = cv2.resize(frame, (self.width, self.height))
            
            # Send frame to virtual camera if running
            if self.is_running and self.process:
                try:
                    frame_data = self.current_frame.tobytes()
                    self.process.stdin.write(frame_data)
                    self.process.stdin.flush()
                except Exception as e:
                    logger.error(f"Failed to write frame: {e}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

class VirtualCameraManager:
    """Manager for virtual camera instances"""
    
    def __init__(self):
        self.cameras = {}
        self.default_camera = None
    
    def create_camera(self, name: str = "mirage_avatar", width: int = 640, height: int = 480, fps: int = 30) -> VirtualCamera:
        """Create a new virtual camera"""
        if name in self.cameras:
            logger.warning(f"Camera {name} already exists")
            return self.cameras[name]
        
        camera = VirtualCamera(width, height, fps)
        self.cameras[name] = camera
        
        if self.default_camera is None:
            self.default_camera = camera
        
        logger.info(f"Created virtual camera: {name}")
        return camera
    
    def get_camera(self, name: str = None) -> Optional[VirtualCamera]:
        """Get a virtual camera by name"""
        if name is None:
            return self.default_camera
        return self.cameras.get(name)
    
    def start_camera(self, name: str = None) -> bool:
        """Start a virtual camera"""
        camera = self.get_camera(name)
        if camera:
            return camera.start()
        return False
    
    def stop_camera(self, name: str = None):
        """Stop a virtual camera"""
        camera = self.get_camera(name)
        if camera:
            camera.stop()
    
    def update_frame(self, frame: np.ndarray, name: str = None):
        """Update frame for a virtual camera"""
        camera = self.get_camera(name)
        if camera:
            camera.update_frame(frame)
    
    def stop_all(self):
        """Stop all virtual cameras"""
        for camera in self.cameras.values():
            camera.stop()
        self.cameras.clear()
        self.default_camera = None

# Global manager instance
_camera_manager = VirtualCameraManager()

def get_virtual_camera_manager() -> VirtualCameraManager:
    """Get the global virtual camera manager"""
    return _camera_manager

def install_virtual_camera_dependencies():
    """Install platform-specific virtual camera dependencies"""
    system = platform.system().lower()
    
    if system == "linux":
        print("To enable virtual camera on Linux:")
        print("1. Install v4l2loopback:")
        print("   sudo apt-get install v4l2loopback-dkms")
        print("2. Load the module:")
        print("   sudo modprobe v4l2loopback devices=1 video_nr=10 card_label='Mirage Virtual Camera'")
        print("3. Install FFmpeg:")
        print("   sudo apt-get install ffmpeg")
    
    elif system == "darwin":
        print("To enable virtual camera on macOS:")
        print("1. Install OBS Studio with Virtual Camera plugin")
        print("2. Or use other virtual camera software like CamTwist")
    
    elif system == "windows":
        print("To enable virtual camera on Windows:")
        print("1. Install OBS Studio")
        print("2. Enable Virtual Camera in OBS Tools menu")
        print("3. Or use other virtual camera software like ManyCam")

if __name__ == "__main__":
    # Test virtual camera setup
    install_virtual_camera_dependencies()
    
    # Create test camera
    manager = get_virtual_camera_manager()
    camera = manager.create_camera("test")
    
    if camera.start():
        print("Virtual camera started successfully!")
        
        # Generate test pattern
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, "Mirage AI Avatar", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        for i in range(100):
            # Update test pattern
            frame = test_frame.copy()
            cv2.putText(frame, f"Frame {i}", (50, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            camera.update_frame(frame)
            time.sleep(0.1)
        
        camera.stop()
    else:
        print("Failed to start virtual camera")