"""
Testing and Validation Suite for Mirage AI Avatar System
Tests end-to-end functionality, latency, and performance
"""
import asyncio
import time
import aiohttp
import json
import numpy as np
import cv2
import logging
from pathlib import Path
import subprocess
import psutil
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MirageSystemTester:
    """Comprehensive testing suite for the AI avatar system"""
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self.session = None
        self.test_results = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_endpoint(self) -> bool:
        """Test basic health endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                data = await response.json()
                
                success = (
                    response.status == 200 and
                    data.get("status") == "ok" and
                    data.get("system") == "real-time-ai-avatar"
                )
                
                self.test_results["health"] = {
                    "success": success,
                    "status": response.status,
                    "data": data
                }
                
                logger.info(f"Health check: {'✅ PASS' if success else '❌ FAIL'}")
                return success
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.test_results["health"] = {"success": False, "error": str(e)}
            return False
    
    async def test_pipeline_initialization(self) -> bool:
        """Test AI pipeline initialization"""
        try:
            start_time = time.time()
            async with self.session.post(f"{self.base_url}/initialize") as response:
                data = await response.json()
                init_time = time.time() - start_time
                
                success = (
                    response.status == 200 and
                    data.get("status") in ["success", "already_initialized"]
                )
                
                self.test_results["initialization"] = {
                    "success": success,
                    "status": response.status,
                    "data": data,
                    "init_time_seconds": init_time
                }
                
                logger.info(f"Pipeline init: {'✅ PASS' if success else '❌ FAIL'} ({init_time:.1f}s)")
                return success
                
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            self.test_results["initialization"] = {"success": False, "error": str(e)}
            return False
    
    async def test_reference_image_upload(self) -> bool:
        """Test reference image upload functionality"""
        try:
            # Create a test image
            test_image = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.circle(test_image, (256, 200), 50, (255, 255, 255), -1)  # Face-like circle
            cv2.circle(test_image, (230, 180), 10, (0, 0, 0), -1)  # Eye
            cv2.circle(test_image, (280, 180), 10, (0, 0, 0), -1)  # Eye
            cv2.ellipse(test_image, (256, 220), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
            
            # Encode as JPEG
            _, encoded = cv2.imencode('.jpg', test_image)
            image_data = encoded.tobytes()
            
            # Upload test image
            form_data = aiohttp.FormData()
            form_data.add_field('file', image_data, filename='test_face.jpg', content_type='image/jpeg')
            
            async with self.session.post(f"{self.base_url}/set_reference", data=form_data) as response:
                data = await response.json()
                
                success = (
                    response.status == 200 and
                    data.get("status") == "success"
                )
                
                self.test_results["reference_upload"] = {
                    "success": success,
                    "status": response.status,
                    "data": data
                }
                
                logger.info(f"Reference upload: {'✅ PASS' if success else '❌ FAIL'}")
                return success
                
        except Exception as e:
            logger.error(f"Reference image upload failed: {e}")
            self.test_results["reference_upload"] = {"success": False, "error": str(e)}
            return False
    
    async def test_websocket_connections(self) -> bool:
        """Test WebSocket connections for audio and video"""
        try:
            import websockets
            
            # Test audio WebSocket
            audio_success = await self._test_websocket_endpoint("/audio")
            
            # Test video WebSocket
            video_success = await self._test_websocket_endpoint("/video")
            
            success = audio_success and video_success
            
            self.test_results["websockets"] = {
                "success": success,
                "audio_success": audio_success,
                "video_success": video_success
            }
            
            logger.info(f"WebSocket connections: {'✅ PASS' if success else '❌ FAIL'}")
            return success
            
        except Exception as e:
            logger.error(f"WebSocket test failed: {e}")
            self.test_results["websockets"] = {"success": False, "error": str(e)}
            return False
    
    async def _test_websocket_endpoint(self, endpoint: str) -> bool:
        """Test a specific WebSocket endpoint"""
        try:
            import websockets
            
            ws_url = self.base_url.replace("http://", "ws://") + endpoint
            
            async with websockets.connect(ws_url) as websocket:
                # Send test data
                if endpoint == "/audio":
                    # Send 160ms of silence (16kHz, 16-bit)
                    test_audio = np.zeros(int(16000 * 0.160), dtype=np.int16)
                    await websocket.send(test_audio.tobytes())
                else:  # video
                    # Send a small test JPEG
                    test_frame = np.zeros((256, 256, 3), dtype=np.uint8)
                    _, encoded = cv2.imencode('.jpg', test_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    await websocket.send(encoded.tobytes())
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                return len(response) > 0
                
        except Exception as e:
            logger.error(f"WebSocket {endpoint} test failed: {e}")
            return False
    
    async def test_performance_metrics(self) -> bool:
        """Test performance metrics endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/pipeline_status") as response:
                data = await response.json()
                
                success = response.status == 200 and data.get("initialized", False)
                
                self.test_results["performance_metrics"] = {
                    "success": success,
                    "status": response.status,
                    "data": data
                }
                
                if success:
                    stats = data.get("stats", {})
                    logger.info(f"Performance metrics: ✅ PASS")
                    logger.info(f"  GPU Memory: {stats.get('gpu_memory_used', 0):.1f} GB")
                    logger.info(f"  Video FPS: {stats.get('video_fps', 0):.1f}")
                    logger.info(f"  Avg Latency: {stats.get('avg_video_latency_ms', 0):.1f} ms")
                else:
                    logger.info("Performance metrics: ❌ FAIL")
                
                return success
                
        except Exception as e:
            logger.error(f"Performance metrics test failed: {e}")
            self.test_results["performance_metrics"] = {"success": False, "error": str(e)}
            return False
    
    async def test_latency_benchmark(self) -> Dict[str, float]:
        """Benchmark system latency"""
        latencies = []
        
        try:
            # Warm up
            for _ in range(5):
                start_time = time.time()
                async with self.session.get(f"{self.base_url}/health") as response:
                    await response.json()
                latencies.append((time.time() - start_time) * 1000)
            
            # Actual benchmark
            latencies = []
            for _ in range(20):
                start_time = time.time()
                async with self.session.get(f"{self.base_url}/pipeline_status") as response:
                    await response.json()
                latencies.append((time.time() - start_time) * 1000)
            
            results = {
                "avg_latency_ms": np.mean(latencies),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99)
            }
            
            self.test_results["latency_benchmark"] = results
            
            logger.info("Latency benchmark results:")
            logger.info(f"  Average: {results['avg_latency_ms']:.1f} ms")
            logger.info(f"  P95: {results['p95_latency_ms']:.1f} ms")
            logger.info(f"  P99: {results['p99_latency_ms']:.1f} ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Latency benchmark failed: {e}")
            return {}
    
    def test_system_requirements(self) -> Dict[str, Any]:
        """Test system requirements and capabilities"""
        results = {}
        
        try:
            # Check GPU availability
            try:
                import torch
                results["gpu_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    results["gpu_name"] = torch.cuda.get_device_name(0)
                    results["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    results["cuda_version"] = torch.version.cuda
            except ImportError:
                results["gpu_available"] = False
            
            # Check system resources
            memory = psutil.virtual_memory()
            results["system_memory_gb"] = memory.total / 1024**3
            results["cpu_count"] = psutil.cpu_count()
            
            # Check disk space
            disk = psutil.disk_usage('/')
            results["disk_free_gb"] = disk.free / 1024**3
            
            # Check required packages
            required_packages = [
                "torch", "torchvision", "torchaudio", "opencv-python", 
                "numpy", "fastapi", "websockets"
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package.replace("-", "_"))
                except ImportError:
                    missing_packages.append(package)
            
            results["missing_packages"] = missing_packages
            results["requirements_met"] = len(missing_packages) == 0
            
            self.test_results["system_requirements"] = results
            
            logger.info("System requirements:")
            logger.info(f"  GPU: {'✅' if results['gpu_available'] else '❌'}")
            logger.info(f"  Memory: {results['system_memory_gb']:.1f} GB")
            logger.info(f"  CPU: {results['cpu_count']} cores")
            logger.info(f"  Packages: {'✅' if results['requirements_met'] else '❌'}")
            
            return results
            
        except Exception as e:
            logger.error(f"System requirements check failed: {e}")
            return {"error": str(e)}
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        logger.info("🧪 Starting comprehensive system test...")
        
        # System requirements (runs first, no server needed)
        self.test_system_requirements()
        
        # Server-dependent tests
        tests = [
            ("Health Check", self.test_health_endpoint()),
            ("Pipeline Initialization", self.test_pipeline_initialization()),
            ("Reference Image Upload", self.test_reference_image_upload()),
            ("WebSocket Connections", self.test_websocket_connections()),
            ("Performance Metrics", self.test_performance_metrics()),
        ]
        
        # Run tests sequentially
        for test_name, test_coro in tests:
            logger.info(f"Running: {test_name}...")
            try:
                result = await test_coro
                if not result:
                    logger.warning(f"{test_name} failed - may affect subsequent tests")
            except Exception as e:
                logger.error(f"{test_name} threw exception: {e}")
        
        # Latency benchmark (runs last)
        logger.info("Running latency benchmark...")
        await self.test_latency_benchmark()
        
        # Calculate overall success rate
        successful_tests = sum(1 for result in self.test_results.values() 
                             if isinstance(result, dict) and result.get("success", False))
        total_tests = len([r for r in self.test_results.values() if isinstance(r, dict) and "success" in r])
        
        overall_success = successful_tests / max(total_tests, 1) >= 0.8  # 80% success rate
        
        summary = {
            "overall_success": overall_success,
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "success_rate": successful_tests / max(total_tests, 1),
            "detailed_results": self.test_results
        }
        
        logger.info(f"🏁 Test completed: {successful_tests}/{total_tests} tests passed")
        logger.info(f"Overall result: {'✅ PASS' if overall_success else '❌ FAIL'}")
        
        return summary

async def main():
    """Main test runner"""
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    
    async with MirageSystemTester(base_url) as tester:
        results = await tester.run_comprehensive_test()
        
        # Save results to file
        results_file = Path("test_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Detailed results saved to: {results_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results["overall_success"] else 1)

if __name__ == "__main__":
    asyncio.run(main())