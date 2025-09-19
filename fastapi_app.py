#!/usr/bin/env python3
"""
Gradio interface for Mirage AI Avatar System
Wraps the existing FastAPI application for HuggingFace Spaces deployment
"""
import gradio as gr
import asyncio
import threading
import uvicorn
import time
import requests
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our existing app
from fastapi_app import app as fastapi_app

class MirageInterface:
    def __init__(self):
        self.server_port = 7860  # Gradio default port
        self.fastapi_port = 8000
        self.server_thread = None
        self.server_running = False
        
    def start_fastapi_server(self):
        """Start the FastAPI server in background"""
        try:
            uvicorn.run(
                fastapi_app, 
                host="0.0.0.0", 
                port=self.fastapi_port,
                log_level="info"
            )
        except Exception as e:
            print(f"FastAPI server error: {e}")
    
    def initialize_system(self):
        """Initialize the AI pipeline"""
        try:
            response = requests.post(f"http://localhost:{self.fastapi_port}/initialize")
            if response.status_code == 200:
                return "✅ AI Pipeline initialized successfully!"
            else:
                return f"❌ Initialization failed: {response.text}"
        except Exception as e:
            return f"❌ Connection error: {str(e)}"
    
    def upload_reference_image(self, image):
        """Upload reference image for avatar"""
        if image is None:
            return "❌ Please upload an image first"
            
        try:
            # Save uploaded image temporarily
            image_path = "/tmp/reference_image.jpg"
            image.save(image_path)
            
            with open(image_path, "rb") as f:
                files = {"file": f}
                response = requests.post(
                    f"http://localhost:{self.fastapi_port}/set_reference",
                    files=files
                )
            
            if response.status_code == 200:
                return "✅ Reference image uploaded successfully!"
            else:
                return f"❌ Upload failed: {response.text}"
        except Exception as e:
            return f"❌ Upload error: {str(e)}"
    
    def get_system_status(self):
        """Get current system status"""
        try:
            response = requests.get(f"http://localhost:{self.fastapi_port}/health")
            if response.status_code == 200:
                data = response.json()
                return f"🟢 System Status: {data.get('status', 'Unknown')}"
            else:
                return "🔴 System offline"
        except:
            return "🔴 Cannot connect to system"

def create_interface():
    """Create the Gradio interface"""
    mirage = MirageInterface()
    
    # Start FastAPI server in background thread
    server_thread = threading.Thread(target=mirage.start_fastapi_server, daemon=True)
    server_thread.start()
    
    # Wait a moment for server to start
    time.sleep(2)
    
    with gr.Blocks(
        title="Mirage AI Avatar System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        """
    ) as interface:
        
        gr.HTML('<h1 class="main-header">🎭 Mirage AI Avatar System</h1>')
        gr.Markdown("""
        **Real-time AI Avatar with Face Animation & Voice Conversion**
        
        Transform your appearance and voice in real-time for video calls. Built with LivePortrait and RVC.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📋 System Setup")
                
                init_btn = gr.Button("🚀 Initialize AI Pipeline", variant="primary")
                init_status = gr.Textbox(label="Initialization Status", interactive=False)
                
                gr.Markdown("## 🖼️ Reference Image")
                reference_image = gr.Image(
                    label="Upload your reference photo",
                    type="pil",
                    height=300
                )
                upload_btn = gr.Button("📤 Set Reference Image", variant="secondary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
            with gr.Column(scale=2):
                gr.Markdown("## 🎥 Live Avatar Interface")
                
                gr.HTML(f"""
                <iframe 
                    src="http://localhost:{mirage.fastapi_port}/" 
                    width="100%" 
                    height="600px" 
                    frameborder="0"
                    style="border-radius: 10px; border: 2px solid #ddd;">
                </iframe>
                """)
                
                status_btn = gr.Button("🔍 Check System Status")
                system_status = gr.Textbox(label="System Status", interactive=False)
        
        gr.Markdown("""
        ## 🎯 How to Use
        
        1. **Initialize**: Click "Initialize AI Pipeline" and wait for confirmation
        2. **Reference**: Upload a clear photo of the person you want to become
        3. **Setup**: Click "Set Reference Image" to configure your avatar
        4. **Go Live**: Use the interface above to start your camera and see your AI avatar!
        
        ## 🚀 Features
        
        - **Real-time Processing**: <250ms latency for smooth interaction
        - **Face Animation**: Powered by LivePortrait technology  
        - **Voice Conversion**: RVC-based voice transformation
        - **GPU Accelerated**: Optimized for NVIDIA A10G hardware
        - **Virtual Camera**: Ready for Zoom, Teams, Discord integration
        
        ## ⚙️ Technical Details
        
        - **Backend**: FastAPI with WebSocket streaming
        - **Models**: InsightFace + LivePortrait + RVC
        - **Hardware**: NVIDIA A10G GPU with CUDA 12.1
        - **Performance**: 20 FPS video, 160ms audio chunks
        """)
        
        # Event handlers
        init_btn.click(
            fn=mirage.initialize_system,
            outputs=init_status
        )
        
        upload_btn.click(
            fn=mirage.upload_reference_image,
            inputs=reference_image,
            outputs=upload_status
        )
        
        status_btn.click(
            fn=mirage.get_system_status,
            outputs=system_status
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    
    # Launch with public sharing enabled for HuggingFace Spaces
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # HF Spaces handles sharing
        show_error=True,
        quiet=False
    )