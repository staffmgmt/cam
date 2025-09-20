#!/usr/bin/env python3
"""
Optional Model Downloader - On-demand only
Safe utility for pre-downloading models when needed
Does NOT run automatically in Docker build
"""

import os
import sys
import logging
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionalModelDownloader:
    """Optional model downloader for on-demand use"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)

        # Conservative model list - only what we actually need
        self.available_models = {
            "scrfd": {
                "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.zip",
                "dir": self.models_dir / "scrfd",
                "description": "SCRFD face detection model",
                "size_mb": 17,
                "required_by": "MIRAGE_ENABLE_SCRFD"
            },
            "liveportrait_appearance": {
                "url": "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/appearance_feature_extractor.onnx",
                "dir": self.models_dir / "liveportrait",
                "description": "LivePortrait appearance extractor",
                "size_mb": 85,
                "required_by": "MIRAGE_ENABLE_LIVEPORTRAIT"
            }
        }

    def print_status(self):
        """Print current model status"""
        print("\n=== Optional Model Status ===")

        for model_key, config in self.available_models.items():
            # Check if feature is enabled
            feature_flag = config["required_by"]
            is_enabled = os.getenv(feature_flag, "0").lower() in ("1", "true", "yes")

            # Check if model exists
            model_dir = config["dir"]
            if model_key == "scrfd":
                model_exists = (model_dir / "scrfd_10g_bnkps.onnx").exists()
            else:
                filename = config["url"].split("/")[-1]
                model_exists = (model_dir / filename).exists()

            enabled_icon = "🟢" if is_enabled else "⚪"
            downloaded_icon = "✅" if model_exists else "❌"

            print(f"{enabled_icon} {downloaded_icon} {model_key:<25} - {config['description']} ({config['size_mb']}MB)")

        print("\n🟢 = Feature enabled | ⚪ = Feature disabled")  
        print("✅ = Downloaded | ❌ = Not downloaded")

def main():
    """CLI interface"""
    downloader = OptionalModelDownloader()

    if len(sys.argv) > 1 and sys.argv[1] == "--download-needed":
        print("On-demand model download - run when needed")
        print("Enable features with: MIRAGE_ENABLE_SCRFD=1 MIRAGE_ENABLE_LIVEPORTRAIT=1")

    downloader.print_status()

if __name__ == "__main__":
    main()
