# LivePortrait (ONNX) Models

Place the following ONNX files in this directory:

- appearance_feature_extractor.onnx (required)
- motion_extractor.onnx (required)
- generator.onnx (optional, improves fidelity; if missing we use motion-based warping or landmark fallback)

Environment variables for runtime downloader (optional):

- MIRAGE_DOWNLOAD_MODELS=1
- MIRAGE_LP_APPEARANCE_URL
- MIRAGE_LP_MOTION_URL
- MIRAGE_LP_GENERATOR_URL (optional)

At startup, the app will try to download models to this folder if URLs are configured. You can also drop files here manually. The /debug/models endpoint reports presence and sizes.
