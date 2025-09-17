FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Ensure consistent locale & no bytecode caching overhead
ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install only Python dependencies (already includes CUDA libs via base image)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
	&& pip cache purge || true

# Copy application source
COPY . /app

EXPOSE 7860

# Default command (can be overridden by Space / docker run)
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","7860"]
