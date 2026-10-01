FROM python:3.11-slim

# Install system dependencies for images/video
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    exiftool \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt /tmp/
# Install PyTorch >= 2.6.0 for CUDA 12.4
RUN pip install --no-cache-dir "torch>=2.6.0" torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r /tmp/requirements.txt
