# ============================================================
# Smart Count Tramway — Production Container
# Optimized for Python 3.12 & Edge Devices (Jetson / RPi / x86)
# ============================================================

FROM python:3.12-slim AS base

# Install system dependencies for OpenCV and SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies (CPU PyTorch by default for lightweight container)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY . .

# Run compatibility patch for DeepSORT on Python 3.12
RUN python patch_deepsort.py

# Create required runtime directories
RUN mkdir -p data/videos data/logs models

EXPOSE 8501

# Default command launches the Streamlit monitoring dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
