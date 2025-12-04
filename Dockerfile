# --- Base image with CUDA 12.1 runtime (works with RTX 40xx + PyTorch cu121) ---
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1) System deps: Python + libs for OpenCV
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


# 2) Workdir inside the container
WORKDIR /workspace

# 3) Copy only requirements first (for Docker layer cache)
COPY requirements.txt .

# 4) Install Python deps (GPU Torch + the rest)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install -r requirements.txt

# 5) Copy the rest of the project into the image
#    (app/, docs/, models/, *.pt, instructions, etc.)
COPY . .

# 6) Default ZMQ endpoints (UDP-style strings; override at runtime if needed)
#    IMPORTANT: These are just strings; your C++ code must use the same endpoints.
ENV ZMQ_IN_ENDPOINT=tcp://0.0.0.0:5555
ENV ZMQ_OUT_ENDPOINT=tcp://0.0.0.0:5556
ENV PYTHONUNBUFFERED=1

# 7) Default command: run the YOLO ZMQ worker
CMD ["python3", "app/yolo_worker_zmq.py"]
