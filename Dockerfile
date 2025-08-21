# ---- Base: PyTorch + CUDA 11.8 runtime ----
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# ---- System deps (useful for PIL/torchvision/OpenCV-like ops) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      ffmpeg \
      libsm6 \
      libxext6 \
      libglib2.0-0 \
      libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps (don’t reinstall torch/torchvision; base image has them) ----
COPY requirements.txt .
RUN python -m pip install --upgrade pip wheel setuptools \
 && pip install -r requirements.txt

# ---- Project files ----
COPY . .

# Create output dirs so volume mounts are optional
RUN mkdir -p outputs/checkpoints outputs/logs

# ---- Default: training entry (override at run-time as needed) ----
# Tip: you can override CMD or pass args when `docker run`
CMD ["python", "train.py", "--amp"]
