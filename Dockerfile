FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install dependencies explicitly to avoid setuptools metadata issues
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "numpy>=1.26.0" \
    "soundfile>=0.12.0" \
    "pydantic>=2.0.0" \
    "pydantic-settings>=2.0.0" \
    "httpx>=0.27.0" \
    "python-multipart>=0.0.9" \
    "nano-vllm-voxcpm>=2.0.0" || \
    pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "numpy>=1.26.0" \
    "soundfile>=0.12.0" \
    "pydantic>=2.0.0" \
    "pydantic-settings>=2.0.0" \
    "httpx>=0.27.0" \
    "python-multipart>=0.0.9"

COPY server/ server/
COPY client/ client/

ENV VOXCPM_DATA_DIR=/data
ENV VOXCPM_MODEL_PATH=/models/VoxCPM2
VOLUME ["/data", "/models"]
EXPOSE 8000

RUN useradd -u 10001 -m appuser
USER appuser

CMD ["python3", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
