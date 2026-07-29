FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/apps/api/src \
    CUDA_MODULE_LOADING=LAZY \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:/usr/local/nvidia/lib64 \
    PATH=/opt/clearframe/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        cuda-compat-12-8 \
        ffmpeg \
        python3 \
        python3-venv \
    && python3 -m venv /opt/clearframe \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.gpu.lock alembic.ini ./
RUN python -m pip install --no-cache-dir -r requirements.gpu.lock

COPY apps/api ./apps/api
COPY configs ./configs
COPY docker/api-entrypoint.sh ./docker/api-entrypoint.sh
COPY migrations ./migrations

RUN groupadd --system clearframe \
    && useradd --system --gid clearframe --home-dir /app clearframe \
    && chmod 755 /app/docker/api-entrypoint.sh \
    && install -d -o clearframe -g clearframe -m 0750 /app/data /app/storage

USER clearframe

ENTRYPOINT ["/app/docker/api-entrypoint.sh"]
CMD ["python", "-m", "clearframe.worker"]
