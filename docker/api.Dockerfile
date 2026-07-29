FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/apps/api/src \
    CUDA_MODULE_LOADING=LAZY \
    PATH=/opt/clearframe/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        ffmpeg \
        python3 \
        python3-venv \
    && python3 -m venv /opt/clearframe \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock alembic.ini ./
RUN python -m pip install --no-cache-dir -r requirements.lock

COPY apps/api ./apps/api
COPY configs ./configs
COPY docker/api-entrypoint.sh ./docker/api-entrypoint.sh
COPY migrations ./migrations

RUN groupadd --system clearframe \
    && useradd --system --gid clearframe --home-dir /app clearframe \
    && chmod 755 /app/docker/api-entrypoint.sh \
    && install -d -o clearframe -g clearframe -m 0750 /app/data /app/storage

USER clearframe

EXPOSE 8000

ENTRYPOINT ["/app/docker/api-entrypoint.sh"]
