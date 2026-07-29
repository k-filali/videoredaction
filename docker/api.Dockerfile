FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock pyproject.toml alembic.ini ./
RUN python -m pip install --no-cache-dir -r requirements.lock

COPY apps/api ./apps/api
COPY configs ./configs
COPY docker/api-entrypoint.sh ./docker/api-entrypoint.sh
COPY migrations ./migrations
RUN python -m pip install --no-cache-dir --no-deps .

RUN groupadd --system clearframe \
    && useradd --system --gid clearframe --home-dir /app clearframe \
    && mkdir -p /app/data /app/storage \
    && chmod 755 /app/docker/api-entrypoint.sh \
    && chown -R clearframe:clearframe /app

USER clearframe

EXPOSE 8000

ENTRYPOINT ["/app/docker/api-entrypoint.sh"]
