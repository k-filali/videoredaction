# ClearFrame

ClearFrame is a local, human-in-the-loop video redaction application. Reviewers can upload video,
inspect detected regions, correct or add redactions, and export a derived copy without modifying the
source file.

## Features

- Streaming video upload with file validation, size limits, checksums, and immutable originals
- H.264 review proxies and thumbnails generated with FFmpeg
- Detector adapters, frame sampling, non-maximum suppression, tracking, and gap interpolation
- Reviewer controls for accepting, rejecting, moving, resizing, trimming, and creating regions
- Context suggestions around edited keyframes without overwriting reviewer decisions
- Pixelated, blurred, or black-box exports rendered from a frozen review revision
- Append-only review history, export manifests, and output checksum verification
- A React review workspace with timeline controls, keyboard shortcuts, job progress, and downloads

## Stack

- React, TypeScript, and Vite
- FastAPI, SQLAlchemy, Alembic, and SQLite
- OpenCV and FFmpeg
- Pytest, Ruff, mypy, ESLint, Vitest, and Docker Compose

## Local setup

Requirements:

- Python 3.12
- Node.js 24 with Corepack
- Git

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
corepack enable
pnpm install --frozen-lockfile
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe scripts\dev.py
```

Open [http://localhost:5173](http://localhost:5173). The API runs at
[http://127.0.0.1:8000](http://127.0.0.1:8000), with interactive API documentation at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Usage

1. Upload an MP4, MOV, M4V, or AVI video.
2. Wait for proxy generation and detection to complete.
3. Open the video and review every suggested region.
4. Accept or reject suggestions, adjust their geometry and time span, or draw manual regions.
5. Resolve any context suggestions created by an edit.
6. Choose a redaction style and export the reviewed video.

The default limits allow files up to 2 GiB and videos up to four hours, with a maximum of
9,000,000 pixels per frame, 8192 pixels per dimension, and 120 fps. These values can be changed in
`.env`.

Longer videos are supported by the same workflow and are covered by an upload-to-export integration
test. Processing and export are CPU-bound and scale with the number, resolution, and codec of the
frames, so rehearse representative footage on the machine used for a presentation.

## Detection

The model registry is stored in `configs/models/registry.yaml`. The enabled detector is a
deterministic plate-shaped-region detector intended to exercise the review pipeline. OpenCV face and
plate adapters are included but disabled by default. Use manual regions for arbitrary footage, or
configure a suitable approved detector before expecting production-quality automatic proposals.

Reviewer decisions remain mandatory regardless of the detector used.

## Docker

Set a non-empty access token and start the application:

```powershell
$env:CLEARFRAME_ACCESS_TOKEN = "replace-with-a-long-random-value"
docker compose up --build --detach --wait
```

Open [http://localhost:5173](http://localhost:5173). Stop the services with:

```powershell
docker compose down
```

The database and media storage use named Docker volumes and remain available across restarts.

## Verification

Run the complete backend and frontend checks:

```powershell
.\.venv\Scripts\python.exe scripts\verify.py
```

Individual commands:

```powershell
.\.venv\Scripts\python.exe -m ruff check apps/api/src tests scripts
.\.venv\Scripts\python.exe -m mypy apps/api/src tests scripts
.\.venv\Scripts\python.exe -m pytest
pnpm lint:web
pnpm typecheck:web
pnpm test:web
pnpm build:web
```

## Project layout

```text
apps/api/        FastAPI application and redaction pipeline
apps/web/        React reviewer workspace
configs/models/  Detector registry
docker/          Container images and web proxy
migrations/      Alembic database migrations
scripts/         Development and verification commands
tests/           Unit, integration, and end-to-end tests
```

## Current limitations

- Automatic detection quality depends on the configured detector; the default is not a field model.
- Jobs run in a local in-process worker pool and are not resumed after an application restart.
- Uploads are not chunked or resumable.
- Redaction applies to video frames only. Source audio is copied to the export unchanged.
