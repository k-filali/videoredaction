# ClearFrame

ClearFrame is a local, human-in-the-loop video redaction application. Reviewers can upload video,
inspect detected regions, correct or add redactions, and export a derived copy without modifying the
source file.

## Features

- Streaming video upload with file validation, size limits, checksums, and immutable originals
- Native-resolution H.264 review proxies and thumbnails generated with FFmpeg
- GPU-accelerated licence-plate and face detection with motion-aware tracking
- Reviewer controls for accepting, rejecting, moving, resizing, trimming, and creating regions
- Context suggestions around edited keyframes without overwriting reviewer decisions
- Pixelated, blurred, or black-box exports rendered from a frozen review revision
- Append-only review history, export manifests, and output checksum verification
- A virtualized React review workspace with smooth native video playback

## Stack

- React, TypeScript, and Vite
- FastAPI, SQLAlchemy, Alembic, and SQLite
- OpenCV, ONNX Runtime GPU, CUDA, and FFmpeg
- Pytest, Ruff, mypy, ESLint, Vitest, and Docker Compose

## Quick start

Install Docker Desktop, use Linux containers, and start Docker Desktop before running Compose. The
default configuration uses an NVIDIA GPU; keep the NVIDIA driver current and enable Docker Desktop
GPU support.

From the repository root:

```powershell
$env:CLEARFRAME_ACCESS_TOKEN = "replace-with-a-long-random-value"
docker compose up --build --detach --wait
```

Open [http://localhost:5173](http://localhost:5173).

If Docker reports that `dockerDesktopLinuxEngine` cannot be found, Docker Desktop is not running or
has not finished starting. Start it, wait for the engine status to show as running, and retry the
command.

For a machine without a compatible NVIDIA GPU:

```powershell
$env:CLEARFRAME_ACCESS_TOKEN = "replace-with-a-long-random-value"
docker compose -f compose.yaml -f compose.cpu.yaml up --build --detach --wait
```

Stop the services with `docker compose down`. The database and media files live in named volumes
and remain available across normal restarts.

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

Long videos use the same workflow. Compatible H.264 inputs are remuxed instead of needlessly
re-encoded, and proxies are never upscaled beyond the source dimensions. Existing proxies created
with an older profile are repaired in the background at startup and replaced only after validation.
Detection skips unsampled frame conversion, bounds concurrent inference, and uses CUDA when
available. The review workspace uses the browser's video renderer, draws only the active overlays,
and virtualizes large track collections.

## Detection

The default pipeline runs two local ONNX models: YOLOv9-t for licence plates and YuNet for faces.
Both prefer the CUDA execution provider and fall back to CPU in the CPU Compose profile. Their
weights are included in `configs/models/weights`, verified by SHA-256 at startup, and configured in
`configs/models/registry.yaml`. Detection samples the review proxy, applies class-specific
suppression, links detections with motion-aware tracking, and interpolates short gaps for review.

The plate model is distributed by
[Open Image Models](https://github.com/ankandrew/open-image-models) under the MIT licence. The face
model is distributed by [OpenCV Zoo](https://github.com/opencv/opencv_zoo) under the MIT licence.
Reviewer decisions remain mandatory: automatic proposals can still be missed or incorrect on
unfamiliar footage.

## Development setup

Local development requires Python 3.12, Node.js 24 with Corepack, Git, FFmpeg, and the model runtime
required by the selected execution provider.

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

The development UI runs at [http://localhost:5173](http://localhost:5173), the API at
[http://127.0.0.1:8000](http://127.0.0.1:8000), and the API documentation at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

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

- Automatic detection quality varies with camera motion, lighting, occlusion, distance, and plate
  style, so every proposal and uncovered interval still requires review.
- The included Compose profile uses SQLite, local volumes, and an in-process dispatcher.
- Uploads are not chunked or resumable.
- Redaction applies to video frames only. Source audio is copied to the export unchanged.
