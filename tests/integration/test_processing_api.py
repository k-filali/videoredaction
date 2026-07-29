import asyncio
from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from tests.helpers import generate_test_video

from clearframe.api.processing import router
from clearframe.database import Database
from clearframe.domain.enums import VideoStatus
from clearframe.jobs import LocalJobRunner
from clearframe.media import MediaProcessor
from clearframe.models import VideoAsset
from clearframe.services.processing import ProcessingService
from clearframe.storage import LocalStorage


def test_processing_api_returns_latest_run_and_metrics(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'api-processing.db').as_posix()}")
    database.create_schema()
    storage = LocalStorage(tmp_path / "storage")
    runner = LocalJobRunner(database, max_workers=1)
    media = MediaProcessor()
    video_id = "api-processing-video"
    proxy_uri = storage.proxy_uri(video_id)
    proxy_path = generate_test_video(
        storage.prepare(proxy_uri),
        media,
        duration_seconds=0.8,
        audio=False,
    )
    metadata = media.probe(proxy_path)
    with database.session() as session:
        session.add(
            VideoAsset(
                id=video_id,
                original_filename="api.mp4",
                safe_filename="api.mp4",
                content_type="video/mp4",
                duration_ms=metadata.duration_ms,
                fps=metadata.fps,
                width=metadata.width,
                height=metadata.height,
                proxy_uri=proxy_uri,
                status=VideoStatus.READY_FOR_REVIEW,
            )
        )
        session.commit()

    app = FastAPI()
    app.state.processing_service = ProcessingService(database, storage, runner)
    app.include_router(router)

    async def exercise() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                f"/api/videos/{video_id}/process",
                json={"sample_every_frames": 3},
            )
            assert response.status_code == 202
            payload = response.json()
            assert payload["run"]["status"] == "QUEUED"
            assert payload["run"]["config_hash"]
            await asyncio.to_thread(runner.wait, payload["job"]["id"], 60)

            latest = await client.get(f"/api/videos/{video_id}/model-runs/latest")
            assert latest.status_code == 200
            assert latest.json()["status"] == "COMPLETED"
            assert latest.json()["metrics"]["detections"] > 0

            metrics = await client.get(
                f"/api/videos/{video_id}/model-runs/latest/metrics"
            )
            assert metrics.status_code == 200
            assert metrics.json()["model_run_id"] == payload["run"]["id"]
            assert metrics.json()["metrics"]["tracks"] == 1

            metrics_alias = await client.get(f"/api/videos/{video_id}/metrics")
            assert metrics_alias.json() == metrics.json()

            missing = await client.get("/api/videos/missing/model-runs/latest")
            assert missing.status_code == 404

    try:
        asyncio.run(exercise())
    finally:
        runner.shutdown()
