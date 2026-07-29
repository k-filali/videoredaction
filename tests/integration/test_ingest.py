import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from httpx import ASGITransport, AsyncClient
from tests.helpers import generate_test_video

from clearframe.config import Settings
from clearframe.database import Database
from clearframe.domain.enums import JobStatus, VideoStatus
from clearframe.main import create_app
from clearframe.media import (
    MediaError,
    MediaMetadata,
    MediaProcessor,
    UnsupportedMediaError,
    sha256_file,
    sniff_media,
)
from clearframe.models import ProcessingJob, VideoAsset
from clearframe.services.container import ServiceContainer
from clearframe.services.ingest import DuplicateVideoError, IngestService
from clearframe.storage import LocalStorage


def build_services(tmp_path: Path) -> tuple[Database, ServiceContainer]:
    database = Database(f"sqlite:///{(tmp_path / 'clearframe.db').as_posix()}")
    database.create_schema()
    settings = Settings(
        database_url=str(database.engine.url),
        storage_root=tmp_path / "storage",
        max_upload_mb=10,
        env="test",
    )
    return database, ServiceContainer.build(settings, database)


def test_media_probe_proxy_and_checksum(tmp_path: Path) -> None:
    media = MediaProcessor()
    source = generate_test_video(tmp_path / "source.mp4", media)
    proxy = tmp_path / "proxy.mp4"

    assert sniff_media(source).content_type == "video/mp4"
    metadata = media.probe(source)
    media.generate_proxy(source, proxy)
    proxy_metadata = media.probe(proxy)

    assert metadata.width == 640
    assert metadata.height == 360
    assert metadata.audio_present is True
    assert metadata.duration_ms == pytest.approx(1200, abs=150)
    assert proxy_metadata.width <= 1280
    assert proxy_metadata.height <= 720
    assert sha256_file(source) != sha256_file(proxy)


def test_media_probe_enforces_resource_limits(tmp_path: Path) -> None:
    media = MediaProcessor()
    source = generate_test_video(tmp_path / "limited.mp4", media)

    with pytest.raises(MediaError, match="duration exceeds"):
        MediaProcessor(
            media.ffmpeg_path,
            max_duration_ms=500,
        ).probe(source)
    with pytest.raises(MediaError, match="pixel count exceeds"):
        MediaProcessor(
            media.ffmpeg_path,
            max_video_pixels=100_000,
        ).probe(source)


def test_corrupt_container_is_rejected(tmp_path: Path) -> None:
    corrupt = tmp_path / "not-video.mp4"
    corrupt.write_bytes(b"MZ" + b"\x00" * 128)

    with pytest.raises(UnsupportedMediaError):
        sniff_media(corrupt)


def test_decode_failure_cleans_temporary_artifacts(tmp_path: Path) -> None:
    database, services = build_services(tmp_path)
    fake_mp4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 256

    async def accept() -> tuple[str, str]:
        result = await services.ingest.accept(
            UploadFile(BytesIO(fake_mp4), filename="crafted.mp4")
        )
        return result.video.id, result.job.id

    video_id, job_id = asyncio.run(accept())
    services.runner.wait(job_id, timeout=30)

    with database.session() as session:
        video = session.get(VideoAsset, video_id)
        job = session.get(ProcessingJob, job_id)
        assert video is not None
        assert job is not None
        assert video.status == VideoStatus.FAILED
        assert job.status == JobStatus.FAILED
    assert not list(services.storage.root.rglob("*.upload"))
    assert not list(services.storage.root.rglob("proxy.mp4"))
    services.runner.shutdown()


def test_ingest_preserves_original_and_rejects_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, services = build_services(tmp_path)
    source = generate_test_video(tmp_path / "upload.mp4", services.media)
    original_checksum = sha256_file(source)
    received_proxy_metadata: list[MediaMetadata | None] = []
    generate_proxy = services.media.generate_proxy

    def record_generate_proxy(
        source_path: Path,
        destination_path: Path,
        *,
        metadata: MediaMetadata | None = None,
    ) -> None:
        received_proxy_metadata.append(metadata)
        generate_proxy(source_path, destination_path, metadata=metadata)

    monkeypatch.setattr(services.media, "generate_proxy", record_generate_proxy)

    async def accept() -> tuple[str, str]:
        with source.open("rb") as stream:
            result = await services.ingest.accept(
                UploadFile(stream, filename="../../incident.mp4")
            )
        return result.video.id, result.job.id

    video_id, job_id = asyncio.run(accept())
    services.runner.wait(job_id, timeout=120)

    assert len(received_proxy_metadata) == 1
    assert received_proxy_metadata[0] is not None
    assert received_proxy_metadata[0].codec == "h264"

    with database.session() as session:
        video = session.get(VideoAsset, video_id)
        job = session.get(ProcessingJob, job_id)
        assert video is not None
        assert job is not None
        assert video.original_filename == "incident.mp4"
        assert video.status == VideoStatus.READY_FOR_REVIEW
        assert job.status == JobStatus.COMPLETED
        assert video.original_uri is not None
        assert video.proxy_uri is not None
        assert sha256_file(services.storage.path_for(video.original_uri)) == original_checksum
        assert services.storage.path_for(video.proxy_uri).is_file()

    async def accept_duplicate() -> None:
        with source.open("rb") as stream, pytest.raises(DuplicateVideoError):
            await services.ingest.accept(UploadFile(stream, filename="copy.mp4"))

    asyncio.run(accept_duplicate())
    services.runner.shutdown()


def test_upload_status_and_proxy_api(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    settings = Settings(
        database_url=str(database.engine.url),
        storage_root=tmp_path / "api-storage",
        max_upload_mb=10,
        env="test",
    )
    services = ServiceContainer.build(settings, database)
    app = create_app(settings=settings, database=database, services=services)
    source = generate_test_video(tmp_path / "api-upload.mp4", services.media)

    async def exercise_api() -> None:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/videos",
                    files={"file": ("clip.mp4", source.read_bytes(), "video/mp4")},
                )
                assert response.status_code == 202
                payload = response.json()
                video_id = payload["video"]["id"]
                job_id = payload["job"]["id"]
                await asyncio.to_thread(services.runner.wait, job_id, 120)

                status_response = await client.get(f"/api/videos/{video_id}/status")
                assert status_response.status_code == 200
                assert status_response.json()["video"]["status"] == "READY_FOR_REVIEW"
                assert status_response.json()["jobs"][0]["status"] == "COMPLETED"

                proxy_response = await client.get(f"/api/videos/{video_id}/proxy")
                assert proxy_response.status_code == 200
                assert proxy_response.headers["content-type"].startswith("video/mp4")
                assert len(proxy_response.content) > 1000

                rejected = await client.post(
                    "/api/videos",
                    files={"file": ("fake.mp4", b"not a video", "video/mp4")},
                )
                assert rejected.status_code == 415

    asyncio.run(exercise_api())


def test_upload_limit_removes_temporary_file(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'limit.db').as_posix()}")
    database.create_schema()
    storage = LocalStorage(tmp_path / "limit-storage")
    media = MediaProcessor()
    services = ServiceContainer.build(
        Settings(
            database_url=str(database.engine.url),
            storage_root=storage.root,
            max_upload_mb=1,
            env="test",
        ),
        database,
    )
    ingest = IngestService(database, storage, media, services.runner, max_upload_mb=1)

    async def upload_large_file() -> None:
        from clearframe.services.ingest import UploadTooLargeError

        with pytest.raises(UploadTooLargeError):
            await ingest.accept(
                UploadFile(BytesIO(b"x" * (1024 * 1024 + 1)), filename="large.mp4")
            )

    asyncio.run(upload_large_file())
    assert not list(storage.root.glob("tmp/uploads/*"))
    services.runner.shutdown()


def test_request_limit_rejects_before_upload_parsing(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'request-limit.db').as_posix()}")
    settings = Settings(
        database_url=str(database.engine.url),
        storage_root=tmp_path / "request-limit-storage",
        max_upload_mb=1,
        env="test",
    )
    services = ServiceContainer.build(settings, database)
    app = create_app(settings=settings, database=database, services=services)

    async def upload() -> None:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/videos",
                    files={"file": ("oversized.mp4", b"x" * (4 * 1024 * 1024), "video/mp4")},
                )
                assert response.status_code == 413

    asyncio.run(upload())
    assert not list(settings.storage_root.rglob("*.upload"))
    with database.session() as session:
        assert session.query(VideoAsset).count() == 0
