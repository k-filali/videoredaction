import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from clearframe.api.uploads import router as uploads_router
from clearframe.database import Database
from clearframe.domain.enums import JobStatus, JobType, VideoStatus
from clearframe.gcs_storage import GCSObjectMetadata, GCSStorage
from clearframe.jobs import JobContext, JobDispatcher
from clearframe.media import MediaMetadata, MediaProcessor, sha256_file
from clearframe.models import ProcessingJob, VideoAsset
from clearframe.services.container import ServiceContainer
from clearframe.services.ingest import DuplicateVideoError, IngestService
from clearframe.services.uploads import UploadService
from clearframe.storage import LocalStorage, original_key, temporary_upload_key


class RecordingDispatcher:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    def enqueue(self, job_id: str) -> None:
        self.job_ids.append(job_id)


class StubMediaProcessor:
    def probe(self, source: Path) -> MediaMetadata:
        assert source.is_file()
        return MediaMetadata(
            duration_ms=1000,
            fps=30.0,
            width=640,
            height=360,
            codec="h264",
            audio_present=False,
            frame_count_estimate=30,
            ffmpeg_version="test-ffmpeg",
        )

    def generate_proxy(
        self,
        source: Path,
        destination: Path,
        *,
        metadata: MediaMetadata | None = None,
    ) -> None:
        assert source.is_file()
        assert metadata is not None
        destination.write_bytes(b"proxy")

    def generate_thumbnail(
        self,
        source: Path,
        destination: Path,
        duration_ms: int,
    ) -> None:
        assert source.is_file()
        assert duration_ms == 1000
        destination.write_bytes(b"thumbnail")


class StubGCSStorage(GCSStorage):
    def __init__(self) -> None:
        self.sessions: dict[str, tuple[int, str, str | None]] = {}
        self.completed: dict[str, GCSObjectMetadata] = {}
        self.verify_calls = 0

    def create_resumable_upload_session(
        self,
        key: str,
        *,
        content_type: str,
        size: int | None = None,
        origin: str | None = None,
    ) -> str:
        assert size is not None
        self.sessions[key] = (size, content_type, origin)
        return f"https://uploads.example/{key}?session=test"

    def verify_object_metadata(
        self,
        key: str,
        *,
        expected_size: int | None = None,
        expected_content_type: str | None = None,
        expected_crc32c: str | None = None,
    ) -> GCSObjectMetadata:
        del expected_crc32c
        self.verify_calls += 1
        try:
            metadata = self.completed[key]
        except KeyError as exc:
            raise FileNotFoundError("artifact is missing") from exc
        if expected_size is not None and metadata.size != expected_size:
            raise AssertionError("unexpected size")
        if (
            expected_content_type is not None
            and metadata.content_type != expected_content_type
        ):
            raise AssertionError("unexpected content type")
        return metadata

    def finish(self, upload_id: str) -> None:
        key = temporary_upload_key(upload_id)
        size, content_type, _ = self.sessions[key]
        self.completed[key] = GCSObjectMetadata(
            key=key,
            size=size,
            generation=17,
            content_type=content_type,
            crc32c="crc32c",
            md5_hash="md5",
            etag="etag-17",
        )


def _build_services(
    tmp_path: Path,
    storage: LocalStorage | StubGCSStorage,
) -> tuple[Database, RecordingDispatcher, IngestService, UploadService]:
    database = Database(f"sqlite:///{(tmp_path / 'uploads.db').as_posix()}")
    database.create_schema()
    dispatcher = RecordingDispatcher()
    ingest = IngestService(
        database,
        storage,
        cast(MediaProcessor, StubMediaProcessor()),
        cast(JobDispatcher, dispatcher),
        max_upload_mb=10,
    )
    uploads = UploadService(
        database,
        storage,
        ingest,
        max_upload_mb=10,
        web_origin="https://clearframe.example",
    )
    return database, dispatcher, ingest, uploads


def _upload_app(uploads: UploadService) -> FastAPI:
    app = FastAPI()
    app.state.services = cast(
        ServiceContainer,
        SimpleNamespace(uploads=uploads),
    )
    app.include_router(uploads_router)
    return app


def test_resumable_api_contract_and_idempotent_completion(tmp_path: Path) -> None:
    storage = StubGCSStorage()
    database, dispatcher, _, uploads = _build_services(tmp_path, storage)
    app = _upload_app(uploads)

    async def exercise() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            capability = await client.get("/api/uploads/capability")
            assert capability.status_code == 200
            assert capability.json() == {
                "mode": "resumable",
                "chunk_size_bytes": 8 * 1024 * 1024,
                "max_upload_bytes": 10 * 1024 * 1024,
            }

            oversized = await client.post(
                "/api/uploads/initiate",
                json={
                    "filename": "large.mp4",
                    "content_type": "video/mp4",
                    "size_bytes": 11 * 1024 * 1024,
                    "hashed_case_id": None,
                },
            )
            assert oversized.status_code == 413

            initiated = await client.post(
                "/api/uploads/initiate",
                json={
                    "filename": "../../incident.mp4",
                    "content_type": "VIDEO/MP4",
                    "size_bytes": 1024,
                    "hashed_case_id": None,
                },
            )
            assert initiated.status_code == 201
            initiation = initiated.json()
            assert set(initiation) == {
                "upload_id",
                "session_url",
                "chunk_size_bytes",
            }
            upload_id = initiation["upload_id"]
            assert initiation["chunk_size_bytes"] == 8 * 1024 * 1024
            assert storage.sessions[temporary_upload_key(upload_id)] == (
                1024,
                "video/mp4",
                "https://clearframe.example",
            )

            storage.finish(upload_id)
            first = await client.post(
                f"/api/uploads/{upload_id}/complete",
                json={},
            )
            second = await client.post(
                f"/api/uploads/{upload_id}/complete",
                json={},
            )
            assert first.status_code == 202
            assert second.status_code == 202
            assert first.json()["job"]["id"] == second.json()["job"]["id"]
            assert first.json()["video"]["status"] == "VALIDATING"

    asyncio.run(exercise())

    with database.session() as session:
        video = session.scalar(select(VideoAsset))
        assert video is not None
        assert video.original_filename == "incident.mp4"
        assert video.status == VideoStatus.VALIDATING
        assert video.metadata_json["upload_generation"] == 17
        assert session.scalar(select(func.count()).select_from(ProcessingJob)) == 1
    assert len(dispatcher.job_ids) == 1
    assert storage.verify_calls == 1


def test_local_capability_keeps_multipart_uploads(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "storage")
    _, _, _, uploads = _build_services(tmp_path, storage)
    app = _upload_app(uploads)

    async def exercise() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            capability = await client.get("/api/uploads/capability")
            assert capability.status_code == 200
            assert capability.json() == {"mode": "multipart"}

            initiated = await client.post(
                "/api/uploads/initiate",
                json={
                    "filename": "incident.mp4",
                    "content_type": "video/mp4",
                    "size_bytes": 1024,
                    "hashed_case_id": None,
                },
            )
            assert initiated.status_code == 409

    asyncio.run(exercise())


def test_resumable_upload_accepts_browser_generic_content_type(
    tmp_path: Path,
) -> None:
    storage = StubGCSStorage()
    _, _, _, uploads = _build_services(tmp_path, storage)
    app = _upload_app(uploads)

    async def exercise() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/uploads/initiate",
                json={
                    "filename": "bodycam.mkv",
                    "content_type": "application/octet-stream",
                    "size_bytes": 1024,
                    "hashed_case_id": None,
                },
            )
            assert response.status_code == 201
            upload_id = response.json()["upload_id"]
            assert storage.sessions[temporary_upload_key(upload_id)][1] == (
                "application/octet-stream"
            )

    asyncio.run(exercise())


def test_direct_ingest_materializes_hashes_and_finalizes(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "direct-storage")
    database, _, ingest, _ = _build_services(tmp_path, storage)
    video_id = "11111111-1111-1111-1111-111111111111"
    job_id = "22222222-2222-2222-2222-222222222222"
    content = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 256
    upload_key = temporary_upload_key(video_id)
    storage.prepare(upload_key).write_bytes(content)

    with database.session() as session:
        session.add(
            VideoAsset(
                id=video_id,
                original_filename="incident.mp4",
                safe_filename="incident.mp4",
                content_type="video/mp4",
                status=VideoStatus.VALIDATING,
                metadata_json={
                    "upload_transport": "gcs_resumable",
                    "declared_upload_bytes": len(content),
                    "declared_content_type": "video/mp4",
                    "upload_generation": 17,
                },
            )
        )
        session.flush()
        session.add(
            ProcessingJob(
                id=job_id,
                video_id=video_id,
                job_type=JobType.INGEST,
                status=JobStatus.RUNNING,
                payload={
                    "temporary_uri": upload_key,
                    "direct_upload": {
                        "size_bytes": len(content),
                        "content_type": "video/mp4",
                        "generation": 17,
                    },
                },
            )
        )
        session.commit()

    ingest.execute(JobContext(database, job_id), job_id)

    with database.session() as session:
        video = session.get(VideoAsset, video_id)
        assert video is not None
        assert video.status == VideoStatus.READY_FOR_REVIEW
        assert video.original_sha256 == sha256_file(
            storage.path_for(original_key(video_id, ".mp4"))
        )
        assert video.proxy_uri is not None
        assert video.thumbnail_uri is not None
    assert not storage.exists(upload_key)


def test_direct_ingest_rejects_duplicate_checksum(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "duplicate-storage")
    database, _, ingest, _ = _build_services(tmp_path, storage)
    existing_id = "11111111-1111-1111-1111-111111111111"
    upload_id = "22222222-2222-2222-2222-222222222222"
    job_id = "33333333-3333-3333-3333-333333333333"
    content = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 256
    duplicate_checksum = sha256_file(_write_file(tmp_path / "checksum.mp4", content))
    upload_key = temporary_upload_key(upload_id)
    storage.prepare(upload_key).write_bytes(content)

    with database.session() as session:
        session.add(
            VideoAsset(
                id=existing_id,
                original_filename="existing.mp4",
                safe_filename="existing.mp4",
                content_type="video/mp4",
                original_sha256=duplicate_checksum,
                status=VideoStatus.READY_FOR_REVIEW,
            )
        )
        session.add(
            VideoAsset(
                id=upload_id,
                original_filename="duplicate.mp4",
                safe_filename="duplicate.mp4",
                content_type="video/mp4",
                status=VideoStatus.VALIDATING,
                metadata_json={
                    "upload_transport": "gcs_resumable",
                    "declared_upload_bytes": len(content),
                    "declared_content_type": "video/mp4",
                    "upload_generation": 18,
                },
            )
        )
        session.flush()
        session.add(
            ProcessingJob(
                id=job_id,
                video_id=upload_id,
                job_type=JobType.INGEST,
                status=JobStatus.RUNNING,
                payload={
                    "temporary_uri": upload_key,
                    "direct_upload": {
                        "size_bytes": len(content),
                        "content_type": "video/mp4",
                        "generation": 18,
                    },
                },
            )
        )
        session.commit()

    try:
        ingest.execute(JobContext(database, job_id), job_id)
    except DuplicateVideoError as exc:
        assert exc.existing_video_id == existing_id
    else:
        raise AssertionError("duplicate upload was accepted")

    assert not storage.exists(upload_key)
    assert not storage.exists(original_key(upload_id, ".mp4"))


def _write_file(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path
