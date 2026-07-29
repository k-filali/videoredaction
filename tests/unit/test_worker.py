import asyncio
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from tests.helpers import generate_test_video

from clearframe.config import Settings
from clearframe.database import Database
from clearframe.domain.enums import JobStatus, VideoStatus
from clearframe.jobs import JobExecutionResult
from clearframe.media import MediaProcessor
from clearframe.models import ProcessingJob, VideoAsset
from clearframe.services.ingest import IngestService
from clearframe.storage import LocalStorage
from clearframe.worker import execute_job


class RecordingDispatcher:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    def enqueue(self, job_id: str) -> None:
        self.job_ids.append(job_id)


def test_worker_rebuilds_ingest_from_persisted_payload(tmp_path: Path) -> None:
    database_path = tmp_path / "worker.db"
    settings = Settings(
        database_url=f"sqlite:///{database_path.as_posix()}",
        storage_root=tmp_path / "storage",
        max_upload_mb=10,
        env="test",
    )
    database = Database(settings.database_url)
    database.create_schema()
    storage = LocalStorage(settings.storage_root)
    media = MediaProcessor()
    dispatcher = RecordingDispatcher()
    ingest = IngestService(
        database,
        storage,
        media,
        dispatcher,
        max_upload_mb=settings.max_upload_mb,
    )
    source = generate_test_video(tmp_path / "worker-source.mp4", media)

    async def stage_upload() -> tuple[str, str]:
        result = await ingest.accept(
            UploadFile(BytesIO(source.read_bytes()), filename="worker-source.mp4")
        )
        return result.video.id, result.job.id

    video_id, job_id = asyncio.run(stage_upload())
    assert dispatcher.job_ids == [job_id]

    assert execute_job(job_id, settings) == JobExecutionResult.COMPLETED

    with database.session() as session:
        video = session.get(VideoAsset, video_id)
        job = session.get(ProcessingJob, job_id)
        assert video is not None
        assert job is not None
        assert video.status == VideoStatus.READY_FOR_REVIEW
        assert video.original_uri is not None
        assert video.proxy_uri is not None
        assert storage.exists(video.original_uri)
        assert storage.exists(video.proxy_uri)
        assert job.status == JobStatus.COMPLETED
        assert job.attempts == 1
