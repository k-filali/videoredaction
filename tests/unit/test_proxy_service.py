import subprocess
from pathlib import Path

import pytest
from sqlalchemy import select
from tests.helpers import generate_test_video

from clearframe.database import Database
from clearframe.domain.enums import JobStatus, JobType, VideoStatus
from clearframe.jobs import LocalJobRunner
from clearframe.media import MediaError, MediaProcessor, sha256_file
from clearframe.models import ProcessingJob, Track, VideoAsset
from clearframe.services.proxy import ProxyService
from clearframe.storage import LocalStorage


def build_legacy_proxy(
    tmp_path: Path,
) -> tuple[Database, LocalStorage, MediaProcessor, LocalJobRunner, str, str]:
    database = Database(f"sqlite:///{(tmp_path / 'proxy.db').as_posix()}")
    database.create_schema()
    storage = LocalStorage(tmp_path / "storage")
    media = MediaProcessor()
    runner = LocalJobRunner(database, max_workers=1)
    video_id = "legacy-proxy"
    original_uri = storage.original_uri(video_id, ".mp4")
    proxy_uri = storage.proxy_uri(video_id)
    original_path = generate_test_video(
        storage.prepare(original_uri),
        media,
        duration_seconds=0.8,
    )
    proxy_path = storage.prepare(proxy_uri)
    subprocess.run(
        [
            str(media.ffmpeg_path),
            "-y",
            "-v",
            "error",
            "-i",
            str(original_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "scale=1280:720",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(proxy_path),
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )
    metadata = media.probe(original_path)
    with database.session() as session:
        session.add(
            VideoAsset(
                id=video_id,
                original_filename="legacy.mp4",
                safe_filename="legacy.mp4",
                content_type="video/mp4",
                duration_ms=metadata.duration_ms,
                fps=metadata.fps,
                width=metadata.width,
                height=metadata.height,
                codec=metadata.codec,
                audio_present=metadata.audio_present,
                original_uri=original_uri,
                proxy_uri=proxy_uri,
                status=VideoStatus.READY_FOR_REVIEW,
            )
        )
        session.flush()
        session.add(
            Track(
                id="existing-track",
                video_id=video_id,
                class_name="face",
                start_frame=0,
                end_frame=max(0, metadata.frame_count_estimate - 1),
                start_ms=0,
                end_ms=metadata.duration_ms,
            )
        )
        session.commit()
    return database, storage, media, runner, video_id, proxy_uri


def proxy_job(database: Database, video_id: str) -> ProcessingJob:
    with database.session() as session:
        job = session.scalar(
            select(ProcessingJob).where(
                ProcessingJob.video_id == video_id,
                ProcessingJob.job_type == JobType.PROXY,
            )
        )
        assert job is not None
        return job


def test_reconcile_replaces_legacy_upscale_without_losing_tracks(tmp_path: Path) -> None:
    database, storage, media, runner, video_id, proxy_uri = build_legacy_proxy(tmp_path)
    service = ProxyService(database, storage, media, runner)
    original_path = storage.path_for(storage.original_uri(video_id, ".mp4"))
    original_hash = sha256_file(original_path)

    try:
        assert service.reconcile_video(video_id) == "scheduled"
        job = proxy_job(database, video_id)
        runner.wait(job.id, timeout=60)

        repaired = media.probe(storage.path_for(proxy_uri))
        assert (repaired.width, repaired.height) == (640, 360)
        assert sha256_file(original_path) == original_hash
        with database.session() as session:
            completed_job = session.get(ProcessingJob, job.id)
            video = session.get(VideoAsset, video_id)
            track = session.get(Track, "existing-track")
            assert completed_job is not None
            assert video is not None
            assert completed_job.status == JobStatus.COMPLETED
            assert video.status == VideoStatus.READY_FOR_REVIEW
            assert video.error_message is None
            assert video.metadata_json["proxy_profile"]["version"] == 2
            assert video.metadata_json["proxy_repair"]["status"] == "complete"
            assert track is not None
    finally:
        runner.shutdown()


def test_failed_repair_keeps_existing_proxy_and_ready_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, storage, media, runner, video_id, proxy_uri = build_legacy_proxy(tmp_path)
    service = ProxyService(database, storage, media, runner)
    proxy_path = storage.path_for(proxy_uri)
    proxy_hash = sha256_file(proxy_path)

    def fail_generation(
        source: Path,
        destination: Path,
        *,
        metadata: object | None = None,
    ) -> None:
        del source, destination, metadata
        raise MediaError("test proxy failure")

    monkeypatch.setattr(media, "generate_proxy", fail_generation)
    try:
        assert service.reconcile_video(video_id) == "scheduled"
        job = proxy_job(database, video_id)
        runner.wait(job.id, timeout=60)

        assert sha256_file(proxy_path) == proxy_hash
        assert not list(proxy_path.parent.glob(".*.repair-*.mp4"))
        with database.session() as session:
            failed_job = session.get(ProcessingJob, job.id)
            video = session.get(VideoAsset, video_id)
            assert failed_job is not None
            assert video is not None
            assert failed_job.status == JobStatus.FAILED
            assert video.status == VideoStatus.READY_FOR_REVIEW
            assert video.error_message is None
            assert video.metadata_json["proxy_repair"]["status"] == "failed"
    finally:
        runner.shutdown()
