import asyncio
import json
from collections.abc import Generator
from pathlib import Path

import cv2
import pytest
from fastapi import UploadFile
from sqlalchemy import select
from tests.helpers import generate_test_video

from clearframe.config import Settings
from clearframe.database import Database
from clearframe.domain.enums import (
    ExportStatus,
    JobStatus,
    JobType,
    RedactionStyle,
    ReprocessingSuggestionStatus,
    ReviewActionType,
    TrackSource,
    VideoStatus,
)
from clearframe.domain.review import ReviewCommand
from clearframe.media import LIBX264_FAST, H264Encoder, sha256_file
from clearframe.models import (
    ExportArtifact,
    ProcessingJob,
    ReprocessingSuggestion,
    ReviewAction,
    Track,
    TrackKeyframe,
    VideoAsset,
)
from clearframe.services.container import ServiceContainer
from clearframe.services.export import ExportValidationError
from clearframe.services.review import append_review_action


@pytest.fixture
def export_environment(
    tmp_path: Path,
) -> Generator[tuple[Database, ServiceContainer, str, str, str], None, None]:
    database = Database(f"sqlite:///{(tmp_path / 'clearframe.db').as_posix()}")
    database.create_schema()
    settings = Settings(
        database_url=str(database.engine.url),
        storage_root=tmp_path / "storage",
        max_upload_mb=10,
        env="test",
        build_id="test-build",
    )
    services = ServiceContainer.build(settings, database)
    upload_path = generate_test_video(tmp_path / "upload.mp4", services.media)

    async def ingest() -> tuple[str, str]:
        with upload_path.open("rb") as stream:
            accepted = await services.ingest.accept(
                UploadFile(stream, filename="synthetic-plate.mp4")
            )
        return accepted.video.id, accepted.job.id

    video_id, ingest_job_id = asyncio.run(ingest())
    services.runner.wait(ingest_job_id, timeout=120)

    with database.session() as session:
        video = session.get(VideoAsset, video_id)
        ingest_job = session.get(ProcessingJob, ingest_job_id)
        assert video is not None
        assert ingest_job is not None
        assert video.status == VideoStatus.READY_FOR_REVIEW
        assert ingest_job.status == JobStatus.COMPLETED
        assert video.original_uri is not None
        original_sha256 = video.original_sha256
        assert original_sha256 is not None

        track = Track(
            video_id=video_id,
            class_name="license_plate",
            start_frame=0,
            end_frame=17,
            start_ms=0,
            end_ms=1133,
            source=TrackSource.MODEL,
            confidence_summary={"mean": 0.99},
        )
        session.add(track)
        session.flush()
        session.add(
            TrackKeyframe(
                track_id=track.id,
                frame_index=0,
                timestamp_ms=0,
                x1=160 / 640,
                y1=220 / 360,
                x2=340 / 640,
                y2=274 / 360,
                confidence=0.99,
                source=TrackSource.MODEL,
            )
        )
        session.commit()
        track_id = track.id

    yield database, services, video_id, track_id, original_sha256
    services.runner.shutdown()


def test_frozen_review_exports_verified_black_box_video(
    export_environment: tuple[Database, ServiceContainer, str, str, str],
) -> None:
    database, services, video_id, track_id, original_sha256 = export_environment
    with database.session() as session:
        _, accepted = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.ACCEPT_PROPOSAL,
                expected_revision=0,
                track_id=track_id,
            ),
            reviewer_session_id="reviewer-export-test",
        )
    assert accepted.revision == 1

    with database.session() as session:
        video = session.get(VideoAsset, video_id)
        assert video is not None
        assert video.original_uri is not None
        original_path = services.storage.path_for(video.original_uri)
    before_export_hash = sha256_file(original_path)

    requested = services.export.request(
        video_id,
        expected_revision=1,
        style=RedactionStyle.BLACK_BOX,
        reviewer_session_id="reviewer-export-test",
    )
    services.runner.wait(requested.job.id, timeout=120)

    with database.session() as session:
        artifact = session.get(ExportArtifact, requested.artifact.id)
        job = session.get(ProcessingJob, requested.job.id)
        video = session.get(VideoAsset, video_id)
        actions = list(
            session.scalars(
                select(ReviewAction)
                .where(ReviewAction.video_id == video_id)
                .order_by(ReviewAction.revision)
            )
        )
        assert artifact is not None
        assert job is not None
        assert video is not None
        assert artifact.status == ExportStatus.COMPLETED
        assert job.status == JobStatus.COMPLETED
        assert video.status == VideoStatus.EXPORTED
        assert artifact.export_uri is not None
        assert artifact.manifest_uri is not None
        assert artifact.export_sha256 is not None
        export_path = services.storage.path_for(artifact.export_uri)
        manifest_path = services.storage.path_for(artifact.manifest_uri)

    assert sha256_file(original_path) == before_export_hash == original_sha256
    assert export_path.is_file()
    assert manifest_path.is_file()
    assert sha256_file(export_path) == artifact.export_sha256

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["original_sha256"] == original_sha256
    assert manifest["export_sha256"] == artifact.export_sha256
    assert manifest["review_revision"] == 1
    assert manifest["redaction_style"] == RedactionStyle.BLACK_BOX
    assert manifest["redaction_track_counts"] == {"license_plate": 1}
    assert manifest["action_count"] == 1
    assert manifest["frames_rendered"] > 0
    assert manifest["video_encoder"] in {"h264_nvenc", "libx264"}
    assert manifest["hardware_video_encoding"] == (
        manifest["video_encoder"] == "h264_nvenc"
    )
    assert isinstance(manifest["encoder_fallback_used"], bool)
    assert manifest["build_id"] == "test-build"
    assert manifest["ffmpeg_version"] == services.media.ffmpeg_version
    assert manifest["model_registry_sha256"] is None
    assert manifest["audio_present"] is True
    assert manifest["audio_redaction_applied"] is False
    assert manifest["audio_policy"] == "preserved_unreviewed"
    assert "Audio was preserved without redaction." in manifest["warnings"]

    metadata = services.media.probe(export_path)
    assert metadata.width == 640
    assert metadata.height == 360
    assert metadata.duration_ms == pytest.approx(1200, abs=200)

    original_capture = cv2.VideoCapture(str(original_path))
    export_capture = cv2.VideoCapture(str(export_path))
    original_ok, original_frame = original_capture.read()
    export_ok, export_frame = export_capture.read()
    original_capture.release()
    export_capture.release()
    assert original_ok and export_ok
    assert original_frame[225, 165].mean() > 200
    assert export_frame[225, 165].mean() < 10

    assert [action.action_type for action in actions] == [
        ReviewActionType.ACCEPT_PROPOSAL,
        ReviewActionType.EXPORT_REQUESTED,
        ReviewActionType.EXPORT_COMPLETED,
    ]
    assert actions[1].after_state["frozen_review_revision"] == 1
    assert actions[2].after_state["export_sha256"] == artifact.export_sha256


def test_failed_hardware_encoder_restarts_cleanly_on_cpu(
    export_environment: tuple[Database, ServiceContainer, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, services, video_id, track_id, _ = export_environment
    with database.session() as session:
        append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.ACCEPT_PROPOSAL,
                expected_revision=0,
                track_id=track_id,
            ),
            reviewer_session_id="reviewer-export-test",
        )

    unusable_nvenc = H264Encoder(
        name="h264_nvenc",
        ffmpeg_arguments=("-c:v", "clearframe_missing_encoder"),
        hardware_accelerated=True,
    )
    monkeypatch.setattr(
        services.media,
        "export_h264_encoders",
        lambda: (unusable_nvenc, LIBX264_FAST),
    )

    requested = services.export.request(
        video_id,
        expected_revision=1,
        style=RedactionStyle.BLACK_BOX,
        reviewer_session_id="reviewer-export-test",
    )
    services.runner.wait(requested.job.id, timeout=120)

    with database.session() as session:
        artifact = session.get(ExportArtifact, requested.artifact.id)
        job = session.get(ProcessingJob, requested.job.id)
        assert artifact is not None
        assert job is not None
        assert artifact.status == ExportStatus.COMPLETED
        assert job.status == JobStatus.COMPLETED
        assert artifact.export_uri is not None
        assert artifact.manifest_uri is not None
        export_path = services.storage.path_for(artifact.export_uri)
        manifest_path = services.storage.path_for(artifact.manifest_uri)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["video_encoder"] == "libx264"
    assert manifest["hardware_video_encoding"] is False
    assert manifest["encoder_fallback_used"] is True
    assert export_path.is_file()
    assert list(export_path.parent.glob("*.part.mp4")) == []
    assert services.media.probe(export_path).audio_present


def test_failed_encoder_attempts_leave_no_export_artifact(
    export_environment: tuple[Database, ServiceContainer, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, services, video_id, track_id, _ = export_environment
    with database.session() as session:
        append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.ACCEPT_PROPOSAL,
                expected_revision=0,
                track_id=track_id,
            ),
            reviewer_session_id="reviewer-export-test",
        )

    broken_hardware = H264Encoder(
        name="h264_nvenc",
        ffmpeg_arguments=("-c:v", "clearframe_missing_nvenc"),
        hardware_accelerated=True,
    )
    broken_cpu = H264Encoder(
        name="libx264",
        ffmpeg_arguments=("-c:v", "clearframe_missing_libx264"),
        hardware_accelerated=False,
    )
    monkeypatch.setattr(
        services.media,
        "export_h264_encoders",
        lambda: (broken_hardware, broken_cpu),
    )

    requested = services.export.request(
        video_id,
        expected_revision=1,
        style=RedactionStyle.BLACK_BOX,
        reviewer_session_id="reviewer-export-test",
    )
    services.runner.wait(requested.job.id, timeout=120)

    export_uri = services.storage.export_video_uri(video_id, requested.artifact.id)
    manifest_uri = services.storage.export_manifest_uri(video_id, requested.artifact.id)
    export_path = services.storage.path_for(export_uri)
    with database.session() as session:
        artifact = session.get(ExportArtifact, requested.artifact.id)
        job = session.get(ProcessingJob, requested.job.id)
        assert artifact is not None
        assert job is not None
        assert artifact.status == ExportStatus.FAILED
        assert job.status == JobStatus.FAILED

    assert not export_path.exists()
    assert not services.storage.path_for(manifest_uri).exists()
    assert list(export_path.parent.glob("*.part.mp4")) == []


def test_unresolved_proposal_rejects_export(
    export_environment: tuple[Database, ServiceContainer, str, str, str],
) -> None:
    database, services, video_id, _, original_sha256 = export_environment

    with pytest.raises(
        ExportValidationError,
        match="still require reviewer confirmation",
    ):
        services.export.request(
            video_id,
            expected_revision=0,
            style=RedactionStyle.BLACK_BOX,
            reviewer_session_id="reviewer-export-test",
        )

    with database.session() as session:
        video = session.get(VideoAsset, video_id)
        artifacts = list(
            session.scalars(
                select(ExportArtifact).where(ExportArtifact.video_id == video_id)
            )
        )
        export_jobs = list(
            session.scalars(
                select(ProcessingJob).where(ProcessingJob.export_id.is_not(None))
            )
        )
        assert video is not None
        assert video.original_uri is not None
        assert video.status == VideoStatus.READY_FOR_REVIEW
        assert artifacts == []
        assert export_jobs == []
        assert sha256_file(services.storage.path_for(video.original_uri)) == original_sha256


def test_pending_context_suggestion_rejects_export(
    export_environment: tuple[Database, ServiceContainer, str, str, str],
) -> None:
    database, services, video_id, track_id, _ = export_environment

    with database.session() as session:
        action, _ = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.ACCEPT_PROPOSAL,
                expected_revision=0,
                track_id=track_id,
            ),
            reviewer_session_id="reviewer-export-test",
        )
        job = ProcessingJob(
            video_id=video_id,
            job_type=JobType.REPROCESS,
            status=JobStatus.COMPLETED,
        )
        session.add(job)
        session.flush()
        session.add(
            ReprocessingSuggestion(
                video_id=video_id,
                source_action_id=action.id,
                job_id=job.id,
                track_id=track_id,
                source_revision=1,
                class_name="license_plate",
                seed_frame_index=0,
                frame_index=1,
                timestamp_ms=67,
                x1=0.25,
                y1=0.61,
                x2=0.53,
                y2=0.76,
                confidence=0.93,
                direction="forward",
                propagation_method="interpolation",
                status=ReprocessingSuggestionStatus.PENDING,
            )
        )
        session.commit()

    with pytest.raises(
        ExportValidationError,
        match=r"context suggestion.*still require reviewer confirmation",
    ):
        services.export.request(
            video_id,
            expected_revision=1,
            style=RedactionStyle.BLACK_BOX,
            reviewer_session_id="reviewer-export-test",
        )
