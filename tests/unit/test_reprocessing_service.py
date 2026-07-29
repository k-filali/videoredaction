from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from threading import Event

import cv2
import pytest
from sqlalchemy import select
from tests.helpers import generate_test_video

from clearframe.database import Database
from clearframe.domain.enums import (
    InterpolationMode,
    JobStatus,
    JobType,
    ReprocessingSuggestionStatus,
    ReviewActionType,
    TrackSource,
    TrackStatus,
    VideoStatus,
)
from clearframe.domain.geometry import NormalizedBox
from clearframe.domain.review import ReviewCommand
from clearframe.jobs import LocalJobRunner
from clearframe.media import MediaProcessor
from clearframe.models import (
    ProcessingJob,
    ReprocessingSuggestion,
    ReviewAction,
    Track,
    TrackKeyframe,
    VideoAsset,
)
from clearframe.services.reprocessing import (
    ReprocessingConflictError,
    ReprocessingNotFoundError,
    ReprocessingService,
    _Candidate,
    _Seed,
    _SourcePoint,
)
from clearframe.services.review import RevisionConflictError, append_review_action
from clearframe.storage import LocalStorage


def _build_context(
    tmp_path: Path,
) -> tuple[Database, LocalStorage, LocalJobRunner, str, float]:
    database = Database(f"sqlite:///{(tmp_path / 'reprocessing.db').as_posix()}")
    database.create_schema()
    storage = LocalStorage(tmp_path / "storage")
    runner = LocalJobRunner(database, max_workers=1)
    media = MediaProcessor()
    video_id = "reprocessing-video"
    proxy_uri = storage.proxy_uri(video_id)
    proxy_path = generate_test_video(
        storage.prepare(proxy_uri),
        media,
        duration_seconds=1.2,
        audio=False,
    )
    metadata = media.probe(proxy_path)
    with database.session() as session:
        session.add(
            VideoAsset(
                id=video_id,
                original_filename="sample.mp4",
                safe_filename="sample.mp4",
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
    return database, storage, runner, video_id, metadata.fps


def _create_corrected_action(database: Database, video_id: str, fps: float) -> ReviewAction:
    track_id = "proposal-track"
    with database.session() as session:
        session.add(
            Track(
                id=track_id,
                video_id=video_id,
                class_name="license_plate",
                start_frame=0,
                end_frame=14,
                start_ms=0,
                end_ms=round(14 * 1000 / fps),
                status=TrackStatus.PROPOSED,
                default_redacted=True,
                source=TrackSource.MODEL,
                confidence_summary={"mean": 0.8},
            )
        )
        session.add_all(
            [
                TrackKeyframe(
                    track_id=track_id,
                    frame_index=0,
                    timestamp_ms=0,
                    x1=0.20,
                    y1=0.60,
                    x2=0.50,
                    y2=0.75,
                    interpolation_mode=InterpolationMode.LINEAR,
                    source=TrackSource.MODEL,
                    locked=False,
                ),
                TrackKeyframe(
                    track_id=track_id,
                    frame_index=14,
                    timestamp_ms=round(14 * 1000 / fps),
                    x1=0.24,
                    y1=0.60,
                    x2=0.54,
                    y2=0.75,
                    interpolation_mode=InterpolationMode.LINEAR,
                    source=TrackSource.MODEL,
                    locked=False,
                ),
            ]
        )
        session.commit()

    with database.session() as session:
        action, _ = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.MOVE_REGION,
                expected_revision=0,
                track_id=track_id,
                frame_index=7,
                timestamp_ms=round(7 * 1000 / fps),
                payload={
                    "bbox": {
                        "x1": 0.30,
                        "y1": 0.58,
                        "x2": 0.60,
                        "y2": 0.73,
                    }
                },
            ),
            reviewer_session_id="reviewer-test",
        )
        return action


def _create_suggestion(
    database: Database,
    video_id: str,
    source_action: ReviewAction,
    fps: float,
    *,
    frame_index: int = 8,
) -> str:
    assert source_action.track_id is not None
    assert source_action.frame_index is not None
    with database.session() as session:
        job = ProcessingJob(
            video_id=video_id,
            job_type=JobType.REPROCESS,
            status=JobStatus.COMPLETED,
        )
        session.add(job)
        session.flush()
        suggestion = ReprocessingSuggestion(
            video_id=video_id,
            source_action_id=source_action.id,
            job_id=job.id,
            track_id=source_action.track_id,
            source_revision=source_action.revision,
            class_name="license_plate",
            seed_frame_index=source_action.frame_index,
            frame_index=frame_index,
            timestamp_ms=round(frame_index * 1000 / fps),
            x1=0.31,
            y1=0.57,
            x2=0.61,
            y2=0.72,
            confidence=0.74,
            direction="forward",
            propagation_method="interpolation",
            seed_locked=True,
            status=ReprocessingSuggestionStatus.PENDING,
            metadata_json={"distance_frames": 1},
        )
        session.add(suggestion)
        session.commit()
        return suggestion.id


def _pause_fallback(
    service: ReprocessingService,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Event, Event]:
    started = Event()
    resume = Event()
    propagate = service._propagate_fallback

    def paused(
        capture: cv2.VideoCapture,
        *,
        seed: _Seed,
        source_points: Sequence[_SourcePoint],
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> list[_Candidate]:
        started.set()
        if not resume.wait(timeout=30):
            raise TimeoutError("test did not release reprocessing")
        return propagate(
            capture,
            seed=seed,
            source_points=source_points,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
        )

    monkeypatch.setattr(service, "_propagate_fallback", paused)
    return started, resume


def test_accept_suggestion_appends_locked_truth_without_reprocessing(
    tmp_path: Path,
) -> None:
    database, storage, runner, video_id, fps = _build_context(tmp_path)
    source_action = _create_corrected_action(database, video_id, fps)
    suggestion_id = _create_suggestion(database, video_id, source_action, fps)
    service = ReprocessingService(database, storage, runner, prefer_csrt=False)
    try:
        result = service.accept_suggestion(
            video_id,
            suggestion_id,
            expected_revision=1,
            reviewer_session_id="reviewer-resolution",
            reason_code="reviewed_context",
        )

        assert result.action is not None
        assert result.action.action_type == ReviewActionType.RESIZE_REGION
        assert result.action.reason_code == "reviewed_context"
        assert result.action.reviewer_session_id == "reviewer-resolution"
        assert result.state.revision == 2
        accepted_track = result.state.tracks["proposal-track"]
        accepted_keyframe = next(
            item for item in accepted_track.keyframes if item.frame_index == 8
        )
        assert accepted_keyframe.locked is True
        assert accepted_keyframe.bbox == NormalizedBox(
            x1=0.31,
            y1=0.57,
            x2=0.61,
            y2=0.72,
        )
        assert result.suggestion.status == ReprocessingSuggestionStatus.ACCEPTED
        assert result.suggestion.resolution_action_id == result.action.id
        assert result.suggestion.resolved_by_session_id == "reviewer-resolution"
        assert result.suggestion.resolution_reason_code == "reviewed_context"
        assert result.suggestion.resolved_at is not None

        with database.session() as session:
            jobs = list(
                session.scalars(
                    select(ProcessingJob).where(
                        ProcessingJob.video_id == video_id,
                        ProcessingJob.job_type == JobType.REPROCESS,
                    )
                )
            )
            assert len(jobs) == 1
    finally:
        runner.shutdown()


def test_dismiss_suggestion_records_resolution_without_changing_truth(
    tmp_path: Path,
) -> None:
    database, storage, runner, video_id, fps = _build_context(tmp_path)
    source_action = _create_corrected_action(database, video_id, fps)
    suggestion_id = _create_suggestion(database, video_id, source_action, fps)
    service = ReprocessingService(database, storage, runner, prefer_csrt=False)
    try:
        result = service.dismiss_suggestion(
            video_id,
            suggestion_id,
            expected_revision=1,
            reviewer_session_id="reviewer-dismiss",
            reason_code="not_useful",
        )

        assert result.action is None
        assert result.state.revision == 1
        assert result.suggestion.status == ReprocessingSuggestionStatus.DISMISSED
        assert result.suggestion.resolution_action_id is None
        assert result.suggestion.resolved_by_session_id == "reviewer-dismiss"
        assert result.suggestion.resolution_reason_code == "not_useful"
        assert result.suggestion.resolved_at is not None
        with database.session() as session:
            video = session.get(VideoAsset, video_id)
            actions = list(
                session.scalars(
                    select(ReviewAction).where(ReviewAction.video_id == video_id)
                )
            )
            assert video is not None
            assert video.review_revision == 1
            assert len(actions) == 1

        with pytest.raises(ReprocessingConflictError, match="already resolved"):
            service.dismiss_suggestion(
                video_id,
                suggestion_id,
                expected_revision=1,
                reviewer_session_id="reviewer-dismiss",
            )
    finally:
        runner.shutdown()


def test_suggestion_resolution_rejects_wrong_video_and_stale_revision(
    tmp_path: Path,
) -> None:
    database, storage, runner, video_id, fps = _build_context(tmp_path)
    source_action = _create_corrected_action(database, video_id, fps)
    suggestion_id = _create_suggestion(database, video_id, source_action, fps)
    service = ReprocessingService(database, storage, runner, prefer_csrt=False)
    try:
        with pytest.raises(ReprocessingNotFoundError, match="not found"):
            service.accept_suggestion(
                "another-video",
                suggestion_id,
                expected_revision=1,
                reviewer_session_id="reviewer-test",
            )
        with pytest.raises(RevisionConflictError) as stale:
            service.accept_suggestion(
                video_id,
                suggestion_id,
                expected_revision=0,
                reviewer_session_id="reviewer-test",
            )
        assert stale.value.actual == 1
        with database.session() as session:
            suggestion = session.get(ReprocessingSuggestion, suggestion_id)
            assert suggestion is not None
            assert suggestion.status == ReprocessingSuggestionStatus.PENDING
            assert suggestion.resolved_at is None
    finally:
        runner.shutdown()


def test_reprocessing_persists_bounded_suggestions_without_mutating_truth(
    tmp_path: Path,
) -> None:
    database, storage, runner, video_id, fps = _build_context(tmp_path)
    action = _create_corrected_action(database, video_id, fps)
    service = ReprocessingService(
        database,
        storage,
        runner,
        window_seconds=1,
        prefer_csrt=False,
    )
    with database.session() as session:
        original_action = session.get(ReviewAction, action.id)
        original_track = session.get(Track, "proposal-track")
        original_keyframes = list(
            session.scalars(
                select(TrackKeyframe)
                .where(TrackKeyframe.track_id == "proposal-track")
                .order_by(TrackKeyframe.frame_index)
            )
        )
        assert original_action is not None
        assert original_track is not None
        action_state = deepcopy(original_action.after_state)
        track_values = (
            original_track.start_frame,
            original_track.end_frame,
            deepcopy(original_track.confidence_summary),
        )
        keyframe_values = [
            (
                item.id,
                item.frame_index,
                item.x1,
                item.y1,
                item.x2,
                item.y2,
                item.locked,
            )
            for item in original_keyframes
        ]

    try:
        requested = service.request(action.id)
        with database.session() as session:
            visible_job = session.get(ProcessingJob, requested.job.id)
            assert visible_job is not None
            assert visible_job.job_type == JobType.REPROCESS
            assert visible_job.payload["source_action_id"] == action.id
            assert visible_job.status in {
                JobStatus.QUEUED,
                JobStatus.RUNNING,
                JobStatus.COMPLETED,
            }
        runner.wait(requested.job.id, timeout=60)

        with database.session() as session:
            stored_action = session.get(ReviewAction, action.id)
            stored_track = session.get(Track, "proposal-track")
            stored_keyframes = list(
                session.scalars(
                    select(TrackKeyframe)
                    .where(TrackKeyframe.track_id == "proposal-track")
                    .order_by(TrackKeyframe.frame_index)
                )
            )
            job = session.get(ProcessingJob, requested.job.id)
            suggestions = list(
                session.scalars(
                    select(ReprocessingSuggestion)
                    .where(ReprocessingSuggestion.source_action_id == action.id)
                    .order_by(ReprocessingSuggestion.frame_index)
                )
            )
            assert stored_action is not None
            assert stored_track is not None
            assert job is not None
            assert job.status == JobStatus.COMPLETED
            assert job.payload["suggestion_count"] == 14
            assert stored_action.after_state == action_state
            assert (
                stored_track.start_frame,
                stored_track.end_frame,
                stored_track.confidence_summary,
            ) == track_values
            assert [
                (
                    item.id,
                    item.frame_index,
                    item.x1,
                    item.y1,
                    item.x2,
                    item.y2,
                    item.locked,
                )
                for item in stored_keyframes
            ] == keyframe_values
            assert len(suggestions) == 14
            assert {item.propagation_method for item in suggestions} == {
                "interpolation"
            }
            assert {item.direction for item in suggestions} == {
                "backward",
                "forward",
            }
            assert all(0 <= item.frame_index <= 14 for item in suggestions)
            assert all(item.frame_index != 7 for item in suggestions)
            assert all(item.seed_locked for item in suggestions)

        with pytest.raises(ReprocessingConflictError, match="already has"):
            service.request(action.id)
    finally:
        runner.shutdown()


def test_reprocessing_discards_suggestions_when_manual_source_is_undone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, storage, runner, video_id, fps = _build_context(tmp_path)
    with database.session() as session:
        action, _ = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.CREATE_MANUAL_REGION,
                expected_revision=0,
                frame_index=7,
                timestamp_ms=round(7 * 1000 / fps),
                payload={
                    "class_name": "face",
                    "start_frame": 4,
                    "end_frame": 10,
                    "start_ms": round(4 * 1000 / fps),
                    "end_ms": round(10 * 1000 / fps),
                    "bbox": {
                        "x1": 0.40,
                        "y1": 0.20,
                        "x2": 0.55,
                        "y2": 0.45,
                    },
                },
            ),
            reviewer_session_id="manual-reviewer",
        )

    service = ReprocessingService(
        database,
        storage,
        runner,
        window_seconds=1,
        prefer_csrt=False,
    )
    started, resume = _pause_fallback(service, monkeypatch)
    try:
        requested = service.request(action.id)
        assert started.wait(timeout=30)
        with database.session() as session:
            append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.UNDO,
                    expected_revision=1,
                    payload={"action_id": action.id},
                ),
                reviewer_session_id="manual-reviewer",
            )
        resume.set()
        runner.wait(requested.job.id, timeout=60)

        with database.session() as session:
            job = session.get(ProcessingJob, requested.job.id)
            suggestions = list(
                session.scalars(
                    select(ReprocessingSuggestion).where(
                        ReprocessingSuggestion.source_action_id == action.id
                    )
                )
            )
            assert job is not None
            assert job.status == JobStatus.COMPLETED
            assert job.payload["suggestion_count"] == 0
            assert job.payload["propagation_methods"] == []
            assert job.payload["discard_reason"] == "source track is no longer current"
            assert suggestions == []
    finally:
        resume.set()
        runner.shutdown()


def test_reprocessing_discards_suggestions_when_seed_geometry_is_superseded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, storage, runner, video_id, fps = _build_context(tmp_path)
    action = _create_corrected_action(database, video_id, fps)
    service = ReprocessingService(
        database,
        storage,
        runner,
        window_seconds=1,
        prefer_csrt=False,
    )
    started, resume = _pause_fallback(service, monkeypatch)
    try:
        requested = service.request(action.id)
        assert started.wait(timeout=30)
        with database.session() as session:
            append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.RESIZE_REGION,
                    expected_revision=1,
                    track_id="proposal-track",
                    frame_index=7,
                    timestamp_ms=round(7 * 1000 / fps),
                    payload={
                        "bbox": {
                            "x1": 0.34,
                            "y1": 0.54,
                            "x2": 0.64,
                            "y2": 0.71,
                        }
                    },
                ),
                reviewer_session_id="reviewer-test",
            )
        resume.set()
        runner.wait(requested.job.id, timeout=60)

        with database.session() as session:
            job = session.get(ProcessingJob, requested.job.id)
            suggestions = list(
                session.scalars(
                    select(ReprocessingSuggestion).where(
                        ReprocessingSuggestion.source_action_id == action.id
                    )
                )
            )
            assert job is not None
            assert job.status == JobStatus.COMPLETED
            assert job.payload["suggestion_count"] == 0
            assert job.payload["discard_reason"] == "source seed geometry changed"
            assert suggestions == []
    finally:
        resume.set()
        runner.shutdown()


def test_manual_region_uses_static_fallback_within_declared_span(tmp_path: Path) -> None:
    database, storage, runner, video_id, fps = _build_context(tmp_path)
    with database.session() as session:
        action, _ = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.CREATE_MANUAL_REGION,
                expected_revision=0,
                frame_index=7,
                timestamp_ms=round(7 * 1000 / fps),
                payload={
                    "class_name": "face",
                    "start_frame": 4,
                    "end_frame": 10,
                    "start_ms": round(4 * 1000 / fps),
                    "end_ms": round(10 * 1000 / fps),
                    "bbox": {
                        "x1": 0.40,
                        "y1": 0.20,
                        "x2": 0.55,
                        "y2": 0.45,
                    },
                },
            ),
            reviewer_session_id="manual-reviewer",
        )

    service = ReprocessingService(
        database,
        storage,
        runner,
        window_seconds=1,
        prefer_csrt=False,
    )
    try:
        requested = service.request(action.id)
        runner.wait(requested.job.id, timeout=60)
        suggestions = service.suggestions_for_action(action.id)

        assert [item.frame_index for item in suggestions] == [4, 5, 6, 8, 9, 10]
        assert {item.propagation_method for item in suggestions} == {"static"}
        assert all(
            (item.x1, item.y1, item.x2, item.y2) == (0.4, 0.2, 0.55, 0.45)
            for item in suggestions
        )
        with database.session() as session:
            assert session.get(Track, action.track_id) is None
            stored_action = session.get(ReviewAction, action.id)
            assert stored_action is not None
            corrected = stored_action.after_state["tracks"][0]["keyframes"][0]
            assert corrected["locked"] is True
    finally:
        runner.shutdown()
