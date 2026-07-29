from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import DatabaseError

from clearframe.database import Database
from clearframe.domain.enums import ReviewActionType, TrackSource, VideoStatus
from clearframe.domain.review import ReviewCommand
from clearframe.models import (
    ImmutableAuditError,
    ReviewAction,
    Track,
    TrackKeyframe,
    VideoAsset,
)
from clearframe.services.review import (
    ReviewError,
    ReviewUnavailableError,
    RevisionConflictError,
    append_review_action,
    build_review_snapshot,
)


def create_database(path: Path) -> Database:
    database = Database(f"sqlite:///{path.as_posix()}")
    database.create_schema()
    return database


def seed_video(database: Database) -> tuple[str, str]:
    with database.session() as session:
        video = VideoAsset(
            original_filename="synthetic.mp4",
            safe_filename="synthetic.mp4",
            content_type="video/mp4",
            status=VideoStatus.READY_FOR_REVIEW,
        )
        session.add(video)
        session.flush()
        track = Track(
            video_id=video.id,
            class_name="license_plate",
            start_frame=0,
            end_frame=10,
            start_ms=0,
            end_ms=1000,
            source=TrackSource.MODEL,
            confidence_summary={"mean": 0.92},
        )
        session.add(track)
        session.flush()
        session.add(
            TrackKeyframe(
                track_id=track.id,
                frame_index=0,
                timestamp_ms=0,
                x1=0.1,
                y1=0.2,
                x2=0.3,
                y2=0.4,
                source=TrackSource.MODEL,
            )
        )
        session.commit()
        return video.id, track.id


def test_schema_and_default_review_state(tmp_path: Path) -> None:
    database = create_database(tmp_path / "schema.db")
    video_id, track_id = seed_video(database)

    assert "review_actions" in inspect(database.engine).get_table_names()
    with database.session() as session:
        snapshot = build_review_snapshot(session, video_id)

    assert snapshot.revision == 0
    assert snapshot.tracks[track_id].redacted is True
    assert snapshot.tracks[track_id].accepted is False
    assert snapshot.tracks[track_id].keyframes[0].bbox.as_list() == [0.1, 0.2, 0.3, 0.4]


def test_actions_persist_and_stale_writes_are_rejected(tmp_path: Path) -> None:
    database = create_database(tmp_path / "actions.db")
    video_id, track_id = seed_video(database)

    with database.session() as session:
        action, snapshot = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.RESTORE_TRACK,
                expected_revision=0,
                track_id=track_id,
            ),
            reviewer_session_id="reviewer-a",
        )

        assert action.revision == 1
        assert snapshot.tracks[track_id].redacted is False
        with pytest.raises(RevisionConflictError):
            append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.REDACT_TRACK,
                    expected_revision=0,
                    track_id=track_id,
                ),
                reviewer_session_id="reviewer-b",
            )

    with database.session() as session:
        replayed = build_review_snapshot(session, video_id)
        assert replayed.revision == 1
        assert replayed.tracks[track_id].redacted is False


def test_manual_region_can_be_undone_and_redone(tmp_path: Path) -> None:
    database = create_database(tmp_path / "undo.db")
    video_id, _ = seed_video(database)

    with database.session() as session:
        created, snapshot = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.CREATE_MANUAL_REGION,
                expected_revision=0,
                frame_index=5,
                timestamp_ms=500,
                payload={
                    "class_name": "license_plate",
                    "bbox": {"x1": 0.55, "y1": 0.6, "x2": 0.75, "y2": 0.72},
                },
            ),
            reviewer_session_id="reviewer-a",
        )
        manual_track_id = created.track_id
        assert manual_track_id is not None
        assert manual_track_id in snapshot.tracks

        _, undone = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.UNDO,
                expected_revision=1,
                payload={"action_id": created.id},
            ),
            reviewer_session_id="reviewer-a",
        )
        assert manual_track_id not in undone.tracks

        _, redone = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.REDO,
                expected_revision=2,
                payload={"action_id": created.id},
            ),
            reviewer_session_id="reviewer-a",
        )
        assert redone.tracks[manual_track_id].source == TrackSource.MANUAL


def test_merge_rejects_the_same_track(tmp_path: Path) -> None:
    database = create_database(tmp_path / "self-merge.db")
    video_id, track_id = seed_video(database)

    with database.session() as session:
        with pytest.raises(ReviewError, match="merged with itself"):
            append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.MERGE_TRACKS,
                    expected_revision=0,
                    track_id=track_id,
                    payload={"other_track_id": track_id},
                ),
                reviewer_session_id="reviewer-a",
            )
        assert session.query(ReviewAction).count() == 0


def test_redo_requires_an_active_undo_chain(tmp_path: Path) -> None:
    database = create_database(tmp_path / "redo-chain.db")
    video_id, track_id = seed_video(database)

    with database.session() as session:
        restored, _ = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.RESTORE_TRACK,
                expected_revision=0,
                track_id=track_id,
            ),
            reviewer_session_id="reviewer-a",
        )
        append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.REDACT_TRACK,
                expected_revision=1,
                track_id=track_id,
            ),
            reviewer_session_id="reviewer-a",
        )

        with pytest.raises(ReviewError, match="not currently undone"):
            append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.REDO,
                    expected_revision=2,
                    payload={"action_id": restored.id},
                ),
                reviewer_session_id="reviewer-a",
            )
        replayed = build_review_snapshot(session, video_id)
        assert replayed.revision == 2
        assert replayed.tracks[track_id].redacted is True


def test_undo_rejects_a_superseded_track_edit(tmp_path: Path) -> None:
    database = create_database(tmp_path / "stale-undo.db")
    video_id, track_id = seed_video(database)

    with database.session() as session:
        restored, _ = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.RESTORE_TRACK,
                expected_revision=0,
                track_id=track_id,
            ),
            reviewer_session_id="reviewer-a",
        )
        append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.REDACT_TRACK,
                expected_revision=1,
                track_id=track_id,
            ),
            reviewer_session_id="reviewer-a",
        )

        with pytest.raises(ReviewError, match="newer action"):
            append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.UNDO,
                    expected_revision=2,
                    payload={"action_id": restored.id},
                ),
                reviewer_session_id="reviewer-a",
            )
        assert build_review_snapshot(session, video_id).tracks[track_id].redacted is True


def test_review_actions_are_append_only(tmp_path: Path) -> None:
    database = create_database(tmp_path / "immutable.db")
    video_id, track_id = seed_video(database)

    with database.session() as session:
        action, _ = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.ACCEPT_PROPOSAL,
                expected_revision=0,
                track_id=track_id,
            ),
            reviewer_session_id="reviewer-a",
        )
        stored = session.get(ReviewAction, action.id)
        assert stored is not None
        action_id = action.id
        stored.reason_code = "changed"
        with pytest.raises(ImmutableAuditError, match="append-only"):
            session.commit()
        session.rollback()

    with database.session() as session:
        with pytest.raises(DatabaseError, match="append-only"):
            session.execute(
                text("UPDATE review_actions SET reason_code = 'bypass' WHERE id = :id"),
                {"id": action_id},
            )
        session.rollback()

        with pytest.raises(DatabaseError, match="append-only"):
            session.execute(
                text("DELETE FROM review_actions WHERE id = :id"),
                {"id": action_id},
            )
        session.rollback()

        with pytest.raises(DatabaseError, match="append-only"):
            session.execute(
                text("DELETE FROM video_assets WHERE id = :id"),
                {"id": video_id},
            )
        session.rollback()


def test_invalid_track_span_is_rejected_before_audit_append(tmp_path: Path) -> None:
    database = create_database(tmp_path / "invalid-span.db")
    video_id, track_id = seed_video(database)

    with database.session() as session:
        with pytest.raises(ReviewError, match="cannot be negative"):
            append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.TRIM_TRACK,
                    expected_revision=0,
                    track_id=track_id,
                    payload={"start_frame": -1, "start_ms": -100},
                ),
                reviewer_session_id="reviewer-a",
            )
        assert session.query(ReviewAction).count() == 0
        assert build_review_snapshot(session, video_id).revision == 0


@pytest.mark.parametrize("invalid_frame", [True, 1.5, float("inf"), "2"])
def test_track_spans_require_strict_integers(
    tmp_path: Path,
    invalid_frame: object,
) -> None:
    database = create_database(tmp_path / f"strict-{type(invalid_frame).__name__}.db")
    video_id, track_id = seed_video(database)

    with database.session() as session:
        with pytest.raises(ReviewError, match="must be an integer"):
            append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.TRIM_TRACK,
                    expected_revision=0,
                    track_id=track_id,
                    payload={"start_frame": invalid_frame},
                ),
                reviewer_session_id="reviewer-a",
            )
        assert session.query(ReviewAction).count() == 0


def test_review_class_and_media_bounds_are_enforced(tmp_path: Path) -> None:
    database = create_database(tmp_path / "review-bounds.db")
    video_id, track_id = seed_video(database)

    with database.session() as session:
        video = session.get(VideoAsset, video_id)
        assert video is not None
        video.duration_ms = 1100
        video.fps = 10
        session.commit()

        with pytest.raises(ReviewError, match="supported redaction class"):
            append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.CHANGE_CLASS,
                    expected_revision=0,
                    track_id=track_id,
                    payload={"class_name": "identity"},
                ),
                reviewer_session_id="reviewer-a",
            )
        with pytest.raises(ReviewError, match="video frame range"):
            append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.EXTEND_TRACK,
                    expected_revision=0,
                    track_id=track_id,
                    payload={"end_frame": 11, "end_ms": 1100},
                ),
                reviewer_session_id="reviewer-a",
            )
        assert session.query(ReviewAction).count() == 0


def test_review_is_blocked_during_processing(tmp_path: Path) -> None:
    database = create_database(tmp_path / "processing-review.db")
    video_id, track_id = seed_video(database)
    with database.session() as session:
        video = session.get(VideoAsset, video_id)
        assert video is not None
        video.status = VideoStatus.PROCESSING
        session.commit()

        with pytest.raises(ReviewUnavailableError):
            append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.ACCEPT_PROPOSAL,
                    expected_revision=0,
                    track_id=track_id,
                ),
                reviewer_session_id="reviewer-a",
            )
        assert session.query(ReviewAction).count() == 0
