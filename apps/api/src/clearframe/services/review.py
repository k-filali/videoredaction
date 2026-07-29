from copy import deepcopy
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from clearframe import __version__
from clearframe.domain.enums import ReviewActionType, TrackSource
from clearframe.domain.geometry import NormalizedBox
from clearframe.domain.review import (
    ReviewCommand,
    ReviewKeyframe,
    ReviewSnapshot,
    TrackReviewState,
)
from clearframe.models import ReviewAction, Track, TrackKeyframe, VideoAsset


class ReviewError(ValueError):
    pass


class VideoNotFoundError(ReviewError):
    pass


class TrackNotFoundError(ReviewError):
    pass


class RevisionConflictError(ReviewError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"review revision conflict: expected {expected}, current {actual}")
        self.expected = expected
        self.actual = actual


def _state_payload(
    states: list[TrackReviewState],
    removed_track_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "tracks": [state.model_dump(mode="json") for state in states],
        "removed_track_ids": removed_track_ids or [],
    }


def _apply_state_payload(snapshot: ReviewSnapshot, payload: dict[str, Any]) -> None:
    for track_id in payload.get("removed_track_ids", []):
        snapshot.tracks.pop(str(track_id), None)
    for raw_state in payload.get("tracks", []):
        state = TrackReviewState.model_validate(raw_state)
        snapshot.tracks[state.track_id] = state


def build_review_snapshot(
    session: Session,
    video_id: str,
    *,
    through_revision: int | None = None,
) -> ReviewSnapshot:
    video = session.get(VideoAsset, video_id)
    if video is None:
        raise VideoNotFoundError(video_id)

    tracks = list(
        session.scalars(select(Track).where(Track.video_id == video_id).order_by(Track.start_ms))
    )
    keyframes = list(
        session.scalars(
            select(TrackKeyframe)
            .join(Track, Track.id == TrackKeyframe.track_id)
            .where(Track.video_id == video_id)
            .order_by(TrackKeyframe.frame_index)
        )
    )
    grouped_keyframes: dict[str, list[ReviewKeyframe]] = {}
    for keyframe in keyframes:
        grouped_keyframes.setdefault(keyframe.track_id, []).append(
            ReviewKeyframe(
                frame_index=keyframe.frame_index,
                timestamp_ms=keyframe.timestamp_ms,
                bbox=NormalizedBox(
                    x1=keyframe.x1,
                    y1=keyframe.y1,
                    x2=keyframe.x2,
                    y2=keyframe.y2,
                ),
                locked=keyframe.locked,
            )
        )

    snapshot = ReviewSnapshot(video_id=video_id, revision=0)
    for track in tracks:
        confidence = track.confidence_summary.get("mean")
        snapshot.tracks[track.id] = TrackReviewState(
            track_id=track.id,
            class_name=track.class_name,
            source=TrackSource(track.source),
            redacted=track.default_redacted,
            start_frame=track.start_frame,
            end_frame=track.end_frame,
            start_ms=track.start_ms,
            end_ms=track.end_ms,
            confidence=float(confidence) if confidence is not None else None,
            warning=track.warning,
            keyframes=grouped_keyframes.get(track.id, []),
        )

    actions_query = select(ReviewAction).where(ReviewAction.video_id == video_id)
    if through_revision is not None:
        actions_query = actions_query.where(ReviewAction.revision <= through_revision)
    actions = session.scalars(actions_query.order_by(ReviewAction.revision))
    for action in actions:
        _apply_state_payload(snapshot, action.after_state)
        snapshot.revision = action.revision

    return snapshot


def build_review_snapshot_at_revision(
    session: Session,
    video_id: str,
    revision: int,
) -> ReviewSnapshot:
    return build_review_snapshot(session, video_id, through_revision=revision)


def _required_target(snapshot: ReviewSnapshot, track_id: str | None) -> TrackReviewState:
    if track_id is None or track_id not in snapshot.tracks:
        raise TrackNotFoundError(track_id or "missing track id")
    return snapshot.tracks[track_id].model_copy(deep=True)


def _bbox_from_payload(command: ReviewCommand) -> NormalizedBox:
    try:
        return NormalizedBox.model_validate(command.payload["bbox"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewError("payload must include a valid normalized bbox") from exc


def _updated_states(
    snapshot: ReviewSnapshot,
    command: ReviewCommand,
) -> tuple[list[TrackReviewState], list[TrackReviewState], str | None]:
    action_type = command.action_type
    inverse_of: str | None = None

    if action_type == ReviewActionType.CREATE_MANUAL_REGION:
        bbox = _bbox_from_payload(command)
        if command.frame_index is None or command.timestamp_ms is None:
            raise ReviewError("manual regions require frame_index and timestamp_ms")
        track_id = command.track_id or str(uuid4())
        if track_id in snapshot.tracks:
            raise ReviewError("manual track id already exists")
        class_name = str(command.payload.get("class_name", "license_plate"))
        start_ms = int(command.payload.get("start_ms", command.timestamp_ms))
        end_ms = int(command.payload.get("end_ms", command.timestamp_ms))
        start_frame = int(command.payload.get("start_frame", command.frame_index))
        end_frame = int(command.payload.get("end_frame", command.frame_index))
        created = TrackReviewState(
            track_id=track_id,
            class_name=class_name,
            source=TrackSource.MANUAL,
            redacted=True,
            accepted=True,
            start_frame=start_frame,
            end_frame=end_frame,
            start_ms=start_ms,
            end_ms=end_ms,
            keyframes=[
                ReviewKeyframe(
                    frame_index=command.frame_index,
                    timestamp_ms=command.timestamp_ms,
                    bbox=bbox,
                )
            ],
        )
        command.track_id = track_id
        return [], [created], inverse_of

    if action_type in {ReviewActionType.UNDO, ReviewActionType.REDO}:
        raise ReviewError("undo and redo must be resolved by the review service")

    current = _required_target(snapshot, command.track_id)
    before = [current.model_copy(deep=True)]
    updated = current.model_copy(deep=True)

    if action_type == ReviewActionType.ACCEPT_PROPOSAL:
        updated.accepted = True
    elif action_type == ReviewActionType.RESTORE_TRACK:
        updated.redacted = False
        updated.accepted = True
    elif action_type == ReviewActionType.REDACT_TRACK:
        updated.active = True
        updated.redacted = True
        updated.accepted = True
    elif action_type == ReviewActionType.DELETE_FALSE_POSITIVE:
        updated.active = False
        updated.redacted = False
        updated.accepted = True
    elif action_type in {ReviewActionType.MOVE_REGION, ReviewActionType.RESIZE_REGION}:
        if command.frame_index is None or command.timestamp_ms is None:
            raise ReviewError("region edits require frame_index and timestamp_ms")
        updated.upsert_keyframe(
            ReviewKeyframe(
                frame_index=command.frame_index,
                timestamp_ms=command.timestamp_ms,
                bbox=_bbox_from_payload(command),
            )
        )
        updated.active = True
        updated.accepted = True
    elif action_type == ReviewActionType.CHANGE_CLASS:
        new_class_name = command.payload.get("class_name")
        if not isinstance(new_class_name, str) or not new_class_name:
            raise ReviewError("class_name is required")
        updated.class_name = new_class_name
        updated.accepted = True
    elif action_type in {ReviewActionType.EXTEND_TRACK, ReviewActionType.TRIM_TRACK}:
        start_ms = int(command.payload.get("start_ms", updated.start_ms))
        end_ms = int(command.payload.get("end_ms", updated.end_ms))
        start_frame = int(command.payload.get("start_frame", updated.start_frame))
        end_frame = int(command.payload.get("end_frame", updated.end_frame))
        if start_ms > end_ms or start_frame > end_frame:
            raise ReviewError("track start cannot follow track end")
        updated.start_ms = start_ms
        updated.end_ms = end_ms
        updated.start_frame = start_frame
        updated.end_frame = end_frame
        updated.accepted = True
    elif action_type == ReviewActionType.SPLIT_TRACK:
        split_ms = int(command.payload.get("split_ms", -1))
        split_frame = int(command.payload.get("split_frame", -1))
        if not (
            updated.start_ms < split_ms < updated.end_ms
            and updated.start_frame < split_frame < updated.end_frame
        ):
            raise ReviewError("split point must be inside the track span")
        child = updated.model_copy(deep=True)
        child.track_id = str(uuid4())
        child.source = TrackSource.MANUAL
        child.start_ms = split_ms
        child.start_frame = split_frame
        child.keyframes = [
            keyframe for keyframe in child.keyframes if keyframe.frame_index >= split_frame
        ]
        updated.end_ms = split_ms
        updated.end_frame = split_frame
        updated.keyframes = [
            keyframe for keyframe in updated.keyframes if keyframe.frame_index <= split_frame
        ]
        updated.accepted = True
        child.accepted = True
        return before, [updated, child], inverse_of
    elif action_type == ReviewActionType.MERGE_TRACKS:
        other_id = command.payload.get("other_track_id")
        if not isinstance(other_id, str):
            raise ReviewError("other_track_id is required")
        other = _required_target(snapshot, other_id)
        if other.class_name != updated.class_name:
            raise ReviewError("tracks must share a class before merging")
        before.append(other.model_copy(deep=True))
        updated.start_ms = min(updated.start_ms, other.start_ms)
        updated.end_ms = max(updated.end_ms, other.end_ms)
        updated.start_frame = min(updated.start_frame, other.start_frame)
        updated.end_frame = max(updated.end_frame, other.end_frame)
        for keyframe in other.keyframes:
            updated.upsert_keyframe(keyframe)
        updated.accepted = True
        other.active = False
        other.redacted = False
        other.accepted = True
        return before, [updated, other], inverse_of
    else:
        raise ReviewError(f"action is not a track edit: {action_type}")

    return before, [updated], inverse_of


def _resolve_inverse(
    session: Session,
    snapshot: ReviewSnapshot,
    command: ReviewCommand,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    target_id = command.payload.get("action_id")
    if not isinstance(target_id, str):
        raise ReviewError("action_id is required")
    target = session.get(ReviewAction, target_id)
    if target is None or target.video_id != snapshot.video_id:
        raise ReviewError("target action was not found")
    desired_payload = (
        target.before_state if command.action_type == ReviewActionType.UNDO else target.after_state
    )
    desired_states = {
        state.track_id: state
        for state in (
            TrackReviewState.model_validate(raw) for raw in desired_payload.get("tracks", [])
        )
    }
    affected_ids = set(desired_states)
    affected_ids.update(str(item) for item in desired_payload.get("removed_track_ids", []))
    affected_ids.update(
        TrackReviewState.model_validate(raw).track_id
        for raw in target.before_state.get("tracks", [])
    )
    affected_ids.update(
        TrackReviewState.model_validate(raw).track_id
        for raw in target.after_state.get("tracks", [])
    )
    before_states: list[TrackReviewState] = []
    before_removed: list[str] = []
    for track_id in sorted(affected_ids):
        current = snapshot.tracks.get(track_id)
        if current is not None:
            before_states.append(current.model_copy(deep=True))
        else:
            before_removed.append(track_id)
    command.track_id = target.track_id
    return (
        _state_payload(before_states, before_removed),
        deepcopy(desired_payload),
        target.id,
    )


def append_review_action(
    session: Session,
    video_id: str,
    command: ReviewCommand,
    reviewer_session_id: str,
) -> tuple[ReviewAction, ReviewSnapshot]:
    video = session.get(VideoAsset, video_id)
    if video is None:
        raise VideoNotFoundError(video_id)
    if video.review_revision != command.expected_revision:
        raise RevisionConflictError(command.expected_revision, video.review_revision)

    snapshot = build_review_snapshot(session, video_id)
    inverse_of: str | None
    if command.action_type in {ReviewActionType.UNDO, ReviewActionType.REDO}:
        before_payload, after_payload, inverse_of = _resolve_inverse(session, snapshot, command)
    else:
        before, after, inverse_of = _updated_states(snapshot, command)
        before_ids = {state.track_id for state in before}
        after_ids = {state.track_id for state in after}
        before_payload = _state_payload(before, sorted(after_ids - before_ids))
        after_payload = _state_payload(after, sorted(before_ids - after_ids))

    next_revision = video.review_revision + 1
    reservation = cast(
        CursorResult[Any],
        session.execute(
            update(VideoAsset)
            .where(
                VideoAsset.id == video_id,
                VideoAsset.review_revision == command.expected_revision,
            )
            .values(review_revision=next_revision)
            .execution_options(synchronize_session=False)
        )
    )
    if reservation.rowcount != 1:
        session.rollback()
        current_revision = session.scalar(
            select(VideoAsset.review_revision).where(VideoAsset.id == video_id)
        )
        raise RevisionConflictError(
            command.expected_revision,
            current_revision if current_revision is not None else 0,
        )

    action = ReviewAction(
        video_id=video_id,
        track_id=command.track_id,
        frame_index=command.frame_index,
        timestamp_ms=command.timestamp_ms,
        action_type=command.action_type,
        before_state=before_payload,
        after_state=after_payload,
        reason_code=command.reason_code,
        reviewer_session_id=reviewer_session_id,
        revision=next_revision,
        model_version=video.active_model_run_id,
        application_version=__version__,
        inverse_of_action_id=inverse_of,
    )
    session.add(action)
    session.commit()

    result = deepcopy(snapshot)
    _apply_state_payload(result, action.after_state)
    result.revision = next_revision
    return action, result


def append_system_audit_event(
    session: Session,
    video_id: str,
    action_type: ReviewActionType,
    payload: dict[str, Any],
    *,
    reviewer_session_id: str,
) -> ReviewAction:
    if action_type not in {
        ReviewActionType.EXPORT_REQUESTED,
        ReviewActionType.EXPORT_COMPLETED,
    }:
        raise ReviewError("unsupported system audit event")
    video = session.get(VideoAsset, video_id)
    if video is None:
        raise VideoNotFoundError(video_id)
    current_revision = video.review_revision
    next_revision = current_revision + 1
    reservation = cast(
        CursorResult[Any],
        session.execute(
            update(VideoAsset)
            .where(
                VideoAsset.id == video_id,
                VideoAsset.review_revision == current_revision,
            )
            .values(review_revision=next_revision)
            .execution_options(synchronize_session=False)
        ),
    )
    if reservation.rowcount != 1:
        session.rollback()
        actual = session.scalar(
            select(VideoAsset.review_revision).where(VideoAsset.id == video_id)
        )
        raise RevisionConflictError(current_revision, actual if actual is not None else 0)

    action = ReviewAction(
        video_id=video_id,
        action_type=action_type,
        before_state={"tracks": [], "removed_track_ids": []},
        after_state={"tracks": [], "removed_track_ids": [], **payload},
        reviewer_session_id=reviewer_session_id,
        revision=next_revision,
        model_version=video.active_model_run_id,
        application_version=__version__,
    )
    session.add(action)
    session.flush()
    return action
