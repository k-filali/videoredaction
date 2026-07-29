from copy import deepcopy
from math import ceil
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from clearframe import __version__
from clearframe.domain.enums import (
    RedactionClass,
    ReviewActionType,
    TrackSource,
    VideoStatus,
)
from clearframe.domain.geometry import NormalizedBox
from clearframe.domain.review import (
    ReviewCommand,
    ReviewKeyframe,
    ReviewSnapshot,
    TrackReviewState,
)
from clearframe.models import ReviewAction, Track, TrackKeyframe, VideoAsset
from clearframe.rendering import box_at_frame


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


class ReviewUnavailableError(ReviewError):
    pass


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

    active_track_scope = or_(
        Track.model_run_id.is_(None),
        Track.model_run_id == video.active_model_run_id,
    )
    tracks = list(
        session.scalars(
            select(Track)
            .where(Track.video_id == video_id, active_track_scope)
            .order_by(Track.start_ms)
        )
    )
    keyframes = list(
        session.scalars(
            select(TrackKeyframe)
            .join(Track, Track.id == TrackKeyframe.track_id)
            .where(Track.video_id == video_id, active_track_scope)
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


def _integer_from_payload(
    command: ReviewCommand,
    key: str,
    default: int,
) -> int:
    value = command.payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReviewError(f"{key} must be an integer")
    if value < 0:
        raise ReviewError(f"{key} cannot be negative")
    return cast(int, value)


def _validated_state(state: TrackReviewState) -> TrackReviewState:
    try:
        return TrackReviewState.model_validate(state.model_dump())
    except ValueError as exc:
        raise ReviewError("review action would create an invalid track span") from exc


def _redaction_class(value: object) -> RedactionClass:
    if isinstance(value, RedactionClass):
        return value
    if not isinstance(value, str):
        raise ReviewError("class_name is not a supported redaction class")
    try:
        return RedactionClass(value)
    except ValueError as exc:
        raise ReviewError("class_name is not a supported redaction class") from exc


def _bulk_accept_targets(
    snapshot: ReviewSnapshot,
    command: ReviewCommand,
) -> list[TrackReviewState]:
    class_filter: RedactionClass | None = None
    raw_class = command.payload.get("class_name")
    if raw_class is not None:
        class_filter = _redaction_class(raw_class)

    min_confidence: float | None = None
    raw_confidence = command.payload.get("min_confidence")
    if raw_confidence is not None:
        if isinstance(raw_confidence, bool) or not isinstance(
            raw_confidence, int | float
        ):
            raise ReviewError("min_confidence must be a number between 0 and 1")
        min_confidence = float(raw_confidence)
        if not 0.0 <= min_confidence <= 1.0:
            raise ReviewError("min_confidence must be a number between 0 and 1")

    start_ms = command.payload.get("start_ms")
    end_ms = command.payload.get("end_ms")
    if (start_ms is None) != (end_ms is None):
        raise ReviewError("time-scoped bulk accept requires both start_ms and end_ms")
    if (start_ms is not None and end_ms is not None) and (
        isinstance(start_ms, bool)
        or isinstance(end_ms, bool)
        or not isinstance(start_ms, int)
        or not isinstance(end_ms, int)
        or start_ms < 0
        or end_ms < start_ms
    ):
        raise ReviewError("bulk accept time range is invalid")

    targets = []
    for state in snapshot.tracks.values():
        if state.accepted:
            continue
        if class_filter is not None and state.class_name != class_filter:
            continue
        if min_confidence is not None and (
            state.confidence is None or state.confidence < min_confidence
        ):
            continue
        if (
            start_ms is not None
            and end_ms is not None
            and (state.end_ms < start_ms or state.start_ms > end_ms)
        ):
            continue
        targets.append(state)
    if not targets:
        raise ReviewError("no unconfirmed tracks match the requested scope")
    return sorted(targets, key=lambda state: state.track_id)


def _validate_video_bounds(
    video: VideoAsset,
    states: list[TrackReviewState],
) -> None:
    if video.duration_ms is not None and any(
        state.end_ms > video.duration_ms for state in states
    ):
        raise ReviewError("track timestamp exceeds the video duration")
    if video.duration_ms is not None and video.fps is not None and video.fps > 0:
        last_frame = max(0, ceil(video.duration_ms * video.fps / 1000) - 1)
        if any(state.end_frame > last_frame for state in states):
            raise ReviewError("track frame exceeds the video frame range")


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
        class_name = _redaction_class(
            command.payload.get("class_name", RedactionClass.LICENSE_PLATE)
        )
        start_ms = _integer_from_payload(command, "start_ms", command.timestamp_ms)
        end_ms = _integer_from_payload(command, "end_ms", command.timestamp_ms)
        start_frame = _integer_from_payload(command, "start_frame", command.frame_index)
        end_frame = _integer_from_payload(command, "end_frame", command.frame_index)
        try:
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
        except ValueError as exc:
            raise ReviewError("manual region has an invalid track span") from exc
        command.track_id = track_id
        return [], [created], inverse_of

    if action_type == ReviewActionType.BULK_ACCEPT:
        targets = _bulk_accept_targets(snapshot, command)
        before = [state.model_copy(deep=True) for state in targets]
        after = []
        for state in targets:
            accepted = state.model_copy(deep=True)
            accepted.accepted = True
            after.append(_validated_state(accepted))
        return before, after, inverse_of

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
        updated.class_name = _redaction_class(command.payload.get("class_name"))
        updated.accepted = True
    elif action_type in {ReviewActionType.EXTEND_TRACK, ReviewActionType.TRIM_TRACK}:
        start_ms = _integer_from_payload(command, "start_ms", updated.start_ms)
        end_ms = _integer_from_payload(command, "end_ms", updated.end_ms)
        start_frame = _integer_from_payload(command, "start_frame", updated.start_frame)
        end_frame = _integer_from_payload(command, "end_frame", updated.end_frame)
        if start_ms > end_ms or start_frame > end_frame:
            raise ReviewError("track start cannot follow track end")
        if action_type == ReviewActionType.TRIM_TRACK:
            if (
                start_ms < current.start_ms
                or end_ms > current.end_ms
                or start_frame < current.start_frame
                or end_frame > current.end_frame
            ):
                raise ReviewError("trim must stay inside the current track span")
            if (
                start_ms == current.start_ms
                and end_ms == current.end_ms
                and start_frame == current.start_frame
                and end_frame == current.end_frame
            ):
                raise ReviewError("trim must narrow the track span")
        else:
            if (
                start_ms > current.start_ms
                or end_ms < current.end_ms
                or start_frame > current.start_frame
                or end_frame < current.end_frame
            ):
                raise ReviewError("extension must contain the current track span")
            if (
                start_ms == current.start_ms
                and end_ms == current.end_ms
                and start_frame == current.start_frame
                and end_frame == current.end_frame
            ):
                raise ReviewError("extension must widen the track span")

        if not current.keyframes:
            raise ReviewError("track span requires existing geometry")

        updated.start_ms = start_ms
        updated.end_ms = end_ms
        updated.start_frame = start_frame
        updated.end_frame = end_frame
        if action_type == ReviewActionType.TRIM_TRACK:
            boundary_source = current.model_copy(
                update={"active": True, "redacted": True}
            )
            start_box = box_at_frame(boundary_source, start_frame)
            end_box = box_at_frame(boundary_source, end_frame)
            if start_box is None or end_box is None:
                raise ReviewError("track span requires existing geometry")
            updated.keyframes = [
                keyframe
                for keyframe in updated.keyframes
                if (
                    start_frame <= keyframe.frame_index <= end_frame
                    and start_ms <= keyframe.timestamp_ms <= end_ms
                )
            ]
            updated.upsert_keyframe(
                ReviewKeyframe(
                    frame_index=start_frame,
                    timestamp_ms=start_ms,
                    bbox=start_box,
                )
            )
            if end_frame != start_frame:
                updated.upsert_keyframe(
                    ReviewKeyframe(
                        frame_index=end_frame,
                        timestamp_ms=end_ms,
                        bbox=end_box,
                    )
                )
        updated.accepted = True
    elif action_type == ReviewActionType.SPLIT_TRACK:
        split_ms = _integer_from_payload(command, "split_ms", -1)
        split_frame = _integer_from_payload(command, "split_frame", -1)
        if not (
            updated.start_ms < split_ms < updated.end_ms
            and updated.start_frame < split_frame < updated.end_frame
        ):
            raise ReviewError("split point must be inside the track span")
        boundary_source = current.model_copy(
            update={"active": True, "redacted": True}
        )
        split_box = box_at_frame(boundary_source, split_frame)
        if split_box is None:
            raise ReviewError("split point requires existing geometry")
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
        boundary_keyframe = ReviewKeyframe(
            frame_index=split_frame,
            timestamp_ms=split_ms,
            bbox=split_box,
        )
        updated.upsert_keyframe(boundary_keyframe)
        child.upsert_keyframe(boundary_keyframe)
        updated.accepted = True
        child.accepted = True
        return before, [_validated_state(updated), _validated_state(child)], inverse_of
    elif action_type == ReviewActionType.MERGE_TRACKS:
        other_id = command.payload.get("other_track_id")
        if not isinstance(other_id, str):
            raise ReviewError("other_track_id is required")
        if other_id == updated.track_id:
            raise ReviewError("a track cannot be merged with itself")
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
        return before, [_validated_state(updated), _validated_state(other)], inverse_of
    else:
        raise ReviewError(f"action is not a track edit: {action_type}")

    return before, [_validated_state(updated)], inverse_of


def _affected_track_ids(payload: dict[str, Any]) -> set[str]:
    track_ids = {
        TrackReviewState.model_validate(raw).track_id
        for raw in payload.get("tracks", [])
    }
    track_ids.update(str(item) for item in payload.get("removed_track_ids", []))
    return track_ids


def _payload_matches_snapshot(
    snapshot: ReviewSnapshot,
    payload: dict[str, Any],
) -> bool:
    for raw_state in payload.get("tracks", []):
        expected = TrackReviewState.model_validate(raw_state)
        if snapshot.tracks.get(expected.track_id) != expected:
            return False
    return all(
        str(track_id) not in snapshot.tracks
        for track_id in payload.get("removed_track_ids", [])
    )


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
    if target.action_type in {
        ReviewActionType.UNDO,
        ReviewActionType.REDO,
        ReviewActionType.EXPORT_REQUESTED,
        ReviewActionType.EXPORT_COMPLETED,
    }:
        raise ReviewError("target action cannot be undone or redone")

    latest_inverse = session.scalar(
        select(ReviewAction)
        .where(
            ReviewAction.video_id == snapshot.video_id,
            ReviewAction.inverse_of_action_id == target.id,
        )
        .order_by(ReviewAction.revision.desc())
    )
    if command.action_type == ReviewActionType.UNDO:
        if (
            latest_inverse is not None
            and latest_inverse.action_type == ReviewActionType.UNDO
        ):
            raise ReviewError("target action is already undone")
        expected_current = target.after_state
    else:
        if (
            latest_inverse is None
            or latest_inverse.action_type != ReviewActionType.UNDO
        ):
            raise ReviewError("target action is not currently undone")
        expected_current = target.before_state

    affected_ids = _affected_track_ids(target.before_state)
    affected_ids.update(_affected_track_ids(target.after_state))
    later_actions = session.scalars(
        select(ReviewAction)
        .where(
            ReviewAction.video_id == snapshot.video_id,
            ReviewAction.revision > target.revision,
        )
        .order_by(ReviewAction.revision)
    )
    latest_touch: ReviewAction | None = None
    for action in later_actions:
        action_ids = _affected_track_ids(action.before_state)
        action_ids.update(_affected_track_ids(action.after_state))
        if affected_ids & action_ids:
            latest_touch = action
    if latest_touch is not None and (
        latest_inverse is None or latest_touch.id != latest_inverse.id
    ):
        raise ReviewError("a newer action has changed the target track")
    if not _payload_matches_snapshot(snapshot, expected_current):
        raise ReviewError("review state no longer matches the target action")

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
    *,
    commit: bool = True,
) -> tuple[ReviewAction, ReviewSnapshot]:
    video = session.get(VideoAsset, video_id)
    if video is None:
        raise VideoNotFoundError(video_id)
    if video.review_revision != command.expected_revision:
        raise RevisionConflictError(command.expected_revision, video.review_revision)
    if video.status not in {
        VideoStatus.READY_FOR_REVIEW,
        VideoStatus.EXPORTING,
        VideoStatus.EXPORTED,
    }:
        raise ReviewUnavailableError("review is unavailable while video processing is active")

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
    _validate_video_bounds(
        video,
        [
            TrackReviewState.model_validate(raw)
            for raw in after_payload.get("tracks", [])
        ],
    )

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
    if commit:
        session.commit()
    else:
        session.flush()

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
