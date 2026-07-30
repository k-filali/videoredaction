from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from math import hypot
from typing import Protocol, cast

import cv2
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from clearframe.database import Database
from clearframe.domain.enums import (
    JobStatus,
    JobType,
    ReprocessingSuggestionStatus,
    ReviewActionType,
)
from clearframe.domain.geometry import NormalizedBox
from clearframe.domain.review import ReviewCommand, ReviewSnapshot, TrackReviewState
from clearframe.jobs import JobContext, JobDispatcher
from clearframe.models import (
    ProcessingJob,
    ReprocessingSuggestion,
    ReviewAction,
    Track,
    TrackKeyframe,
    VideoAsset,
)
from clearframe.pipeline.detection import Frame
from clearframe.services.review import (
    RevisionConflictError,
    append_review_action,
    build_review_snapshot,
)
from clearframe.storage import ArtifactStorage


class ReprocessingError(ValueError):
    pass


class ReprocessingNotFoundError(ReprocessingError):
    pass


class ReprocessingValidationError(ReprocessingError):
    pass


class ReprocessingConflictError(ReprocessingError):
    pass


@dataclass(frozen=True, slots=True)
class RequestedReprocessing:
    job: ProcessingJob
    source_action_id: str


@dataclass(frozen=True, slots=True)
class SuggestionResolution:
    suggestion: ReprocessingSuggestion
    state: ReviewSnapshot
    action: ReviewAction | None


@dataclass(frozen=True, slots=True)
class _Seed:
    action_id: str
    video_id: str
    proxy_uri: str
    track_id: str
    source_revision: int
    class_name: str
    frame_index: int
    timestamp_ms: int
    bbox: NormalizedBox
    track_start_frame: int
    track_end_frame: int
    track_start_ms: int
    track_end_ms: int
    fps: float


@dataclass(frozen=True, slots=True)
class _SourcePoint:
    frame_index: int
    bbox: NormalizedBox


@dataclass(frozen=True, slots=True)
class _Candidate:
    frame_index: int
    timestamp_ms: int
    bbox: NormalizedBox
    confidence: float
    direction: str
    method: str
    distance_frames: int


class _CVTracker(Protocol):
    def init(self, frame: Frame, box: tuple[int, int, int, int]) -> bool | None: ...

    def update(
        self,
        frame: Frame,
    ) -> tuple[bool, tuple[float, float, float, float]]: ...


TrackerFactory = Callable[[], _CVTracker]
# Tracked candidates per second of footage. Reviewers do not need a suggestion
# on every frame of 60fps video, and each one costs a decode plus a tracker
# update.
_PROPAGATION_SAMPLES_PER_SECOND = 20
# Upper bound on frames held in memory for one propagation pass.
_MAX_PROPAGATION_FRAMES = 400
ALLOWED_ACTIONS = frozenset(
    {
        ReviewActionType.MOVE_REGION,
        ReviewActionType.RESIZE_REGION,
        ReviewActionType.CREATE_MANUAL_REGION,
    }
)


class ReprocessingService:
    def __init__(
        self,
        database: Database,
        storage: ArtifactStorage,
        runner: JobDispatcher,
        *,
        window_seconds: int = 3,
        prefer_csrt: bool = True,
    ) -> None:
        if not 1 <= window_seconds <= 30:
            raise ValueError("window_seconds must be between 1 and 30")
        self.database = database
        self.storage = storage
        self.runner = runner
        self.window_seconds = window_seconds
        self.prefer_csrt = prefer_csrt

    def request(self, source_action_id: str) -> RequestedReprocessing:
        with self.database.session() as session:
            seed = self._load_seed(session, source_action_id)
            existing = session.scalar(
                select(ReprocessingSuggestion.id).where(
                    ReprocessingSuggestion.source_action_id == source_action_id
                )
            )
            if existing is not None:
                raise ReprocessingConflictError("action already has reprocessing suggestions")
            active_jobs = session.scalars(
                select(ProcessingJob).where(
                    ProcessingJob.video_id == seed.video_id,
                    ProcessingJob.job_type == JobType.REPROCESS,
                    ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
                )
            )
            if any(
                job.payload.get("source_action_id") == source_action_id
                for job in active_jobs
            ):
                raise ReprocessingConflictError("action is already being reprocessed")

            window_frames = max(1, round(self.window_seconds * seed.fps))
            start_frame = max(seed.track_start_frame, seed.frame_index - window_frames)
            end_frame = min(seed.track_end_frame, seed.frame_index + window_frames)
            job = ProcessingJob(
                video_id=seed.video_id,
                job_type=JobType.REPROCESS,
                payload={
                    "source_action_id": source_action_id,
                    "source_revision": seed.source_revision,
                    "track_id": seed.track_id,
                    "seed_frame_index": seed.frame_index,
                    "window_seconds": self.window_seconds,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                },
                stage="queued",
            )
            session.add(job)
            session.commit()

        self.runner.enqueue(job.id)
        return RequestedReprocessing(job=job, source_action_id=source_action_id)

    def execute(self, context: JobContext, job_id: str) -> None:
        with self.database.session() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None or job.job_type != JobType.REPROCESS:
                raise ReprocessingValidationError("reprocessing job is invalid")
            source_action_id = job.payload.get("source_action_id")
        if not isinstance(source_action_id, str):
            raise ReprocessingValidationError("reprocessing job payload is invalid")
        self._run(
            context,
            job_id=job_id,
            source_action_id=source_action_id,
        )

    def suggestions_for_action(
        self,
        source_action_id: str,
    ) -> tuple[ReprocessingSuggestion, ...]:
        with self.database.session() as session:
            suggestions = tuple(
                session.scalars(
                    select(ReprocessingSuggestion)
                    .where(
                        ReprocessingSuggestion.source_action_id == source_action_id
                    )
                    .order_by(ReprocessingSuggestion.frame_index)
                )
            )
            for suggestion in suggestions:
                session.expunge(suggestion)
            return suggestions

    def suggestions_for_video(
        self,
        video_id: str,
    ) -> tuple[ReprocessingSuggestion, ...]:
        with self.database.session() as session:
            if session.get(VideoAsset, video_id) is None:
                raise ReprocessingNotFoundError("video not found")
            suggestions = tuple(
                session.scalars(
                    select(ReprocessingSuggestion)
                    .where(ReprocessingSuggestion.video_id == video_id)
                    .order_by(
                        ReprocessingSuggestion.source_revision,
                        ReprocessingSuggestion.frame_index,
                    )
                )
            )
            for suggestion in suggestions:
                session.expunge(suggestion)
            return suggestions

    def accept_suggestion(
        self,
        video_id: str,
        suggestion_id: str,
        *,
        expected_revision: int,
        reviewer_session_id: str,
        reason_code: str | None = None,
    ) -> SuggestionResolution:
        resolved_reason = self._resolution_metadata(
            reviewer_session_id,
            reason_code or "accepted_context_suggestion",
        )
        with self.database.session() as session:
            suggestion = self._pending_suggestion(session, video_id, suggestion_id)
            snapshot = self._review_snapshot(
                session,
                video_id,
                expected_revision,
            )
            track = snapshot.tracks.get(suggestion.track_id)
            if track is None:
                raise ReprocessingValidationError("suggestion track is not available")
            if not track.active:
                raise ReprocessingConflictError("suggestion track is no longer active")
            if track.class_name != suggestion.class_name:
                raise ReprocessingConflictError("suggestion class no longer matches the track")
            if not (
                track.start_frame <= suggestion.frame_index <= track.end_frame
                and track.start_ms <= suggestion.timestamp_ms <= track.end_ms
            ):
                raise ReprocessingConflictError("suggestion is outside the current track span")
            if any(
                keyframe.frame_index == suggestion.frame_index and keyframe.locked
                for keyframe in track.keyframes
            ):
                raise ReprocessingConflictError(
                    "a reviewer keyframe already exists at the suggestion frame"
                )
            if not suggestion.seed_locked:
                raise ReprocessingValidationError("suggestion provenance is not locked")

            action, state = append_review_action(
                session,
                video_id,
                ReviewCommand(
                    action_type=ReviewActionType.RESIZE_REGION,
                    expected_revision=expected_revision,
                    track_id=suggestion.track_id,
                    frame_index=suggestion.frame_index,
                    timestamp_ms=suggestion.timestamp_ms,
                    reason_code=resolved_reason,
                    payload={
                        "bbox": {
                            "x1": suggestion.x1,
                            "y1": suggestion.y1,
                            "x2": suggestion.x2,
                            "y2": suggestion.y2,
                        }
                    },
                ),
                reviewer_session_id=reviewer_session_id,
                commit=False,
            )
            self._resolve_pending(
                session,
                video_id,
                suggestion_id,
                status=ReprocessingSuggestionStatus.ACCEPTED,
                reviewer_session_id=reviewer_session_id,
                reason_code=resolved_reason,
                action_id=action.id,
            )
            session.commit()
            session.refresh(suggestion)
            session.expunge(suggestion)
            return SuggestionResolution(
                suggestion=suggestion,
                state=state,
                action=action,
            )

    def dismiss_suggestion(
        self,
        video_id: str,
        suggestion_id: str,
        *,
        expected_revision: int,
        reviewer_session_id: str,
        reason_code: str | None = None,
    ) -> SuggestionResolution:
        resolved_reason = self._resolution_metadata(
            reviewer_session_id,
            reason_code or "dismissed_context_suggestion",
        )
        with self.database.session() as session:
            suggestion = self._pending_suggestion(session, video_id, suggestion_id)
            snapshot = self._review_snapshot(
                session,
                video_id,
                expected_revision,
            )
            current_revision = (
                select(VideoAsset.review_revision)
                .where(VideoAsset.id == video_id)
                .scalar_subquery()
            )
            result = cast(
                CursorResult[object],
                session.execute(
                    update(ReprocessingSuggestion)
                    .where(
                        ReprocessingSuggestion.id == suggestion_id,
                        ReprocessingSuggestion.video_id == video_id,
                        ReprocessingSuggestion.status
                        == ReprocessingSuggestionStatus.PENDING,
                        current_revision == expected_revision,
                    )
                    .values(
                        status=ReprocessingSuggestionStatus.DISMISSED,
                        resolved_at=datetime.now(UTC),
                        resolved_by_session_id=reviewer_session_id,
                        resolution_reason_code=resolved_reason,
                        resolution_action_id=None,
                    )
                    .execution_options(synchronize_session=False)
                ),
            )
            if result.rowcount != 1:
                self._raise_resolution_race(
                    session,
                    video_id,
                    suggestion_id,
                    expected_revision,
                )
            session.commit()
            session.refresh(suggestion)
            session.expunge(suggestion)
            return SuggestionResolution(
                suggestion=suggestion,
                state=snapshot,
                action=None,
            )

    @staticmethod
    def _resolution_metadata(
        reviewer_session_id: str,
        reason_code: str,
    ) -> str:
        if not reviewer_session_id or len(reviewer_session_id) > 64:
            raise ReprocessingValidationError("reviewer session is invalid")
        if not reason_code or len(reason_code) > 64:
            raise ReprocessingValidationError("resolution reason is invalid")
        return reason_code

    @staticmethod
    def _pending_suggestion(
        session: Session,
        video_id: str,
        suggestion_id: str,
    ) -> ReprocessingSuggestion:
        suggestion = session.get(ReprocessingSuggestion, suggestion_id)
        if suggestion is None or suggestion.video_id != video_id:
            raise ReprocessingNotFoundError("reprocessing suggestion not found")
        if suggestion.status != ReprocessingSuggestionStatus.PENDING:
            raise ReprocessingConflictError("reprocessing suggestion is already resolved")
        return suggestion

    @staticmethod
    def _review_snapshot(
        session: Session,
        video_id: str,
        expected_revision: int,
    ) -> ReviewSnapshot:
        video = session.get(VideoAsset, video_id)
        if video is None:
            raise ReprocessingNotFoundError("video not found")
        if video.review_revision != expected_revision:
            raise RevisionConflictError(expected_revision, video.review_revision)
        return build_review_snapshot(session, video_id)

    @staticmethod
    def _resolve_pending(
        session: Session,
        video_id: str,
        suggestion_id: str,
        *,
        status: ReprocessingSuggestionStatus,
        reviewer_session_id: str,
        reason_code: str,
        action_id: str | None,
    ) -> None:
        result = cast(
            CursorResult[object],
            session.execute(
                update(ReprocessingSuggestion)
                .where(
                    ReprocessingSuggestion.id == suggestion_id,
                    ReprocessingSuggestion.video_id == video_id,
                    ReprocessingSuggestion.status
                    == ReprocessingSuggestionStatus.PENDING,
                )
                .values(
                    status=status,
                    resolved_at=datetime.now(UTC),
                    resolved_by_session_id=reviewer_session_id,
                    resolution_reason_code=reason_code,
                    resolution_action_id=action_id,
                )
                .execution_options(synchronize_session=False)
            ),
        )
        if result.rowcount != 1:
            session.rollback()
            raise ReprocessingConflictError(
                "reprocessing suggestion was resolved concurrently"
            )

    @staticmethod
    def _raise_resolution_race(
        session: Session,
        video_id: str,
        suggestion_id: str,
        expected_revision: int,
    ) -> None:
        session.rollback()
        current_revision = session.scalar(
            select(VideoAsset.review_revision).where(VideoAsset.id == video_id)
        )
        if current_revision is not None and current_revision != expected_revision:
            raise RevisionConflictError(expected_revision, current_revision)
        suggestion_status = session.scalar(
            select(ReprocessingSuggestion.status).where(
                ReprocessingSuggestion.id == suggestion_id,
                ReprocessingSuggestion.video_id == video_id,
            )
        )
        if suggestion_status is None:
            raise ReprocessingNotFoundError("reprocessing suggestion not found")
        raise ReprocessingConflictError("reprocessing suggestion is already resolved")

    def _load_seed(self, session: Session, source_action_id: str) -> _Seed:
        action = session.get(ReviewAction, source_action_id)
        if action is None:
            raise ReprocessingNotFoundError("review action not found")
        try:
            action_type = ReviewActionType(action.action_type)
        except ValueError as exc:
            raise ReprocessingValidationError("review action type is invalid") from exc
        if action_type not in ALLOWED_ACTIONS:
            raise ReprocessingValidationError(
                "only move, resize, and manual-region actions can seed reprocessing"
            )
        if (
            action.track_id is None
            or action.frame_index is None
            or action.timestamp_ms is None
        ):
            raise ReprocessingValidationError("review action is missing seed context")

        state = self._state_from_action(action)
        keyframe = next(
            (
                item
                for item in state.keyframes
                if item.frame_index == action.frame_index
            ),
            None,
        )
        if keyframe is None or not keyframe.locked:
            raise ReprocessingValidationError(
                "review action must contain a locked corrected keyframe"
            )
        if not state.start_frame <= action.frame_index <= state.end_frame:
            raise ReprocessingValidationError("seed frame is outside the corrected track span")

        video = session.get(VideoAsset, action.video_id)
        if video is None:
            raise ReprocessingNotFoundError("video not found")
        if not video.proxy_uri or not self.storage.exists(video.proxy_uri):
            raise ReprocessingValidationError("review proxy is not ready")
        if video.fps is None or video.fps <= 0:
            raise ReprocessingValidationError("video frame rate is unavailable")
        # The tracking proxy shares the review proxy's frame timing but is an
        # order of magnitude smaller to fetch. Videos ingested before it
        # existed fall back to the review proxy.
        propagation_uri = video.proxy_uri
        if video.tracking_proxy_uri and self.storage.exists(video.tracking_proxy_uri):
            propagation_uri = video.tracking_proxy_uri
        return _Seed(
            action_id=action.id,
            video_id=action.video_id,
            proxy_uri=propagation_uri,
            track_id=action.track_id,
            source_revision=action.revision,
            class_name=state.class_name,
            frame_index=action.frame_index,
            timestamp_ms=action.timestamp_ms,
            bbox=keyframe.bbox,
            track_start_frame=state.start_frame,
            track_end_frame=state.end_frame,
            track_start_ms=state.start_ms,
            track_end_ms=state.end_ms,
            fps=video.fps,
        )

    @staticmethod
    def _state_from_action(action: ReviewAction) -> TrackReviewState:
        raw_states = action.after_state.get("tracks", [])
        try:
            states = [
                TrackReviewState.model_validate(raw)
                for raw in raw_states
                if isinstance(raw, dict)
            ]
        except ValueError as exc:
            raise ReprocessingValidationError(
                "review action contains invalid corrected geometry"
            ) from exc
        for state in states:
            if state.track_id == action.track_id:
                return state
        raise ReprocessingValidationError(
            "review action does not contain its corrected track state"
        )

    def _run(
        self,
        context: JobContext,
        *,
        job_id: str,
        source_action_id: str,
    ) -> None:
        with self.database.session() as session:
            seed = self._load_seed(session, source_action_id)
            source_points = self._source_points(session, seed.track_id)

        window_frames = max(1, round(self.window_seconds * seed.fps))
        start_frame = max(seed.track_start_frame, seed.frame_index - window_frames)
        end_frame = min(seed.track_end_frame, seed.frame_index + window_frames)
        context.update(0.08, "loading corrected context")

        materialized_input = ExitStack()
        proxy_path = materialized_input.enter_context(
            self.storage.materialize_input(seed.proxy_uri)
        )
        capture = cv2.VideoCapture(str(proxy_path))
        if not capture.isOpened():
            materialized_input.close()
            raise ReprocessingValidationError("review proxy could not be opened")
        capture_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = capture_fps if capture_fps > 0 else seed.fps
        try:
            candidates: list[_Candidate] | None = None
            tracker_factory = self._csrt_factory() if self.prefer_csrt else None
            if tracker_factory is not None:
                candidates = self._propagate_csrt(
                    capture,
                    seed=seed,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    fps=fps,
                    tracker_factory=tracker_factory,
                )
            if candidates is None:
                candidates = self._propagate_fallback(
                    capture,
                    seed=seed,
                    source_points=source_points,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    fps=fps,
                )
        finally:
            capture.release()
            materialized_input.close()

        context.update(0.78, "storing reprocessing suggestions")
        with self.database.session() as session:
            reservation = cast(
                CursorResult[object],
                session.execute(
                    update(ProcessingJob)
                    .where(ProcessingJob.id == job_id)
                    .values(progress=ProcessingJob.progress)
                    .execution_options(synchronize_session=False)
                ),
            )
            if reservation.rowcount != 1:
                raise ReprocessingNotFoundError("reprocessing records disappeared")
            video = session.scalar(
                select(VideoAsset)
                .where(VideoAsset.id == seed.video_id)
                .with_for_update()
            )
            action = session.get(ReviewAction, source_action_id)
            job = session.get(ProcessingJob, job_id)
            if video is None or action is None or job is None:
                raise ReprocessingNotFoundError("reprocessing records disappeared")
            discard_reason = self._source_discard_reason(session, seed)
            if discard_reason is not None:
                job.payload = {
                    **job.payload,
                    "suggestion_count": 0,
                    "propagation_methods": [],
                    "discard_reason": discard_reason,
                }
                session.commit()
                context.update(0.96, "stale source discarded")
                return
            session.add_all(
                ReprocessingSuggestion(
                    video_id=seed.video_id,
                    source_action_id=source_action_id,
                    job_id=job_id,
                    track_id=seed.track_id,
                    source_revision=seed.source_revision,
                    class_name=seed.class_name,
                    seed_frame_index=seed.frame_index,
                    frame_index=candidate.frame_index,
                    timestamp_ms=candidate.timestamp_ms,
                    x1=candidate.bbox.x1,
                    y1=candidate.bbox.y1,
                    x2=candidate.bbox.x2,
                    y2=candidate.bbox.y2,
                    confidence=candidate.confidence,
                    direction=candidate.direction,
                    propagation_method=candidate.method,
                    seed_locked=True,
                    status="PENDING",
                    metadata_json={
                        "distance_frames": candidate.distance_frames,
                        "window_seconds": self.window_seconds,
                    },
                )
                for candidate in candidates
            )
            job.payload = {
                **job.payload,
                "suggestion_count": len(candidates),
                "propagation_methods": sorted(
                    {candidate.method for candidate in candidates}
                ),
            }
            session.commit()
        context.update(0.96, "suggestions ready")

    @staticmethod
    def _source_discard_reason(session: Session, seed: _Seed) -> str | None:
        snapshot = build_review_snapshot(session, seed.video_id)
        track = snapshot.tracks.get(seed.track_id)
        if track is None:
            return "source track is no longer current"
        if not track.active:
            return "source track is no longer active"
        if track.class_name != seed.class_name:
            return "source track class changed"
        if (
            track.start_frame != seed.track_start_frame
            or track.end_frame != seed.track_end_frame
            or track.start_ms != seed.track_start_ms
            or track.end_ms != seed.track_end_ms
        ):
            return "source track span changed"
        keyframe = next(
            (
                item
                for item in track.keyframes
                if item.frame_index == seed.frame_index
            ),
            None,
        )
        if keyframe is None or not keyframe.locked or keyframe.bbox != seed.bbox:
            return "source seed geometry changed"
        return None

    @staticmethod
    def _source_points(session: Session, track_id: str) -> tuple[_SourcePoint, ...]:
        if session.get(Track, track_id) is None:
            return ()
        rows = session.scalars(
            select(TrackKeyframe)
            .where(TrackKeyframe.track_id == track_id)
            .order_by(TrackKeyframe.frame_index)
        )
        return tuple(
            _SourcePoint(
                frame_index=row.frame_index,
                bbox=NormalizedBox(x1=row.x1, y1=row.y1, x2=row.x2, y2=row.y2),
            )
            for row in rows
        )

    @staticmethod
    def _csrt_factory() -> TrackerFactory | None:
        module = cast(object, cv2)
        direct = getattr(module, "TrackerCSRT_create", None)
        if callable(direct):
            return cast(TrackerFactory, direct)
        legacy = getattr(module, "legacy", None)
        legacy_factory = (
            getattr(legacy, "TrackerCSRT_create", None) if legacy is not None else None
        )
        return cast(TrackerFactory, legacy_factory) if callable(legacy_factory) else None

    def _propagate_csrt(
        self,
        capture: cv2.VideoCapture,
        *,
        seed: _Seed,
        start_frame: int,
        end_frame: int,
        fps: float,
        tracker_factory: TrackerFactory,
    ) -> list[_Candidate] | None:
        stride = self._propagation_stride(
            fps,
            span_frames=end_frame - start_frame,
        )
        backward = list(range(seed.frame_index - stride, start_frame - 1, -stride))
        forward = list(range(seed.frame_index + stride, end_frame + 1, stride))
        buffered = self._read_ascending(
            capture,
            sorted({seed.frame_index, *backward, *forward}),
        )

        seed_frame = buffered.get(seed.frame_index)
        if seed_frame is None:
            raise ReprocessingValidationError("corrected seed frame could not be decoded")
        height, width = seed_frame.shape[:2]
        seed_pixels = seed.bbox.to_pixels(width, height)
        initial_box = (
            seed_pixels.x1,
            seed_pixels.y1,
            seed_pixels.width,
            seed_pixels.height,
        )
        candidates: list[_Candidate] = []
        initialized = False
        for direction, indices in (
            ("backward", backward),
            ("forward", forward),
        ):
            tracker = tracker_factory()
            result = tracker.init(seed_frame, initial_box)
            if result is False:
                continue
            initialized = True
            previous = seed.bbox
            for frame_index in indices:
                frame = buffered.get(frame_index)
                if frame is None:
                    break
                success, raw_box = tracker.update(frame)
                if not success:
                    break
                candidate_box = self._box_from_pixels(
                    raw_box,
                    width=frame.shape[1],
                    height=frame.shape[0],
                )
                if candidate_box is None or not self._conservative_step(
                    previous,
                    candidate_box,
                    steps=stride,
                ):
                    break
                distance = abs(frame_index - seed.frame_index)
                candidates.append(
                    _Candidate(
                        frame_index=frame_index,
                        timestamp_ms=round(frame_index * 1000 / fps),
                        bbox=candidate_box,
                        confidence=max(
                            0.45,
                            0.88 - 0.38 * distance / max(1, end_frame - start_frame),
                        ),
                        direction=direction,
                        method="csrt",
                        distance_frames=distance,
                    )
                )
                previous = candidate_box
        return sorted(candidates, key=lambda item: item.frame_index) if initialized else None

    def _propagate_fallback(
        self,
        capture: cv2.VideoCapture,
        *,
        seed: _Seed,
        source_points: Sequence[_SourcePoint],
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> list[_Candidate]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        seed_source = self._box_at_frame(source_points, seed.frame_index)
        method = "interpolation" if seed_source is not None else "static"
        deltas = (
            (
                seed.bbox.x1 - seed_source.x1,
                seed.bbox.y1 - seed_source.y1,
                seed.bbox.x2 - seed_source.x2,
                seed.bbox.y2 - seed_source.y2,
            )
            if seed_source is not None
            else None
        )
        maximum_distance = max(
            1,
            seed.frame_index - start_frame,
            end_frame - seed.frame_index,
        )
        candidates: list[_Candidate] = []
        for frame_index in range(start_frame, end_frame + 1):
            success, _ = capture.read()
            if not success:
                break
            if frame_index == seed.frame_index:
                continue
            distance = abs(frame_index - seed.frame_index)
            baseline = self._box_at_frame(source_points, frame_index)
            if baseline is not None and deltas is not None:
                influence = max(0.25, 1.0 - 0.75 * distance / maximum_distance)
                bbox = self._clamped_box(
                    baseline.x1 + deltas[0] * influence,
                    baseline.y1 + deltas[1] * influence,
                    baseline.x2 + deltas[2] * influence,
                    baseline.y2 + deltas[3] * influence,
                )
            else:
                bbox = seed.bbox
            candidates.append(
                _Candidate(
                    frame_index=frame_index,
                    timestamp_ms=round(frame_index * 1000 / fps),
                    bbox=bbox,
                    confidence=max(0.4, 0.78 - 0.33 * distance / maximum_distance),
                    direction=(
                        "backward" if frame_index < seed.frame_index else "forward"
                    ),
                    method=method,
                    distance_frames=distance,
                )
            )
        return candidates

    @staticmethod
    def _propagation_stride(fps: float, *, span_frames: int) -> int:
        """Frames to advance between tracked candidates.

        High frame rates carry far more frames than reviewers need, so sample
        at a fixed rate in time and widen further if the window would exceed
        the buffered-frame budget.
        """
        rate = fps if fps > 0 else float(_PROPAGATION_SAMPLES_PER_SECOND)
        stride = max(1, round(rate / _PROPAGATION_SAMPLES_PER_SECOND))
        while span_frames // stride > _MAX_PROPAGATION_FRAMES:
            stride += 1
        return stride

    @staticmethod
    def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> Frame | None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        return cast(Frame, frame) if success else None

    @staticmethod
    def _read_ascending(
        capture: cv2.VideoCapture,
        indices: Sequence[int],
    ) -> dict[int, Frame]:
        """Decode the requested frames while seeking exactly once.

        Seeking to an arbitrary frame makes H.264 decode forward from the
        previous keyframe, so seeking per frame costs orders of magnitude more
        than walking the stream once and grabbing what is needed.
        """
        if not indices:
            return {}
        wanted = set(indices)
        first, last = indices[0], indices[-1]
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames: dict[int, Frame] = {}
        current = first
        while current <= last:
            if current in wanted:
                success, frame = capture.read()
                if not success:
                    break
                frames[current] = cast(Frame, frame)
            elif not capture.grab():
                break
            current += 1
        return frames

    @classmethod
    def _box_at_frame(
        cls,
        points: Sequence[_SourcePoint],
        frame_index: int,
    ) -> NormalizedBox | None:
        if not points:
            return None
        if frame_index <= points[0].frame_index:
            return points[0].bbox
        if frame_index >= points[-1].frame_index:
            return points[-1].bbox
        for before, after in pairwise(points):
            if before.frame_index <= frame_index <= after.frame_index:
                distance = after.frame_index - before.frame_index
                progress = (frame_index - before.frame_index) / distance
                return before.bbox.interpolate(after.bbox, progress)
        return None

    @staticmethod
    def _clamped_box(x1: float, y1: float, x2: float, y2: float) -> NormalizedBox:
        left = min(0.999999, max(0.0, x1))
        top = min(0.999999, max(0.0, y1))
        right = min(1.0, max(left + 0.000001, x2))
        bottom = min(1.0, max(top + 0.000001, y2))
        return NormalizedBox(x1=left, y1=top, x2=right, y2=bottom)

    @classmethod
    def _box_from_pixels(
        cls,
        raw: tuple[float, float, float, float],
        *,
        width: int,
        height: int,
    ) -> NormalizedBox | None:
        x, y, box_width, box_height = raw
        if width <= 0 or height <= 0 or box_width <= 1 or box_height <= 1:
            return None
        return cls._clamped_box(
            x / width,
            y / height,
            (x + box_width) / width,
            (y + box_height) / height,
        )

    @staticmethod
    def _conservative_step(
        previous: NormalizedBox,
        current: NormalizedBox,
        *,
        steps: int = 1,
    ) -> bool:
        # Allowances scale with the frame gap: sampling every Nth frame means
        # a subject legitimately travels N times as far between candidates.
        span = max(1, steps)
        previous_area = previous.area()
        area_ratio = current.area() / previous_area
        previous_center = ((previous.x1 + previous.x2) / 2, (previous.y1 + previous.y2) / 2)
        current_center = ((current.x1 + current.x2) / 2, (current.y1 + current.y2) / 2)
        displacement = hypot(
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        )
        return (
            0.5 <= area_ratio <= 2.0
            and displacement <= min(0.45, 0.15 * span)
            and previous.iou(current) >= 0.05 / span
        )
