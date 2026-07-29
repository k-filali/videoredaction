from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select

from clearframe.api.dependencies import get_database, get_services
from clearframe.api.schemas import (
    AuditActionRead,
    AuditLogRead,
    JobRead,
    ManualRegionCreate,
    ReprocessingSuggestionRead,
    ReviewMutationRead,
)
from clearframe.database import Database
from clearframe.domain.enums import ReviewActionType
from clearframe.domain.review import ReviewCommand, ReviewSnapshot
from clearframe.models import ReviewAction
from clearframe.services.container import ServiceContainer
from clearframe.services.reprocessing import ReprocessingError
from clearframe.services.review import (
    ReviewError,
    RevisionConflictError,
    TrackNotFoundError,
    VideoNotFoundError,
    append_review_action,
    build_review_snapshot,
)

router = APIRouter(prefix="/api/videos", tags=["review"])
DatabaseDependency = Annotated[Database, Depends(get_database)]
ServicesDependency = Annotated[ServiceContainer, Depends(get_services)]
ReviewerSession = Annotated[
    str,
    Header(
        alias="X-Reviewer-Session",
        min_length=1,
        max_length=64,
        description="Opaque local reviewer session identifier",
    ),
]


def _raise_review_http_error(error: ReviewError) -> None:
    if isinstance(error, RevisionConflictError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(error),
                "expected_revision": error.expected,
                "current_revision": error.actual,
            },
        ) from error
    if isinstance(error, (VideoNotFoundError, TrackNotFoundError)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=str(error),
    ) from error


def _append(
    services: ServiceContainer,
    video_id: str,
    command: ReviewCommand,
    reviewer_session: str,
) -> ReviewMutationRead:
    try:
        with services.database.session() as session:
            action, snapshot = append_review_action(
                session,
                video_id,
                command,
                reviewer_session_id=reviewer_session,
            )
            action_read = AuditActionRead.from_model(action)
    except ReviewError as exc:
        _raise_review_http_error(exc)
        raise AssertionError("unreachable") from exc

    reprocessing_job: JobRead | None = None
    reprocessing_note: str | None = None
    if action_read.action_type in {
        ReviewActionType.MOVE_REGION,
        ReviewActionType.RESIZE_REGION,
        ReviewActionType.CREATE_MANUAL_REGION,
    }:
        try:
            requested = services.reprocess.request(action_read.id)
            reprocessing_job = JobRead.from_model(requested.job)
        except ReprocessingError:
            reprocessing_note = "context reprocessing was not available for this edit"
    return ReviewMutationRead(
        action=action_read,
        state=snapshot,
        reprocessing_job=reprocessing_job,
        reprocessing_note=reprocessing_note,
    )


@router.get("/{video_id}/tracks", response_model=ReviewSnapshot)
def get_tracks(video_id: str, database: DatabaseDependency) -> ReviewSnapshot:
    try:
        with database.session() as session:
            return build_review_snapshot(session, video_id)
    except ReviewError as exc:
        _raise_review_http_error(exc)
        raise AssertionError("unreachable") from exc


@router.patch("/{video_id}/tracks/{track_id}", response_model=ReviewMutationRead)
def update_track(
    video_id: str,
    track_id: str,
    command: ReviewCommand,
    services: ServicesDependency,
    reviewer_session: ReviewerSession = "local-demo",
) -> ReviewMutationRead:
    if command.action_type in {
        ReviewActionType.CREATE_MANUAL_REGION,
        ReviewActionType.UNDO,
        ReviewActionType.REDO,
        ReviewActionType.EXPORT_REQUESTED,
        ReviewActionType.EXPORT_COMPLETED,
    }:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="action is not valid for the track endpoint",
        )
    command = command.model_copy(update={"track_id": track_id})
    return _append(services, video_id, command, reviewer_session)


@router.post(
    "/{video_id}/manual-regions",
    response_model=ReviewMutationRead,
    status_code=status.HTTP_201_CREATED,
)
def create_manual_region(
    video_id: str,
    region: ManualRegionCreate,
    services: ServicesDependency,
    reviewer_session: ReviewerSession = "local-demo",
) -> ReviewMutationRead:
    payload = {
        "class_name": region.class_name,
        "bbox": region.bbox.model_dump(),
        "start_frame": region.start_frame
        if region.start_frame is not None
        else region.frame_index,
        "end_frame": region.end_frame if region.end_frame is not None else region.frame_index,
        "start_ms": region.start_ms if region.start_ms is not None else region.timestamp_ms,
        "end_ms": region.end_ms if region.end_ms is not None else region.timestamp_ms,
    }
    command = ReviewCommand(
        action_type=ReviewActionType.CREATE_MANUAL_REGION,
        expected_revision=region.expected_revision,
        frame_index=region.frame_index,
        timestamp_ms=region.timestamp_ms,
        reason_code=region.reason_code,
        payload=payload,
    )
    return _append(services, video_id, command, reviewer_session)


@router.post(
    "/{video_id}/review-actions",
    response_model=ReviewMutationRead,
    status_code=status.HTTP_201_CREATED,
)
def create_review_action(
    video_id: str,
    command: ReviewCommand,
    services: ServicesDependency,
    reviewer_session: ReviewerSession = "local-demo",
) -> ReviewMutationRead:
    return _append(services, video_id, command, reviewer_session)


@router.get(
    "/{video_id}/reprocessing-suggestions",
    response_model=list[ReprocessingSuggestionRead],
)
def get_reprocessing_suggestions(
    video_id: str,
    services: ServicesDependency,
) -> list[ReprocessingSuggestionRead]:
    try:
        suggestions = services.reprocess.suggestions_for_video(video_id)
    except ReprocessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return [
        ReprocessingSuggestionRead.from_model(suggestion)
        for suggestion in suggestions
    ]


@router.get("/{video_id}/audit", response_model=AuditLogRead)
def get_audit_log(video_id: str, database: DatabaseDependency) -> AuditLogRead:
    try:
        with database.session() as session:
            snapshot = build_review_snapshot(session, video_id)
            actions = list(
                session.scalars(
                    select(ReviewAction)
                    .where(ReviewAction.video_id == video_id)
                    .order_by(ReviewAction.revision)
                )
            )
    except ReviewError as exc:
        _raise_review_http_error(exc)
        raise AssertionError("unreachable") from exc
    return AuditLogRead(
        video_id=video_id,
        revision=snapshot.revision,
        actions=[AuditActionRead.from_model(action) for action in actions],
    )
