from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from clearframe.domain.enums import (
    ExportStatus,
    JobStatus,
    JobType,
    RedactionStyle,
    ReviewActionType,
    VideoStatus,
)
from clearframe.domain.geometry import NormalizedBox
from clearframe.domain.review import ReviewSnapshot
from clearframe.models import (
    ExportArtifact,
    ProcessingJob,
    ReprocessingSuggestion,
    ReviewAction,
    VideoAsset,
)


class VideoRead(BaseModel):
    id: str
    original_filename: str
    content_type: str
    source_type: str
    duration_ms: int | None
    fps: float | None
    width: int | None
    height: int | None
    codec: str | None
    audio_present: bool | None
    original_sha256: str | None
    status: VideoStatus
    review_revision: int
    active_model_run_id: str | None
    proxy_url: str | None
    thumbnail_url: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, video: VideoAsset) -> "VideoRead":
        return cls(
            id=video.id,
            original_filename=video.original_filename,
            content_type=video.content_type,
            source_type=video.source_type,
            duration_ms=video.duration_ms,
            fps=video.fps,
            width=video.width,
            height=video.height,
            codec=video.codec,
            audio_present=video.audio_present,
            original_sha256=video.original_sha256,
            status=VideoStatus(video.status),
            review_revision=video.review_revision,
            active_model_run_id=video.active_model_run_id,
            proxy_url=f"/api/videos/{video.id}/proxy" if video.proxy_uri else None,
            thumbnail_url=f"/api/videos/{video.id}/thumbnail" if video.thumbnail_uri else None,
            error_message=video.error_message,
            created_at=video.created_at,
            updated_at=video.updated_at,
        )


class JobRead(BaseModel):
    id: str
    job_type: JobType
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    stage: str
    attempts: int
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    @classmethod
    def from_model(cls, job: ProcessingJob) -> "JobRead":
        return cls(
            id=job.id,
            job_type=JobType(job.job_type),
            status=JobStatus(job.status),
            progress=job.progress,
            stage=job.stage,
            attempts=job.attempts,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )


class UploadAccepted(BaseModel):
    video: VideoRead
    job: JobRead


class VideoStatusRead(BaseModel):
    video: VideoRead
    jobs: list[JobRead]


class VideoList(BaseModel):
    items: list[VideoRead]
    total: int


class ManualRegionCreate(BaseModel):
    expected_revision: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    class_name: str = Field(default="license_plate", min_length=1, max_length=48)
    bbox: NormalizedBox
    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(default=None, ge=0)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    reason_code: str | None = Field(default=None, max_length=64)


class AuditActionRead(BaseModel):
    id: str
    video_id: str
    track_id: str | None
    frame_index: int | None
    timestamp_ms: int | None
    action_type: ReviewActionType
    before_state: dict[str, Any]
    after_state: dict[str, Any]
    reason_code: str | None
    reviewer_session_id: str
    revision: int
    model_version: str | None
    application_version: str
    inverse_of_action_id: str | None
    created_at: datetime

    @classmethod
    def from_model(cls, action: ReviewAction) -> "AuditActionRead":
        return cls(
            id=action.id,
            video_id=action.video_id,
            track_id=action.track_id,
            frame_index=action.frame_index,
            timestamp_ms=action.timestamp_ms,
            action_type=ReviewActionType(action.action_type),
            before_state=action.before_state,
            after_state=action.after_state,
            reason_code=action.reason_code,
            reviewer_session_id=action.reviewer_session_id,
            revision=action.revision,
            model_version=action.model_version,
            application_version=action.application_version,
            inverse_of_action_id=action.inverse_of_action_id,
            created_at=action.created_at,
        )


class ReviewMutationRead(BaseModel):
    action: AuditActionRead
    state: ReviewSnapshot
    reprocessing_job: JobRead | None = None
    reprocessing_note: str | None = None


class ReprocessingSuggestionRead(BaseModel):
    id: str
    source_action_id: str
    job_id: str
    track_id: str
    source_revision: int
    class_name: str
    seed_frame_index: int
    frame_index: int
    timestamp_ms: int
    bbox: NormalizedBox
    confidence: float
    direction: str
    propagation_method: str
    seed_locked: bool
    status: str
    metadata: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_model(
        cls,
        suggestion: ReprocessingSuggestion,
    ) -> "ReprocessingSuggestionRead":
        return cls(
            id=suggestion.id,
            source_action_id=suggestion.source_action_id,
            job_id=suggestion.job_id,
            track_id=suggestion.track_id,
            source_revision=suggestion.source_revision,
            class_name=suggestion.class_name,
            seed_frame_index=suggestion.seed_frame_index,
            frame_index=suggestion.frame_index,
            timestamp_ms=suggestion.timestamp_ms,
            bbox=NormalizedBox(
                x1=suggestion.x1,
                y1=suggestion.y1,
                x2=suggestion.x2,
                y2=suggestion.y2,
            ),
            confidence=suggestion.confidence,
            direction=suggestion.direction,
            propagation_method=suggestion.propagation_method,
            seed_locked=suggestion.seed_locked,
            status=suggestion.status,
            metadata=suggestion.metadata_json,
            created_at=suggestion.created_at,
        )


class AuditLogRead(BaseModel):
    video_id: str
    revision: int
    actions: list[AuditActionRead]


class ExportCreate(BaseModel):
    expected_revision: int = Field(ge=0)
    redaction_style: RedactionStyle = RedactionStyle.PIXELATE


class ExportRead(BaseModel):
    id: str
    video_id: str
    status: ExportStatus
    redaction_style: RedactionStyle
    source_model_run_id: str | None
    review_revision: int
    export_sha256: str | None
    download_url: str | None
    manifest_url: str | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None

    @classmethod
    def from_model(cls, artifact: ExportArtifact) -> "ExportRead":
        return cls(
            id=artifact.id,
            video_id=artifact.video_id,
            status=ExportStatus(artifact.status),
            redaction_style=RedactionStyle(artifact.redaction_style),
            source_model_run_id=artifact.source_model_run_id,
            review_revision=artifact.review_revision,
            export_sha256=artifact.export_sha256,
            download_url=(
                f"/api/exports/{artifact.id}/download" if artifact.export_uri else None
            ),
            manifest_url=(
                f"/api/exports/{artifact.id}/manifest" if artifact.manifest_uri else None
            ),
            error_message=artifact.error_message,
            created_at=artifact.created_at,
            completed_at=artifact.completed_at,
        )


class ExportAccepted(BaseModel):
    export: ExportRead
    job: JobRead
