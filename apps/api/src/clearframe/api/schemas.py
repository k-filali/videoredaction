from datetime import datetime

from pydantic import BaseModel, Field

from clearframe.domain.enums import JobStatus, JobType, VideoStatus
from clearframe.models import ProcessingJob, VideoAsset


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

