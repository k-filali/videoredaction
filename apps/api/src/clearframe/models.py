from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, Session, mapped_column

from clearframe.database import Base
from clearframe.domain.enums import (
    ExportStatus,
    InterpolationMode,
    JobStatus,
    JobType,
    ProposalSource,
    RedactionStyle,
    RunStatus,
    TrackSource,
    TrackStatus,
    VideoStatus,
)


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC)


class VideoAsset(Base):
    __tablename__ = "video_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    hashed_case_id: Mapped[str | None] = mapped_column(String(64))
    original_filename: Mapped[str] = mapped_column(String(255))
    safe_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(128))
    source_type: Mapped[str] = mapped_column(String(32), default="upload")
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    codec: Mapped[str | None] = mapped_column(String(64))
    audio_present: Mapped[bool | None] = mapped_column(Boolean)
    original_uri: Mapped[str | None] = mapped_column(Text)
    proxy_uri: Mapped[str | None] = mapped_column(Text)
    thumbnail_uri: Mapped[str | None] = mapped_column(Text)
    original_sha256: Mapped[str | None] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(32), default=VideoStatus.UPLOADING)
    review_revision: Mapped[int] = mapped_column(Integer, default=0)
    active_model_run_id: Mapped[str | None] = mapped_column(String(36))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("video_assets.id", ondelete="CASCADE"),
        index=True,
    )
    detector_versions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tracker_name: Mapped[str] = mapped_column(String(64))
    tracker_version: Mapped[str] = mapped_column(String(64))
    thresholds: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    config_hash: Mapped[str] = mapped_column(String(64))
    device: Mapped[str] = mapped_column(String(64), default="cpu")
    status: Mapped[str] = mapped_column(String(24), default=RunStatus.QUEUED)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Detection(Base):
    __tablename__ = "detections"
    __table_args__ = (
        Index("ix_detections_video_frame", "video_id", "frame_index"),
        Index("ix_detections_run_class", "model_run_id", "class_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("video_assets.id", ondelete="CASCADE"),
        index=True,
    )
    model_run_id: Mapped[str] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"),
        index=True,
    )
    frame_index: Mapped[int] = mapped_column(Integer)
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(48))
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    detector_name: Mapped[str] = mapped_column(String(64))
    detector_version: Mapped[str] = mapped_column(String(64))
    proposal_source: Mapped[str] = mapped_column(String(24), default=ProposalSource.MODEL)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    inference_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Track(Base):
    __tablename__ = "tracks"
    __table_args__ = (Index("ix_tracks_video_span", "video_id", "start_frame", "end_frame"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("video_assets.id", ondelete="CASCADE"),
        index=True,
    )
    model_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_runs.id", ondelete="SET NULL"),
        index=True,
    )
    class_name: Mapped[str] = mapped_column(String(48))
    start_frame: Mapped[int] = mapped_column(Integer)
    end_frame: Mapped[int] = mapped_column(Integer)
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), default=TrackStatus.PROPOSED)
    default_redacted: Mapped[bool] = mapped_column(Boolean, default=True)
    source: Mapped[str] = mapped_column(String(24), default=TrackSource.MODEL)
    confidence_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    parent_track_id: Mapped[str | None] = mapped_column(
        ForeignKey("tracks.id", ondelete="SET NULL"),
    )
    reviewed_until_frame: Mapped[int | None] = mapped_column(Integer)
    warning: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrackKeyframe(Base):
    __tablename__ = "track_keyframes"
    __table_args__ = (
        UniqueConstraint("track_id", "frame_index", "source", name="uq_track_keyframe_source"),
        Index("ix_track_keyframes_track_time", "track_id", "timestamp_ms"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    track_id: Mapped[str] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"),
        index=True,
    )
    frame_index: Mapped[int] = mapped_column(Integer)
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    interpolation_mode: Mapped[str] = mapped_column(
        String(24),
        default=InterpolationMode.LINEAR,
    )
    visibility: Mapped[bool] = mapped_column(Boolean, default=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(24), default=TrackSource.MODEL)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ReviewAction(Base):
    __tablename__ = "review_actions"
    __table_args__ = (
        UniqueConstraint("video_id", "revision", name="uq_review_action_revision"),
        Index("ix_review_actions_video_created", "video_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("video_assets.id", ondelete="CASCADE"),
        index=True,
    )
    track_id: Mapped[str | None] = mapped_column(String(36), index=True)
    frame_index: Mapped[int | None] = mapped_column(Integer)
    timestamp_ms: Mapped[int | None] = mapped_column(Integer)
    action_type: Mapped[str] = mapped_column(String(40))
    before_state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    after_state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    reason_code: Mapped[str | None] = mapped_column(String(64))
    reviewer_session_id: Mapped[str] = mapped_column(String(64))
    revision: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str | None] = mapped_column(String(128))
    application_version: Mapped[str] = mapped_column(String(32))
    inverse_of_action_id: Mapped[str | None] = mapped_column(
        ForeignKey("review_actions.id", ondelete="RESTRICT"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ExportArtifact(Base):
    __tablename__ = "export_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "video_id",
            "review_revision",
            "redaction_style",
            name="uq_export_revision_style",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("video_assets.id", ondelete="CASCADE"),
        index=True,
    )
    export_uri: Mapped[str | None] = mapped_column(Text)
    export_sha256: Mapped[str | None] = mapped_column(String(64))
    manifest_uri: Mapped[str | None] = mapped_column(Text)
    redaction_style: Mapped[str] = mapped_column(String(24), default=RedactionStyle.PIXELATE)
    source_model_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_runs.id", ondelete="SET NULL"),
    )
    review_revision: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), default=ExportStatus.QUEUED)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (Index("ix_processing_jobs_video_status", "video_id", "status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    video_id: Mapped[str | None] = mapped_column(
        ForeignKey("video_assets.id", ondelete="CASCADE"),
        index=True,
    )
    export_id: Mapped[str | None] = mapped_column(
        ForeignKey("export_artifacts.id", ondelete="CASCADE"),
    )
    job_type: Mapped[str] = mapped_column(String(24), default=JobType.INGEST)
    status: Mapped[str] = mapped_column(String(24), default=JobStatus.QUEUED)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str] = mapped_column(String(80), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ReprocessingSuggestion(Base):
    __tablename__ = "reprocessing_suggestions"
    __table_args__ = (
        UniqueConstraint(
            "source_action_id",
            "frame_index",
            name="uq_reprocessing_suggestion_action_frame",
        ),
        Index(
            "ix_reprocessing_suggestions_video_track_frame",
            "video_id",
            "track_id",
            "frame_index",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("video_assets.id", ondelete="CASCADE"),
        index=True,
    )
    source_action_id: Mapped[str] = mapped_column(
        ForeignKey("review_actions.id", ondelete="RESTRICT"),
        index=True,
    )
    job_id: Mapped[str] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="CASCADE"),
        index=True,
    )
    track_id: Mapped[str] = mapped_column(String(36), index=True)
    source_revision: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(48))
    seed_frame_index: Mapped[int] = mapped_column(Integer)
    frame_index: Mapped[int] = mapped_column(Integer)
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(16))
    propagation_method: Mapped[str] = mapped_column(String(32))
    seed_locked: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(24), default="PENDING")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ImmutableAuditError(RuntimeError):
    pass


@event.listens_for(ReviewAction.__table__, "after_create")
def install_sqlite_audit_guards(
    _: Any,
    connection: Connection,
    **__: Any,
) -> None:
    if connection.dialect.name != "sqlite":
        return
    connection.exec_driver_sql(
        """
        CREATE TRIGGER IF NOT EXISTS prevent_review_action_update
        BEFORE UPDATE ON review_actions
        BEGIN
            SELECT RAISE(ABORT, 'review actions are append-only');
        END
        """
    )
    connection.exec_driver_sql(
        """
        CREATE TRIGGER IF NOT EXISTS prevent_review_action_delete
        BEFORE DELETE ON review_actions
        BEGIN
            SELECT RAISE(ABORT, 'review actions are append-only');
        END
        """
    )


@event.listens_for(Session, "before_flush")
def prevent_audit_mutation(
    session: Session,
    _: Any,
    __: Any,
) -> None:
    changed = set(session.dirty).union(session.deleted)
    if any(isinstance(item, ReviewAction) for item in changed):
        raise ImmutableAuditError("review actions are append-only")
