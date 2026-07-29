from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from clearframe.api.schemas import JobRead
from clearframe.domain.enums import RunStatus
from clearframe.models import ModelRun
from clearframe.services.processing import (
    DetectorSelectionError,
    ModelRunNotFoundError,
    ProcessingConflictError,
    ProcessingNotFoundError,
    ProcessingService,
    ProcessingValidationError,
)

router = APIRouter(prefix="/api/videos", tags=["processing"])


def get_processing_service(request: Request) -> ProcessingService:
    try:
        return cast(ProcessingService, request.app.state.processing_service)
    except AttributeError as exc:
        raise RuntimeError("processing service is not configured") from exc


ProcessingDependency = Annotated[ProcessingService, Depends(get_processing_service)]


class ProcessVideoRequest(BaseModel):
    model_ids: list[str] | None = Field(default=None, min_length=1)
    sample_every_frames: int = Field(default=5, ge=1, le=300)


class ModelRunRead(BaseModel):
    id: str
    video_id: str
    detector_versions: dict[str, Any]
    tracker_name: str
    tracker_version: str
    thresholds: dict[str, Any]
    config_hash: str
    device: str
    status: RunStatus
    metrics: dict[str, Any]
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    @classmethod
    def from_model(cls, run: ModelRun) -> ModelRunRead:
        return cls(
            id=run.id,
            video_id=run.video_id,
            detector_versions=run.detector_versions,
            tracker_name=run.tracker_name,
            tracker_version=run.tracker_version,
            thresholds=run.thresholds,
            config_hash=run.config_hash,
            device=run.device,
            status=RunStatus(run.status),
            metrics=run.metrics,
            started_at=run.started_at,
            completed_at=run.completed_at,
            created_at=run.created_at,
        )


class ProcessingAccepted(BaseModel):
    run: ModelRunRead
    job: JobRead


class ModelRunMetricsRead(BaseModel):
    model_run_id: str
    status: RunStatus
    config_hash: str
    metrics: dict[str, Any]


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (ProcessingNotFoundError, ModelRunNotFoundError)):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, DetectorSelectionError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )
    if isinstance(exc, ProcessingConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, ProcessingValidationError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="processing failed",
    )


@router.post(
    "/{video_id}/process",
    response_model=ProcessingAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def process_video(
    video_id: str,
    service: ProcessingDependency,
    request: ProcessVideoRequest | None = None,
) -> ProcessingAccepted:
    options = request or ProcessVideoRequest()
    try:
        requested = service.request(
            video_id,
            model_ids=options.model_ids,
            sample_every_frames=options.sample_every_frames,
        )
    except (
        ProcessingNotFoundError,
        DetectorSelectionError,
        ProcessingConflictError,
        ProcessingValidationError,
    ) as exc:
        raise _translate_error(exc) from exc
    return ProcessingAccepted(
        run=ModelRunRead.from_model(requested.run),
        job=JobRead.from_model(requested.job),
    )


@router.get("/{video_id}/model-runs/latest", response_model=ModelRunRead)
def get_latest_model_run(
    video_id: str,
    service: ProcessingDependency,
) -> ModelRunRead:
    try:
        return ModelRunRead.from_model(service.latest_run(video_id))
    except (ProcessingNotFoundError, ModelRunNotFoundError) as exc:
        raise _translate_error(exc) from exc


@router.get(
    "/{video_id}/metrics",
    response_model=ModelRunMetricsRead,
)
@router.get(
    "/{video_id}/model-runs/latest/metrics",
    response_model=ModelRunMetricsRead,
)
def get_latest_model_metrics(
    video_id: str,
    service: ProcessingDependency,
) -> ModelRunMetricsRead:
    try:
        run = service.latest_run(video_id)
    except (ProcessingNotFoundError, ModelRunNotFoundError) as exc:
        raise _translate_error(exc) from exc
    return ModelRunMetricsRead(
        model_run_id=run.id,
        status=RunStatus(run.status),
        config_hash=run.config_hash,
        metrics=run.metrics,
    )
