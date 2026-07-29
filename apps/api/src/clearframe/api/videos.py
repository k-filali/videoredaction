from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from clearframe.api.dependencies import get_database, get_services
from clearframe.api.schemas import (
    JobRead,
    UploadAccepted,
    VideoList,
    VideoRead,
    VideoStatusRead,
)
from clearframe.database import Database
from clearframe.media import UnsupportedMediaError
from clearframe.models import ProcessingJob, VideoAsset
from clearframe.services.container import ServiceContainer
from clearframe.services.ingest import (
    DuplicateVideoError,
    EmptyUploadError,
    IngestError,
    UploadTooLargeError,
)

router = APIRouter(prefix="/api/videos", tags=["videos"])

DatabaseDependency = Annotated[Database, Depends(get_database)]
ServicesDependency = Annotated[ServiceContainer, Depends(get_services)]


def _video_or_404(database: Database, video_id: str) -> VideoAsset:
    with database.session() as session:
        video = session.get(VideoAsset, video_id)
        if video is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video not found")
        session.expunge(video)
        return video


@router.post("", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    file: Annotated[UploadFile, File(description="Supported video file")],
    services: ServicesDependency,
    hashed_case_id: Annotated[str | None, Form()] = None,
) -> UploadAccepted:
    try:
        accepted = await services.ingest.accept(file, hashed_case_id)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    except UnsupportedMediaError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except EmptyUploadError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DuplicateVideoError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(exc),
                "existing_video_id": exc.existing_video_id,
            },
        ) from exc
    except IngestError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return UploadAccepted(
        video=VideoRead.from_model(accepted.video),
        job=JobRead.from_model(accepted.job),
    )


@router.get("", response_model=VideoList)
def list_videos(database: DatabaseDependency) -> VideoList:
    with database.session() as session:
        videos = list(
            session.scalars(select(VideoAsset).order_by(VideoAsset.created_at.desc()))
        )
    return VideoList(items=[VideoRead.from_model(video) for video in videos], total=len(videos))


@router.get("/{video_id}", response_model=VideoRead)
def get_video(video_id: str, database: DatabaseDependency) -> VideoRead:
    return VideoRead.from_model(_video_or_404(database, video_id))


@router.get("/{video_id}/status", response_model=VideoStatusRead)
def get_video_status(video_id: str, database: DatabaseDependency) -> VideoStatusRead:
    video = _video_or_404(database, video_id)
    with database.session() as session:
        jobs = list(
            session.scalars(
                select(ProcessingJob)
                .where(ProcessingJob.video_id == video_id)
                .order_by(ProcessingJob.created_at.desc())
            )
        )
    return VideoStatusRead(
        video=VideoRead.from_model(video),
        jobs=[JobRead.from_model(job) for job in jobs],
    )


@router.get("/{video_id}/proxy", response_class=FileResponse)
def get_proxy(
    video_id: str,
    database: DatabaseDependency,
    services: ServicesDependency,
) -> FileResponse:
    video = _video_or_404(database, video_id)
    if not video.proxy_uri or not services.storage.exists(video.proxy_uri):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="proxy is not ready")
    return FileResponse(
        services.storage.path_for(video.proxy_uri),
        media_type="video/mp4",
        headers={"Cache-Control": "private, no-store"},
    )


@router.get("/{video_id}/thumbnail", response_class=FileResponse)
def get_thumbnail(
    video_id: str,
    database: DatabaseDependency,
    services: ServicesDependency,
) -> FileResponse:
    video = _video_or_404(database, video_id)
    if not video.thumbnail_uri or not services.storage.exists(video.thumbnail_uri):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="thumbnail not found")
    return FileResponse(
        services.storage.path_for(video.thumbnail_uri),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-store"},
    )

