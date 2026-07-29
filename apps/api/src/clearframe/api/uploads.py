from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, status

from clearframe.api.dependencies import get_services
from clearframe.api.schemas import (
    JobRead,
    MultipartUploadCapability,
    ResumableUploadCapability,
    UploadAccepted,
    UploadComplete,
    UploadInitiate,
    UploadSessionRead,
    VideoRead,
)
from clearframe.services.container import ServiceContainer
from clearframe.services.ingest import IngestError
from clearframe.services.uploads import (
    ResumableUploadUnavailableError,
    UploadIncompleteError,
    UploadNotFoundError,
    UploadSizeExceededError,
    UploadStateError,
    UploadValidationError,
    UploadVerificationError,
)

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

ServicesDependency = Annotated[ServiceContainer, Depends(get_services)]


@router.get(
    "/capability",
    response_model=MultipartUploadCapability | ResumableUploadCapability,
)
def get_upload_capability(
    services: ServicesDependency,
) -> MultipartUploadCapability | ResumableUploadCapability:
    capability = services.uploads.capability()
    if capability.mode == "resumable":
        if (
            capability.chunk_size_bytes is None
            or capability.max_upload_bytes is None
        ):
            raise RuntimeError("resumable upload capability is incomplete")
        return ResumableUploadCapability(
            chunk_size_bytes=capability.chunk_size_bytes,
            max_upload_bytes=capability.max_upload_bytes,
        )
    return MultipartUploadCapability()


@router.post(
    "/initiate",
    response_model=UploadSessionRead,
    status_code=status.HTTP_201_CREATED,
)
def initiate_upload(
    request: UploadInitiate,
    services: ServicesDependency,
) -> UploadSessionRead:
    try:
        upload = services.uploads.initiate(
            filename=request.filename,
            content_type=request.content_type,
            size_bytes=request.size_bytes,
            hashed_case_id=request.hashed_case_id,
        )
    except ResumableUploadUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except UploadSizeExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except (UploadValidationError, IngestError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return UploadSessionRead(
        upload_id=upload.upload_id,
        session_url=upload.session_url,
        chunk_size_bytes=upload.chunk_size_bytes,
    )


@router.post(
    "/{upload_id}/complete",
    response_model=UploadAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def complete_upload(
    upload_id: str,
    _: Annotated[UploadComplete, Body()],
    services: ServicesDependency,
) -> UploadAccepted:
    try:
        accepted = services.uploads.complete(upload_id)
    except UploadNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (ResumableUploadUnavailableError, UploadIncompleteError, UploadStateError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except UploadVerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return UploadAccepted(
        video=VideoRead.from_model(accepted.video),
        job=JobRead.from_model(accepted.job),
    )
