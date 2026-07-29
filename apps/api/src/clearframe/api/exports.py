import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from sqlalchemy import select

from clearframe.api.dependencies import get_database, get_services
from clearframe.api.schemas import ExportAccepted, ExportCreate, ExportRead, JobRead
from clearframe.database import Database
from clearframe.domain.enums import ExportStatus
from clearframe.models import ExportArtifact
from clearframe.services.container import ServiceContainer
from clearframe.services.export import (
    ExportNotFoundError,
    ExportValidationError,
)
from clearframe.services.review import RevisionConflictError
from clearframe.storage import ArtifactDelivery

router = APIRouter(prefix="/api", tags=["exports"])
DatabaseDependency = Annotated[Database, Depends(get_database)]
ServicesDependency = Annotated[ServiceContainer, Depends(get_services)]
ReviewerSession = Annotated[
    str,
    Header(alias="X-Reviewer-Session", min_length=1, max_length=64),
]
private_download_headers = {"Cache-Control": "private, no-store"}


def _download_response(
    delivery: ArtifactDelivery,
    *,
    filename: str,
) -> Response:
    if delivery.kind == "redirect":
        if delivery.url is None:
            raise RuntimeError("redirect delivery is missing a URL")
        return RedirectResponse(
            delivery.url,
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={
                **private_download_headers,
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )
    if delivery.path is None:
        raise RuntimeError("local delivery is missing a path")
    return FileResponse(
        delivery.path,
        media_type="video/mp4",
        filename=filename,
        headers=private_download_headers,
    )


def _export_or_404(database: Database, export_id: str) -> ExportArtifact:
    with database.session() as session:
        artifact = session.get(ExportArtifact, export_id)
        if artifact is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="export not found")
        session.expunge(artifact)
        return artifact


@router.post(
    "/videos/{video_id}/exports",
    response_model=ExportAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_export(
    video_id: str,
    request: ExportCreate,
    services: ServicesDependency,
    reviewer_session: ReviewerSession = "local-reviewer",
) -> ExportAccepted:
    try:
        requested = services.export.request(
            video_id,
            expected_revision=request.expected_revision,
            style=request.redaction_style,
            reviewer_session_id=reviewer_session,
        )
    except RevisionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(exc),
                "expected_revision": exc.expected,
                "current_revision": exc.actual,
            },
        ) from exc
    except ExportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ExportValidationError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ExportAccepted(
        export=ExportRead.from_model(requested.artifact),
        job=JobRead.from_model(requested.job),
        warnings=list(requested.warnings),
    )


@router.get("/exports/{export_id}", response_model=ExportRead)
def get_export(export_id: str, database: DatabaseDependency) -> ExportRead:
    return ExportRead.from_model(_export_or_404(database, export_id))


@router.get("/exports/{export_id}/download", response_class=FileResponse)
def download_export(
    export_id: str,
    database: DatabaseDependency,
    services: ServicesDependency,
) -> Response:
    artifact = _export_or_404(database, export_id)
    if (
        artifact.status != ExportStatus.COMPLETED
        or not artifact.export_uri
        or not services.storage.exists(artifact.export_uri)
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="export is not ready")
    filename = f"clearframe-{artifact.video_id[:8]}-redacted.mp4"
    return _download_response(
        services.storage.delivery_for(artifact.export_uri, filename=filename),
        filename=filename,
    )


@router.get("/exports/{export_id}/manifest", response_class=JSONResponse)
def get_export_manifest(
    export_id: str,
    database: DatabaseDependency,
    services: ServicesDependency,
) -> JSONResponse:
    artifact = _export_or_404(database, export_id)
    if not artifact.manifest_uri or not services.storage.exists(artifact.manifest_uri):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="manifest is not ready")
    return JSONResponse(_load_manifest(services, artifact.manifest_uri))


@router.get("/videos/{video_id}/manifest", response_class=JSONResponse)
def get_latest_manifest(
    video_id: str,
    database: DatabaseDependency,
    services: ServicesDependency,
) -> JSONResponse:
    with database.session() as session:
        artifact = session.scalar(
            select(ExportArtifact)
            .where(
                ExportArtifact.video_id == video_id,
                ExportArtifact.status == ExportStatus.COMPLETED,
            )
            .order_by(ExportArtifact.completed_at.desc())
        )
    if artifact is None or not artifact.manifest_uri:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="manifest not found")
    return JSONResponse(_load_manifest(services, artifact.manifest_uri))


def _load_manifest(services: ServiceContainer, manifest_uri: str) -> dict[str, Any]:
    try:
        with services.storage.materialize_input(manifest_uri) as path:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="manifest could not be read",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="manifest is invalid",
        )
    return payload
