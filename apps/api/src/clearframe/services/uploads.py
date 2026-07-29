from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select

from clearframe.database import Database
from clearframe.domain.enums import JobType, VideoStatus
from clearframe.gcs_storage import (
    GCSObjectMetadata,
    GCSObjectVerificationError,
    GCSStorage,
)
from clearframe.models import ProcessingJob, VideoAsset, new_id
from clearframe.services.ingest import (
    AcceptedUpload,
    DirectUploadStateError,
    IngestService,
    validate_case_hash,
)
from clearframe.storage import ArtifactStorage, sanitize_filename, temporary_upload_key

RESUMABLE_CHUNK_SIZE_BYTES = 8 * 1024 * 1024
SUPPORTED_UPLOAD_CONTENT_TYPES = frozenset(
    {
        "application/octet-stream",
        "video/mp4",
        "video/quicktime",
        "video/webm",
        "video/x-matroska",
        "video/x-msvideo",
    }
)


class UploadFlowError(ValueError):
    pass


class ResumableUploadUnavailableError(UploadFlowError):
    pass


class UploadValidationError(UploadFlowError):
    pass


class UploadSizeExceededError(UploadValidationError):
    pass


class UploadNotFoundError(UploadFlowError):
    pass


class UploadIncompleteError(UploadFlowError):
    pass


class UploadVerificationError(UploadFlowError):
    pass


class UploadStateError(UploadFlowError):
    pass


@dataclass(frozen=True, slots=True)
class UploadCapability:
    mode: Literal["multipart", "resumable"]
    chunk_size_bytes: int | None = None
    max_upload_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ResumableUploadSession:
    upload_id: str
    session_url: str
    chunk_size_bytes: int


class UploadService:
    def __init__(
        self,
        database: Database,
        storage: ArtifactStorage,
        ingest: IngestService,
        *,
        max_upload_mb: int,
        web_origin: str,
    ) -> None:
        self.database = database
        self.storage = storage
        self.ingest = ingest
        self.max_upload_bytes = max_upload_mb * 1024 * 1024
        self.web_origin = web_origin

    def capability(self) -> UploadCapability:
        if isinstance(self.storage, GCSStorage):
            return UploadCapability(
                mode="resumable",
                chunk_size_bytes=RESUMABLE_CHUNK_SIZE_BYTES,
                max_upload_bytes=self.max_upload_bytes,
            )
        return UploadCapability(mode="multipart")

    def initiate(
        self,
        *,
        filename: str,
        content_type: str,
        size_bytes: int,
        hashed_case_id: str | None,
    ) -> ResumableUploadSession:
        storage = self._gcs_storage()
        display_name = self._validate_filename(filename)
        normalized_content_type = self._validate_content_type(content_type)
        self._validate_size(size_bytes)
        validated_case_hash = validate_case_hash(hashed_case_id)
        video_id = new_id()
        upload_key = temporary_upload_key(video_id)

        session_url = storage.create_resumable_upload_session(
            upload_key,
            content_type=normalized_content_type,
            size=size_bytes,
            origin=self.web_origin,
        )
        with self.database.session() as session:
            session.add(
                VideoAsset(
                    id=video_id,
                    hashed_case_id=validated_case_hash,
                    original_filename=display_name,
                    safe_filename=display_name,
                    content_type=normalized_content_type,
                    status=VideoStatus.UPLOADING,
                    metadata_json={
                        "upload_transport": "gcs_resumable",
                        "declared_upload_bytes": size_bytes,
                        "declared_content_type": normalized_content_type,
                    },
                )
            )
            session.commit()

        return ResumableUploadSession(
            upload_id=video_id,
            session_url=session_url,
            chunk_size_bytes=RESUMABLE_CHUNK_SIZE_BYTES,
        )

    def complete(self, upload_id: str) -> AcceptedUpload:
        storage = self._gcs_storage()
        video, accepted = self._load_upload(upload_id)
        if accepted is not None:
            return accepted

        expected_size = video.metadata_json.get("declared_upload_bytes")
        expected_content_type = video.metadata_json.get("declared_content_type")
        if not isinstance(expected_size, int) or not isinstance(
            expected_content_type, str
        ):
            raise UploadStateError("upload declaration is incomplete")

        try:
            metadata = storage.verify_object_metadata(
                temporary_upload_key(upload_id),
                expected_size=expected_size,
                expected_content_type=expected_content_type,
            )
        except FileNotFoundError as exc:
            _, accepted = self._load_upload(upload_id)
            if accepted is not None:
                return accepted
            raise UploadIncompleteError("upload has not completed") from exc
        except GCSObjectVerificationError as exc:
            raise UploadVerificationError(str(exc)) from exc

        self._validate_generation(metadata)
        try:
            return self.ingest.enqueue_direct_upload(
                upload_id,
                metadata=metadata,
            )
        except DirectUploadStateError as exc:
            raise UploadStateError(str(exc)) from exc

    def _load_upload(
        self,
        upload_id: str,
    ) -> tuple[VideoAsset, AcceptedUpload | None]:
        with self.database.session() as session:
            video = session.get(VideoAsset, upload_id)
            if video is None:
                raise UploadNotFoundError("upload was not found")
            if video.metadata_json.get("upload_transport") != "gcs_resumable":
                raise UploadStateError("video is not a resumable upload")
            job = session.scalar(
                select(ProcessingJob)
                .where(
                    ProcessingJob.video_id == upload_id,
                    ProcessingJob.job_type == JobType.INGEST,
                )
                .order_by(ProcessingJob.created_at.asc())
            )
            session.expunge(video)
            if job is not None:
                session.expunge(job)
                return video, AcceptedUpload(video=video, job=job)
            return video, None

    def _gcs_storage(self) -> GCSStorage:
        if not isinstance(self.storage, GCSStorage):
            raise ResumableUploadUnavailableError(
                "resumable uploads are unavailable for local storage"
            )
        return self.storage

    @staticmethod
    def _validate_filename(filename: str) -> str:
        if not filename.strip():
            raise UploadValidationError("filename is required")
        return sanitize_filename(filename)

    @staticmethod
    def _validate_content_type(content_type: str) -> str:
        normalized = content_type.strip().lower()
        if normalized not in SUPPORTED_UPLOAD_CONTENT_TYPES:
            raise UploadValidationError("unsupported video content type")
        return normalized

    def _validate_size(self, size_bytes: int) -> None:
        if isinstance(size_bytes, bool) or size_bytes <= 0:
            raise UploadValidationError("upload size must be positive")
        if size_bytes > self.max_upload_bytes:
            raise UploadSizeExceededError(
                f"video exceeds the {self.max_upload_bytes // (1024 * 1024)} MB limit"
            )

    @staticmethod
    def _validate_generation(metadata: GCSObjectMetadata) -> None:
        if metadata.generation <= 0:
            raise UploadVerificationError("upload generation is invalid")
