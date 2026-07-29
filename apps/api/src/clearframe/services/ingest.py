import hashlib
import re
import shutil
from dataclasses import dataclass

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from clearframe.database import Database
from clearframe.domain.enums import JobType, VideoStatus
from clearframe.jobs import JobContext, JobDispatcher
from clearframe.media import MediaError, MediaKind, MediaProcessor, sha256_file, sniff_media
from clearframe.models import ProcessingJob, VideoAsset, new_id
from clearframe.storage import (
    ArtifactStorage,
    original_key,
    proxy_key,
    sanitize_filename,
    temporary_upload_key,
    thumbnail_key,
)


class IngestError(ValueError):
    pass


class UploadTooLargeError(IngestError):
    pass


class EmptyUploadError(IngestError):
    pass


class DuplicateVideoError(IngestError):
    def __init__(self, existing_video_id: str) -> None:
        super().__init__("this video has already been uploaded")
        self.existing_video_id = existing_video_id


@dataclass(frozen=True, slots=True)
class AcceptedUpload:
    video: VideoAsset
    job: ProcessingJob


class IngestService:
    def __init__(
        self,
        database: Database,
        storage: ArtifactStorage,
        media: MediaProcessor,
        runner: JobDispatcher,
        max_upload_mb: int,
    ) -> None:
        self.database = database
        self.storage = storage
        self.media = media
        self.runner = runner
        self.max_upload_bytes = max_upload_mb * 1024 * 1024

    @staticmethod
    def _validate_case_hash(value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = value.strip().lower()
        if not re.fullmatch(r"[a-f0-9]{64}", normalized):
            raise IngestError("hashed_case_id must be a SHA-256 hex digest")
        return normalized

    async def accept(
        self,
        upload: UploadFile,
        hashed_case_id: str | None = None,
    ) -> AcceptedUpload:
        validated_case_hash = self._validate_case_hash(hashed_case_id)
        display_name = sanitize_filename(upload.filename)
        video_id = new_id()
        temporary_uri = temporary_upload_key(video_id)
        digest = hashlib.sha256()
        total_bytes = 0

        try:
            with self.storage.publish_output(temporary_uri) as temporary_path:
                with temporary_path.open("xb") as stream:
                    while chunk := await upload.read(1024 * 1024):
                        total_bytes += len(chunk)
                        if total_bytes > self.max_upload_bytes:
                            raise UploadTooLargeError(
                                "video exceeds the "
                                f"{self.max_upload_bytes // (1024 * 1024)} MB limit"
                            )
                        digest.update(chunk)
                        stream.write(chunk)
                if total_bytes == 0:
                    raise EmptyUploadError("video is empty")
                media_kind = sniff_media(temporary_path)
        finally:
            await upload.close()

        checksum = digest.hexdigest()
        with self.database.session() as session:
            duplicate = session.scalar(
                select(VideoAsset).where(VideoAsset.original_sha256 == checksum)
            )
            if duplicate is not None:
                self.storage.remove_file(temporary_uri)
                raise DuplicateVideoError(duplicate.id)

            video = VideoAsset(
                id=video_id,
                hashed_case_id=validated_case_hash,
                original_filename=display_name,
                safe_filename=display_name,
                content_type=media_kind.content_type,
                original_sha256=checksum,
                status=VideoStatus.VALIDATING,
                metadata_json={
                    "container": media_kind.container,
                    "upload_bytes": total_bytes,
                },
            )
            job = ProcessingJob(
                video_id=video_id,
                job_type=JobType.INGEST,
                stage="queued",
                payload={
                    "temporary_uri": temporary_uri,
                    "expected_checksum": checksum,
                    "media_kind": {
                        "container": media_kind.container,
                        "extension": media_kind.extension,
                        "content_type": media_kind.content_type,
                    },
                },
            )
            session.add(video)
            session.flush()
            session.add(job)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                existing = session.scalar(
                    select(VideoAsset).where(VideoAsset.original_sha256 == checksum)
                )
                self.storage.remove_file(temporary_uri)
                if existing is not None:
                    raise DuplicateVideoError(existing.id) from exc
                raise IngestError("video record could not be created") from exc

        self.runner.enqueue(job.id)
        return AcceptedUpload(video=video, job=job)

    def execute(self, context: JobContext, job_id: str) -> None:
        with self.database.session() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None or job.job_type != JobType.INGEST or not job.video_id:
                raise IngestError("ingest job is invalid")
            temporary_uri = job.payload.get("temporary_uri")
            expected_checksum = job.payload.get("expected_checksum")
            media_payload = job.payload.get("media_kind")
            video_id = job.video_id

        if (
            not isinstance(temporary_uri, str)
            or not isinstance(expected_checksum, str)
            or not isinstance(media_payload, dict)
        ):
            raise IngestError("ingest job payload is invalid")
        container = media_payload.get("container")
        extension = media_payload.get("extension")
        content_type = media_payload.get("content_type")
        if (
            not isinstance(container, str)
            or not isinstance(extension, str)
            or not isinstance(content_type, str)
        ):
            raise IngestError("ingest media payload is invalid")

        self._finalize_with_cleanup(
            context,
            video_id=video_id,
            temporary_uri=temporary_uri,
            media_kind=MediaKind(
                container=container,
                extension=extension,
                content_type=content_type,
            ),
            expected_checksum=expected_checksum,
        )

    def _finalize_with_cleanup(
        self,
        context: JobContext,
        *,
        video_id: str,
        temporary_uri: str,
        media_kind: MediaKind,
        expected_checksum: str,
    ) -> None:
        proxy_uri = proxy_key(video_id)
        thumbnail_uri = thumbnail_key(video_id)
        try:
            self._finalize(
                context,
                video_id=video_id,
                temporary_uri=temporary_uri,
                media_kind=media_kind,
                expected_checksum=expected_checksum,
            )
        except Exception:
            self.storage.remove_file(proxy_uri)
            self.storage.remove_file(thumbnail_uri)
            raise
        finally:
            self.storage.remove_file(temporary_uri)

    def _finalize(
        self,
        context: JobContext,
        *,
        video_id: str,
        temporary_uri: str,
        media_kind: MediaKind,
        expected_checksum: str,
    ) -> None:
        with self.storage.materialize_input(temporary_uri) as temporary_path:
            context.update(0.08, "validating media")
            metadata = self.media.probe(temporary_path)

            original_uri = original_key(video_id, media_kind.extension)
            with self.storage.publish_output(original_uri) as original_path:
                shutil.copyfile(temporary_path, original_path)
                if sha256_file(original_path) != expected_checksum:
                    raise MediaError("original checksum changed during ingest")

            with self.database.session() as session:
                video = session.get(VideoAsset, video_id)
                if video is None:
                    raise IngestError("video record disappeared during ingest")
                video.duration_ms = metadata.duration_ms
                video.fps = metadata.fps
                video.width = metadata.width
                video.height = metadata.height
                video.codec = metadata.codec
                video.audio_present = metadata.audio_present
                video.original_uri = original_uri
                video.status = VideoStatus.PROXYING
                video.metadata_json = {
                    **video.metadata_json,
                    "frame_count_estimate": metadata.frame_count_estimate,
                    "ffmpeg_version": metadata.ffmpeg_version,
                }
                session.commit()

            context.update(0.35, "generating review proxy")
            proxy_uri = proxy_key(video_id)
            with self.storage.publish_output(proxy_uri) as proxy_path:
                self.media.generate_proxy(
                    temporary_path,
                    proxy_path,
                    metadata=metadata,
                )

            context.update(0.88, "generating thumbnail")
            thumbnail_uri = thumbnail_key(video_id)
            try:
                with self.storage.publish_output(thumbnail_uri) as thumbnail_path:
                    self.media.generate_thumbnail(
                        temporary_path,
                        thumbnail_path,
                        metadata.duration_ms,
                    )
            except MediaError:
                thumbnail_uri = ""

        with self.database.session() as session:
            video = session.get(VideoAsset, video_id)
            if video is None:
                raise IngestError("video record disappeared during proxy generation")
            video.proxy_uri = proxy_uri
            video.thumbnail_uri = thumbnail_uri or None
            video.status = VideoStatus.READY_FOR_REVIEW
            video.error_message = None
            session.commit()
        context.update(0.98, "finalizing")
