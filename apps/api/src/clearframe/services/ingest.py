import hashlib
import re
from dataclasses import dataclass

from fastapi import UploadFile
from sqlalchemy import select

from clearframe.database import Database
from clearframe.domain.enums import JobType, VideoStatus
from clearframe.jobs import JobContext, LocalJobRunner
from clearframe.media import MediaError, MediaKind, MediaProcessor, sha256_file, sniff_media
from clearframe.models import ProcessingJob, VideoAsset, new_id
from clearframe.storage import LocalStorage, sanitize_filename


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
        storage: LocalStorage,
        media: MediaProcessor,
        runner: LocalJobRunner,
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
        temporary_uri = self.storage.temporary_upload_uri(video_id)
        temporary_path = self.storage.prepare(temporary_uri)
        digest = hashlib.sha256()
        total_bytes = 0

        try:
            with temporary_path.open("xb") as stream:
                while chunk := await upload.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > self.max_upload_bytes:
                        raise UploadTooLargeError(
                            f"video exceeds the {self.max_upload_bytes // (1024 * 1024)} MB limit"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
        except Exception:
            self.storage.remove_file(temporary_uri)
            raise
        finally:
            await upload.close()

        if total_bytes == 0:
            self.storage.remove_file(temporary_uri)
            raise EmptyUploadError("video is empty")

        try:
            media_kind = sniff_media(temporary_path)
        except Exception:
            self.storage.remove_file(temporary_uri)
            raise

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
                payload={"temporary_uri": temporary_uri},
            )
            session.add(video)
            session.flush()
            session.add(job)
            session.commit()

        self.runner.submit(
            job.id,
            lambda context: self._finalize(
                context,
                video_id=video_id,
                temporary_uri=temporary_uri,
                media_kind=media_kind,
                expected_checksum=checksum,
            ),
        )
        return AcceptedUpload(video=video, job=job)

    def _finalize(
        self,
        context: JobContext,
        *,
        video_id: str,
        temporary_uri: str,
        media_kind: MediaKind,
        expected_checksum: str,
    ) -> None:
        temporary_path = self.storage.path_for(temporary_uri)
        context.update(0.08, "validating media")
        metadata = self.media.probe(temporary_path)

        original_uri = self.storage.original_uri(video_id, media_kind.extension)
        original_path = self.storage.promote(temporary_uri, original_uri)
        if sha256_file(original_path) != expected_checksum:
            raise MediaError("original checksum changed during ingest")
        self.storage.make_read_only(original_uri)

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
        proxy_uri = self.storage.proxy_uri(video_id)
        self.media.generate_proxy(original_path, self.storage.prepare(proxy_uri))

        context.update(0.88, "generating thumbnail")
        thumbnail_uri = self.storage.thumbnail_uri(video_id)
        try:
            self.media.generate_thumbnail(
                original_path,
                self.storage.prepare(thumbnail_uri),
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
