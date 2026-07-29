import json
import subprocess
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, cast

import cv2
from sqlalchemy import func, select

from clearframe import __version__
from clearframe.database import Database
from clearframe.domain.enums import (
    ExportStatus,
    JobType,
    RedactionStyle,
    ReprocessingSuggestionStatus,
    ReviewActionType,
    VideoStatus,
)
from clearframe.domain.review import ReviewSnapshot
from clearframe.jobs import JobContext, JobDispatcher
from clearframe.media import (
    H264Encoder,
    MediaError,
    MediaMetadata,
    MediaProcessor,
    sha256_file,
)
from clearframe.models import (
    ExportArtifact,
    ModelRun,
    ProcessingJob,
    ReprocessingSuggestion,
    ReviewAction,
    VideoAsset,
    new_id,
)
from clearframe.rendering import (
    ClassTreatment,
    Frame,
    SequentialRedactionLookup,
    parse_class_treatments,
    redact_frame,
)
from clearframe.services.review import (
    RevisionConflictError,
    append_system_audit_event,
    build_review_snapshot_at_revision,
)
from clearframe.storage import (
    ArtifactStorage,
    export_manifest_key,
    export_video_key,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class ExportError(ValueError):
    pass


class ExportNotFoundError(ExportError):
    pass


class ExportValidationError(ExportError):
    pass


class _EncoderFailure(ExportValidationError):
    pass


@dataclass(frozen=True, slots=True)
class RequestedExport:
    artifact: ExportArtifact
    job: ProcessingJob
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _RenderedVideo:
    frame_count: int
    encoder_name: str
    hardware_accelerated: bool
    fallback_used: bool
    metadata: MediaMetadata


class ExportService:
    def __init__(
        self,
        database: Database,
        storage: ArtifactStorage,
        media: MediaProcessor,
        runner: JobDispatcher,
        build_id: str,
    ) -> None:
        self.database = database
        self.storage = storage
        self.media = media
        self.runner = runner
        self.build_id = build_id

    @staticmethod
    def _unconfirmed_track_count(snapshot: ReviewSnapshot) -> int:
        return sum(1 for track in snapshot.tracks.values() if not track.accepted)

    @classmethod
    def _review_warnings(
        cls,
        snapshot: ReviewSnapshot,
        *,
        pending_suggestions: int,
    ) -> list[str]:
        warnings = []
        if not snapshot.tracks:
            warnings.append("The review contains no redaction tracks.")
        unconfirmed = cls._unconfirmed_track_count(snapshot)
        if unconfirmed:
            warnings.append(
                f"{unconfirmed} redaction track(s) were not individually "
                "confirmed by a reviewer."
            )
        missing_geometry = sum(
            1
            for track in snapshot.tracks.values()
            if track.active and track.redacted and not track.keyframes
        )
        if missing_geometry:
            warnings.append(
                f"{missing_geometry} active redaction(s) have no geometry "
                "and will not be rendered."
            )
        if pending_suggestions:
            warnings.append(
                f"{pending_suggestions} context suggestion(s) were still "
                "pending at export time."
            )
        return warnings

    def request(
        self,
        video_id: str,
        *,
        expected_revision: int,
        style: RedactionStyle,
        class_treatments: dict[str, ClassTreatment] | None = None,
        reviewer_session_id: str,
    ) -> RequestedExport:
        treatments_payload = {
            class_name: {
                key: value
                for key, value in (
                    ("style", treatment.style),
                    ("shape", treatment.shape),
                )
                if value is not None
            }
            for class_name, treatment in (class_treatments or {}).items()
        }
        with self.database.session() as session:
            video = session.get(VideoAsset, video_id)
            if video is None:
                raise ExportNotFoundError("video not found")
            if not video.original_uri or not video.original_sha256:
                raise ExportValidationError("immutable original is not ready")
            if video.review_revision != expected_revision:
                raise RevisionConflictError(expected_revision, video.review_revision)
            snapshot = build_review_snapshot_at_revision(
                session,
                video_id,
                expected_revision,
            )
            pending_context = (
                session.scalar(
                    select(func.count(ReprocessingSuggestion.id)).where(
                        ReprocessingSuggestion.video_id == video_id,
                        ReprocessingSuggestion.status
                        == ReprocessingSuggestionStatus.PENDING,
                    )
                )
                or 0
            )
            review_warnings = self._review_warnings(
                snapshot,
                pending_suggestions=pending_context,
            )

            export_id = new_id()
            artifact = ExportArtifact(
                id=export_id,
                video_id=video_id,
                redaction_style=style,
                class_treatments=treatments_payload,
                source_model_run_id=video.active_model_run_id,
                review_revision=expected_revision,
                status=ExportStatus.QUEUED,
            )
            session.add(artifact)
            session.flush()
            job = ProcessingJob(
                video_id=video_id,
                export_id=export_id,
                job_type=JobType.EXPORT,
                payload={
                    "review_revision": expected_revision,
                    "redaction_style": style,
                },
            )
            session.add(job)
            append_system_audit_event(
                session,
                video_id,
                ReviewActionType.EXPORT_REQUESTED,
                {
                    "export_id": export_id,
                    "frozen_review_revision": expected_revision,
                    "redaction_style": style,
                    "class_treatments": treatments_payload,
                    "review_warnings": review_warnings,
                },
                reviewer_session_id=reviewer_session_id,
            )
            video.status = VideoStatus.EXPORTING
            session.commit()

        self.runner.enqueue(job.id)
        return RequestedExport(artifact=artifact, job=job, warnings=tuple(review_warnings))

    def execute(self, context: JobContext, job_id: str) -> None:
        with self.database.session() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None or job.job_type != JobType.EXPORT or not job.export_id:
                raise ExportValidationError("export job is invalid")
            export_id = job.export_id
        self._render_with_cleanup(context, export_id=export_id)

    def _render_with_cleanup(self, context: JobContext, *, export_id: str) -> None:
        with self.database.session() as session:
            artifact = session.get(ExportArtifact, export_id)
            if artifact is None:
                raise ExportNotFoundError(export_id)
            video_uri = export_video_key(artifact.video_id, export_id)
            manifest_uri = export_manifest_key(artifact.video_id, export_id)
            artifact.status = ExportStatus.RENDERING
            session.commit()
        try:
            self._render(context, export_id=export_id)
        except Exception:
            self.storage.remove_file(video_uri)
            self.storage.remove_file(manifest_uri)
            raise

    def _render(self, context: JobContext, *, export_id: str) -> None:
        with self.database.session() as session:
            artifact = session.get(ExportArtifact, export_id)
            if artifact is None:
                raise ExportNotFoundError(export_id)
            video = session.get(VideoAsset, artifact.video_id)
            if video is None or not video.original_uri or not video.original_sha256:
                raise ExportValidationError("immutable original is unavailable")
            snapshot = build_review_snapshot_at_revision(
                session,
                video.id,
                artifact.review_revision,
            )
            action_count = int(
                session.scalar(
                    select(func.count(ReviewAction.id)).where(
                        ReviewAction.video_id == video.id,
                        ReviewAction.revision <= artifact.review_revision,
                    )
                )
                or 0
            )
            original_uri = video.original_uri
            expected_original_hash = video.original_sha256
            expected_duration_ms = video.duration_ms or 0
            expected_width = video.width or 0
            expected_height = video.height or 0
            fps = video.fps or 0.0
            model_run_id = artifact.source_model_run_id
            model_run = (
                session.get(ModelRun, model_run_id)
                if model_run_id is not None
                else None
            )
            model_registry_sha256 = model_run.config_hash if model_run is not None else None
            video_id = video.id
            original_filename = video.original_filename
            pending_context = (
                session.scalar(
                    select(func.count(ReprocessingSuggestion.id)).where(
                        ReprocessingSuggestion.video_id == video.id,
                        ReprocessingSuggestion.status
                        == ReprocessingSuggestionStatus.PENDING,
                    )
                )
                or 0
            )

        review_warnings = self._review_warnings(
            snapshot,
            pending_suggestions=pending_context,
        )
        if fps <= 0 or expected_width <= 0 or expected_height <= 0:
            raise ExportValidationError("source media metadata is incomplete")

        export_uri = export_video_key(video_id, export_id)
        with (
            self.storage.materialize_input(original_uri) as original_path,
            self.storage.publish_output(export_uri) as export_path,
        ):
            if sha256_file(original_path) != expected_original_hash:
                raise ExportValidationError("original checksum verification failed")

            context.update(0.05, "rendering reviewed frames")
            rendered_video = self._render_video(
                source=original_path,
                destination=export_path,
                snapshot=snapshot,
                style=RedactionStyle(artifact.redaction_style),
                treatments=parse_class_treatments(artifact.class_treatments),
                fps=fps,
                width=expected_width,
                height=expected_height,
                estimated_frames=max(
                    1,
                    round(expected_duration_ms * fps / 1000),
                ),
                context=context,
            )

            context.update(0.9, "verifying export")
            with self.database.session() as session:
                stored_artifact = session.get(ExportArtifact, export_id)
                if stored_artifact is None:
                    raise ExportNotFoundError(export_id)
                stored_artifact.status = ExportStatus.VERIFYING
                session.commit()
            output_metadata = rendered_video.metadata
            frame_tolerance_ms = max(150, round(2000 / fps))
            if abs(output_metadata.duration_ms - expected_duration_ms) > frame_tolerance_ms:
                raise ExportValidationError("export duration is outside tolerance")
            if (
                output_metadata.width != expected_width
                or output_metadata.height != expected_height
            ):
                raise ExportValidationError("export dimensions do not match the original")
            if sha256_file(original_path) != expected_original_hash:
                raise ExportValidationError("original changed during export")

            export_hash = sha256_file(export_path)
        track_counts = Counter(
            track.class_name
            for track in snapshot.tracks.values()
            if track.active and track.redacted
        )
        warnings = sorted(
            {
                track.warning
                for track in snapshot.tracks.values()
                if track.warning is not None
            }
        )
        warnings.extend(review_warnings)
        if output_metadata.audio_present:
            warnings.append("Audio was preserved without redaction.")
        manifest = {
            "schema_version": 1,
            "export_id": export_id,
            "video_id": video_id,
            "source_filename": original_filename,
            "original_sha256": expected_original_hash,
            "export_sha256": export_hash,
            "application_version": __version__,
            "build_id": self.build_id,
            "ffmpeg_version": self.media.ffmpeg_version,
            "model_run_id": model_run_id,
            "model_registry_sha256": model_registry_sha256,
            "review_revision": artifact.review_revision,
            "redaction_style": artifact.redaction_style,
            "class_treatments": artifact.class_treatments,
            "action_count": action_count,
            "unconfirmed_track_count": self._unconfirmed_track_count(snapshot),
            "redaction_track_counts": dict(sorted(track_counts.items())),
            "frames_rendered": rendered_video.frame_count,
            "video_encoder": rendered_video.encoder_name,
            "hardware_video_encoding": rendered_video.hardware_accelerated,
            "encoder_fallback_used": rendered_video.fallback_used,
            "duration_ms": output_metadata.duration_ms,
            "width": output_metadata.width,
            "height": output_metadata.height,
            "fps": output_metadata.fps,
            "audio_present": output_metadata.audio_present,
            "audio_redaction_applied": False,
            "audio_policy": "preserved_unreviewed",
            "warnings": warnings,
            "created_at": utc_now().isoformat(),
        }
        manifest_uri = export_manifest_key(video_id, export_id)
        with self.storage.publish_output(manifest_uri) as manifest_path:
            self._write_manifest(manifest_path, manifest)

        with self.database.session() as session:
            stored_artifact = session.get(ExportArtifact, export_id)
            stored_video = session.get(VideoAsset, video_id)
            if stored_artifact is None or stored_video is None:
                raise ExportNotFoundError(export_id)
            stored_artifact.export_uri = export_uri
            stored_artifact.export_sha256 = export_hash
            stored_artifact.manifest_uri = manifest_uri
            stored_artifact.status = ExportStatus.COMPLETED
            stored_artifact.completed_at = utc_now()
            stored_video.status = VideoStatus.EXPORTED
            stored_video.error_message = None
            append_system_audit_event(
                session,
                video_id,
                ReviewActionType.EXPORT_COMPLETED,
                {
                    "export_id": export_id,
                    "export_sha256": export_hash,
                    "frozen_review_revision": stored_artifact.review_revision,
                },
                reviewer_session_id="system:export",
            )
            session.commit()
        context.update(0.98, "export verified")

    def _render_video(
        self,
        *,
        source: Path,
        destination: Path,
        snapshot: ReviewSnapshot,
        style: RedactionStyle,
        treatments: dict[str, ClassTreatment],
        fps: float,
        width: int,
        height: int,
        estimated_frames: int,
        context: JobContext,
    ) -> _RenderedVideo:
        encoders = self.media.export_h264_encoders()
        for attempt, encoder in enumerate(encoders):
            temporary = destination.with_name(
                f"{destination.stem}.{attempt}.{encoder.name}.part{destination.suffix}"
            )
            temporary.unlink(missing_ok=True)
            try:
                frame_count = self._render_video_attempt(
                    source=source,
                    destination=temporary,
                    snapshot=snapshot,
                    style=style,
                    treatments=treatments,
                    fps=fps,
                    width=width,
                    height=height,
                    estimated_frames=estimated_frames,
                    context=context,
                    encoder=encoder,
                )
            except _EncoderFailure as exc:
                temporary.unlink(missing_ok=True)
                if encoder.hardware_accelerated and attempt + 1 < len(encoders):
                    context.update(0.08, "hardware encoder failed; retrying on CPU")
                    continue
                raise ExportValidationError("video renderer failed") from exc
            except Exception:
                temporary.unlink(missing_ok=True)
                raise

            try:
                attempt_metadata = self.media.probe(temporary)
            except MediaError as exc:
                temporary.unlink(missing_ok=True)
                if encoder.hardware_accelerated and attempt + 1 < len(encoders):
                    context.update(0.08, "hardware output invalid; retrying on CPU")
                    continue
                raise ExportValidationError("rendered video validation failed") from exc
            if (
                attempt_metadata.width != width
                or attempt_metadata.height != height
            ):
                temporary.unlink(missing_ok=True)
                if encoder.hardware_accelerated and attempt + 1 < len(encoders):
                    context.update(0.08, "hardware output invalid; retrying on CPU")
                    continue
                raise ExportValidationError("rendered video dimensions changed")
            expected_duration_ms = round(estimated_frames * 1000 / fps)
            frame_tolerance_ms = max(150, round(2000 / fps))
            if (
                abs(attempt_metadata.duration_ms - expected_duration_ms)
                > frame_tolerance_ms
            ):
                temporary.unlink(missing_ok=True)
                if encoder.hardware_accelerated and attempt + 1 < len(encoders):
                    context.update(0.08, "hardware output incomplete; retrying on CPU")
                    continue
                raise ExportValidationError("rendered video duration is incomplete")

            try:
                temporary.replace(destination)
            except OSError as exc:
                temporary.unlink(missing_ok=True)
                raise ExportValidationError("rendered video could not be finalized") from exc
            return _RenderedVideo(
                frame_count=frame_count,
                encoder_name=encoder.name,
                hardware_accelerated=encoder.hardware_accelerated,
                fallback_used=attempt > 0,
                metadata=attempt_metadata,
            )
        raise ExportValidationError("no H.264 video encoder is available")

    def _render_video_attempt(
        self,
        *,
        source: Path,
        destination: Path,
        snapshot: ReviewSnapshot,
        style: RedactionStyle,
        treatments: dict[str, ClassTreatment],
        fps: float,
        width: int,
        height: int,
        estimated_frames: int,
        context: JobContext,
        encoder: H264Encoder,
    ) -> int:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ExportValidationError("original video could not be opened")

        command = [
            str(self.media.ffmpeg_path),
            "-nostdin",
            "-max_alloc",
            str(256 * 1024 * 1024),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "pipe:0",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            *encoder.ffmpeg_arguments,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        try:
            process: subprocess.Popen[bytes] = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            capture.release()
            raise _EncoderFailure(f"{encoder.name} could not be started") from exc
        if process.stdin is None:
            capture.release()
            process.kill()
            raise _EncoderFailure(f"{encoder.name} pipe could not be created")

        redaction_lookup = SequentialRedactionLookup(snapshot)
        frame_index = 0
        pipe_closed = False
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    raise ExportValidationError("decoded frame dimensions changed")
                redactions = redaction_lookup.at_frame(frame_index)
                rendered = redact_frame(
                    cast(Frame, frame),
                    redactions,
                    style,
                    treatments,
                )
                self._write_frame(
                    process.stdin,
                    rendered.tobytes(),
                    encoder_name=encoder.name,
                )
                frame_index += 1
                if frame_index % max(1, estimated_frames // 20) == 0:
                    progress = min(0.87, 0.08 + 0.79 * frame_index / estimated_frames)
                    context.update(progress, "rendering reviewed frames")
        except Exception:
            if process.poll() is None:
                process.kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            raise
        finally:
            capture.release()
            try:
                process.stdin.close()
                pipe_closed = True
            except OSError:
                pass

        try:
            return_code = process.wait(timeout=120)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            raise _EncoderFailure(f"{encoder.name} timed out") from exc
        if return_code != 0:
            raise _EncoderFailure(f"{encoder.name} failed")
        if not pipe_closed:
            raise _EncoderFailure(f"{encoder.name} stopped unexpectedly")
        if frame_index == 0:
            raise ExportValidationError("source contained no decodable frames")
        return frame_index

    @staticmethod
    def _write_frame(
        stream: IO[bytes],
        frame_bytes: bytes,
        *,
        encoder_name: str,
    ) -> None:
        try:
            stream.write(frame_bytes)
        except OSError as exc:
            raise _EncoderFailure(f"{encoder_name} stopped unexpectedly") from exc

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
        temporary_path = path.with_suffix(".json.tmp")
        with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_path.replace(path)
