import json
import subprocess
from collections import Counter
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
    ReviewActionType,
    VideoStatus,
)
from clearframe.domain.review import ReviewSnapshot
from clearframe.jobs import JobContext, LocalJobRunner
from clearframe.media import MediaProcessor, sha256_file
from clearframe.models import ExportArtifact, ProcessingJob, ReviewAction, VideoAsset, new_id
from clearframe.rendering import Frame, redact_frame, redactions_at_frame
from clearframe.services.review import (
    RevisionConflictError,
    append_system_audit_event,
    build_review_snapshot_at_revision,
)
from clearframe.storage import LocalStorage


def utc_now() -> datetime:
    return datetime.now(UTC)


class ExportError(ValueError):
    pass


class ExportNotFoundError(ExportError):
    pass


class ExportValidationError(ExportError):
    pass


@dataclass(frozen=True, slots=True)
class RequestedExport:
    artifact: ExportArtifact
    job: ProcessingJob


class ExportService:
    def __init__(
        self,
        database: Database,
        storage: LocalStorage,
        media: MediaProcessor,
        runner: LocalJobRunner,
    ) -> None:
        self.database = database
        self.storage = storage
        self.media = media
        self.runner = runner

    @staticmethod
    def _validate_snapshot(snapshot: ReviewSnapshot) -> None:
        if not snapshot.tracks:
            raise ExportValidationError("review has no redaction tracks")
        unresolved = [
            track.track_id for track in snapshot.tracks.values() if not track.accepted
        ]
        if unresolved:
            raise ExportValidationError(
                f"{len(unresolved)} track(s) still require reviewer confirmation"
            )
        missing_geometry = [
            track.track_id
            for track in snapshot.tracks.values()
            if track.active and track.redacted and not track.keyframes
        ]
        if missing_geometry:
            raise ExportValidationError("active redactions are missing geometry")

    def request(
        self,
        video_id: str,
        *,
        expected_revision: int,
        style: RedactionStyle,
        reviewer_session_id: str,
    ) -> RequestedExport:
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
            self._validate_snapshot(snapshot)

            export_id = new_id()
            artifact = ExportArtifact(
                id=export_id,
                video_id=video_id,
                redaction_style=style,
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
                },
                reviewer_session_id=reviewer_session_id,
            )
            video.status = VideoStatus.EXPORTING
            session.commit()

        self.runner.submit(
            job.id,
            lambda context: self._render_with_cleanup(
                context,
                export_id=export_id,
            ),
        )
        return RequestedExport(artifact=artifact, job=job)

    def _render_with_cleanup(self, context: JobContext, *, export_id: str) -> None:
        with self.database.session() as session:
            artifact = session.get(ExportArtifact, export_id)
            if artifact is None:
                raise ExportNotFoundError(export_id)
            video_uri = self.storage.export_video_uri(artifact.video_id, export_id)
            manifest_uri = self.storage.export_manifest_uri(artifact.video_id, export_id)
            artifact.status = ExportStatus.RENDERING
            session.commit()
        try:
            self._render(context, export_id=export_id)
        except Exception:
            self.storage.remove_file(video_uri)
            self.storage.remove_file(manifest_uri)
            temporary_manifest = self.storage.path_for(manifest_uri).with_suffix(".json.tmp")
            if temporary_manifest.is_file():
                temporary_manifest.unlink()
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
            model_run_id = video.active_model_run_id
            video_id = video.id
            original_filename = video.original_filename

        self._validate_snapshot(snapshot)
        if fps <= 0 or expected_width <= 0 or expected_height <= 0:
            raise ExportValidationError("source media metadata is incomplete")

        original_path = self.storage.path_for(original_uri)
        if sha256_file(original_path) != expected_original_hash:
            raise ExportValidationError("original checksum verification failed")

        export_uri = self.storage.export_video_uri(video_id, export_id)
        export_path = self.storage.prepare(export_uri)
        context.update(0.05, "rendering reviewed frames")
        frame_count = self._render_video(
            source=original_path,
            destination=export_path,
            snapshot=snapshot,
            style=RedactionStyle(artifact.redaction_style),
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
        output_metadata = self.media.probe(export_path)
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
        manifest = {
            "schema_version": 1,
            "export_id": export_id,
            "video_id": video_id,
            "source_filename": original_filename,
            "original_sha256": expected_original_hash,
            "export_sha256": export_hash,
            "application_version": __version__,
            "model_run_id": model_run_id,
            "review_revision": artifact.review_revision,
            "redaction_style": artifact.redaction_style,
            "action_count": action_count,
            "redaction_track_counts": dict(sorted(track_counts.items())),
            "frames_rendered": frame_count,
            "duration_ms": output_metadata.duration_ms,
            "width": output_metadata.width,
            "height": output_metadata.height,
            "fps": output_metadata.fps,
            "audio_present": output_metadata.audio_present,
            "warnings": warnings,
            "created_at": utc_now().isoformat(),
        }
        manifest_uri = self.storage.export_manifest_uri(video_id, export_id)
        self._write_manifest(self.storage.prepare(manifest_uri), manifest)

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
        fps: float,
        width: int,
        height: int,
        estimated_frames: int,
        context: JobContext,
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
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
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
        process: subprocess.Popen[bytes] = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if process.stdin is None:
            capture.release()
            process.kill()
            raise ExportValidationError("renderer pipe could not be created")

        frame_index = 0
        pipe_closed = False
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    raise ExportValidationError("decoded frame dimensions changed")
                redactions = redactions_at_frame(snapshot, frame_index)
                rendered = redact_frame(cast(Frame, frame), redactions, style)
                self._write_frame(process.stdin, rendered.tobytes())
                frame_index += 1
                if frame_index % max(1, estimated_frames // 20) == 0:
                    progress = min(0.87, 0.08 + 0.79 * frame_index / estimated_frames)
                    context.update(progress, "rendering reviewed frames")
        except Exception:
            process.kill()
            process.wait(timeout=10)
            raise
        finally:
            capture.release()
            try:
                process.stdin.close()
                pipe_closed = True
            except BrokenPipeError:
                pass

        error_output = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait(timeout=120)
        if return_code != 0:
            raise ExportValidationError("video renderer failed")
        if not pipe_closed and error_output:
            raise ExportValidationError("video renderer stopped unexpectedly")
        if frame_index == 0:
            raise ExportValidationError("source contained no decodable frames")
        return frame_index

    @staticmethod
    def _write_frame(stream: IO[bytes], frame_bytes: bytes) -> None:
        try:
            stream.write(frame_bytes)
        except BrokenPipeError as exc:
            raise ExportValidationError("video renderer stopped unexpectedly") from exc

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
        temporary_path = path.with_suffix(".json.tmp")
        with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_path.replace(path)
