from dataclasses import dataclass

import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from clearframe.database import Database
from clearframe.domain.enums import JobStatus, JobType
from clearframe.jobs import JobContext, LocalJobRunner, utc_now
from clearframe.media import (
    PROXY_PROFILE_VERSION,
    MediaMetadata,
    MediaProcessor,
    ProxyAssessment,
)
from clearframe.models import ProcessingJob, Track, VideoAsset, new_id
from clearframe.storage import LocalStorage


class ProxyRepairError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProxyReconcileSummary:
    scanned: int = 0
    current: int = 0
    scheduled: int = 0
    blocked: int = 0
    skipped: int = 0


class ProxyService:
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
        self.logger = structlog.get_logger("clearframe.proxy")

    def reconcile_all(self) -> ProxyReconcileSummary:
        with self.database.session() as session:
            video_ids = list(
                session.scalars(
                    select(VideoAsset.id).where(
                        VideoAsset.original_uri.is_not(None),
                        VideoAsset.proxy_uri.is_not(None),
                    )
                )
            )

        counts = {
            "current": 0,
            "scheduled": 0,
            "blocked": 0,
            "skipped": 0,
        }
        for video_id in video_ids:
            try:
                outcome = self.reconcile_video(video_id)
            except Exception as exc:
                outcome = "skipped"
                self.logger.warning(
                    "proxy_reconcile_failed",
                    video_id=video_id,
                    error_type=type(exc).__name__,
                )
            counts[outcome] += 1

        summary = ProxyReconcileSummary(
            scanned=len(video_ids),
            current=counts["current"],
            scheduled=counts["scheduled"],
            blocked=counts["blocked"],
            skipped=counts["skipped"],
        )
        if summary.scanned:
            self.logger.info(
                "proxy_reconcile_complete",
                scanned=summary.scanned,
                current=summary.current,
                scheduled=summary.scheduled,
                blocked=summary.blocked,
                skipped=summary.skipped,
                profile_version=PROXY_PROFILE_VERSION,
            )
        return summary

    def reconcile_video(self, video_id: str) -> str:
        with self.database.session() as session:
            video = session.get(VideoAsset, video_id)
            if (
                video is None
                or not video.original_uri
                or not video.proxy_uri
                or not self.storage.exists(video.original_uri)
                or not self.storage.exists(video.proxy_uri)
            ):
                return "skipped"
            original_uri = video.original_uri
            proxy_uri = video.proxy_uri

        source_metadata = self.media.probe(self.storage.path_for(original_uri))
        proxy_metadata = self.media.probe(self.storage.path_for(proxy_uri))
        assessment = self.media.assess_proxy(source_metadata, proxy_metadata)

        with self.database.session() as session:
            video = session.get(VideoAsset, video_id)
            if video is None:
                return "skipped"
            if assessment.current:
                self._record_current(
                    video,
                    source_metadata,
                    proxy_metadata,
                    repair=None,
                )
                session.commit()
                return "current"

            active_job = session.scalar(
                select(ProcessingJob).where(
                    ProcessingJob.video_id == video_id,
                    ProcessingJob.job_type == JobType.PROXY,
                    ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
                )
            )
            if active_job is not None:
                return "scheduled"

            track_count = session.scalar(
                select(func.count(Track.id)).where(Track.video_id == video_id)
            )
            if track_count and not assessment.timeline_compatible:
                return self._record_blocked(
                    session,
                    video,
                    assessment,
                    track_count,
                )

            job = ProcessingJob(
                id=new_id(),
                video_id=video_id,
                job_type=JobType.PROXY,
                stage="queued for proxy repair",
                payload={
                    "action": "repair",
                    "profile_version": PROXY_PROFILE_VERSION,
                    "reasons": list(assessment.reasons),
                },
            )
            video.metadata_json = {
                **video.metadata_json,
                "proxy_repair": {
                    "status": "queued",
                    "job_id": job.id,
                    "profile_version": PROXY_PROFILE_VERSION,
                    "reasons": list(assessment.reasons),
                },
            }
            session.add(job)
            session.commit()

        self.logger.info(
            "proxy_repair_scheduled",
            video_id=video_id,
            job_id=job.id,
            reasons=assessment.reasons,
        )
        self.runner.submit(
            job.id,
            lambda context: self._repair_with_tracking(
                context,
                video_id=video_id,
                job_id=job.id,
            ),
        )
        return "scheduled"

    def _record_blocked(
        self,
        session: Session,
        video: VideoAsset,
        assessment: ProxyAssessment,
        track_count: int,
    ) -> str:
        previous = video.metadata_json.get("proxy_repair")
        reasons = list(assessment.reasons)
        if (
            isinstance(previous, dict)
            and previous.get("status") == "blocked"
            and previous.get("profile_version") == PROXY_PROFILE_VERSION
            and previous.get("reasons") == reasons
        ):
            return "blocked"

        message = (
            "Proxy repair requires detection to be rerun because the existing "
            "track timeline does not match the original."
        )
        job = ProcessingJob(
            id=new_id(),
            video_id=video.id,
            job_type=JobType.PROXY,
            status=JobStatus.FAILED,
            progress=0.0,
            stage="proxy repair blocked",
            payload={
                "action": "repair",
                "profile_version": PROXY_PROFILE_VERSION,
                "reasons": reasons,
                "track_count": track_count,
            },
            error_message=message,
            completed_at=utc_now(),
        )
        video.metadata_json = {
            **video.metadata_json,
            "proxy_repair": {
                "status": "blocked",
                "job_id": job.id,
                "profile_version": PROXY_PROFILE_VERSION,
                "reasons": reasons,
                "track_count": track_count,
            },
        }
        session.add(job)
        session.commit()
        self.logger.warning(
            "proxy_repair_blocked",
            video_id=video.id,
            job_id=job.id,
            track_count=track_count,
            reasons=assessment.reasons,
        )
        return "blocked"

    def _repair_with_tracking(
        self,
        context: JobContext,
        *,
        video_id: str,
        job_id: str,
    ) -> None:
        try:
            self._repair(context, video_id=video_id, job_id=job_id)
        except Exception as exc:
            with self.database.session() as session:
                video = session.get(VideoAsset, video_id)
                if video is not None:
                    video.metadata_json = {
                        **video.metadata_json,
                        "proxy_repair": {
                            "status": "failed",
                            "job_id": job_id,
                            "profile_version": PROXY_PROFILE_VERSION,
                            "error_type": type(exc).__name__,
                        },
                    }
                    session.commit()
            self.logger.error(
                "proxy_repair_failed",
                video_id=video_id,
                job_id=job_id,
                error_type=type(exc).__name__,
            )
            raise

    def _repair(
        self,
        context: JobContext,
        *,
        video_id: str,
        job_id: str,
    ) -> None:
        with self.database.session() as session:
            video = session.get(VideoAsset, video_id)
            if video is None or not video.original_uri or not video.proxy_uri:
                raise ProxyRepairError("proxy repair input disappeared")
            original_uri = video.original_uri
            proxy_uri = video.proxy_uri
            video.metadata_json = {
                **video.metadata_json,
                "proxy_repair": {
                    "status": "running",
                    "job_id": job_id,
                    "profile_version": PROXY_PROFILE_VERSION,
                },
            }
            session.commit()

        original_path = self.storage.path_for(original_uri)
        proxy_path = self.storage.path_for(proxy_uri)
        if not original_path.is_file() or not proxy_path.is_file():
            raise ProxyRepairError("proxy repair input is missing")

        context.update(0.08, "validating stale proxy")
        source_metadata = self.media.probe(original_path)
        existing_metadata = self.media.probe(proxy_path)
        existing_assessment = self.media.assess_proxy(
            source_metadata,
            existing_metadata,
        )
        if existing_assessment.current:
            with self.database.session() as session:
                video = session.get(VideoAsset, video_id)
                if video is None:
                    raise ProxyRepairError("video disappeared during proxy repair")
                self._record_current(
                    video,
                    source_metadata,
                    existing_metadata,
                    repair={"status": "complete", "job_id": job_id},
                )
                session.commit()
            return

        with self.database.session() as session:
            track_count = session.scalar(
                select(func.count(Track.id)).where(Track.video_id == video_id)
            )
        if track_count and not existing_assessment.timeline_compatible:
            raise ProxyRepairError("existing tracks are incompatible with the source timeline")

        candidate = proxy_path.with_name(f".{proxy_path.stem}.repair-{job_id}.mp4")
        candidate.unlink(missing_ok=True)
        try:
            context.update(0.2, "regenerating review proxy")
            self.media.generate_proxy(
                original_path,
                candidate,
                metadata=source_metadata,
            )
            context.update(0.85, "validating regenerated proxy")
            candidate_metadata = self.media.probe(candidate)
            candidate_assessment = self.media.assess_proxy(
                source_metadata,
                candidate_metadata,
            )
            if not candidate_assessment.current:
                raise ProxyRepairError(
                    "regenerated proxy does not match the current profile"
                )

            context.update(0.94, "activating regenerated proxy")
            candidate.replace(proxy_path)
            with self.database.session() as session:
                video = session.get(VideoAsset, video_id)
                if video is None:
                    raise ProxyRepairError("video disappeared during proxy repair")
                self._record_current(
                    video,
                    source_metadata,
                    candidate_metadata,
                    repair={
                        "status": "complete",
                        "job_id": job_id,
                        "reasons": list(existing_assessment.reasons),
                    },
                )
                session.commit()
            self.logger.info(
                "proxy_repair_complete",
                video_id=video_id,
                job_id=job_id,
                strategy=self.media.proxy_strategy(source_metadata),
                width=candidate_metadata.width,
                height=candidate_metadata.height,
            )
        finally:
            candidate.unlink(missing_ok=True)

    def _record_current(
        self,
        video: VideoAsset,
        source: MediaMetadata,
        proxy: MediaMetadata,
        *,
        repair: dict[str, object] | None,
    ) -> None:
        metadata = {
            **video.metadata_json,
            "proxy_profile": {
                "version": PROXY_PROFILE_VERSION,
                "strategy": self.media.proxy_strategy(source),
                "width": proxy.width,
                "height": proxy.height,
                "fps": proxy.fps,
                "duration_ms": proxy.duration_ms,
                "audio_present": proxy.audio_present,
            },
        }
        if repair is not None:
            metadata["proxy_repair"] = {
                **repair,
                "profile_version": PROXY_PROFILE_VERSION,
            }
        video.metadata_json = metadata
