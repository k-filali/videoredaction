from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Event, Lock, Thread
from typing import Protocol, cast

import structlog
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from clearframe.database import Database
from clearframe.domain.enums import ExportStatus, JobStatus, JobType, VideoStatus
from clearframe.models import ExportArtifact, ProcessingJob, VideoAsset


def utc_now() -> datetime:
    return datetime.now(UTC)


class JobContext:
    def __init__(
        self,
        database: Database,
        job_id: str,
        lease_duration: timedelta = timedelta(minutes=5),
    ) -> None:
        self.database = database
        self.job_id = job_id
        self.lease_duration = lease_duration

    def update(self, progress: float, stage: str) -> None:
        now = utc_now()
        with self.database.session() as session:
            update_result = cast(
                CursorResult[object],
                session.execute(
                    update(ProcessingJob)
                    .where(
                        ProcessingJob.id == self.job_id,
                        ProcessingJob.status == JobStatus.RUNNING,
                    )
                    .values(
                        progress=min(1.0, max(0.0, progress)),
                        stage=stage[:80],
                        heartbeat_at=now,
                        lease_expires_at=now + self.lease_duration,
                    )
                    .execution_options(synchronize_session=False)
                ),
            )
            if update_result.rowcount != 1:
                session.rollback()
                raise RuntimeError("processing job is no longer running")
            session.commit()


class JobDispatcher(Protocol):
    def enqueue(self, job_id: str) -> None: ...


class ManagedJobDispatcher(JobDispatcher, Protocol):
    def recover_interrupted_jobs(self) -> None: ...

    def wait(self, job_id: str, timeout: float = 30.0) -> None: ...

    def shutdown(self) -> None: ...


JobHandler = Callable[[JobContext, str], None]


class JobExecutionResult(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_CLAIMED = "not_claimed"


def _fail_job(
    session: Session,
    job: ProcessingJob,
    *,
    stage: str,
    message: str,
) -> None:
    job.status = JobStatus.FAILED
    job.stage = stage
    job.error_message = message
    job.completed_at = utc_now()
    job.lease_expires_at = None
    if job.video_id:
        video = session.get(VideoAsset, job.video_id)
        if video is not None and job.job_type != JobType.PROXY:
            video.status = (
                VideoStatus.READY_FOR_REVIEW
                if job.job_type in {JobType.EXPORT, JobType.REPROCESS}
                else VideoStatus.FAILED
            )
            video.error_message = message
    if job.export_id:
        export = session.get(ExportArtifact, job.export_id)
        if export is not None:
            export.status = ExportStatus.FAILED
            export.error_message = message


class JobExecutor:
    def __init__(
        self,
        database: Database,
        *,
        lease_duration: timedelta = timedelta(minutes=5),
        heartbeat_interval_seconds: float = 30.0,
    ) -> None:
        if lease_duration <= timedelta(0):
            raise ValueError("lease duration must be positive")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        if heartbeat_interval_seconds >= lease_duration.total_seconds():
            raise ValueError("heartbeat interval must be shorter than the lease")
        self.database = database
        self.lease_duration = lease_duration
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._handlers: dict[JobType, JobHandler] = {}
        self.logger = structlog.get_logger("clearframe.jobs")

    def register(self, job_type: JobType, handler: JobHandler) -> None:
        if job_type in self._handlers:
            raise ValueError(f"a handler is already registered for {job_type}")
        self._handlers[job_type] = handler

    def execute(self, job_id: str) -> JobExecutionResult:
        job_type = self._claim(job_id)
        if job_type is None:
            return JobExecutionResult.NOT_CLAIMED

        handler = self._handlers.get(job_type)
        if handler is None:
            self.logger.error(
                "job_handler_missing",
                job_id=job_id,
                job_type=job_type,
            )
            self._mark_failed(job_id)
            return JobExecutionResult.FAILED

        heartbeat_stop = Event()
        heartbeat = Thread(
            target=self._heartbeat_loop,
            args=(job_id, heartbeat_stop),
            name=f"clearframe-heartbeat-{job_id[:8]}",
            daemon=True,
        )
        heartbeat.start()
        handler_failed = False
        try:
            handler(
                JobContext(self.database, job_id, self.lease_duration),
                job_id,
            )
        except Exception as exc:
            handler_failed = True
            self.logger.exception(
                "job_execution_failed",
                job_id=job_id,
                job_type=job_type,
                error_type=type(exc).__name__,
            )
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=self.heartbeat_interval_seconds + 5.0)
            if heartbeat.is_alive():
                self.logger.warning("job_heartbeat_stop_timed_out", job_id=job_id)

        if handler_failed:
            self._mark_failed(job_id)
            return JobExecutionResult.FAILED
        if not self._mark_completed(job_id):
            self.logger.error("job_completion_rejected", job_id=job_id)
            return JobExecutionResult.FAILED
        return JobExecutionResult.COMPLETED

    def _claim(self, job_id: str) -> JobType | None:
        now = utc_now()
        with self.database.session() as session:
            claim = cast(
                CursorResult[object],
                session.execute(
                    update(ProcessingJob)
                    .where(
                        ProcessingJob.id == job_id,
                        ProcessingJob.status == JobStatus.QUEUED,
                    )
                    .values(
                        status=JobStatus.RUNNING,
                        stage="starting",
                        attempts=ProcessingJob.attempts + 1,
                        started_at=now,
                        heartbeat_at=now,
                        lease_expires_at=now + self.lease_duration,
                        completed_at=None,
                        error_message=None,
                    )
                    .execution_options(synchronize_session=False)
                ),
            )
            if claim.rowcount != 1:
                session.rollback()
                return None
            job_type = session.scalar(
                select(ProcessingJob.job_type).where(ProcessingJob.id == job_id)
            )
            session.commit()
        return JobType(job_type) if job_type is not None else None

    def _mark_completed(self, job_id: str) -> bool:
        with self.database.session() as session:
            completion = cast(
                CursorResult[object],
                session.execute(
                    update(ProcessingJob)
                    .where(
                        ProcessingJob.id == job_id,
                        ProcessingJob.status == JobStatus.RUNNING,
                    )
                    .values(
                        status=JobStatus.COMPLETED,
                        progress=1.0,
                        stage="complete",
                        completed_at=utc_now(),
                        lease_expires_at=None,
                    )
                    .execution_options(synchronize_session=False)
                ),
            )
            session.commit()
            return completion.rowcount == 1

    def _mark_failed(self, job_id: str) -> None:
        with self.database.session() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                return
            _fail_job(
                session,
                job,
                stage="failed",
                message="Processing failed. Review the input and retry.",
            )
            session.commit()

    def _heartbeat_loop(self, job_id: str, stop: Event) -> None:
        while not stop.wait(self.heartbeat_interval_seconds):
            try:
                if not self._renew_lease(job_id):
                    return
            except Exception as exc:
                self.logger.warning(
                    "job_heartbeat_failed",
                    job_id=job_id,
                    error_type=type(exc).__name__,
                )

    def _renew_lease(self, job_id: str) -> bool:
        now = utc_now()
        with self.database.session() as session:
            renewal = cast(
                CursorResult[object],
                session.execute(
                    update(ProcessingJob)
                    .where(
                        ProcessingJob.id == job_id,
                        ProcessingJob.status == JobStatus.RUNNING,
                    )
                    .values(
                        heartbeat_at=now,
                        lease_expires_at=now + self.lease_duration,
                    )
                    .execution_options(synchronize_session=False)
                ),
            )
            session.commit()
            return renewal.rowcount == 1

    def recover_interrupted_jobs(self) -> None:
        with self.database.session() as session:
            interrupted = list(
                session.scalars(
                    select(ProcessingJob).where(
                        ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
                    )
                )
            )
            for job in interrupted:
                _fail_job(
                    session,
                    job,
                    stage="interrupted",
                    message="Processing was interrupted by an application restart.",
                )
            session.commit()


@dataclass(frozen=True, slots=True)
class JobReconcileSummary:
    queued_found: int = 0
    redispatched: int = 0
    dispatch_failed: int = 0
    expired_running: int = 0


class JobReconciler:
    def __init__(
        self,
        database: Database,
        dispatcher: JobDispatcher,
        *,
        queued_age: timedelta = timedelta(minutes=2),
    ) -> None:
        if queued_age < timedelta(0):
            raise ValueError("queued age cannot be negative")
        self.database = database
        self.dispatcher = dispatcher
        self.queued_age = queued_age
        self.logger = structlog.get_logger("clearframe.job_reconciler")

    def reconcile(self, *, now: datetime | None = None) -> JobReconcileSummary:
        checked_at = now or utc_now()
        expired_running = self._expire_running(checked_at)
        queued_ids = self._aged_queued_ids(checked_at - self.queued_age)
        redispatched = 0
        dispatch_failed = 0
        for job_id in queued_ids:
            try:
                self.dispatcher.enqueue(job_id)
            except Exception as exc:
                dispatch_failed += 1
                self.logger.warning(
                    "queued_job_redispatch_failed",
                    job_id=job_id,
                    error_type=type(exc).__name__,
                )
            else:
                redispatched += 1

        summary = JobReconcileSummary(
            queued_found=len(queued_ids),
            redispatched=redispatched,
            dispatch_failed=dispatch_failed,
            expired_running=expired_running,
        )
        self.logger.info(
            "job_reconcile_complete",
            queued_found=summary.queued_found,
            redispatched=summary.redispatched,
            dispatch_failed=summary.dispatch_failed,
            expired_running=summary.expired_running,
        )
        return summary

    def _aged_queued_ids(self, cutoff: datetime) -> list[str]:
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(ProcessingJob.id)
                    .where(
                        ProcessingJob.status == JobStatus.QUEUED,
                        ProcessingJob.created_at <= cutoff,
                    )
                    .order_by(ProcessingJob.created_at)
                )
            )

    def _expire_running(self, now: datetime) -> int:
        with self.database.session() as session:
            expired = list(
                session.scalars(
                    select(ProcessingJob)
                    .where(
                        ProcessingJob.status == JobStatus.RUNNING,
                        ProcessingJob.lease_expires_at.is_not(None),
                        ProcessingJob.lease_expires_at <= now,
                    )
                    .with_for_update(skip_locked=True)
                )
            )
            for job in expired:
                _fail_job(
                    session,
                    job,
                    stage="worker lease expired",
                    message=(
                        "Processing worker stopped responding. "
                        "Review the input and retry."
                    ),
                )
            session.commit()
            return len(expired)


class LocalJobRunner:
    def __init__(
        self,
        database: Database,
        max_workers: int = 2,
        *,
        job_executor: JobExecutor | None = None,
    ) -> None:
        self.database = database
        self.job_executor = job_executor or JobExecutor(database)
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="clearframe",
        )
        self._futures: dict[str, Future[JobExecutionResult]] = {}
        self._lock = Lock()

    def register(self, job_type: JobType, handler: JobHandler) -> None:
        self.job_executor.register(job_type, handler)

    def recover_interrupted_jobs(self) -> None:
        self.job_executor.recover_interrupted_jobs()

    def enqueue(self, job_id: str) -> None:
        future = self._pool.submit(self.job_executor.execute, job_id)
        with self._lock:
            self._futures[job_id] = future
        future.add_done_callback(lambda _: self._discard(job_id))

    def _discard(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)

    def wait(self, job_id: str, timeout: float = 30.0) -> None:
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=False)
