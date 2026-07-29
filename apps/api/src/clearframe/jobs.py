from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock
from typing import Protocol, cast

import structlog
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult

from clearframe.database import Database
from clearframe.domain.enums import ExportStatus, JobStatus, JobType, VideoStatus
from clearframe.models import ExportArtifact, ProcessingJob, VideoAsset


def utc_now() -> datetime:
    return datetime.now(UTC)


class JobContext:
    def __init__(self, database: Database, job_id: str) -> None:
        self.database = database
        self.job_id = job_id

    def update(self, progress: float, stage: str) -> None:
        with self.database.session() as session:
            job = session.get(ProcessingJob, self.job_id)
            if job is None:
                raise RuntimeError("processing job no longer exists")
            job.progress = min(1.0, max(0.0, progress))
            job.stage = stage[:80]
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


class JobExecutor:
    def __init__(self, database: Database) -> None:
        self.database = database
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

        try:
            handler(JobContext(self.database, job_id), job_id)
        except Exception as exc:
            self.logger.exception(
                "job_execution_failed",
                job_id=job_id,
                job_type=job_type,
                error_type=type(exc).__name__,
            )
            self._mark_failed(job_id)
            return JobExecutionResult.FAILED

        self._mark_completed(job_id)
        return JobExecutionResult.COMPLETED

    def _claim(self, job_id: str) -> JobType | None:
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
                        started_at=utc_now(),
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

    def _mark_completed(self, job_id: str) -> None:
        with self.database.session() as session:
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
                )
                .execution_options(synchronize_session=False)
            )
            session.commit()

    def _mark_failed(self, job_id: str) -> None:
        with self.database.session() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED
            job.stage = "failed"
            job.error_message = "Processing failed. Review the input and retry."
            job.completed_at = utc_now()
            if job.video_id:
                video = session.get(VideoAsset, job.video_id)
                if video is not None and job.job_type != JobType.PROXY:
                    video.status = (
                        VideoStatus.READY_FOR_REVIEW
                        if job.job_type in {JobType.EXPORT, JobType.REPROCESS}
                        else VideoStatus.FAILED
                    )
                    video.error_message = job.error_message
            if job.export_id:
                export = session.get(ExportArtifact, job.export_id)
                if export is not None:
                    export.status = ExportStatus.FAILED
                    export.error_message = job.error_message
            session.commit()

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
                job.status = JobStatus.FAILED
                job.stage = "interrupted"
                job.error_message = "Processing was interrupted by an application restart."
                job.completed_at = utc_now()
                if job.video_id:
                    video = session.get(VideoAsset, job.video_id)
                    if video is not None and job.job_type != JobType.PROXY:
                        video.status = (
                            VideoStatus.READY_FOR_REVIEW
                            if job.job_type in {JobType.EXPORT, JobType.REPROCESS}
                            else VideoStatus.FAILED
                        )
                        video.error_message = job.error_message
                if job.export_id:
                    export = session.get(ExportArtifact, job.export_id)
                    if export is not None:
                        export.status = ExportStatus.FAILED
                        export.error_message = job.error_message
            session.commit()


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
