from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Lock

from sqlalchemy import select

from clearframe.database import Database
from clearframe.domain.enums import ExportStatus, JobStatus, VideoStatus
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


JobHandler = Callable[[JobContext], None]


class LocalJobRunner:
    def __init__(self, database: Database, max_workers: int = 2) -> None:
        self.database = database
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="clearframe",
        )
        self._futures: dict[str, Future[None]] = {}
        self._lock = Lock()

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
                    if video is not None:
                        video.status = VideoStatus.FAILED
                        video.error_message = job.error_message
                if job.export_id:
                    export = session.get(ExportArtifact, job.export_id)
                    if export is not None:
                        export.status = ExportStatus.FAILED
                        export.error_message = job.error_message
            session.commit()

    def submit(self, job_id: str, handler: JobHandler) -> None:
        future = self.executor.submit(self._run, job_id, handler)
        with self._lock:
            self._futures[job_id] = future
        future.add_done_callback(lambda _: self._discard(job_id))

    def _discard(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)

    def _run(self, job_id: str, handler: JobHandler) -> None:
        with self.database.session() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                return
            job.status = JobStatus.RUNNING
            job.stage = "starting"
            job.attempts += 1
            job.started_at = utc_now()
            session.commit()

        try:
            handler(JobContext(self.database, job_id))
        except Exception:
            self._mark_failed(job_id)
            return

        with self.database.session() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                return
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.stage = "complete"
            job.completed_at = utc_now()
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
                if video is not None:
                    video.status = VideoStatus.FAILED
                    video.error_message = job.error_message
            if job.export_id:
                export = session.get(ExportArtifact, job.export_id)
                if export is not None:
                    export.status = ExportStatus.FAILED
                    export.error_message = job.error_message
            session.commit()

    def wait(self, job_id: str, timeout: float = 30.0) -> None:
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
