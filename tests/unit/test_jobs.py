from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from clearframe.database import Database
from clearframe.domain.enums import JobStatus, JobType, VideoStatus
from clearframe.jobs import (
    JobContext,
    JobExecutionResult,
    JobExecutor,
    LocalJobRunner,
)
from clearframe.models import ProcessingJob, VideoAsset


def create_queued_job(
    database: Database,
    job_type: JobType,
) -> tuple[str, str]:
    with database.session() as session:
        video = VideoAsset(
            original_filename="job.mp4",
            safe_filename="job.mp4",
            content_type="video/mp4",
            status=VideoStatus.PROXYING,
        )
        session.add(video)
        session.flush()
        job = ProcessingJob(
            video_id=video.id,
            job_type=job_type,
            status=JobStatus.QUEUED,
        )
        session.add(job)
        session.commit()
        return video.id, job.id


def test_executor_claims_job_once(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'claim.db').as_posix()}")
    database.create_schema()
    _, job_id = create_queued_job(database, JobType.INGEST)
    executor = JobExecutor(database)
    claimed = Event()
    release = Event()
    handled: list[str] = []

    def handle(context: JobContext, received_job_id: str) -> None:
        handled.append(received_job_id)
        context.update(0.5, "working")
        claimed.set()
        assert release.wait(timeout=5)

    executor.register(JobType.INGEST, handle)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.execute, job_id)
        try:
            assert claimed.wait(timeout=5)
            second = pool.submit(executor.execute, job_id)
            assert second.result(timeout=5) == JobExecutionResult.NOT_CLAIMED
        finally:
            release.set()
        assert first.result(timeout=5) == JobExecutionResult.COMPLETED

    assert executor.execute(job_id) == JobExecutionResult.NOT_CLAIMED
    assert handled == [job_id]
    with database.session() as session:
        job = session.get(ProcessingJob, job_id)
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert job.attempts == 1
        assert job.progress == 1.0
        assert job.started_at is not None
        assert job.completed_at is not None


def test_executor_fails_handler_errors_and_missing_handlers(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'failure.db').as_posix()}")
    database.create_schema()
    error_video_id, error_job_id = create_queued_job(database, JobType.INGEST)
    missing_video_id, missing_job_id = create_queued_job(database, JobType.DETECT)
    executor = JobExecutor(database)

    def fail(_: JobContext, __: str) -> None:
        raise RuntimeError("worker failure")

    executor.register(JobType.INGEST, fail)

    assert executor.execute(error_job_id) == JobExecutionResult.FAILED
    assert executor.execute(missing_job_id) == JobExecutionResult.FAILED

    with database.session() as session:
        for job_id, video_id in (
            (error_job_id, error_video_id),
            (missing_job_id, missing_video_id),
        ):
            job = session.get(ProcessingJob, job_id)
            video = session.get(VideoAsset, video_id)
            assert job is not None
            assert video is not None
            assert job.status == JobStatus.FAILED
            assert job.stage == "failed"
            assert job.attempts == 1
            assert job.completed_at is not None
            assert video.status == VideoStatus.FAILED


def test_startup_recovery_marks_orphaned_jobs_and_video_failed(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'jobs.db').as_posix()}")
    database.create_schema()
    with database.session() as session:
        video = VideoAsset(
            original_filename="interrupted.mp4",
            safe_filename="interrupted.mp4",
            content_type="video/mp4",
            status=VideoStatus.PROXYING,
        )
        session.add(video)
        session.flush()
        queued = ProcessingJob(
            video_id=video.id,
            job_type=JobType.INGEST,
            status=JobStatus.QUEUED,
        )
        session.add(queued)
        session.commit()
        video_id = video.id
        job_id = queued.id

    runner = LocalJobRunner(database)
    runner.recover_interrupted_jobs()

    with database.session() as session:
        recovered_job = session.get(ProcessingJob, job_id)
        recovered_video = session.get(VideoAsset, video_id)
        assert recovered_job is not None
        assert recovered_video is not None
        assert recovered_job.status == JobStatus.FAILED
        assert recovered_job.stage == "interrupted"
        assert recovered_video.status == VideoStatus.FAILED

    runner.shutdown()


def test_actionable_failures_reach_the_reviewer_verbatim(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'reviewer-message.db').as_posix()}")
    database.create_schema()
    actionable_video_id, actionable_job_id = create_queued_job(database, JobType.INGEST)
    opaque_video_id, opaque_job_id = create_queued_job(database, JobType.DETECT)
    executor = JobExecutor(database)

    class _DuplicateLike(RuntimeError):
        reviewer_message = "This video has already been uploaded. Open the existing copy instead."

    def duplicate(_: JobContext, __: str) -> None:
        raise _DuplicateLike("internal detail the reviewer must not see")

    def leaky(_: JobContext, __: str) -> None:
        raise RuntimeError("psycopg.OperationalError: connection refused at 10.0.0.4")

    executor.register(JobType.INGEST, duplicate)
    executor.register(JobType.DETECT, leaky)

    assert executor.execute(actionable_job_id) == JobExecutionResult.FAILED
    assert executor.execute(opaque_job_id) == JobExecutionResult.FAILED

    with database.session() as session:
        actionable_job = session.get(ProcessingJob, actionable_job_id)
        actionable_video = session.get(VideoAsset, actionable_video_id)
        opaque_job = session.get(ProcessingJob, opaque_job_id)
        opaque_video = session.get(VideoAsset, opaque_video_id)
        assert actionable_job is not None
        assert actionable_video is not None
        assert opaque_job is not None
        assert opaque_video is not None

        assert actionable_job.error_message == _DuplicateLike.reviewer_message
        assert actionable_video.error_message == _DuplicateLike.reviewer_message

        # Unrecognised failures must not leak internals to the workspace.
        assert opaque_job.error_message == "Processing failed. Review the input and retry."
        assert "psycopg" not in (opaque_video.error_message or "")
