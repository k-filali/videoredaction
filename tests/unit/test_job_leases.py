from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import sleep

from clearframe.database import Database
from clearframe.domain.enums import JobStatus, JobType, VideoStatus
from clearframe.jobs import (
    JobContext,
    JobExecutionResult,
    JobExecutor,
    JobReconciler,
)
from clearframe.models import ProcessingJob, VideoAsset


class RecordingDispatcher:
    def __init__(self, failing_job_id: str | None = None) -> None:
        self.failing_job_id = failing_job_id
        self.job_ids: list[str] = []

    def enqueue(self, job_id: str) -> None:
        self.job_ids.append(job_id)
        if job_id == self.failing_job_id:
            raise RuntimeError("dispatch unavailable")


def create_database(tmp_path: Path) -> Database:
    database = Database(f"sqlite:///{(tmp_path / 'leases.db').as_posix()}")
    database.create_schema()
    return database


def create_job(
    database: Database,
    *,
    job_type: JobType,
    status: JobStatus,
    created_at: datetime,
    heartbeat_at: datetime | None = None,
    lease_expires_at: datetime | None = None,
) -> tuple[str, str]:
    with database.session() as session:
        video = VideoAsset(
            original_filename=f"{job_type}.mp4",
            safe_filename=f"{job_type}.mp4",
            content_type="video/mp4",
            status=(
                VideoStatus.PROCESSING
                if status == JobStatus.RUNNING
                else VideoStatus.PROXYING
            ),
        )
        session.add(video)
        session.flush()
        job = ProcessingJob(
            video_id=video.id,
            job_type=job_type,
            status=status,
            created_at=created_at,
            started_at=created_at if status == JobStatus.RUNNING else None,
            heartbeat_at=heartbeat_at,
            lease_expires_at=lease_expires_at,
        )
        session.add(job)
        session.commit()
        return video.id, job.id


def test_executor_renews_long_running_job_lease_and_clears_it(
    tmp_path: Path,
) -> None:
    database = create_database(tmp_path)
    _, job_id = create_job(
        database,
        job_type=JobType.DETECT,
        status=JobStatus.QUEUED,
        created_at=datetime.now(UTC),
    )
    executor = JobExecutor(
        database,
        lease_duration=timedelta(milliseconds=250),
        heartbeat_interval_seconds=0.02,
    )
    heartbeats: list[datetime] = []

    def slow_handler(_: JobContext, received_job_id: str) -> None:
        with database.session() as session:
            claimed = session.get(ProcessingJob, received_job_id)
            assert claimed is not None
            assert claimed.heartbeat_at is not None
            assert claimed.lease_expires_at is not None
            heartbeats.append(claimed.heartbeat_at)
        sleep(0.09)
        with database.session() as session:
            renewed = session.get(ProcessingJob, received_job_id)
            assert renewed is not None
            assert renewed.heartbeat_at is not None
            assert renewed.lease_expires_at is not None
            heartbeats.append(renewed.heartbeat_at)

    executor.register(JobType.DETECT, slow_handler)

    assert executor.execute(job_id) == JobExecutionResult.COMPLETED
    assert heartbeats[1] > heartbeats[0]
    with database.session() as session:
        completed = session.get(ProcessingJob, job_id)
        assert completed is not None
        assert completed.status == JobStatus.COMPLETED
        assert completed.heartbeat_at is not None
        assert completed.lease_expires_at is None


def test_failed_execution_clears_the_worker_lease(tmp_path: Path) -> None:
    database = create_database(tmp_path)
    _, job_id = create_job(
        database,
        job_type=JobType.INGEST,
        status=JobStatus.QUEUED,
        created_at=datetime.now(UTC),
    )
    executor = JobExecutor(database)

    def fail(_: JobContext, __: str) -> None:
        raise RuntimeError("media failure")

    executor.register(JobType.INGEST, fail)

    assert executor.execute(job_id) == JobExecutionResult.FAILED
    with database.session() as session:
        failed = session.get(ProcessingJob, job_id)
        assert failed is not None
        assert failed.status == JobStatus.FAILED
        assert failed.heartbeat_at is not None
        assert failed.lease_expires_at is None


def test_reconciler_redispatches_only_aged_queued_and_expires_stale_running(
    tmp_path: Path,
) -> None:
    database = create_database(tmp_path)
    now = datetime.now(UTC)
    _, aged_queued_id = create_job(
        database,
        job_type=JobType.EXPORT,
        status=JobStatus.QUEUED,
        created_at=now - timedelta(minutes=10),
    )
    _, fresh_queued_id = create_job(
        database,
        job_type=JobType.DETECT,
        status=JobStatus.QUEUED,
        created_at=now - timedelta(seconds=30),
    )
    expired_video_id, expired_running_id = create_job(
        database,
        job_type=JobType.INGEST,
        status=JobStatus.RUNNING,
        created_at=now - timedelta(minutes=8),
        heartbeat_at=now - timedelta(minutes=6),
        lease_expires_at=now - timedelta(minutes=5),
    )
    _, live_running_id = create_job(
        database,
        job_type=JobType.DETECT,
        status=JobStatus.RUNNING,
        created_at=now - timedelta(minutes=8),
        heartbeat_at=now - timedelta(seconds=10),
        lease_expires_at=now + timedelta(minutes=4),
    )
    _, unleased_running_id = create_job(
        database,
        job_type=JobType.PROXY,
        status=JobStatus.RUNNING,
        created_at=now - timedelta(minutes=8),
    )
    dispatcher = RecordingDispatcher()

    summary = JobReconciler(
        database,
        dispatcher,
        queued_age=timedelta(minutes=2),
    ).reconcile(now=now)

    assert summary.queued_found == 1
    assert summary.redispatched == 1
    assert summary.dispatch_failed == 0
    assert summary.expired_running == 1
    assert dispatcher.job_ids == [aged_queued_id]
    assert fresh_queued_id not in dispatcher.job_ids
    assert expired_running_id not in dispatcher.job_ids

    with database.session() as session:
        aged_queued = session.get(ProcessingJob, aged_queued_id)
        expired_running = session.get(ProcessingJob, expired_running_id)
        live_running = session.get(ProcessingJob, live_running_id)
        unleased_running = session.get(ProcessingJob, unleased_running_id)
        expired_video = session.get(VideoAsset, expired_video_id)
        assert aged_queued is not None
        assert expired_running is not None
        assert live_running is not None
        assert unleased_running is not None
        assert expired_video is not None
        assert aged_queued.status == JobStatus.QUEUED
        assert expired_running.status == JobStatus.FAILED
        assert expired_running.stage == "worker lease expired"
        assert expired_running.lease_expires_at is None
        assert expired_video.status == VideoStatus.FAILED
        assert live_running.status == JobStatus.RUNNING
        assert unleased_running.status == JobStatus.RUNNING


def test_failed_redispatch_leaves_queued_job_available_for_next_sweep(
    tmp_path: Path,
) -> None:
    database = create_database(tmp_path)
    now = datetime.now(UTC)
    _, job_id = create_job(
        database,
        job_type=JobType.REPROCESS,
        status=JobStatus.QUEUED,
        created_at=now - timedelta(minutes=5),
    )
    dispatcher = RecordingDispatcher(failing_job_id=job_id)

    summary = JobReconciler(
        database,
        dispatcher,
        queued_age=timedelta(minutes=1),
    ).reconcile(now=now)

    assert summary.queued_found == 1
    assert summary.redispatched == 0
    assert summary.dispatch_failed == 1
    with database.session() as session:
        queued = session.get(ProcessingJob, job_id)
        assert queued is not None
        assert queued.status == JobStatus.QUEUED
