from pathlib import Path

from clearframe.database import Database
from clearframe.domain.enums import JobStatus, JobType, VideoStatus
from clearframe.jobs import LocalJobRunner
from clearframe.models import ProcessingJob, VideoAsset


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

