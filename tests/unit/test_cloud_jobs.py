from pathlib import Path
from types import SimpleNamespace

import pytest

from clearframe.cloud_jobs import (
    CloudRunDispatchError,
    CloudRunJobConfigurationError,
    CloudRunJobDispatcher,
    CloudRunJobNotFoundError,
    CloudRunJobsClient,
    RunJobRequest,
)
from clearframe.config import Settings
from clearframe.database import Database
from clearframe.domain.enums import JobType, VideoStatus
from clearframe.models import ProcessingJob, VideoAsset
from clearframe.services.container import ServiceContainer

CPU_JOB = "projects/clearframe/locations/northamerica-northeast1/jobs/worker-cpu"
GPU_JOB = "projects/clearframe/locations/northamerica-northeast1/jobs/worker-gpu"


class RecordingClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[RunJobRequest] = []
        self.closed = False

    def run_job(self, *, request: RunJobRequest) -> object:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            operation=SimpleNamespace(name="operations/dispatch-1")
        )

    def close(self) -> None:
        self.closed = True


def create_database(tmp_path: Path) -> Database:
    database = Database(f"sqlite:///{(tmp_path / 'cloud-jobs.db').as_posix()}")
    database.create_schema()
    return database


def create_job(database: Database, job_type: JobType) -> str:
    with database.session() as session:
        video = VideoAsset(
            original_filename="dispatch.mp4",
            safe_filename="dispatch.mp4",
            content_type="video/mp4",
            status=VideoStatus.READY_FOR_REVIEW,
        )
        session.add(video)
        session.flush()
        job = ProcessingJob(video_id=video.id, job_type=job_type)
        session.add(job)
        session.commit()
        return job.id


def build_dispatcher(
    database: Database,
    client: CloudRunJobsClient,
) -> CloudRunJobDispatcher:
    return CloudRunJobDispatcher(
        database,
        cpu_job=CPU_JOB,
        gpu_job=GPU_JOB,
        client=client,
    )


def test_detection_jobs_dispatch_to_gpu_with_job_id_override(
    tmp_path: Path,
) -> None:
    database = create_database(tmp_path)
    job_id = create_job(database, JobType.DETECT)
    client = RecordingClient()

    build_dispatcher(database, client).enqueue(job_id)

    assert client.requests == [
        {
            "name": GPU_JOB,
            "overrides": {
                "container_overrides": [
                    {
                        "env": [
                            {
                                "name": "CLEARFRAME_JOB_ID",
                                "value": job_id,
                            }
                        ]
                    }
                ]
            },
        }
    ]


@pytest.mark.parametrize(
    "job_type",
    [
        JobType.INGEST,
        JobType.PROXY,
        JobType.REPROCESS,
        JobType.EXPORT,
    ],
)
def test_non_detection_jobs_dispatch_to_cpu(
    tmp_path: Path,
    job_type: JobType,
) -> None:
    database = create_database(tmp_path)
    job_id = create_job(database, job_type)
    client = RecordingClient()

    build_dispatcher(database, client).enqueue(job_id)

    assert client.requests[0]["name"] == CPU_JOB
    assert (
        client.requests[0]["overrides"]["container_overrides"][0]["env"][0][
            "value"
        ]
        == job_id
    )


def test_client_factory_is_lazy_and_reused(tmp_path: Path) -> None:
    database = create_database(tmp_path)
    first_job_id = create_job(database, JobType.INGEST)
    second_job_id = create_job(database, JobType.DETECT)
    client = RecordingClient()
    factory_calls = 0

    def factory() -> CloudRunJobsClient:
        nonlocal factory_calls
        factory_calls += 1
        return client

    dispatcher = CloudRunJobDispatcher(
        database,
        cpu_job=CPU_JOB,
        gpu_job=GPU_JOB,
        client_factory=factory,
    )
    assert factory_calls == 0

    dispatcher.enqueue(first_job_id)
    dispatcher.enqueue(second_job_id)

    assert factory_calls == 1
    assert [request["name"] for request in client.requests] == [
        CPU_JOB,
        GPU_JOB,
    ]


def test_missing_job_is_rejected_before_cloud_call(tmp_path: Path) -> None:
    database = create_database(tmp_path)
    client = RecordingClient()

    with pytest.raises(CloudRunJobNotFoundError, match="missing"):
        build_dispatcher(database, client).enqueue("missing")

    assert client.requests == []


def test_cloud_api_errors_have_typed_context(tmp_path: Path) -> None:
    database = create_database(tmp_path)
    job_id = create_job(database, JobType.EXPORT)
    client = RecordingClient(RuntimeError("upstream unavailable"))

    with pytest.raises(CloudRunDispatchError) as caught:
        build_dispatcher(database, client).enqueue(job_id)

    assert caught.value.job_id == job_id
    assert caught.value.target_job == CPU_JOB
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_wait_observes_terminal_database_state(tmp_path: Path) -> None:
    database = create_database(tmp_path)
    job_id = create_job(database, JobType.INGEST)
    client = RecordingClient()
    dispatcher = build_dispatcher(database, client)
    with database.session() as session:
        job = session.get(ProcessingJob, job_id)
        assert job is not None
        job.status = "COMPLETED"
        session.commit()

    dispatcher.wait(job_id, timeout=0)
    dispatcher.shutdown()

    assert client.closed


@pytest.mark.parametrize(
    "target",
    [
        "",
        "worker-cpu",
        "projects/p/locations/r",
        "projects/p/locations/r/jobs/name/extra",
    ],
)
def test_job_targets_must_be_full_resource_names(
    tmp_path: Path,
    target: str,
) -> None:
    database = create_database(tmp_path)

    with pytest.raises(CloudRunJobConfigurationError):
        CloudRunJobDispatcher(
            database,
            cpu_job=target,
            gpu_job=GPU_JOB,
            client=RecordingClient(),
        )


def test_service_container_selects_cloud_dispatcher(tmp_path: Path) -> None:
    database = create_database(tmp_path)
    settings = Settings(
        env="test",
        database_url=str(database.engine.url),
        storage_root=tmp_path / "storage",
        job_backend="cloud_run",
        gcp_project_id="clearframe-test1",
        gcp_region="us-central1",
    )

    services = ServiceContainer.build(settings, database)
    try:
        assert isinstance(services.runner, CloudRunJobDispatcher)
        assert services.runner.cpu_job == (
            "projects/clearframe-test1/locations/us-central1/"
            "jobs/clearframe-cpu-worker"
        )
    finally:
        services.runner.shutdown()
