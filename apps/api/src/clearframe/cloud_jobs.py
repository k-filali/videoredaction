import re
from collections.abc import Callable
from importlib import import_module
from threading import Lock
from time import monotonic, sleep
from typing import Protocol, TypedDict, cast

import structlog

from clearframe.database import Database
from clearframe.domain.enums import JobStatus, JobType
from clearframe.models import ProcessingJob


class CloudRunJobError(RuntimeError):
    pass


class CloudRunJobNotFoundError(CloudRunJobError):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"processing job {job_id} was not found")


class CloudRunJobTypeError(CloudRunJobError):
    def __init__(self, job_id: str, job_type: str) -> None:
        self.job_id = job_id
        self.job_type = job_type
        super().__init__(f"processing job {job_id} has unsupported type {job_type}")


class CloudRunJobConfigurationError(CloudRunJobError):
    pass


class CloudRunClientUnavailableError(CloudRunJobError):
    pass


class CloudRunDispatchError(CloudRunJobError):
    def __init__(self, job_id: str, target_job: str) -> None:
        self.job_id = job_id
        self.target_job = target_job
        super().__init__(f"could not dispatch processing job {job_id}")


class EnvironmentVariable(TypedDict):
    name: str
    value: str


class ContainerOverride(TypedDict):
    env: list[EnvironmentVariable]


class ExecutionOverrides(TypedDict):
    container_overrides: list[ContainerOverride]


class RunJobRequest(TypedDict):
    name: str
    overrides: ExecutionOverrides


class CloudRunJobsClient(Protocol):
    def run_job(self, *, request: RunJobRequest) -> object: ...


CloudRunJobsClientFactory = Callable[[], CloudRunJobsClient]

_JOB_RESOURCE_PATTERN = re.compile(
    r"^projects/[^/]+/locations/[^/]+/jobs/[^/]+$"
)


def create_cloud_run_jobs_client() -> CloudRunJobsClient:
    try:
        run_v2 = import_module("google.cloud.run_v2")
        jobs_client = run_v2.JobsClient
    except (AttributeError, ModuleNotFoundError) as exc:
        raise CloudRunClientUnavailableError(
            "google-cloud-run is required for Cloud Run dispatch"
        ) from exc
    return cast(CloudRunJobsClient, jobs_client())


class CloudRunJobDispatcher:
    def __init__(
        self,
        database: Database,
        *,
        cpu_job: str,
        gpu_job: str,
        client: CloudRunJobsClient | None = None,
        client_factory: CloudRunJobsClientFactory | None = None,
    ) -> None:
        if client is not None and client_factory is not None:
            raise CloudRunJobConfigurationError(
                "provide either a Cloud Run client or client factory"
            )
        self._validate_target("cpu_job", cpu_job)
        self._validate_target("gpu_job", gpu_job)
        self.database = database
        self.cpu_job = cpu_job
        self.gpu_job = gpu_job
        self._client = client
        self._client_factory = client_factory or create_cloud_run_jobs_client
        self._client_lock = Lock()
        self.logger = structlog.get_logger("clearframe.cloud_jobs")

    def enqueue(self, job_id: str) -> None:
        job_type = self._job_type(job_id)
        target_job = self.gpu_job if job_type is JobType.DETECT else self.cpu_job
        request = RunJobRequest(
            name=target_job,
            overrides=ExecutionOverrides(
                container_overrides=[
                    ContainerOverride(
                        env=[
                            EnvironmentVariable(
                                name="CLEARFRAME_JOB_ID",
                                value=job_id,
                            )
                        ]
                    )
                ]
            ),
        )
        self.logger.info(
            "cloud_job_dispatching",
            job_id=job_id,
            job_type=job_type,
            target_job=target_job,
        )
        try:
            operation = self._get_client().run_job(request=request)
        except CloudRunClientUnavailableError:
            self.logger.error(
                "cloud_job_client_unavailable",
                job_id=job_id,
                job_type=job_type,
                target_job=target_job,
            )
            raise
        except Exception as exc:
            self.logger.error(
                "cloud_job_dispatch_failed",
                job_id=job_id,
                job_type=job_type,
                target_job=target_job,
                error_type=type(exc).__name__,
            )
            raise CloudRunDispatchError(job_id, target_job) from exc

        self.logger.info(
            "cloud_job_dispatched",
            job_id=job_id,
            job_type=job_type,
            target_job=target_job,
            operation_name=self._operation_name(operation),
        )

    def recover_interrupted_jobs(self) -> None:
        return

    def wait(self, job_id: str, timeout: float = 30.0) -> None:
        deadline = monotonic() + timeout
        terminal = {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
        while True:
            with self.database.session() as session:
                job = session.get(ProcessingJob, job_id)
                if job is None:
                    raise CloudRunJobNotFoundError(job_id)
                status = JobStatus(job.status)
            if status in terminal:
                return
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"processing job {job_id} did not finish")
            sleep(min(0.5, remaining))

    def shutdown(self) -> None:
        client = self._client
        close = getattr(client, "close", None)
        if callable(close):
            close()

    def _job_type(self, job_id: str) -> JobType:
        with self.database.session() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                raise CloudRunJobNotFoundError(job_id)
            stored_type = str(job.job_type)
        try:
            return JobType(stored_type)
        except ValueError as exc:
            raise CloudRunJobTypeError(job_id, stored_type) from exc

    def _get_client(self) -> CloudRunJobsClient:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                self._client = self._client_factory()
            return self._client

    @staticmethod
    def _validate_target(field: str, value: str) -> None:
        if not _JOB_RESOURCE_PATTERN.fullmatch(value):
            raise CloudRunJobConfigurationError(
                f"{field} must be a full Cloud Run job resource name"
            )

    @staticmethod
    def _operation_name(operation: object) -> str | None:
        raw_operation = getattr(operation, "operation", None)
        name = getattr(raw_operation, "name", None)
        if isinstance(name, str):
            return name
        direct_name = getattr(operation, "name", None)
        return direct_name if isinstance(direct_name, str) else None
