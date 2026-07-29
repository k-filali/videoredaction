from dataclasses import dataclass
from datetime import timedelta

from clearframe.cloud_jobs import CloudRunJobDispatcher
from clearframe.config import Settings
from clearframe.database import Database
from clearframe.domain.enums import JobType
from clearframe.gcs_storage import GCSStorage
from clearframe.jobs import JobExecutor, LocalJobRunner, ManagedJobDispatcher
from clearframe.media import MediaProcessor
from clearframe.services.export import ExportService
from clearframe.services.ingest import IngestService
from clearframe.services.processing import ProcessingService
from clearframe.services.proxy import ProxyService
from clearframe.services.reprocessing import ReprocessingService
from clearframe.storage import ArtifactStorage, LocalStorage


@dataclass(slots=True)
class ServiceContainer:
    database: Database
    storage: ArtifactStorage
    media: MediaProcessor
    executor: JobExecutor
    runner: ManagedJobDispatcher
    ingest: IngestService
    proxy: ProxyService
    processing: ProcessingService
    export: ExportService
    reprocess: ReprocessingService

    @classmethod
    def build(cls, settings: Settings, database: Database) -> "ServiceContainer":
        storage: ArtifactStorage
        if settings.storage_backend == "gcs":
            if settings.gcs_bucket is None:
                raise ValueError("a GCS bucket is required")
            storage = GCSStorage(
                settings.gcs_bucket,
                settings.storage_scratch_root,
                signed_url_ttl=timedelta(
                    minutes=settings.artifact_url_ttl_minutes
                ),
                signing_service_account=settings.gcs_signing_service_account,
            )
        else:
            storage = LocalStorage(settings.storage_root)
        media = MediaProcessor(
            settings.ffmpeg_path,
            max_duration_ms=settings.max_duration_minutes * 60 * 1000,
            max_video_pixels=settings.max_video_pixels,
            max_video_dimension=settings.max_video_dimension,
            max_video_fps=settings.max_video_fps,
        )
        executor = JobExecutor(database)
        runner: ManagedJobDispatcher
        if settings.job_backend == "cloud_run":
            if settings.gcp_project_id is None:
                raise ValueError("a Google Cloud project is required")
            resource_prefix = (
                f"projects/{settings.gcp_project_id}/locations/{settings.gcp_region}"
            )
            runner = CloudRunJobDispatcher(
                database,
                cpu_job=f"{resource_prefix}/jobs/{settings.cloud_run_cpu_job}",
                gpu_job=f"{resource_prefix}/jobs/{settings.cloud_run_gpu_job}",
            )
        else:
            runner = LocalJobRunner(database, job_executor=executor)
        ingest = IngestService(
            database=database,
            storage=storage,
            media=media,
            runner=runner,
            max_upload_mb=settings.max_upload_mb,
        )
        proxy = ProxyService(
            database=database,
            storage=storage,
            media=media,
            runner=runner,
        )
        processing = ProcessingService(
            database=database,
            storage=storage,
            runner=runner,
            registry_path=settings.model_registry_path,
        )
        export = ExportService(
            database=database,
            storage=storage,
            media=media,
            runner=runner,
            build_id=settings.build_id,
        )
        reprocess = ReprocessingService(
            database=database,
            storage=storage,
            runner=runner,
            window_seconds=settings.reprocess_window_seconds,
        )
        executor.register(JobType.INGEST, ingest.execute)
        executor.register(JobType.PROXY, proxy.execute)
        executor.register(JobType.DETECT, processing.execute)
        executor.register(JobType.REPROCESS, reprocess.execute)
        executor.register(JobType.EXPORT, export.execute)
        return cls(
            database=database,
            storage=storage,
            media=media,
            executor=executor,
            runner=runner,
            ingest=ingest,
            proxy=proxy,
            processing=processing,
            export=export,
            reprocess=reprocess,
        )
