from dataclasses import dataclass

from clearframe.config import Settings
from clearframe.database import Database
from clearframe.jobs import LocalJobRunner
from clearframe.media import MediaProcessor
from clearframe.services.export import ExportService
from clearframe.services.ingest import IngestService
from clearframe.services.reprocessing import ReprocessingService
from clearframe.storage import LocalStorage


@dataclass(slots=True)
class ServiceContainer:
    database: Database
    storage: LocalStorage
    media: MediaProcessor
    runner: LocalJobRunner
    ingest: IngestService
    export: ExportService
    reprocess: ReprocessingService

    @classmethod
    def build(cls, settings: Settings, database: Database) -> "ServiceContainer":
        storage = LocalStorage(settings.storage_root)
        media = MediaProcessor(settings.ffmpeg_path)
        runner = LocalJobRunner(database)
        ingest = IngestService(
            database=database,
            storage=storage,
            media=media,
            runner=runner,
            max_upload_mb=settings.max_upload_mb,
        )
        export = ExportService(
            database=database,
            storage=storage,
            media=media,
            runner=runner,
        )
        reprocess = ReprocessingService(
            database=database,
            storage=storage,
            runner=runner,
            window_seconds=settings.reprocess_window_seconds,
        )
        return cls(
            database=database,
            storage=storage,
            media=media,
            runner=runner,
            ingest=ingest,
            export=export,
            reprocess=reprocess,
        )
