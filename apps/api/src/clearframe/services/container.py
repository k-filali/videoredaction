from dataclasses import dataclass

from clearframe.config import Settings
from clearframe.database import Database
from clearframe.jobs import LocalJobRunner
from clearframe.media import MediaProcessor
from clearframe.services.ingest import IngestService
from clearframe.storage import LocalStorage


@dataclass(slots=True)
class ServiceContainer:
    storage: LocalStorage
    media: MediaProcessor
    runner: LocalJobRunner
    ingest: IngestService

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
        return cls(storage=storage, media=media, runner=runner, ingest=ingest)

