from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLEARFRAME_",
        extra="ignore",
    )

    env: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    web_origin: str = "http://localhost:5173"
    database_url: str = "sqlite:///./data/clearframe.db"
    storage_root: Path = Path("./storage")
    max_upload_mb: int = Field(default=2048, gt=0)
    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None
    detector: str = "mock"
    enable_face_detector: bool = False
    enable_text_detector: bool = False
    reprocess_window_seconds: int = Field(default=3, ge=1, le=30)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()

