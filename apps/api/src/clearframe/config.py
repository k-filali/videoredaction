from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
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
    access_token: SecretStr | None = None
    database_url: str = "sqlite:///./data/clearframe.db"
    storage_root: Path = Path("./storage")
    max_upload_mb: int = Field(default=2048, gt=0)
    max_duration_minutes: int = Field(default=240, ge=1, le=1440)
    max_video_pixels: int = Field(default=9_000_000, ge=307_200)
    max_video_dimension: int = Field(default=8192, ge=640)
    max_video_fps: float = Field(default=120.0, ge=1.0, le=240.0)
    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None
    model_registry_path: Path = Path("configs/models/registry.yaml")
    detector: str = "mock"
    enable_face_detector: bool = False
    enable_text_detector: bool = False
    reprocess_window_seconds: int = Field(default=3, ge=1, le=30)
    log_level: str = "INFO"

    @field_validator("access_token", mode="before")
    @classmethod
    def empty_access_token_is_unset(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def prevent_unprotected_remote_binding(self) -> "Settings":
        loopback_hosts = {"127.0.0.1", "::1", "localhost"}
        if self.api_host not in loopback_hosts and self.access_token is None:
            raise ValueError("an access token is required when binding beyond localhost")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
