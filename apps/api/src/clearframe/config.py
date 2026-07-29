from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    storage_backend: Literal["local", "gcs"] = "local"
    storage_root: Path = Path("./storage")
    storage_scratch_root: Path = Path("./storage/tmp/cloud")
    gcs_bucket: str | None = None
    gcs_signing_service_account: str | None = None
    artifact_url_ttl_minutes: int = Field(default=360, ge=5, le=10080)
    max_upload_mb: int = Field(default=2048, gt=0)
    max_api_body_kb: int = Field(default=1024, ge=16, le=10240)
    max_duration_minutes: int = Field(default=240, ge=1, le=1440)
    max_video_pixels: int = Field(default=9_000_000, ge=307_200)
    max_video_dimension: int = Field(default=8192, ge=640)
    max_video_fps: float = Field(default=120.0, ge=1.0, le=240.0)
    ffmpeg_path: Path | None = None
    model_registry_path: Path = Path("configs/models/registry.yaml")
    reprocess_window_seconds: int = Field(default=3, ge=1, le=30)
    log_level: str = "INFO"
    build_id: str = Field(default="development", min_length=1, max_length=128)

    @field_validator(
        "access_token",
        "gcs_bucket",
        "gcs_signing_service_account",
        mode="before",
    )
    @classmethod
    def empty_optional_value_is_unset(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def prevent_unprotected_remote_binding(self) -> "Settings":
        loopback_hosts = {"127.0.0.1", "::1", "localhost"}
        if self.api_host not in loopback_hosts and self.access_token is None:
            raise ValueError("an access token is required when binding beyond localhost")
        if self.storage_backend == "gcs" and not self.gcs_bucket:
            raise ValueError("a GCS bucket is required for the GCS storage backend")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
