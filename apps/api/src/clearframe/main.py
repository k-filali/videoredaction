import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.responses import JSONResponse

from clearframe import __version__
from clearframe.api.exports import router as exports_router
from clearframe.api.processing import router as processing_router
from clearframe.api.review import router as review_router
from clearframe.api.videos import router as videos_router
from clearframe.config import Settings, get_settings
from clearframe.database import Database
from clearframe.middleware import (
    AccessTokenMiddleware,
    RequestBodyLimitMiddleware,
    UploadLimitMiddleware,
)
from clearframe.observability import RequestTraceMiddleware, configure_observability
from clearframe.services.container import ServiceContainer

SCHEMA_HEAD = "c7b28f914d62"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    app.state.database.create_schema()
    app.state.services.runner.recover_interrupted_jobs()
    app.state.services.proxy.reconcile_all()
    try:
        yield
    finally:
        app.state.services.runner.shutdown()


def create_app(
    settings: Settings | None = None,
    database: Database | None = None,
    services: ServiceContainer | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_observability(resolved_settings.log_level)
    resolved_database = database or Database(resolved_settings.database_url)
    resolved_services = services or ServiceContainer.build(
        resolved_settings,
        resolved_database,
    )
    app = FastAPI(
        title="ClearFrame API",
        version=__version__,
        description="Local-first, human-reviewed video redaction research prototype.",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.database = resolved_database
    app.state.services = resolved_services
    app.state.processing_service = resolved_services.processing
    app.add_middleware(
        UploadLimitMiddleware,
        max_upload_bytes=resolved_settings.max_upload_mb * 1024 * 1024,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_request_bytes=resolved_settings.max_api_body_kb * 1024,
    )
    app.add_middleware(
        AccessTokenMiddleware,
        access_token=(
            resolved_settings.access_token.get_secret_value()
            if resolved_settings.access_token is not None
            else None
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[resolved_settings.web_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestTraceMiddleware)
    app.include_router(videos_router)
    app.include_router(review_router)
    app.include_router(exports_router)
    app.include_router(processing_router)

    @app.get("/api/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/ready", tags=["system"])
    async def ready() -> JSONResponse:
        revision: str | None = None
        database_ready = False
        try:
            with resolved_database.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                revision = connection.scalar(
                    text("SELECT version_num FROM alembic_version")
                )
                database_ready = True
        except SQLAlchemyError:
            pass
        checks = {
            "database": database_ready,
            "schema": revision == SCHEMA_HEAD,
            "storage": resolved_services.storage.healthcheck(),
            "ffmpeg": resolved_services.media.ffmpeg_path.is_file(),
        }
        is_ready = all(checks.values())
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={
                "status": "ready" if is_ready else "not_ready",
                "checks": checks,
                "schema_revision": revision,
            },
        )

    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    port = int(os.environ.get("PORT", settings.api_port))
    uvicorn.run(
        "clearframe.main:app",
        host=settings.api_host,
        port=port,
        reload=settings.env == "development",
    )


if __name__ == "__main__":
    run()
