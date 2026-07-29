from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from clearframe import __version__
from clearframe.api.exports import router as exports_router
from clearframe.api.review import router as review_router
from clearframe.api.videos import router as videos_router
from clearframe.config import Settings, get_settings
from clearframe.database import Database
from clearframe.middleware import AccessTokenMiddleware, UploadLimitMiddleware
from clearframe.observability import RequestTraceMiddleware, configure_observability
from clearframe.services.container import ServiceContainer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    app.state.database.create_schema()
    app.state.services.runner.recover_interrupted_jobs()
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
    app.add_middleware(
        UploadLimitMiddleware,
        max_upload_bytes=resolved_settings.max_upload_mb * 1024 * 1024,
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

    @app.get("/api/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "clearframe.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.env == "development",
    )


if __name__ == "__main__":
    run()
