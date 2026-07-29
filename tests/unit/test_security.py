import asyncio
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from clearframe.config import Settings
from clearframe.database import Database
from clearframe.main import create_app
from clearframe.services.container import ServiceContainer


def test_remote_binding_requires_access_token() -> None:
    with pytest.raises(ValidationError, match="access token"):
        Settings(api_host="0.0.0.0", access_token=None)


def test_access_token_protects_sensitive_api(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'secured.db').as_posix()}")
    settings = Settings(
        database_url=str(database.engine.url),
        storage_root=tmp_path / "storage",
        access_token="test-secret",
        env="test",
    )
    services = ServiceContainer.build(settings, database)
    app = create_app(settings=settings, database=database, services=services)

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                health = await client.get("/api/health")
                assert health.status_code == 200
                assert len(health.headers["x-request-id"]) == 32

                traced = await client.get(
                    "/api/health",
                    headers={"X-Request-ID": "test-request-1234"},
                )
                assert traced.headers["x-request-id"] == "test-request-1234"

                sanitized = await client.get(
                    "/api/health",
                    headers={"X-Request-ID": "../../unsafe"},
                )
                assert sanitized.headers["x-request-id"] != "../../unsafe"

                unauthorized = await client.get("/api/videos")
                assert unauthorized.status_code == 401
                assert unauthorized.headers["www-authenticate"] == "Bearer"

                authorized = await client.get(
                    "/api/videos",
                    headers={"Authorization": "Bearer test-secret"},
                )
                assert authorized.status_code == 200
                assert authorized.json() == {"items": [], "total": 0}

    asyncio.run(exercise())
