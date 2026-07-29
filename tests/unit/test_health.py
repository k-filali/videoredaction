import asyncio
from pathlib import Path

from alembic.script import ScriptDirectory
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from clearframe.config import Settings
from clearframe.database import Database
from clearframe.main import SCHEMA_HEAD, app, create_app
from clearframe.services.container import ServiceContainer

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_schema_head_matches_the_latest_migration() -> None:
    """The readiness pin must track the migration chain.

    Without this, adding a migration leaves every freshly migrated
    deployment reporting not_ready.
    """
    script = ScriptDirectory(str(REPO_ROOT / "migrations"))

    assert script.get_current_head() == SCHEMA_HEAD


def test_health_endpoint() -> None:
    async def request() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": "0.1.0"}

    asyncio.run(request())


def test_readiness_checks_migration_and_runtime_dependencies(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'ready.db').as_posix()}")
    settings = Settings(
        database_url=str(database.engine.url),
        storage_root=tmp_path / "storage",
        env="test",
    )
    services = ServiceContainer.build(settings, database)
    ready_app = create_app(settings=settings, database=database, services=services)

    async def request() -> None:
        async with ready_app.router.lifespan_context(ready_app):
            transport = ASGITransport(app=ready_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                missing_revision = await client.get("/api/ready")
                assert missing_revision.status_code == 503
                assert missing_revision.json()["checks"]["schema"] is False

                with database.engine.begin() as connection:
                    connection.execute(
                        text(
                            "CREATE TABLE alembic_version "
                            "(version_num VARCHAR(32) NOT NULL)"
                        )
                    )
                    connection.execute(
                        text(
                            "INSERT INTO alembic_version (version_num) "
                            "VALUES (:revision)"
                        ),
                        {"revision": SCHEMA_HEAD},
                    )

                response = await client.get("/api/ready")
                assert response.status_code == 200
                assert response.json() == {
                    "status": "ready",
                    "checks": {
                        "database": True,
                        "schema": True,
                        "storage": True,
                        "ffmpeg": True,
                    },
                    "schema_revision": SCHEMA_HEAD,
                }

    asyncio.run(request())
