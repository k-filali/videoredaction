from typing import Any

import pytest
from sqlalchemy import Engine, create_engine

import clearframe.database as database_module
from clearframe.database import Database


def _capture_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[Engine]]:
    captured: dict[str, Any] = {}
    engines: list[Engine] = []

    def create_engine_spy(url: str, **kwargs: Any) -> Engine:
        captured["url"] = url
        captured["options"] = kwargs
        engine = create_engine("sqlite://")
        engines.append(engine)
        return engine

    monkeypatch.setattr(database_module, "create_engine", create_engine_spy)
    return captured, engines


def test_sqlite_keeps_existing_engine_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured, engines = _capture_engine(monkeypatch)

    Database("sqlite:///test.db")

    assert captured == {
        "url": "sqlite:///test.db",
        "options": {"connect_args": {"check_same_thread": False}},
    }
    engines[0].dispose()


@pytest.mark.parametrize(
    ("url", "expected_url"),
    [
        (
            "postgresql://clearframe:secret@db.example/clearframe?sslmode=require",
            "postgresql+psycopg://clearframe:secret@db.example/clearframe?sslmode=require",
        ),
        (
            "postgres://clearframe:secret@db.example/clearframe?sslmode=require",
            "postgresql+psycopg://clearframe:secret@db.example/clearframe?sslmode=require",
        ),
        (
            "postgresql+psycopg://clearframe:secret@db.example/clearframe?sslmode=require",
            "postgresql+psycopg://clearframe:secret@db.example/clearframe?sslmode=require",
        ),
    ],
)
def test_postgres_uses_psycopg_and_bounded_pool(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    expected_url: str,
) -> None:
    captured, engines = _capture_engine(monkeypatch)

    Database(url)

    assert captured == {
        "url": expected_url,
        "options": {
            "connect_args": {},
            "pool_pre_ping": True,
            "pool_size": 5,
            "max_overflow": 2,
            "pool_recycle": 1800,
            "pool_timeout": 30,
        },
    }
    engines[0].dispose()


def test_sqlite_connection_pragmas_remain_enabled() -> None:
    database = Database("sqlite://")

    with database.engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 5000

    database.engine.dispose()
