from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from clearframe.config import get_settings


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, url: str) -> None:
        connect_args: dict[str, Any] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self.engine = create_engine(url, connect_args=connect_args)
        if url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )

    @staticmethod
    def _enable_sqlite_foreign_keys(
        dbapi_connection: Any,
        _: Any,
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def create_schema(self) -> None:
        from clearframe import models  # noqa: F401

        if self.engine.url.database and self.engine.url.drivername == "sqlite":
            Path(self.engine.url.database).parent.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(self.engine)

    def drop_schema(self) -> None:
        from clearframe import models  # noqa: F401

        Base.metadata.drop_all(self.engine)

    def session(self) -> Session:
        return self.session_factory()


database = Database(get_settings().database_url)


def get_session() -> Iterator[Session]:
    with database.session() as session:
        yield session


def dispose_database(engine: Engine) -> None:
    engine.dispose()
