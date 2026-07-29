from typing import cast

from fastapi import Request

from clearframe.database import Database
from clearframe.services.container import ServiceContainer


def get_database(request: Request) -> Database:
    return cast(Database, request.app.state.database)


def get_services(request: Request) -> ServiceContainer:
    return cast(ServiceContainer, request.app.state.services)

