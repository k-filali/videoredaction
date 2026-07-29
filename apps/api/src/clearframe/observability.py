import logging
import re
import sys
import time
from uuid import uuid4

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def configure_observability(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=False,
    )
    structlog.configure(
        processors=[
            merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


class RequestTraceMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.logger = structlog.get_logger("clearframe.request")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._request_id(scope)
        bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status_code = 500

        async def traced_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, traced_send)
        except Exception:
            self.logger.exception(
                "request_failed",
                method=scope.get("method"),
                path=scope.get("path"),
            )
            raise
        finally:
            self.logger.info(
                "request_complete",
                method=scope.get("method"),
                path=scope.get("path"),
                status_code=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            clear_contextvars()

    @staticmethod
    def _request_id(scope: Scope) -> str:
        for raw_name, raw_value in scope.get("headers", []):
            if raw_name.lower() != b"x-request-id":
                continue
            candidate = bytes(raw_value).decode("ascii", errors="ignore")
            if REQUEST_ID_PATTERN.fullmatch(candidate):
                return candidate
        return uuid4().hex
