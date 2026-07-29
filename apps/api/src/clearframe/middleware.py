import hmac

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class PayloadTooLargeError(Exception):
    pass


class AccessTokenMiddleware:
    def __init__(self, app: ASGIApp, *, access_token: str | None) -> None:
        self.app = app
        self.access_token = access_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            self.access_token is None
            or scope["type"] != "http"
            or scope.get("path") == "/api/health"
            or scope.get("method") == "OPTIONS"
        ):
            await self.app(scope, receive, send)
            return

        supplied_token = self._read_token(scope)
        if supplied_token is None or not hmac.compare_digest(
            supplied_token,
            self.access_token,
        ):
            response = JSONResponse(
                status_code=401,
                content={"detail": "a valid access token is required"},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    @staticmethod
    def _read_token(scope: Scope) -> str | None:
        for raw_name, raw_value in scope.get("headers", []):
            if raw_name.lower() == b"authorization":
                value = bytes(raw_value).decode("latin-1")
                scheme, _, token = value.partition(" ")
                if scheme.lower() == "bearer" and token:
                    return token
            if raw_name.lower() == b"x-access-token":
                return bytes(raw_value).decode("latin-1")
        return None


class UploadLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_upload_bytes: int,
        multipart_overhead_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        self.app = app
        self.max_request_bytes = max_upload_bytes + multipart_overhead_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._is_upload(scope):
            await self.app(scope, receive, send)
            return

        content_length = self._content_length(scope)
        if content_length is not None and content_length > self.max_request_bytes:
            await self._reject(scope, receive, send)
            return

        consumed = 0

        async def limited_receive() -> Message:
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > self.max_request_bytes:
                raise PayloadTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except PayloadTooLargeError:
            await self._reject(scope, receive, send)

    @staticmethod
    def _is_upload(scope: Scope) -> bool:
        return (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/api/videos"
        )

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        for raw_name, raw_value in scope.get("headers", []):
            if raw_name.lower() == b"content-length":
                try:
                    return int(raw_value)
                except ValueError:
                    return None
        return None

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={"detail": "request exceeds the configured upload limit"},
        )
        await response(scope, receive, send)
