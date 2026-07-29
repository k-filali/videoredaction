import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from clearframe.api.exports import router as exports_router
from clearframe.api.videos import router as videos_router
from clearframe.database import Database
from clearframe.domain.enums import ExportStatus, RedactionStyle, VideoStatus
from clearframe.models import ExportArtifact, VideoAsset
from clearframe.services.container import ServiceContainer
from clearframe.storage import ArtifactDelivery, LocalStorage


class RedirectStorage(LocalStorage):
    def __init__(self, root: Path, url: str) -> None:
        super().__init__(root)
        self.url = url

    def exists(self, uri: str) -> bool:
        return True

    def delivery_for(self, key: str) -> ArtifactDelivery:
        return ArtifactDelivery.redirect(self.url)


def build_app(
    tmp_path: Path,
    storage: LocalStorage,
) -> tuple[FastAPI, str, str]:
    database = Database(f"sqlite:///{(tmp_path / 'delivery.db').as_posix()}")
    database.create_schema()
    video_id = "11111111-1111-1111-1111-111111111111"
    export_id = "22222222-2222-2222-2222-222222222222"
    with database.session() as session:
        session.add(
            VideoAsset(
                id=video_id,
                original_filename="incident.mp4",
                safe_filename="incident.mp4",
                content_type="video/mp4",
                proxy_uri=storage.proxy_uri(video_id),
                thumbnail_uri=storage.thumbnail_uri(video_id),
                status=VideoStatus.READY_FOR_REVIEW,
            )
        )
        session.flush()
        session.add(
            ExportArtifact(
                id=export_id,
                video_id=video_id,
                export_uri=storage.export_video_uri(video_id, export_id),
                redaction_style=RedactionStyle.PIXELATE,
                review_revision=0,
                status=ExportStatus.COMPLETED,
            )
        )
        session.commit()

    app = FastAPI()
    app.state.database = database
    app.state.services = cast(
        ServiceContainer,
        SimpleNamespace(storage=storage),
    )
    app.include_router(videos_router)
    app.include_router(exports_router)
    return app, video_id, export_id


def test_local_delivery_preserves_file_response_range_support(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "storage")
    app, video_id, export_id = build_app(tmp_path, storage)
    proxy = storage.prepare(storage.proxy_uri(video_id))
    proxy.write_bytes(b"0123456789")
    thumbnail = storage.prepare(storage.thumbnail_uri(video_id))
    thumbnail.write_bytes(b"jpeg")
    exported = storage.prepare(storage.export_video_uri(video_id, export_id))
    exported.write_bytes(b"export")

    async def exercise() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                f"/api/videos/{video_id}/proxy",
                headers={"Range": "bytes=2-5"},
            )
            assert response.status_code == 206
            assert response.content == b"2345"
            assert response.headers["accept-ranges"] == "bytes"
            assert response.headers["content-range"] == "bytes 2-5/10"
            assert response.headers["cache-control"] == "private, no-store"

            thumbnail_response = await client.get(f"/api/videos/{video_id}/thumbnail")
            assert thumbnail_response.status_code == 200
            assert thumbnail_response.content == b"jpeg"

            download = await client.get(f"/api/exports/{export_id}/download")
            assert download.status_code == 200
            assert download.content == b"export"
            assert "attachment;" in download.headers["content-disposition"]

    asyncio.run(exercise())


def test_cloud_delivery_uses_temporary_redirects_with_private_headers(
    tmp_path: Path,
) -> None:
    signed_url = "https://objects.example/signed-artifact?signature=secret"
    storage = RedirectStorage(tmp_path / "storage", signed_url)
    app, video_id, export_id = build_app(tmp_path, storage)

    async def exercise() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            follow_redirects=False,
        ) as client:
            for endpoint in (
                f"/api/videos/{video_id}/proxy",
                f"/api/videos/{video_id}/thumbnail",
                f"/api/exports/{export_id}/download",
            ):
                response = await client.get(endpoint)
                assert response.status_code == 307
                assert response.headers["location"] == signed_url
                assert response.headers["cache-control"] == "private, no-store"

            download = await client.get(f"/api/exports/{export_id}/download")
            assert "attachment;" in download.headers["content-disposition"]

    asyncio.run(exercise())
