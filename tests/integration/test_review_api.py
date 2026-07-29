import asyncio
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from clearframe.config import Settings
from clearframe.database import Database
from clearframe.domain.enums import TrackSource, VideoStatus
from clearframe.main import create_app
from clearframe.models import Track, TrackKeyframe, VideoAsset
from clearframe.services.container import ServiceContainer


def seed_review_project(database: Database) -> tuple[str, str]:
    database.create_schema()
    with database.session() as session:
        video = VideoAsset(
            original_filename="review.mp4",
            safe_filename="review.mp4",
            content_type="video/mp4",
            status=VideoStatus.READY_FOR_REVIEW,
            duration_ms=2000,
            fps=10,
            width=640,
            height=360,
        )
        session.add(video)
        session.flush()
        track = Track(
            video_id=video.id,
            class_name="license_plate",
            start_frame=0,
            end_frame=19,
            start_ms=0,
            end_ms=1900,
            source=TrackSource.MODEL,
            confidence_summary={"mean": 0.83},
        )
        session.add(track)
        session.flush()
        session.add(
            TrackKeyframe(
                track_id=track.id,
                frame_index=0,
                timestamp_ms=0,
                x1=0.2,
                y1=0.5,
                x2=0.4,
                y2=0.65,
                source=TrackSource.MODEL,
            )
        )
        session.commit()
        return video.id, track.id


def test_review_endpoints_enforce_revisions_and_record_audit(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'review-api.db').as_posix()}")
    video_id, track_id = seed_review_project(database)
    settings = Settings(
        database_url=str(database.engine.url),
        storage_root=tmp_path / "storage",
        env="test",
    )
    services = ServiceContainer.build(settings, database)
    app = create_app(settings=settings, database=database, services=services)

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                initial = await client.get(f"/api/videos/{video_id}/tracks")
                assert initial.status_code == 200
                assert initial.json()["revision"] == 0
                assert initial.json()["tracks"][track_id]["redacted"] is True

                restored = await client.patch(
                    f"/api/videos/{video_id}/tracks/{track_id}",
                    headers={"X-Reviewer-Session": "reviewer-test"},
                    json={
                        "action_type": "RESTORE_TRACK",
                        "expected_revision": 0,
                        "payload": {},
                    },
                )
                assert restored.status_code == 200
                assert restored.json()["state"]["revision"] == 1
                assert restored.json()["state"]["tracks"][track_id]["redacted"] is False

                stale = await client.patch(
                    f"/api/videos/{video_id}/tracks/{track_id}",
                    json={
                        "action_type": "REDACT_TRACK",
                        "expected_revision": 0,
                        "payload": {},
                    },
                )
                assert stale.status_code == 409
                assert stale.json()["detail"]["current_revision"] == 1

                manual = await client.post(
                    f"/api/videos/{video_id}/manual-regions",
                    json={
                        "expected_revision": 1,
                        "frame_index": 10,
                        "timestamp_ms": 1000,
                        "class_name": "license_plate",
                        "bbox": {"x1": 0.6, "y1": 0.4, "x2": 0.8, "y2": 0.55},
                        "end_frame": 15,
                        "end_ms": 1500,
                    },
                )
                assert manual.status_code == 201
                assert manual.json()["state"]["revision"] == 2
                assert len(manual.json()["state"]["tracks"]) == 2

                audit = await client.get(f"/api/videos/{video_id}/audit")
                assert audit.status_code == 200
                assert audit.json()["revision"] == 2
                assert [item["action_type"] for item in audit.json()["actions"]] == [
                    "RESTORE_TRACK",
                    "CREATE_MANUAL_REGION",
                ]
                assert audit.json()["actions"][0]["reviewer_session_id"] == "reviewer-test"

    asyncio.run(exercise())

