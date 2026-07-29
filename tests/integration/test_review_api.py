import asyncio
from pathlib import Path

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from clearframe.config import Settings
from clearframe.database import Database
from clearframe.domain.enums import (
    JobStatus,
    JobType,
    ReprocessingSuggestionStatus,
    ReviewActionType,
    TrackSource,
    VideoStatus,
)
from clearframe.domain.review import ReviewCommand
from clearframe.main import create_app
from clearframe.models import (
    ProcessingJob,
    ReprocessingSuggestion,
    Track,
    TrackKeyframe,
    VideoAsset,
)
from clearframe.services.container import ServiceContainer
from clearframe.services.review import append_review_action


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


def seed_reprocessing_suggestions(
    database: Database,
    video_id: str,
    track_id: str,
) -> tuple[str, str]:
    with database.session() as session:
        source_action, _ = append_review_action(
            session,
            video_id,
            ReviewCommand(
                action_type=ReviewActionType.MOVE_REGION,
                expected_revision=0,
                track_id=track_id,
                frame_index=5,
                timestamp_ms=500,
                payload={
                    "bbox": {
                        "x1": 0.24,
                        "y1": 0.49,
                        "x2": 0.44,
                        "y2": 0.64,
                    }
                },
            ),
            reviewer_session_id="seed-reviewer",
        )

    with database.session() as session:
        job = ProcessingJob(
            video_id=video_id,
            job_type=JobType.REPROCESS,
            status=JobStatus.COMPLETED,
        )
        session.add(job)
        session.flush()
        suggestions = [
            ReprocessingSuggestion(
                video_id=video_id,
                source_action_id=source_action.id,
                job_id=job.id,
                track_id=track_id,
                source_revision=source_action.revision,
                class_name="license_plate",
                seed_frame_index=5,
                frame_index=frame_index,
                timestamp_ms=frame_index * 100,
                x1=0.24 + offset,
                y1=0.49,
                x2=0.44 + offset,
                y2=0.64,
                confidence=0.72,
                direction="forward",
                propagation_method="interpolation",
                seed_locked=True,
                status=ReprocessingSuggestionStatus.PENDING,
                metadata_json={"distance_frames": frame_index - 5},
            )
            for frame_index, offset in ((6, 0.01), (7, 0.02))
        ]
        session.add_all(suggestions)
        session.commit()
        return suggestions[0].id, suggestions[1].id


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


def test_reprocessing_suggestion_resolution_endpoints(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'suggestion-api.db').as_posix()}")
    video_id, track_id = seed_review_project(database)
    accepted_id, dismissed_id = seed_reprocessing_suggestions(
        database,
        video_id,
        track_id,
    )
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
                listed = await client.get(
                    f"/api/videos/{video_id}/reprocessing-suggestions"
                )
                assert listed.status_code == 200
                assert [item["status"] for item in listed.json()] == [
                    "PENDING",
                    "PENDING",
                ]
                assert all(
                    item["resolved_by_session_id"] is None
                    for item in listed.json()
                )

                wrong_video = await client.post(
                    f"/api/videos/not-this-video/reprocessing-suggestions/"
                    f"{accepted_id}/accept",
                    json={"expected_revision": 1},
                )
                assert wrong_video.status_code == 404

                stale = await client.post(
                    f"/api/videos/{video_id}/reprocessing-suggestions/"
                    f"{accepted_id}/accept",
                    json={"expected_revision": 0},
                )
                assert stale.status_code == 409
                assert stale.json()["detail"]["current_revision"] == 1

                accepted = await client.post(
                    f"/api/videos/{video_id}/reprocessing-suggestions/"
                    f"{accepted_id}/accept",
                    headers={"X-Reviewer-Session": "reviewer-api"},
                    json={
                        "expected_revision": 1,
                        "reason_code": "approved_context",
                    },
                )
                assert accepted.status_code == 200
                accepted_payload = accepted.json()
                assert accepted_payload["suggestion"]["status"] == "ACCEPTED"
                assert (
                    accepted_payload["suggestion"]["resolved_by_session_id"]
                    == "reviewer-api"
                )
                assert (
                    accepted_payload["suggestion"]["resolution_reason_code"]
                    == "approved_context"
                )
                assert accepted_payload["action"]["action_type"] == "RESIZE_REGION"
                assert accepted_payload["state"]["revision"] == 2
                accepted_track = accepted_payload["state"]["tracks"][track_id]
                accepted_keyframe = next(
                    item
                    for item in accepted_track["keyframes"]
                    if item["frame_index"] == 6
                )
                assert accepted_keyframe["locked"] is True

                already_accepted = await client.post(
                    f"/api/videos/{video_id}/reprocessing-suggestions/"
                    f"{accepted_id}/accept",
                    json={"expected_revision": 2},
                )
                assert already_accepted.status_code == 409

                dismissed = await client.post(
                    f"/api/videos/{video_id}/reprocessing-suggestions/"
                    f"{dismissed_id}/dismiss",
                    headers={"X-Reviewer-Session": "reviewer-dismiss-api"},
                    json={
                        "expected_revision": 2,
                        "reason_code": "drifted_box",
                    },
                )
                assert dismissed.status_code == 200
                dismissed_payload = dismissed.json()
                assert dismissed_payload["suggestion"]["status"] == "DISMISSED"
                assert dismissed_payload["suggestion"]["resolution_action_id"] is None
                assert (
                    dismissed_payload["suggestion"]["resolved_by_session_id"]
                    == "reviewer-dismiss-api"
                )
                assert dismissed_payload["state"]["revision"] == 2
                assert dismissed_payload["action"] is None

                already_dismissed = await client.post(
                    f"/api/videos/{video_id}/reprocessing-suggestions/"
                    f"{dismissed_id}/dismiss",
                    json={"expected_revision": 2},
                )
                assert already_dismissed.status_code == 409

                refreshed = await client.get(
                    f"/api/videos/{video_id}/reprocessing-suggestions"
                )
                assert refreshed.status_code == 200
                assert {item["status"] for item in refreshed.json()} == {
                    "ACCEPTED",
                    "DISMISSED",
                }

                audit = await client.get(f"/api/videos/{video_id}/audit")
                assert audit.status_code == 200
                assert [item["action_type"] for item in audit.json()["actions"]] == [
                    "MOVE_REGION",
                    "RESIZE_REGION",
                ]

            with database.session() as session:
                reprocess_jobs = list(
                    session.scalars(
                        select(ProcessingJob).where(
                            ProcessingJob.video_id == video_id,
                            ProcessingJob.job_type == JobType.REPROCESS,
                        )
                    )
                )
                assert len(reprocess_jobs) == 1

    asyncio.run(exercise())
