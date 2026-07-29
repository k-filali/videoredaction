import asyncio
from pathlib import Path

from httpx import ASGITransport, AsyncClient
from tests.helpers import MOCK_MODEL_REGISTRY_PATH, generate_test_video

from clearframe.config import Settings
from clearframe.database import Database
from clearframe.main import create_app
from clearframe.services.container import ServiceContainer


def test_thirty_second_upload_completes_reviewed_export(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'long-video.db').as_posix()}")
    settings = Settings(
        database_url=str(database.engine.url),
        storage_root=tmp_path / "storage",
        max_upload_mb=10,
        env="test",
        build_id="long-video-integration",
        log_level="WARNING",
        model_registry_path=MOCK_MODEL_REGISTRY_PATH,
    )
    services = ServiceContainer.build(settings, database)
    app = create_app(settings=settings, database=database, services=services)
    source = generate_test_video(
        tmp_path / "user-supplied-30s.mp4",
        services.media,
        duration_seconds=30,
        audio=False,
    )

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
                timeout=180,
            ) as client:
                uploaded = await client.post(
                    "/api/videos",
                    files={
                        "file": (
                            "user-supplied-30s.mp4",
                            source.read_bytes(),
                            "video/mp4",
                        )
                    },
                )
                assert uploaded.status_code == 202
                upload_payload = uploaded.json()
                video_id = upload_payload["video"]["id"]
                await asyncio.to_thread(
                    services.runner.wait,
                    upload_payload["job"]["id"],
                    180,
                )

                ingested = await client.get(f"/api/videos/{video_id}/status")
                assert ingested.status_code == 200
                ingested_video = ingested.json()["video"]
                assert ingested_video["status"] == "READY_FOR_REVIEW"
                assert ingested_video["original_filename"] == "user-supplied-30s.mp4"
                assert ingested_video["duration_ms"] > 6_000
                assert ingested_video["duration_ms"] >= 29_800
                assert ingested_video["proxy_url"] is not None

                processing = await client.post(
                    f"/api/videos/{video_id}/process",
                    json={
                        "model_ids": ["mock-plate-v1"],
                        "sample_every_frames": 15,
                    },
                )
                assert processing.status_code == 202
                processing_payload = processing.json()
                await asyncio.to_thread(
                    services.runner.wait,
                    processing_payload["job"]["id"],
                    180,
                )

                model_run = await client.get(
                    f"/api/videos/{video_id}/model-runs/latest"
                )
                assert model_run.status_code == 200
                run_payload = model_run.json()
                assert run_payload["status"] == "COMPLETED"
                assert run_payload["metrics"]["detections"] >= 10
                assert run_payload["metrics"]["tracks"] >= 1

                review = await client.get(f"/api/videos/{video_id}/tracks")
                assert review.status_code == 200
                review_payload = review.json()
                revision = review_payload["revision"]
                tracks = review_payload["tracks"]
                assert tracks

                for track_id in tracks:
                    confirmed = await client.patch(
                        f"/api/videos/{video_id}/tracks/{track_id}",
                        headers={"X-Reviewer-Session": "integration-reviewer"},
                        json={
                            "action_type": "ACCEPT_PROPOSAL",
                            "expected_revision": revision,
                            "payload": {},
                        },
                    )
                    assert confirmed.status_code == 200
                    confirmed_state = confirmed.json()["state"]
                    revision = confirmed_state["revision"]
                    assert confirmed_state["tracks"][track_id]["accepted"] is True

                exported = await client.post(
                    f"/api/videos/{video_id}/exports",
                    headers={"X-Reviewer-Session": "integration-reviewer"},
                    json={
                        "expected_revision": revision,
                        "redaction_style": "black_box",
                    },
                )
                assert exported.status_code == 202
                export_payload = exported.json()
                export_id = export_payload["export"]["id"]
                await asyncio.to_thread(
                    services.runner.wait,
                    export_payload["job"]["id"],
                    180,
                )

                artifact = await client.get(f"/api/exports/{export_id}")
                assert artifact.status_code == 200
                artifact_payload = artifact.json()
                assert artifact_payload["status"] == "COMPLETED"
                assert artifact_payload["review_revision"] == revision
                assert artifact_payload["source_model_run_id"] == run_payload["id"]
                assert artifact_payload["export_sha256"] is not None

                manifest = await client.get(f"/api/exports/{export_id}/manifest")
                assert manifest.status_code == 200
                manifest_payload = manifest.json()
                assert manifest_payload["duration_ms"] > 6_000
                assert manifest_payload["duration_ms"] >= 29_800
                assert manifest_payload["frames_rendered"] >= 445
                assert manifest_payload["review_revision"] == revision
                assert manifest_payload["model_run_id"] == run_payload["id"]
                assert manifest_payload["build_id"] == "long-video-integration"

                download = await client.get(f"/api/exports/{export_id}/download")
                assert download.status_code == 200
                assert download.headers["content-type"].startswith("video/mp4")
                assert len(download.content) > 5_000

                completed = await client.get(f"/api/videos/{video_id}/status")
                assert completed.status_code == 200
                completed_payload = completed.json()
                assert completed_payload["video"]["status"] == "EXPORTED"
                job_statuses = {
                    item["job_type"]: item["status"]
                    for item in completed_payload["jobs"]
                }
                assert job_statuses == {
                    "INGEST": "COMPLETED",
                    "DETECT": "COMPLETED",
                    "EXPORT": "COMPLETED",
                }

                audit = await client.get(f"/api/videos/{video_id}/audit")
                assert audit.status_code == 200
                action_types = [
                    item["action_type"] for item in audit.json()["actions"]
                ]
                assert action_types[0] == "ACCEPT_PROPOSAL"
                assert action_types[-2:] == [
                    "EXPORT_REQUESTED",
                    "EXPORT_COMPLETED",
                ]

    asyncio.run(exercise())
