import asyncio
from pathlib import Path

import cv2
from httpx import ASGITransport, AsyncClient
from tests.helpers import MOCK_MODEL_REGISTRY_PATH, generate_test_video

from clearframe.config import Settings
from clearframe.database import Database
from clearframe.main import create_app
from clearframe.media import sha256_file
from clearframe.services.container import ServiceContainer


def test_upload_detect_review_export_workflow(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'workflow.db').as_posix()}")
    settings = Settings(
        database_url=str(database.engine.url),
        storage_root=tmp_path / "storage",
        max_upload_mb=10,
        env="test",
        model_registry_path=MOCK_MODEL_REGISTRY_PATH,
    )
    services = ServiceContainer.build(settings, database)
    app = create_app(settings=settings, database=database, services=services)
    source = generate_test_video(tmp_path / "test-evidence.mp4", services.media)
    original_hash = sha256_file(source)

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
                timeout=30,
            ) as client:
                upload = await client.post(
                    "/api/videos",
                    files={
                        "file": (
                            "test-evidence.mp4",
                            source.read_bytes(),
                            "video/mp4",
                        )
                    },
                )
                assert upload.status_code == 202
                video_id = upload.json()["video"]["id"]
                await asyncio.to_thread(
                    services.runner.wait,
                    upload.json()["job"]["id"],
                    120,
                )

                processing = await client.post(
                    f"/api/videos/{video_id}/process",
                    json={"sample_every_frames": 1},
                )
                assert processing.status_code == 202
                await asyncio.to_thread(
                    services.runner.wait,
                    processing.json()["job"]["id"],
                    120,
                )

                tracks_response = await client.get(f"/api/videos/{video_id}/tracks")
                assert tracks_response.status_code == 200
                state = tracks_response.json()
                assert len(state["tracks"]) == 1
                track_id, plate = next(iter(state["tracks"].items()))
                assert plate["class_name"] == "license_plate"
                assert plate["redacted"] is True
                assert plate["accepted"] is False

                accepted = await client.patch(
                    f"/api/videos/{video_id}/tracks/{track_id}",
                    headers={"X-Reviewer-Session": "e2e-reviewer"},
                    json={
                        "action_type": "ACCEPT_PROPOSAL",
                        "expected_revision": state["revision"],
                    },
                )
                assert accepted.status_code == 200
                state = accepted.json()["state"]

                corrected = await client.patch(
                    f"/api/videos/{video_id}/tracks/{track_id}",
                    headers={"X-Reviewer-Session": "e2e-reviewer"},
                    json={
                        "action_type": "RESIZE_REGION",
                        "expected_revision": state["revision"],
                        "frame_index": 5,
                        "timestamp_ms": 333,
                        "payload": {
                            "bbox": next(
                                keyframe["bbox"]
                                for keyframe in plate["keyframes"]
                                if keyframe["frame_index"] == 5
                            )
                        },
                    },
                )
                assert corrected.status_code == 200
                corrected_payload = corrected.json()
                assert corrected_payload["reprocessing_job"]["job_type"] == "REPROCESS"
                await asyncio.to_thread(
                    services.runner.wait,
                    corrected_payload["reprocessing_job"]["id"],
                    120,
                )
                suggestions = await client.get(
                    f"/api/videos/{video_id}/reprocessing-suggestions"
                )
                assert suggestions.status_code == 200
                suggestion_payloads = suggestions.json()
                assert suggestion_payloads
                assert {
                    suggestion["source_action_id"]
                    for suggestion in suggestion_payloads
                } == {corrected_payload["action"]["id"]}
                assert all(
                    suggestion["frame_index"] != 5
                    for suggestion in suggestion_payloads
                )
                state = corrected_payload["state"]
                for suggestion in suggestion_payloads:
                    dismissed = await client.post(
                        (
                            f"/api/videos/{video_id}/reprocessing-suggestions/"
                            f"{suggestion['id']}/dismiss"
                        ),
                        headers={"X-Reviewer-Session": "e2e-reviewer"},
                        json={
                            "expected_revision": state["revision"],
                            "reason_code": "e2e_context_reviewed",
                        },
                    )
                    assert dismissed.status_code == 200
                    assert dismissed.json()["suggestion"]["status"] == "DISMISSED"

                manual = await client.post(
                    f"/api/videos/{video_id}/manual-regions",
                    headers={"X-Reviewer-Session": "e2e-reviewer"},
                    json={
                        "expected_revision": state["revision"],
                        "frame_index": 0,
                        "timestamp_ms": 0,
                        "class_name": "license_plate",
                        "bbox": {"x1": 0.05, "y1": 0.1, "x2": 0.15, "y2": 0.2},
                        "end_frame": 17,
                        "end_ms": 1133,
                        "reason_code": "missed_region",
                    },
                )
                assert manual.status_code == 201
                state = manual.json()["state"]
                manual_track_id = next(
                    candidate_id
                    for candidate_id, candidate in state["tracks"].items()
                    if candidate["source"] == "MANUAL"
                )

                restored = await client.patch(
                    f"/api/videos/{video_id}/tracks/{manual_track_id}",
                    headers={"X-Reviewer-Session": "e2e-reviewer"},
                    json={
                        "action_type": "RESTORE_TRACK",
                        "expected_revision": state["revision"],
                        "reason_code": "false_positive",
                    },
                )
                assert restored.status_code == 200
                frozen_revision = restored.json()["state"]["revision"]

                export = await client.post(
                    f"/api/videos/{video_id}/exports",
                    headers={"X-Reviewer-Session": "e2e-reviewer"},
                    json={
                        "expected_revision": frozen_revision,
                        "redaction_style": "black_box",
                    },
                )
                assert export.status_code == 202
                export_id = export.json()["export"]["id"]
                await asyncio.to_thread(
                    services.runner.wait,
                    export.json()["job"]["id"],
                    120,
                )

                artifact = await client.get(f"/api/exports/{export_id}")
                assert artifact.status_code == 200
                assert artifact.json()["status"] == "COMPLETED"

                download = await client.get(f"/api/exports/{export_id}/download")
                assert download.status_code == 200
                export_path = tmp_path / "downloaded-redacted.mp4"
                export_path.write_bytes(download.content)

                manifest_response = await client.get(
                    f"/api/exports/{export_id}/manifest"
                )
                assert manifest_response.status_code == 200
                manifest = manifest_response.json()
                assert manifest["review_revision"] == frozen_revision
                assert manifest["original_sha256"] == original_hash
                assert manifest["export_sha256"] == sha256_file(export_path)
                assert manifest["action_count"] == 4

                audit = await client.get(f"/api/videos/{video_id}/audit")
                assert audit.status_code == 200
                assert [action["action_type"] for action in audit.json()["actions"]] == [
                    "ACCEPT_PROPOSAL",
                    "RESIZE_REGION",
                    "CREATE_MANUAL_REGION",
                    "RESTORE_TRACK",
                    "EXPORT_REQUESTED",
                    "EXPORT_COMPLETED",
                ]

                source_capture = cv2.VideoCapture(str(source))
                export_capture = cv2.VideoCapture(str(export_path))
                source_ok, source_frame = source_capture.read()
                export_ok, export_frame = export_capture.read()
                source_capture.release()
                export_capture.release()
                assert source_ok and export_ok
                assert source_frame[225, 165].mean() > 200
                assert export_frame[225, 165].mean() < 10
                assert export_frame[45, 45].mean() > 10

                video = await client.get(f"/api/videos/{video_id}")
                assert video.status_code == 200
                assert video.json()["original_sha256"] == original_hash

    asyncio.run(exercise())
