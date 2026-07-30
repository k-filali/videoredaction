import hashlib
from pathlib import Path

import pytest
from scripts.fetch_models import (
    ARTIFACTS,
    ModelArtifact,
    ModelIntegrityError,
    fetch_artifact,
    sha256_file,
    verify_artifact,
)


def artifact_for(payload: bytes) -> ModelArtifact:
    source = ARTIFACTS["face"]
    temporary = ModelArtifact(
        key="test",
        filename="test.onnx",
        url="https://example.invalid/test.onnx",
        size_bytes=len(payload),
        sha256="",
        license_spdx=source.license_spdx,
        license_url=source.license_url,
    )
    return ModelArtifact(
        key=temporary.key,
        filename=temporary.filename,
        url=temporary.url,
        size_bytes=temporary.size_bytes,
        sha256=hashlib.sha256(payload).hexdigest(),
        license_spdx=temporary.license_spdx,
        license_url=temporary.license_url,
    )


def test_model_artifacts_are_immutable_and_verified() -> None:
    for artifact in ARTIFACTS.values():
        assert "/main/" not in artifact.url
        assert len(artifact.sha256) == 64
        assert artifact.size_bytes > 0
    assert "/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/" in (
        ARTIFACTS["face"].url
    )
    assert "/releases/download/assets/" in ARTIFACTS["plate"].url


def test_primary_plate_artifact_uses_semantic_local_filename() -> None:
    artifact = ARTIFACTS["plate"]

    assert artifact.filename == "license_plate_detection_yolov9s_608.onnx"
    assert artifact.url.endswith("yolo-v9-s-608-license-plates-end2end.onnx")
    assert artifact.license_spdx == "MIT"


def test_fetch_is_idempotent_after_verified_download(tmp_path: Path) -> None:
    payload = b"verified model"
    artifact = artifact_for(payload)
    calls = 0

    def download(_url: str, destination: Path, _timeout: float) -> None:
        nonlocal calls
        calls += 1
        destination.write_bytes(payload)

    path, downloaded = fetch_artifact(artifact, tmp_path, downloader=download)
    same_path, downloaded_again = fetch_artifact(artifact, tmp_path, downloader=download)

    assert path == same_path
    assert path.read_bytes() == payload
    assert downloaded
    assert not downloaded_again
    assert calls == 1


def test_fetch_replaces_corrupt_existing_file_atomically(tmp_path: Path) -> None:
    payload = b"replacement"
    artifact = artifact_for(payload)
    destination = tmp_path / artifact.filename
    destination.write_bytes(b"corrupt")

    def download(_url: str, temporary_path: Path, _timeout: float) -> None:
        assert temporary_path != destination
        temporary_path.write_bytes(payload)

    path, downloaded = fetch_artifact(artifact, tmp_path, downloader=download)

    assert downloaded
    assert path.read_bytes() == payload
    verify_artifact(path, artifact)


def test_failed_integrity_check_preserves_existing_file(tmp_path: Path) -> None:
    artifact = artifact_for(b"expected")
    destination = tmp_path / artifact.filename
    destination.write_bytes(b"existing")

    def download(_url: str, temporary_path: Path, _timeout: float) -> None:
        temporary_path.write_bytes(b"wrong")

    with pytest.raises(ModelIntegrityError):
        fetch_artifact(artifact, tmp_path, downloader=download)

    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.glob("*.part")) == []


def test_sha256_file_streams_expected_digest(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"abc")

    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
