from pathlib import Path

import pytest

from clearframe.storage import (
    ArtifactDelivery,
    ArtifactStorage,
    LocalStorage,
    StorageSecurityError,
    export_manifest_key,
    export_video_key,
    original_key,
    proxy_key,
    sanitize_filename,
    temporary_upload_key,
    thumbnail_key,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("../../private.mp4", "private.mp4"),
        (r"C:\fakepath\incident 01.MP4", "incident 01.mp4"),
        ("résumé?.mov", "r_sum_.mov"),
        ("\x00", "video"),
    ],
)
def test_filename_sanitization(source: str, expected: str) -> None:
    assert sanitize_filename(source) == expected


def test_storage_uris_cannot_escape_root(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "storage")

    with pytest.raises(StorageSecurityError):
        storage.path_for("../outside.mp4")
    with pytest.raises(StorageSecurityError):
        storage.path_for("/absolute/path.mp4")

    safe = storage.prepare("proxies/video-id/proxy.mp4")
    assert safe.is_relative_to(storage.root)
    assert safe.parent.is_dir()


def test_local_storage_satisfies_provider_neutral_contract(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "storage")

    assert isinstance(storage, ArtifactStorage)
    assert storage.healthcheck()


def test_materialized_local_input_is_the_range_compatible_file(
    tmp_path: Path,
) -> None:
    storage = LocalStorage(tmp_path / "storage")
    key = proxy_key("video-id")
    stored = storage.prepare(key)
    stored.write_bytes(b"video-bytes")

    with storage.materialize_input(key) as materialized:
        assert materialized == stored
        assert materialized.read_bytes() == b"video-bytes"

    delivery = storage.delivery_for(key)
    assert delivery == ArtifactDelivery.local_file(stored)
    assert delivery.kind == "local_file"
    assert delivery.path == stored
    assert delivery.url is None


def test_missing_artifacts_cannot_be_materialized_or_delivered(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "storage")

    with (
        pytest.raises(FileNotFoundError),
        storage.materialize_input(proxy_key("missing")),
    ):
        pass
    with pytest.raises(FileNotFoundError):
        storage.delivery_for(proxy_key("missing"))


def test_publish_output_is_atomic(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "storage")
    key = export_video_key("video-id", "export-id")
    destination = storage.path_for(key)

    with storage.publish_output(key) as temporary:
        assert temporary.parent == destination.parent
        assert temporary != destination
        assert not destination.exists()
        temporary.write_bytes(b"complete-export")
        assert not destination.exists()

    assert destination.read_bytes() == b"complete-export"
    assert not list(destination.parent.glob("*.part*"))


def test_failed_publish_preserves_previous_artifact_and_cleans_partial(
    tmp_path: Path,
) -> None:
    storage = LocalStorage(tmp_path / "storage")
    key = proxy_key("video-id")
    destination = storage.prepare(key)
    destination.write_bytes(b"previous")

    with (
        pytest.raises(RuntimeError, match="encoding failed"),
        storage.publish_output(key) as temporary,
    ):
        temporary.write_bytes(b"partial")
        raise RuntimeError("encoding failed")

    assert destination.read_bytes() == b"previous"
    assert not list(destination.parent.glob("*.part*"))


def test_immutable_original_cannot_be_republished_or_removed(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "storage")
    key = original_key("video-id", ".mp4")
    original = storage.prepare(key)
    original.write_bytes(b"evidence")

    with pytest.raises(FileExistsError), storage.publish_output(key):
        pass
    with pytest.raises(StorageSecurityError):
        storage.remove_file(key)
    assert original.read_bytes() == b"evidence"


def test_delivery_descriptor_supports_future_redirects() -> None:
    delivery = ArtifactDelivery.redirect("https://storage.example/artifact")

    assert delivery.kind == "redirect"
    assert delivery.url == "https://storage.example/artifact"
    assert delivery.path is None


def test_logical_keys_are_provider_neutral_and_keep_local_uri_compatibility() -> None:
    assert temporary_upload_key("video-id") == LocalStorage.temporary_upload_uri("video-id")
    assert original_key("video-id", "MP4") == LocalStorage.original_uri("video-id", "MP4")
    assert proxy_key("video-id") == LocalStorage.proxy_uri("video-id")
    assert thumbnail_key("video-id") == LocalStorage.thumbnail_uri("video-id")
    assert export_video_key("video-id", "export-id") == LocalStorage.export_video_uri(
        "video-id", "export-id"
    )
    assert export_manifest_key("video-id", "export-id") == LocalStorage.export_manifest_uri(
        "video-id", "export-id"
    )
