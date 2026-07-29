from pathlib import Path
from typing import Any

import pytest

from clearframe.gcs_storage import (
    GCSObjectVerificationError,
    GCSStorage,
)
from clearframe.storage import ArtifactStorage, StorageSecurityError, original_key, proxy_key


class FakeGoogleError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"google error {code}")
        self.code = code


class FakeBlob:
    def __init__(self, bucket: "FakeBucket", name: str) -> None:
        self.bucket = bucket
        self.name = name
        self.size: int | None = None
        self.generation: int | None = None
        self.content_type: str | None = None
        self.crc32c: str | None = None
        self.md5_hash: str | None = None
        self.etag: str | None = None

    def exists(self, client: object | None = None) -> bool:
        del client
        self.bucket.read_calls += 1
        return self.name in self.bucket.objects

    def delete(self, client: object | None = None) -> None:
        del client
        self.bucket.delete_calls.append(self.name)
        if self.name not in self.bucket.objects:
            raise FakeGoogleError(404)
        del self.bucket.objects[self.name]

    def download_to_filename(
        self,
        filename: str,
        *,
        client: object | None = None,
        checksum: str | None = "auto",
    ) -> None:
        del client, checksum
        self.bucket.read_calls += 1
        try:
            content = self.bucket.objects[self.name]["content"]
        except KeyError as exc:
            raise FakeGoogleError(404) from exc
        Path(filename).write_bytes(content)

    def upload_from_filename(
        self,
        filename: str,
        *,
        content_type: str | None = None,
        client: object | None = None,
        if_generation_match: int | None = None,
        checksum: str | None = "auto",
    ) -> None:
        del client
        if if_generation_match == 0 and self.name in self.bucket.objects:
            raise FakeGoogleError(412)
        content = Path(filename).read_bytes()
        generation = self.bucket.next_generation
        self.bucket.next_generation += 1
        self.bucket.objects[self.name] = {
            "content": content,
            "content_type": content_type,
            "generation": generation,
            "crc32c": "uploaded-crc",
            "md5_hash": "uploaded-md5",
            "etag": f"etag-{generation}",
        }
        self.bucket.upload_calls.append(
            {
                "name": self.name,
                "content_type": content_type,
                "if_generation_match": if_generation_match,
                "checksum": checksum,
            }
        )

    def generate_signed_url(
        self,
        *,
        expiration: Any,
        method: str,
        version: str,
        response_disposition: str | None = None,
        client: object | None = None,
        credentials: object | None = None,
    ) -> str:
        del client
        self.bucket.signed_url_calls.append(
            {
                "name": self.name,
                "expiration": expiration,
                "method": method,
                "version": version,
                "response_disposition": response_disposition,
                "credentials": credentials,
            }
        )
        return f"https://storage.example/{self.name}?signature=test"

    def create_resumable_upload_session(
        self,
        *,
        content_type: str,
        size: int | None = None,
        origin: str | None = None,
        client: object | None = None,
        checksum: str | None = "auto",
        if_generation_match: int | None = None,
    ) -> str:
        del client
        self.bucket.resumable_calls.append(
            {
                "name": self.name,
                "content_type": content_type,
                "size": size,
                "origin": origin,
                "checksum": checksum,
                "if_generation_match": if_generation_match,
            }
        )
        return f"https://upload.example/{self.name}?session=test"

    def reload(self, *, client: object | None = None) -> None:
        del client
        self.bucket.read_calls += 1
        try:
            stored = self.bucket.objects[self.name]
        except KeyError as exc:
            raise FakeGoogleError(404) from exc
        content = stored["content"]
        self.size = len(content)
        self.generation = stored["generation"]
        self.content_type = stored["content_type"]
        self.crc32c = stored["crc32c"]
        self.md5_hash = stored["md5_hash"]
        self.etag = stored["etag"]


class FakeBucket:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.objects: dict[str, dict[str, Any]] = {}
        self.next_generation = 1
        self.read_calls = 0
        self.healthcheck_calls = 0
        self.upload_calls: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self.signed_url_calls: list[dict[str, Any]] = []
        self.resumable_calls: list[dict[str, Any]] = []

    def blob(self, blob_name: str) -> FakeBlob:
        return FakeBlob(self, blob_name)

    def exists(self, client: object | None = None) -> bool:
        del client
        self.healthcheck_calls += 1
        return self.available

    def seed(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str = "video/mp4",
        crc32c: str = "seed-crc",
    ) -> None:
        generation = self.next_generation
        self.next_generation += 1
        self.objects[key] = {
            "content": content,
            "content_type": content_type,
            "generation": generation,
            "crc32c": crc32c,
            "md5_hash": "seed-md5",
            "etag": f"etag-{generation}",
        }


class FakeClient:
    def __init__(self, bucket: FakeBucket) -> None:
        self.fake_bucket = bucket
        self.requested_buckets: list[str] = []

    def bucket(self, bucket_name: str) -> FakeBucket:
        self.requested_buckets.append(bucket_name)
        return self.fake_bucket


@pytest.fixture
def gcs(tmp_path: Path) -> tuple[GCSStorage, FakeBucket]:
    bucket = FakeBucket()
    storage = GCSStorage(
        "clearframe-test",
        tmp_path / "scratch",
        client=FakeClient(bucket),
    )
    return storage, bucket


def test_gcs_storage_satisfies_contract_and_healthcheck_is_read_only(
    gcs: tuple[GCSStorage, FakeBucket],
) -> None:
    storage, bucket = gcs

    assert isinstance(storage, ArtifactStorage)
    assert storage.healthcheck()
    assert bucket.healthcheck_calls == 1
    assert bucket.upload_calls == []
    assert bucket.delete_calls == []


def test_materialize_downloads_to_scratch_and_always_cleans_up(
    gcs: tuple[GCSStorage, FakeBucket],
) -> None:
    storage, bucket = gcs
    key = proxy_key("video-id")
    bucket.seed(key, b"proxy")

    with storage.materialize_input(key) as path:
        assert path.parent == storage.scratch_dir
        assert path.read_bytes() == b"proxy"
        materialized = path

    assert not materialized.exists()
    assert list(storage.scratch_dir.iterdir()) == []


def test_missing_materialized_input_is_reported_and_cleans_scratch(
    gcs: tuple[GCSStorage, FakeBucket],
) -> None:
    storage, _ = gcs

    with pytest.raises(FileNotFoundError), storage.materialize_input("missing.mp4"):
        pass

    assert list(storage.scratch_dir.iterdir()) == []


def test_publish_uploads_complete_file_and_cleans_local_temporary_path(
    gcs: tuple[GCSStorage, FakeBucket],
) -> None:
    storage, bucket = gcs
    key = proxy_key("video-id")

    with storage.publish_output(key) as path:
        path.write_bytes(b"complete")
        temporary = path
        assert key not in bucket.objects

    assert not temporary.exists()
    assert bucket.objects[key]["content"] == b"complete"
    assert bucket.upload_calls == [
        {
            "name": key,
            "content_type": "video/mp4",
            "if_generation_match": None,
            "checksum": "auto",
        }
    ]


def test_original_upload_uses_generation_precondition_and_is_immutable(
    gcs: tuple[GCSStorage, FakeBucket],
) -> None:
    storage, bucket = gcs
    key = original_key("video-id", ".mp4")

    with storage.publish_output(key) as path:
        path.write_bytes(b"evidence")

    assert bucket.upload_calls[0]["if_generation_match"] == 0
    with pytest.raises(FileExistsError), storage.publish_output(key):
        pass
    with pytest.raises(StorageSecurityError):
        storage.remove_file(key)
    assert bucket.objects[key]["content"] == b"evidence"


def test_missing_published_output_does_not_upload(
    gcs: tuple[GCSStorage, FakeBucket],
) -> None:
    storage, bucket = gcs

    with pytest.raises(FileNotFoundError), storage.publish_output(proxy_key("video-id")):
        pass

    assert bucket.upload_calls == []
    assert list(storage.scratch_dir.iterdir()) == []


def test_remove_is_idempotent_for_derived_artifacts(
    gcs: tuple[GCSStorage, FakeBucket],
) -> None:
    storage, bucket = gcs
    key = proxy_key("video-id")
    bucket.seed(key, b"proxy")

    storage.remove_file(key)
    storage.remove_file(key)

    assert key not in bucket.objects
    assert bucket.delete_calls == [key, key]


def test_delivery_returns_v4_signed_redirect_with_download_filename(
    gcs: tuple[GCSStorage, FakeBucket],
) -> None:
    storage, bucket = gcs
    key = proxy_key("video-id")
    bucket.seed(key, b"proxy")

    delivery = storage.delivery_for(key, filename="../../incident.MP4")

    assert delivery.kind == "redirect"
    assert delivery.url == f"https://storage.example/{key}?signature=test"
    assert bucket.signed_url_calls[0]["method"] == "GET"
    assert bucket.signed_url_calls[0]["version"] == "v4"
    assert (
        bucket.signed_url_calls[0]["response_disposition"]
        == 'attachment; filename="incident.mp4"'
    )


def test_resumable_session_is_origin_bound_and_create_only(
    gcs: tuple[GCSStorage, FakeBucket],
) -> None:
    storage, bucket = gcs
    key = "tmp/uploads/video-id.upload"

    session_url = storage.create_resumable_upload_session(
        key,
        content_type="video/mp4",
        size=1234,
        origin="https://clearframe.example",
    )

    assert session_url.endswith("?session=test")
    assert bucket.resumable_calls == [
        {
            "name": key,
            "content_type": "video/mp4",
            "size": 1234,
            "origin": "https://clearframe.example",
            "checksum": "auto",
            "if_generation_match": 0,
        }
    ]


def test_delivery_uses_explicit_signing_credentials(tmp_path: Path) -> None:
    bucket = FakeBucket()
    signing_credentials = object()
    storage = GCSStorage(
        "clearframe-test",
        tmp_path / "scratch",
        client=FakeClient(bucket),
        signing_credentials=signing_credentials,
    )
    key = proxy_key("video-id")
    bucket.seed(key, b"proxy")

    storage.delivery_for(key)

    assert bucket.signed_url_calls[0]["credentials"] is signing_credentials


def test_metadata_can_be_verified_without_downloading_object(
    gcs: tuple[GCSStorage, FakeBucket],
) -> None:
    storage, bucket = gcs
    key = proxy_key("video-id")
    bucket.seed(key, b"proxy", crc32c="expected-crc")

    metadata = storage.verify_object_metadata(
        key,
        expected_size=5,
        expected_content_type="video/mp4",
        expected_crc32c="expected-crc",
    )

    assert metadata.key == key
    assert metadata.size == 5
    assert metadata.generation == 1
    assert metadata.md5_hash == "seed-md5"
    with pytest.raises(GCSObjectVerificationError, match="size"):
        storage.verify_object_metadata(key, expected_size=6)


@pytest.mark.parametrize("key", ["../escape.mp4", "/absolute.mp4", ""])
def test_object_keys_cannot_escape_logical_storage_root(
    gcs: tuple[GCSStorage, FakeBucket],
    key: str,
) -> None:
    storage, _ = gcs

    with pytest.raises(StorageSecurityError):
        storage.exists(key)
