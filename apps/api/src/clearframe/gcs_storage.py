import mimetypes
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Protocol, cast
from uuid import uuid4

from clearframe.storage import (
    ArtifactDelivery,
    StorageSecurityError,
    sanitize_filename,
)


class GCSObjectVerificationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GCSObjectMetadata:
    key: str
    size: int
    generation: int
    content_type: str | None
    crc32c: str | None
    md5_hash: str | None
    etag: str | None


class GCSBlob(Protocol):
    size: int | None
    generation: int | None
    content_type: str | None
    crc32c: str | None
    md5_hash: str | None
    etag: str | None

    def exists(self, client: object | None = None) -> bool: ...

    def delete(self, client: object | None = None) -> None: ...

    def download_to_filename(
        self,
        filename: str,
        *,
        client: object | None = None,
        checksum: str | None = "auto",
    ) -> None: ...

    def upload_from_filename(
        self,
        filename: str,
        *,
        content_type: str | None = None,
        client: object | None = None,
        if_generation_match: int | None = None,
        checksum: str | None = "auto",
    ) -> None: ...

    def generate_signed_url(
        self,
        *,
        expiration: timedelta,
        method: str,
        version: str,
        response_disposition: str | None = None,
        client: object | None = None,
        credentials: object | None = None,
    ) -> str: ...

    def create_resumable_upload_session(
        self,
        *,
        content_type: str,
        size: int | None = None,
        origin: str | None = None,
        client: object | None = None,
        checksum: str | None = "auto",
        if_generation_match: int | None = None,
    ) -> str: ...

    def reload(self, *, client: object | None = None) -> None: ...


class GCSBucket(Protocol):
    def blob(self, blob_name: str) -> GCSBlob: ...

    def exists(self, client: object | None = None) -> bool: ...


class GCSClient(Protocol):
    def bucket(self, bucket_name: str) -> GCSBucket: ...


class GCSStorage:
    def __init__(
        self,
        bucket_name: str,
        scratch_dir: Path,
        *,
        client: GCSClient | None = None,
        signed_url_ttl: timedelta = timedelta(minutes=15),
        signing_service_account: str | None = None,
        signing_credentials: object | None = None,
    ) -> None:
        if not bucket_name.strip():
            raise ValueError("bucket name is required")
        if signed_url_ttl <= timedelta(0):
            raise ValueError("signed URL lifetime must be positive")
        self.bucket_name = bucket_name
        self.scratch_dir = scratch_dir.resolve()
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.signed_url_ttl = signed_url_ttl
        self.client = client or self._default_client()
        self.bucket = self.client.bucket(bucket_name)
        if signing_service_account and signing_credentials is not None:
            raise ValueError(
                "provide either a signing service account or signing credentials"
            )
        self.signing_credentials = signing_credentials
        if signing_service_account:
            self.signing_credentials = self._impersonated_credentials(
                signing_service_account
            )

    def exists(self, key: str) -> bool:
        return self._blob(key).exists(client=self.client)

    def remove_file(self, key: str) -> None:
        validated = self._validate_key(key)
        if self._is_original(validated):
            raise StorageSecurityError("immutable originals cannot be removed")
        blob = self.bucket.blob(validated)
        try:
            blob.delete(client=self.client)
        except Exception as exc:
            if not self._has_status(exc, 404):
                raise

    @contextmanager
    def materialize_input(self, key: str) -> Iterator[Path]:
        validated = self._validate_key(key)
        temporary = self._scratch_path(validated, "input")
        try:
            try:
                self.bucket.blob(validated).download_to_filename(
                    str(temporary),
                    client=self.client,
                    checksum="auto",
                )
            except Exception as exc:
                if self._has_status(exc, 404):
                    raise FileNotFoundError("artifact is missing") from exc
                raise
            if not temporary.is_file():
                raise FileNotFoundError("artifact is missing")
            yield temporary
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def publish_output(self, key: str) -> Iterator[Path]:
        validated = self._validate_key(key)
        blob = self.bucket.blob(validated)
        is_original = self._is_original(validated)
        if is_original and blob.exists(client=self.client):
            raise FileExistsError("immutable original already exists")

        temporary = self._scratch_path(validated, "output")
        try:
            yield temporary
            if not temporary.is_file():
                raise FileNotFoundError("published artifact was not created")
            try:
                blob.upload_from_filename(
                    str(temporary),
                    content_type=mimetypes.guess_type(validated)[0],
                    client=self.client,
                    if_generation_match=0 if is_original else None,
                    checksum="auto",
                )
            except Exception as exc:
                if is_original and self._has_status(exc, 412):
                    raise FileExistsError("immutable original already exists") from exc
                raise
        finally:
            temporary.unlink(missing_ok=True)

    def delivery_for(
        self,
        key: str,
        *,
        filename: str | None = None,
    ) -> ArtifactDelivery:
        blob = self._blob(key)
        if not blob.exists(client=self.client):
            raise FileNotFoundError("artifact is missing")
        response_disposition = None
        if filename is not None:
            response_disposition = (
                f'attachment; filename="{sanitize_filename(filename)}"'
            )
        url = blob.generate_signed_url(
            version="v4",
            expiration=self.signed_url_ttl,
            method="GET",
            response_disposition=response_disposition,
            client=self.client,
            credentials=self.signing_credentials,
        )
        return ArtifactDelivery.redirect(url)

    def healthcheck(self) -> bool:
        if not self.scratch_dir.is_dir() or not os.access(self.scratch_dir, os.W_OK):
            return False
        try:
            return self.bucket.exists(client=self.client)
        except Exception:
            return False

    def create_resumable_upload_session(
        self,
        key: str,
        *,
        content_type: str,
        size: int | None = None,
        origin: str | None = None,
    ) -> str:
        validated = self._validate_key(key)
        if not content_type.strip():
            raise ValueError("content type is required")
        if size is not None and size < 0:
            raise ValueError("upload size cannot be negative")
        blob = self.bucket.blob(validated)
        try:
            return blob.create_resumable_upload_session(
                content_type=content_type,
                size=size,
                origin=origin,
                client=self.client,
                checksum="auto",
                if_generation_match=0,
            )
        except Exception as exc:
            if self._has_status(exc, 412):
                raise FileExistsError("upload destination already exists") from exc
            raise

    def metadata_for(self, key: str) -> GCSObjectMetadata:
        validated = self._validate_key(key)
        blob = self.bucket.blob(validated)
        try:
            blob.reload(client=self.client)
        except Exception as exc:
            if self._has_status(exc, 404):
                raise FileNotFoundError("artifact is missing") from exc
            raise
        if blob.size is None or blob.generation is None:
            raise GCSObjectVerificationError("artifact metadata is incomplete")
        return GCSObjectMetadata(
            key=validated,
            size=blob.size,
            generation=blob.generation,
            content_type=blob.content_type,
            crc32c=blob.crc32c,
            md5_hash=blob.md5_hash,
            etag=blob.etag,
        )

    def verify_object_metadata(
        self,
        key: str,
        *,
        expected_size: int | None = None,
        expected_content_type: str | None = None,
        expected_crc32c: str | None = None,
    ) -> GCSObjectMetadata:
        metadata = self.metadata_for(key)
        if expected_size is not None and metadata.size != expected_size:
            raise GCSObjectVerificationError("artifact size does not match")
        if (
            expected_content_type is not None
            and metadata.content_type != expected_content_type
        ):
            raise GCSObjectVerificationError("artifact content type does not match")
        if expected_crc32c is not None and metadata.crc32c != expected_crc32c:
            raise GCSObjectVerificationError("artifact checksum does not match")
        return metadata

    def _blob(self, key: str) -> GCSBlob:
        return self.bucket.blob(self._validate_key(key))

    def _scratch_path(self, key: str, purpose: str) -> Path:
        suffix = PurePosixPath(key).suffix
        return self.scratch_dir / f".gcs-{purpose}-{uuid4().hex}{suffix}"

    @staticmethod
    def _validate_key(key: str) -> str:
        pure_path = PurePosixPath(key)
        if not key or pure_path.is_absolute() or ".." in pure_path.parts:
            raise StorageSecurityError("storage URI must be relative")
        return pure_path.as_posix()

    @staticmethod
    def _is_original(key: str) -> bool:
        return PurePosixPath(key).parts[:1] == ("originals",)

    @staticmethod
    def _has_status(exc: Exception, expected: int) -> bool:
        code = getattr(exc, "code", None)
        if callable(code):
            code = code()
        if code == expected:
            return True
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None) == expected

    @staticmethod
    def _default_client() -> GCSClient:
        try:
            module = import_module("google.cloud.storage")
        except ModuleNotFoundError as exc:
            raise RuntimeError("google-cloud-storage is required for GCS storage") from exc
        return cast(GCSClient, module.Client())

    @staticmethod
    def _impersonated_credentials(service_account: str) -> object:
        try:
            auth = import_module("google.auth")
            impersonated = import_module("google.auth.impersonated_credentials")
        except ModuleNotFoundError as exc:
            raise RuntimeError("google-auth is required for signed GCS URLs") from exc
        source, _ = auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return impersonated.Credentials(
            source_credentials=source,
            target_principal=service_account,
            target_scopes=[
                "https://www.googleapis.com/auth/devstorage.read_only"
            ],
            lifetime=3600,
        )
