import os
import re
import stat
import unicodedata
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, runtime_checkable
from uuid import uuid4


class StorageSecurityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactDelivery:
    kind: Literal["local_file", "redirect"]
    path: Path | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "local_file" and (self.path is None or self.url is not None):
            raise ValueError("local file delivery requires only a path")
        if self.kind == "redirect" and (not self.url or self.path is not None):
            raise ValueError("redirect delivery requires only a URL")

    @classmethod
    def local_file(cls, path: Path) -> "ArtifactDelivery":
        return cls(kind="local_file", path=path)

    @classmethod
    def redirect(cls, url: str) -> "ArtifactDelivery":
        return cls(kind="redirect", url=url)


@runtime_checkable
class ArtifactStorage(Protocol):
    def exists(self, key: str) -> bool: ...

    def remove_file(self, key: str) -> None: ...

    def materialize_input(self, key: str) -> AbstractContextManager[Path]: ...

    def publish_output(self, key: str) -> AbstractContextManager[Path]: ...

    def delivery_for(
        self,
        key: str,
        *,
        filename: str | None = None,
    ) -> ArtifactDelivery: ...

    def healthcheck(self) -> bool: ...


def sanitize_filename(filename: str | None) -> str:
    candidate = (filename or "video").replace("\\", "/")
    candidate = PurePosixPath(candidate).name
    candidate = unicodedata.normalize("NFKC", candidate).replace("\x00", "")
    candidate = re.sub(r"[^A-Za-z0-9._ -]+", "_", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .")
    if not candidate or candidate in {".", ".."}:
        candidate = "video"
    stem = Path(candidate).stem[:160].strip(" .") or "video"
    suffix = Path(candidate).suffix[:16].lower()
    return f"{stem}{suffix}"


def temporary_upload_key(video_id: str) -> str:
    return f"tmp/uploads/{video_id}.upload"


def original_key(video_id: str, extension: str) -> str:
    clean_extension = extension if extension.startswith(".") else f".{extension}"
    return f"originals/{video_id}/original{clean_extension.lower()}"


def proxy_key(video_id: str) -> str:
    return f"proxies/{video_id}/proxy.mp4"


def tracking_proxy_key(video_id: str) -> str:
    """Small copy used for reviewer-edit propagation.

    Context propagation only needs a few seconds of frames, so downloading the
    full review proxy per edit dominates job time. This artifact keeps the
    review proxy's frame timing so track frame indices stay valid.
    """
    return f"tracking/{video_id}/tracking.mp4"


def thumbnail_key(video_id: str) -> str:
    return f"thumbnails/{video_id}/poster.jpg"


def export_video_key(video_id: str, export_id: str) -> str:
    return f"exports/{video_id}/{export_id}/redacted.mp4"


def export_manifest_key(video_id: str, export_id: str) -> str:
    return f"exports/{video_id}/{export_id}/manifest.json"


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, uri: str) -> Path:
        pure_path = PurePosixPath(uri)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise StorageSecurityError("storage URI must be relative")
        resolved = self.root.joinpath(*pure_path.parts).resolve()
        if not resolved.is_relative_to(self.root):
            raise StorageSecurityError("storage URI escapes the configured root")
        return resolved

    def prepare(self, uri: str) -> Path:
        path = self.path_for(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def exists(self, uri: str) -> bool:
        return self.path_for(uri).is_file()

    def remove_file(self, uri: str) -> None:
        if self._is_original(uri):
            raise StorageSecurityError("immutable originals cannot be removed")
        path = self.path_for(uri)
        if path.is_file():
            if not os.access(path, os.W_OK):
                path.chmod(stat.S_IWRITE | stat.S_IREAD)
            path.unlink()

    def promote(self, temporary_uri: str, destination_uri: str) -> Path:
        source = self.path_for(temporary_uri)
        destination = self.prepare(destination_uri)
        if not source.is_file():
            raise FileNotFoundError("temporary upload is missing")
        if destination.exists():
            raise FileExistsError("destination artifact already exists")
        source.replace(destination)
        return destination

    @contextmanager
    def materialize_input(self, key: str) -> Iterator[Path]:
        path = self.path_for(key)
        if not path.is_file():
            raise FileNotFoundError("artifact is missing")
        yield path

    @contextmanager
    def publish_output(self, key: str) -> Iterator[Path]:
        destination = self.path_for(key)
        if self._is_original(key) and destination.exists():
            raise FileExistsError("immutable original already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._temporary_output_path(destination)
        try:
            yield temporary
            if not temporary.is_file():
                raise FileNotFoundError("published artifact was not created")
            if self._is_original(key) and destination.exists():
                raise FileExistsError("immutable original already exists")
            temporary.replace(destination)
            if self._is_original(key):
                destination.chmod(stat.S_IREAD)
        finally:
            if temporary.is_file():
                temporary.unlink()

    def delivery_for(
        self,
        key: str,
        *,
        filename: str | None = None,
    ) -> ArtifactDelivery:
        del filename
        path = self.path_for(key)
        if not path.is_file():
            raise FileNotFoundError("artifact is missing")
        return ArtifactDelivery.local_file(path)

    def healthcheck(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.W_OK)

    def make_read_only(self, uri: str) -> None:
        path = self.path_for(uri)
        if not path.is_file():
            raise FileNotFoundError("artifact is missing")
        path.chmod(stat.S_IREAD)

    @staticmethod
    def temporary_upload_uri(video_id: str) -> str:
        return temporary_upload_key(video_id)

    @staticmethod
    def original_uri(video_id: str, extension: str) -> str:
        return original_key(video_id, extension)

    @staticmethod
    def proxy_uri(video_id: str) -> str:
        return proxy_key(video_id)

    @staticmethod
    def tracking_proxy_uri(video_id: str) -> str:
        return tracking_proxy_key(video_id)

    @staticmethod
    def thumbnail_uri(video_id: str) -> str:
        return thumbnail_key(video_id)

    @staticmethod
    def export_video_uri(video_id: str, export_id: str) -> str:
        return export_video_key(video_id, export_id)

    @staticmethod
    def export_manifest_uri(video_id: str, export_id: str) -> str:
        return export_manifest_key(video_id, export_id)

    @staticmethod
    def _is_original(key: str) -> bool:
        return PurePosixPath(key).parts[:1] == ("originals",)

    @staticmethod
    def _temporary_output_path(destination: Path) -> Path:
        suffix = destination.suffix
        stem = destination.name[: -len(suffix)] if suffix else destination.name
        return destination.with_name(f".{stem}.{uuid4().hex}.part{suffix}")
