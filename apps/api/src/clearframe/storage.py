import os
import re
import stat
import unicodedata
from pathlib import Path, PurePosixPath


class StorageSecurityError(ValueError):
    pass


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
        if PurePosixPath(uri).parts[:1] == ("originals",):
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

    def make_read_only(self, uri: str) -> None:
        path = self.path_for(uri)
        if not path.is_file():
            raise FileNotFoundError("artifact is missing")
        path.chmod(stat.S_IREAD)

    @staticmethod
    def temporary_upload_uri(video_id: str) -> str:
        return f"tmp/uploads/{video_id}.upload"

    @staticmethod
    def original_uri(video_id: str, extension: str) -> str:
        clean_extension = extension if extension.startswith(".") else f".{extension}"
        return f"originals/{video_id}/original{clean_extension.lower()}"

    @staticmethod
    def proxy_uri(video_id: str) -> str:
        return f"proxies/{video_id}/proxy.mp4"

    @staticmethod
    def thumbnail_uri(video_id: str) -> str:
        return f"thumbnails/{video_id}/poster.jpg"
