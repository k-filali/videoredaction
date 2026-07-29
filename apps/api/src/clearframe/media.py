import hashlib
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg  # type: ignore[import-untyped]


class MediaError(ValueError):
    pass


class UnsupportedMediaError(MediaError):
    pass


@dataclass(frozen=True, slots=True)
class MediaKind:
    container: str
    extension: str
    content_type: str


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    duration_ms: int
    fps: float
    width: int
    height: int
    codec: str
    audio_present: bool
    frame_count_estimate: int
    ffmpeg_version: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sniff_media(path: Path) -> MediaKind:
    with path.open("rb") as stream:
        header = stream.read(64)

    if len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12]
        if brand == b"qt  ":
            return MediaKind("quicktime", ".mov", "video/quicktime")
        return MediaKind("mp4", ".mp4", "video/mp4")
    if header.startswith(b"\x1aE\xdf\xa3"):
        if b"webm" in header.lower():
            return MediaKind("webm", ".webm", "video/webm")
        return MediaKind("matroska", ".mkv", "video/x-matroska")
    if len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"AVI ":
        return MediaKind("avi", ".avi", "video/x-msvideo")
    raise UnsupportedMediaError("unsupported or unrecognized video container")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class MediaProcessor:
    def __init__(self, ffmpeg_path: Path | None = None) -> None:
        self.ffmpeg_path = self._resolve_ffmpeg(ffmpeg_path)

    @staticmethod
    def _resolve_ffmpeg(configured_path: Path | None) -> Path:
        if configured_path is not None:
            resolved = configured_path.resolve()
            if not resolved.is_file():
                raise MediaError("configured FFmpeg executable was not found")
            return resolved
        system_path = shutil.which("ffmpeg")
        if system_path:
            return Path(system_path).resolve()
        return Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()

    def _run(
        self,
        arguments: list[str],
        *,
        timeout: int,
        failure_message: str,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                [str(self.ffmpeg_path), *arguments],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MediaError(failure_message) from exc
        if result.returncode != 0:
            raise MediaError(failure_message)
        return result

    def _has_audio(self, path: Path) -> bool:
        try:
            self._run(
                [
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:a:0",
                    "-t",
                    "0.05",
                    "-f",
                    "null",
                    "-",
                ],
                timeout=30,
                failure_message="audio stream not found",
            )
        except MediaError:
            return False
        return True

    def probe(self, path: Path) -> MediaMetadata:
        try:
            reader = imageio_ffmpeg.read_frames(
                str(path),
                output_params=["-frames:v", "1"],
            )
            metadata = next(reader)
            first_frame = next(reader)
            reader.close()
        except (OSError, RuntimeError, StopIteration, ValueError) as exc:
            raise MediaError("video could not be decoded") from exc

        source_size = metadata.get("source_size") or metadata.get("size")
        if (
            not isinstance(source_size, tuple)
            or len(source_size) != 2
            or not all(isinstance(value, int) and value > 0 for value in source_size)
        ):
            raise MediaError("video dimensions are invalid")
        width, height = source_size
        fps = float(metadata.get("fps") or 0.0)
        duration_seconds = float(metadata.get("duration") or 0.0)
        if fps <= 0 or duration_seconds <= 0 or not first_frame:
            raise MediaError("video duration or frame rate is invalid")

        self._run(
            [
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-t",
                f"{min(duration_seconds, 3.0):.3f}",
                "-f",
                "null",
                "-",
            ],
            timeout=90,
            failure_message="video validation failed",
        )
        return MediaMetadata(
            duration_ms=round(duration_seconds * 1000),
            fps=fps,
            width=width,
            height=height,
            codec=str(metadata.get("codec") or "unknown"),
            audio_present=self._has_audio(path),
            frame_count_estimate=max(1, round(duration_seconds * fps)),
            ffmpeg_version=str(metadata.get("ffmpeg_version") or "unknown"),
        )

    def generate_proxy(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            timeout=1800,
            failure_message="proxy generation failed",
        )
        self.probe(destination)

    def generate_thumbnail(
        self,
        source: Path,
        destination: Path,
        duration_ms: int,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        seek_seconds = min(max(duration_ms / 2000, 0.05), 3.0)
        self._run(
            [
                "-y",
                "-v",
                "error",
                "-ss",
                f"{seek_seconds:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                "scale=640:-2",
                "-q:v",
                "3",
                str(destination),
            ],
            timeout=120,
            failure_message="thumbnail generation failed",
        )
