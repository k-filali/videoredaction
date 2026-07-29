import hashlib
import re
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


@dataclass(frozen=True, slots=True)
class H264Encoder:
    name: str
    ffmpeg_arguments: tuple[str, ...]
    hardware_accelerated: bool


LIBX264_FAST = H264Encoder(
    name="libx264",
    ffmpeg_arguments=(
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
    ),
    hardware_accelerated=False,
)
H264_NVENC = H264Encoder(
    name="h264_nvenc",
    ffmpeg_arguments=(
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "19",
        "-b:v",
        "0",
    ),
    hardware_accelerated=True,
)


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
    def __init__(
        self,
        ffmpeg_path: Path | None = None,
        *,
        max_duration_ms: int = 240 * 60 * 1000,
        max_video_pixels: int = 9_000_000,
        max_video_dimension: int = 8192,
        max_video_fps: float = 120.0,
    ) -> None:
        self.ffmpeg_path = self._resolve_ffmpeg(ffmpeg_path)
        self.ffmpeg_version = self._detect_version()
        self.max_duration_ms = max_duration_ms
        self.max_video_pixels = max_video_pixels
        self.max_video_dimension = max_video_dimension
        self.max_video_fps = max_video_fps
        if (
            max_duration_ms <= 0
            or max_video_pixels <= 0
            or max_video_dimension <= 0
            or max_video_fps <= 0
        ):
            raise ValueError("media limits must be positive")
        self._h264_nvenc_available = self._detect_h264_nvenc()

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

    def _detect_version(self) -> str:
        result = self._run(
            ["-version"],
            timeout=10,
            failure_message="FFmpeg version could not be read",
        )
        output = result.stdout or result.stderr
        first_line = output.splitlines()[0].strip() if output.splitlines() else ""
        if not first_line:
            raise MediaError("FFmpeg version could not be read")
        return first_line[:256]

    def _run(
        self,
        arguments: list[str],
        *,
        timeout: int,
        failure_message: str,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                [
                    str(self.ffmpeg_path),
                    "-nostdin",
                    "-max_alloc",
                    str(256 * 1024 * 1024),
                    *arguments,
                ],
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

    def _detect_h264_nvenc(self) -> bool:
        try:
            self._run(
                [
                    "-hide_banner",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=256x256:r=1:d=0.1",
                    "-frames:v",
                    "1",
                    "-an",
                    *H264_NVENC.ffmpeg_arguments,
                    "-pix_fmt",
                    "yuv420p",
                    "-f",
                    "null",
                    "-",
                ],
                timeout=20,
                failure_message="NVIDIA H.264 encoder probe failed",
            )
        except MediaError:
            return False
        return True

    @property
    def h264_nvenc_available(self) -> bool:
        return self._h264_nvenc_available

    def export_h264_encoders(self) -> tuple[H264Encoder, ...]:
        if self.h264_nvenc_available:
            return H264_NVENC, LIBX264_FAST
        return (LIBX264_FAST,)

    def probe(self, path: Path) -> MediaMetadata:
        inspection = self._run(
            [
                "-hide_banner",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-t",
                "0",
                "-f",
                "null",
                "-",
            ],
            timeout=90,
            failure_message="video could not be decoded",
        )
        probe_output = inspection.stderr
        duration_match = re.search(
            r"Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)",
            probe_output,
        )
        video_line = next(
            (
                line
                for line in probe_output.splitlines()
                if "Stream #" in line and " Video:" in line
            ),
            "",
        )
        size_match = re.search(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)", video_line)
        fps_match = re.search(r"(\d+(?:\.\d+)?)\s+fps\b", video_line)
        codec_match = re.search(r"Video:\s*([^,\s]+)", video_line)
        if duration_match is None or size_match is None or fps_match is None:
            raise MediaError("video metadata is invalid")

        hours, minutes, seconds = duration_match.groups()
        duration_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        width, height = (int(value) for value in size_match.groups())
        fps = float(fps_match.group(1))
        if width <= 0 or height <= 0:
            raise MediaError("video dimensions are invalid")
        if fps <= 0 or duration_seconds <= 0:
            raise MediaError("video duration or frame rate is invalid")
        duration_ms = round(duration_seconds * 1000)
        if duration_ms > self.max_duration_ms:
            raise MediaError("video duration exceeds the configured limit")
        if width > self.max_video_dimension or height > self.max_video_dimension:
            raise MediaError("video dimensions exceed the configured limit")
        if width * height > self.max_video_pixels:
            raise MediaError("video pixel count exceeds the configured limit")
        if fps > self.max_video_fps:
            raise MediaError("video frame rate exceeds the configured limit")

        self._run(
            [
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
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
            duration_ms=duration_ms,
            fps=fps,
            width=width,
            height=height,
            codec=codec_match.group(1) if codec_match else "unknown",
            audio_present=any(
                "Stream #" in line and " Audio:" in line
                for line in probe_output.splitlines()
            ),
            frame_count_estimate=max(1, round(duration_seconds * fps)),
            ffmpeg_version=self.ffmpeg_version,
        )

    @staticmethod
    def _can_remux_proxy(metadata: MediaMetadata) -> bool:
        return (
            metadata.codec.casefold() == "h264"
            and metadata.width <= 1280
            and metadata.height <= 720
        )

    @staticmethod
    def _proxy_remux_arguments(source: Path, destination: Path) -> list[str]:
        return [
            "-y",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(destination),
        ]

    @staticmethod
    def _proxy_transcode_arguments(source: Path, destination: Path) -> list[str]:
        return [
            "-y",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
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
        ]

    def generate_proxy(
        self,
        source: Path,
        destination: Path,
        *,
        metadata: MediaMetadata | None = None,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_metadata = metadata or self.probe(source)
        if self._can_remux_proxy(source_metadata):
            try:
                self._run(
                    self._proxy_remux_arguments(source, destination),
                    timeout=14400,
                    failure_message="proxy remux failed",
                )
                self.probe(destination)
                return
            except MediaError:
                destination.unlink(missing_ok=True)

        self._run(
            self._proxy_transcode_arguments(source, destination),
            timeout=14400,
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
                "-protocol_whitelist",
                "file,pipe",
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
