import subprocess
from pathlib import Path

import pytest

from clearframe.media import MediaError, MediaMetadata, MediaProcessor


def metadata(*, codec: str = "h264", width: int = 1280, height: int = 720) -> MediaMetadata:
    return MediaMetadata(
        duration_ms=60_000,
        fps=30.0,
        width=width,
        height=height,
        codec=codec,
        audio_present=True,
        frame_count_estimate=1_800,
        ffmpeg_version="ffmpeg test",
    )


class RecordingMediaProcessor(MediaProcessor):
    def __init__(
        self,
        source_metadata: MediaMetadata,
        *,
        fail_remux: bool = False,
    ) -> None:
        self.ffmpeg_path = Path("ffmpeg")
        self.source_metadata = source_metadata
        self.fail_remux = fail_remux
        self.commands: list[list[str]] = []
        self.probed_paths: list[Path] = []
        self.destination_existed_before_run: list[bool] = []

    def _run(
        self,
        arguments: list[str],
        *,
        timeout: int,
        failure_message: str,
    ) -> subprocess.CompletedProcess[str]:
        del timeout, failure_message
        self.commands.append(arguments)
        destination = Path(arguments[-1])
        self.destination_existed_before_run.append(destination.exists())
        video_codec = arguments[arguments.index("-c:v") + 1]
        if self.fail_remux and video_codec == "copy":
            destination.write_bytes(b"partial")
            raise MediaError("remux failed")
        destination.write_bytes(b"complete")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def probe(self, path: Path) -> MediaMetadata:
        self.probed_paths.append(path)
        return self.source_metadata


def video_codec(arguments: list[str]) -> str:
    return arguments[arguments.index("-c:v") + 1]


def test_h264_proxy_uses_stream_copy_with_known_metadata(tmp_path: Path) -> None:
    processor = RecordingMediaProcessor(metadata())
    source = tmp_path / "source.mp4"
    destination = tmp_path / "proxy.mp4"

    processor.generate_proxy(source, destination, metadata=processor.source_metadata)

    assert len(processor.commands) == 1
    command = processor.commands[0]
    assert video_codec(command) == "copy"
    assert "-vf" not in command
    assert command[command.index("-c:a") + 1] == "aac"
    assert processor.probed_paths == [destination]


@pytest.mark.parametrize(
    ("codec", "width", "height"),
    [
        ("hevc", 1280, 720),
        ("h264", 1920, 1080),
        ("h264", 1280, 721),
    ],
)
def test_proxy_transcodes_when_stream_copy_is_ineligible(
    tmp_path: Path,
    codec: str,
    width: int,
    height: int,
) -> None:
    processor = RecordingMediaProcessor(
        metadata(codec=codec, width=width, height=height)
    )
    destination = tmp_path / "proxy.mp4"

    processor.generate_proxy(
        tmp_path / "source.mp4",
        destination,
        metadata=processor.source_metadata,
    )

    assert len(processor.commands) == 1
    command = processor.commands[0]
    assert video_codec(command) == "libx264"
    assert "-vf" in command
    assert processor.probed_paths == [destination]


def test_failed_remux_cleans_partial_output_and_transcodes(tmp_path: Path) -> None:
    processor = RecordingMediaProcessor(metadata(), fail_remux=True)
    destination = tmp_path / "proxy.mp4"

    processor.generate_proxy(
        tmp_path / "source.mp4",
        destination,
        metadata=processor.source_metadata,
    )

    assert [video_codec(command) for command in processor.commands] == ["copy", "libx264"]
    assert processor.destination_existed_before_run == [False, False]
    assert destination.read_bytes() == b"complete"
    assert processor.probed_paths == [destination]


def test_direct_generate_proxy_probes_source_for_backward_compatibility(
    tmp_path: Path,
) -> None:
    processor = RecordingMediaProcessor(metadata())
    source = tmp_path / "source.mp4"
    destination = tmp_path / "proxy.mp4"

    processor.generate_proxy(source, destination)

    assert video_codec(processor.commands[0]) == "copy"
    assert processor.probed_paths == [source, destination]
