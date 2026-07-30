import subprocess
from pathlib import Path

import cv2
import pytest
from tests.helpers import generate_test_video

from clearframe.media import (
    H264_NVENC,
    LIBX264_FAST,
    PROXY_PROFILE_VERSION,
    MediaError,
    MediaMetadata,
    MediaProcessor,
)


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


class EncoderProbeMediaProcessor(MediaProcessor):
    def __init__(self, *, succeeds: bool) -> None:
        self.ffmpeg_path = Path("ffmpeg")
        self.succeeds = succeeds
        self.commands: list[list[str]] = []

    def _run(
        self,
        arguments: list[str],
        *,
        timeout: int,
        failure_message: str,
    ) -> subprocess.CompletedProcess[str]:
        del timeout
        self.commands.append(arguments)
        if not self.succeeds:
            raise MediaError(failure_message)
        return subprocess.CompletedProcess(arguments, 0, "", "")


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


def test_small_non_h264_proxy_is_not_upscaled(tmp_path: Path) -> None:
    processor = RecordingMediaProcessor(
        metadata(codec="hevc", width=640, height=360)
    )
    destination = tmp_path / "proxy.mp4"

    processor.generate_proxy(
        tmp_path / "source.mp4",
        destination,
        metadata=processor.source_metadata,
    )

    filter_value = processor.commands[0][processor.commands[0].index("-vf") + 1]
    assert "min(1280,iw)" in filter_value
    assert "min(720,ih)" in filter_value
    assert processor.expected_proxy_dimensions(processor.source_metadata) == (640, 360)


def test_proxy_assessment_marks_legacy_upscale_as_stale_but_timeline_safe() -> None:
    source = metadata(width=640, height=360)
    legacy_proxy = metadata(width=1280, height=720)

    assessment = MediaProcessor.assess_proxy(source, legacy_proxy)

    assert PROXY_PROFILE_VERSION == 2
    assert assessment.current is False
    assert assessment.timeline_compatible is True
    assert "proxy upscales the source" in assessment.reasons


def test_proxy_assessment_blocks_track_reuse_when_frame_rate_changes() -> None:
    source = metadata(width=640, height=360)
    mismatched_proxy = MediaMetadata(
        **{
            **source.as_dict(),
            "fps": 24.0,
            "frame_count_estimate": 1_440,
        }
    )

    assessment = MediaProcessor.assess_proxy(source, mismatched_proxy)

    assert assessment.current is False
    assert assessment.timeline_compatible is False


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


@pytest.mark.parametrize("probe_succeeds", [False, True])
def test_nvenc_requires_a_successful_encode_probe(probe_succeeds: bool) -> None:
    processor = EncoderProbeMediaProcessor(succeeds=probe_succeeds)

    processor._h264_nvenc_available = processor._detect_h264_nvenc()

    assert processor.h264_nvenc_available is probe_succeeds
    assert processor.commands
    assert "-f" in processor.commands[0]
    assert "lavfi" in processor.commands[0]
    assert "h264_nvenc" in processor.commands[0]
    expected = (
        (H264_NVENC, LIBX264_FAST)
        if probe_succeeds
        else (LIBX264_FAST,)
    )
    assert processor.export_h264_encoders() == expected


def test_cpu_export_encoder_uses_fast_high_quality_settings() -> None:
    arguments = list(LIBX264_FAST.ffmpeg_arguments)

    assert arguments[arguments.index("-c:v") + 1] == "libx264"
    assert arguments[arguments.index("-preset") + 1] == "veryfast"
    assert arguments[arguments.index("-crf") + 1] == "18"


def test_tracking_proxy_is_smaller_but_frame_aligned(tmp_path: Path) -> None:
    """Track frame indices address the review proxy, so timing must match.

    A tracking proxy with a different frame rate or frame count would make
    propagation land on the wrong moment and redact the wrong region.
    """
    media = MediaProcessor(None)
    review = generate_test_video(
        tmp_path / "review.mp4",
        media,
        duration_seconds=2.0,
    )
    tracking = tmp_path / "tracking.mp4"

    media.generate_tracking_proxy(review, tracking)

    review_meta = media.probe(review)
    tracking_meta = media.probe(tracking)

    assert tracking_meta.fps == pytest.approx(review_meta.fps, abs=0.01)
    assert tracking_meta.duration_ms == pytest.approx(
        review_meta.duration_ms, abs=100
    )
    assert not tracking_meta.audio_present
    assert tracking_meta.width <= min(960, review_meta.width)
    assert tracking_meta.height <= min(540, review_meta.height)

    review_frames = int(
        cv2.VideoCapture(str(review)).get(cv2.CAP_PROP_FRAME_COUNT)
    )
    tracking_frames = int(
        cv2.VideoCapture(str(tracking)).get(cv2.CAP_PROP_FRAME_COUNT)
    )
    assert tracking_frames == review_frames
