from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import sleep

import numpy as np
import pytest
from numpy.typing import NDArray

from clearframe.pipeline import (
    DetectionContext,
    DetectorUnavailableError,
    OnnxRuntimeYoloV9PlateDetector,
)


@dataclass(frozen=True)
class _StubNodeArgument:
    name: str
    shape: Sequence[int | str | None]
    type: str


class _StubSession:
    def __init__(
        self,
        output: object,
        *,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self.output = output
        self.providers = providers
        self.input_spec = _StubNodeArgument(
            name="images",
            shape=(1, 3, 384, 384),
            type="tensor(float)",
        )
        self.output_spec = _StubNodeArgument(
            name="detections",
            shape=("detections", 7),
            type="tensor(float)",
        )
        self.run_calls: list[
            tuple[Sequence[str] | None, Mapping[str, NDArray[np.float32]]]
        ] = []

    def get_inputs(self) -> Sequence[_StubNodeArgument]:
        return [self.input_spec]

    def get_outputs(self) -> Sequence[_StubNodeArgument]:
        return [self.output_spec]

    def get_providers(self) -> Sequence[str]:
        return self.providers

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, NDArray[np.float32]],
    ) -> Sequence[object]:
        copied_feed = {name: values.copy() for name, values in input_feed.items()}
        self.run_calls.append((output_names, copied_feed))
        return [self.output]


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "regional_plate_yolov9.onnx"
    path.write_bytes(b"controlled test model")
    return path


def test_yolov9_preprocesses_decodes_filters_and_sorts(tmp_path: Path) -> None:
    detections = np.asarray(
        [
            [0, 96, 144, 288, 192, 0, 0.95],
            [0, 48, 120, 144, 158.4, 0, 0.95],
            [0, -20, 70, 420, 330, 0, 0.99],
            [1, 96, 144, 288, 192, 0, 0.99],
            [0, 96, 144, 288, 192, 1, 0.99],
            [0, 96, 144, 288, 192, 0, 0.49],
            [0, 100, 100, 103, 110, 0, 0.99],
            [0, np.nan, 144, 288, 192, 0, 0.99],
            [0, 200, 144, 100, 192, 0, 0.99],
            [0, 96, 144, 288, 192, 0, 1.01],
        ],
        dtype=np.float32,
    )
    session = _StubSession(detections)
    factory_calls: list[tuple[str, tuple[str, ...]]] = []

    def factory(path: str, providers: tuple[str, ...]) -> _StubSession:
        factory_calls.append((path, providers))
        return session

    model_path = _model_file(tmp_path)
    detector = OnnxRuntimeYoloV9PlateDetector(
        model_path,
        confidence_threshold=0.5,
        min_plate_size_pixels=8,
        _factory=factory,
        _available_providers=lambda: ("CPUExecutionProvider",),
    )
    context = DetectionContext(frame_index=15, timestamp_ms=600)
    frame = np.full((200, 400, 3), (10, 20, 30), dtype=np.uint8)

    proposals = detector.detect(frame, context)

    assert detector.availability.available
    assert factory_calls == [
        (str(model_path.resolve()), ("CPUExecutionProvider",))
    ]
    output_names, input_feed = session.run_calls[0]
    assert output_names is None
    tensor = input_feed["images"]
    assert tensor.shape == (1, 3, 384, 384)
    assert tensor.dtype == np.float32
    assert tensor[0, :, 100, 100] == pytest.approx(
        np.asarray([30, 20, 10], dtype=np.float32) / 255.0
    )
    assert tensor[0, :, 0, 0] == pytest.approx(
        np.asarray([114, 114, 114], dtype=np.float32) / 255.0
    )

    assert [proposal.confidence for proposal in proposals] == pytest.approx(
        [0.99, 0.95, 0.95]
    )
    expected_boxes = [
        [0.0, 0.0, 1.0, 1.0],
        [0.125, 0.125, 0.375, 0.325],
        [0.25, 0.25, 0.75, 0.5],
    ]
    for proposal, expected in zip(proposals, expected_boxes, strict=True):
        assert proposal.bbox.as_list() == pytest.approx(expected)
    assert all(proposal.class_name == "license_plate" for proposal in proposals)
    assert all(proposal.frame_index == 15 for proposal in proposals)
    assert all(proposal.timestamp_ms == 600 for proposal in proposals)
    assert detector.provider == "CPUExecutionProvider"
    assert detector.device == "cpu"


def test_yolov9_prefers_cuda_and_reports_runtime_fallback(tmp_path: Path) -> None:
    output = np.empty((0, 7), dtype=np.float32)
    requested: list[tuple[str, ...]] = []

    def cuda_factory(
        _path: str,
        providers: tuple[str, ...],
    ) -> _StubSession:
        requested.append(providers)
        return _StubSession(
            output,
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        )

    cuda_detector = OnnxRuntimeYoloV9PlateDetector(
        _model_file(tmp_path),
        _factory=cuda_factory,
        _available_providers=lambda: (
            "AzureExecutionProvider",
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ),
    )
    fallback_detector = OnnxRuntimeYoloV9PlateDetector(
        _model_file(tmp_path),
        _factory=lambda _path, _providers: _StubSession(output),
        _available_providers=lambda: (
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ),
    )
    fallback_requests: list[tuple[str, ...]] = []

    def failing_cuda_factory(
        _path: str,
        providers: tuple[str, ...],
    ) -> _StubSession:
        fallback_requests.append(providers)
        if providers[0] == "CUDAExecutionProvider":
            raise RuntimeError("CUDA runtime unavailable")
        return _StubSession(output)

    retry_detector = OnnxRuntimeYoloV9PlateDetector(
        _model_file(tmp_path),
        _factory=failing_cuda_factory,
        _available_providers=lambda: (
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ),
    )

    assert requested == [
        ("CUDAExecutionProvider", "CPUExecutionProvider")
    ]
    assert cuda_detector.provider == "CUDAExecutionProvider"
    assert cuda_detector.device == "cuda"
    assert fallback_detector.provider == "CPUExecutionProvider"
    assert fallback_detector.device == "cpu"
    assert fallback_requests == [
        ("CUDAExecutionProvider", "CPUExecutionProvider"),
        ("CPUExecutionProvider",),
    ]
    assert retry_detector.availability.available
    assert retry_detector.provider == "CPUExecutionProvider"
    assert retry_detector.device == "cpu"


def test_yolov9_reports_unavailable_supported_provider(tmp_path: Path) -> None:
    detector = OnnxRuntimeYoloV9PlateDetector(
        _model_file(tmp_path),
        _available_providers=lambda: ("AzureExecutionProvider",),
    )

    assert not detector.availability.available
    assert detector.provider is None
    assert detector.device is None
    assert detector.availability.reason is not None
    assert "no supported CUDA or CPU" in detector.availability.reason


def test_yolov9_reports_unavailable_model_paths(tmp_path: Path) -> None:
    missing_detector = OnnxRuntimeYoloV9PlateDetector(tmp_path / "missing.onnx")
    wrong_format = tmp_path / "model.pb"
    wrong_format.write_bytes(b"not ONNX")
    wrong_format_detector = OnnxRuntimeYoloV9PlateDetector(wrong_format)
    empty_model = tmp_path / "empty.onnx"
    empty_model.touch()
    empty_detector = OnnxRuntimeYoloV9PlateDetector(empty_model)

    for detector in (missing_detector, wrong_format_detector, empty_detector):
        assert not detector.availability.available
        assert detector.availability.reason is not None
        with pytest.raises(DetectorUnavailableError):
            detector.detect(
                np.zeros((100, 100, 3), dtype=np.uint8),
                DetectionContext(frame_index=0, timestamp_ms=0),
            )


def test_yolov9_reports_model_load_and_contract_failures(tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)

    def corrupt_factory(
        _path: str,
        _providers: tuple[str, ...],
    ) -> _StubSession:
        raise RuntimeError("invalid protobuf")

    corrupt_detector = OnnxRuntimeYoloV9PlateDetector(
        model_path,
        _factory=corrupt_factory,
    )
    incompatible_session = _StubSession(np.empty((0, 7), dtype=np.float32))
    incompatible_session.input_spec = _StubNodeArgument(
        name="images",
        shape=(1, 3, "height", "width"),
        type="tensor(float)",
    )
    incompatible_detector = OnnxRuntimeYoloV9PlateDetector(
        model_path,
        _factory=lambda _path, _providers: incompatible_session,
    )

    assert not corrupt_detector.availability.available
    assert corrupt_detector.availability.reason is not None
    assert "invalid protobuf" in corrupt_detector.availability.reason
    assert not incompatible_detector.availability.available
    assert incompatible_detector.availability.reason is not None
    assert "input shape" in incompatible_detector.availability.reason


def test_yolov9_accepts_higher_resolution_input(tmp_path: Path) -> None:
    # A 640x640 export must load and letterbox to its own input size, so a
    # higher-recall model can be swapped in via the registry alone.
    session = _StubSession(np.empty((0, 7), dtype=np.float32))
    session.input_spec = _StubNodeArgument(
        name="images",
        shape=(1, 3, 640, 640),
        type="tensor(float)",
    )
    detector = OnnxRuntimeYoloV9PlateDetector(
        _model_file(tmp_path),
        _factory=lambda _path, _providers: session,
        _available_providers=lambda: ("CPUExecutionProvider",),
    )
    frame = np.full((200, 400, 3), (10, 20, 30), dtype=np.uint8)

    proposals = detector.detect(frame, DetectionContext(frame_index=0, timestamp_ms=0))

    assert detector.availability.available
    assert proposals == []
    _output_names, input_feed = session.run_calls[0]
    assert input_feed["images"].shape == (1, 3, 640, 640)


def test_yolov9_rejects_invalid_configuration(tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)

    with pytest.raises(ValueError, match="confidence_threshold"):
        OnnxRuntimeYoloV9PlateDetector(model_path, confidence_threshold=-0.1)
    with pytest.raises(ValueError, match="min_plate_size_pixels"):
        OnnxRuntimeYoloV9PlateDetector(model_path, min_plate_size_pixels=0)


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((1, 2, 7), dtype=np.float32),
        np.zeros((5, 6), dtype=np.float32),
        [["not", "numeric"]],
    ],
)
def test_yolov9_rejects_malformed_runtime_output(
    tmp_path: Path,
    output: object,
) -> None:
    detector = OnnxRuntimeYoloV9PlateDetector(
        _model_file(tmp_path),
        _factory=lambda _path, _providers: _StubSession(output),
    )

    with pytest.raises(RuntimeError, match="malformed detection output"):
        detector.detect(
            np.zeros((100, 100, 4), dtype=np.uint8),
            DetectionContext(frame_index=0, timestamp_ms=0),
        )


class _ConcurrentSession(_StubSession):
    def __init__(
        self,
        *,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        super().__init__(
            np.empty((0, 7), dtype=np.float32),
            providers=providers,
        )
        self.active_calls = 0
        self.maximum_active_calls = 0
        self.guard = Lock()

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, NDArray[np.float32]],
    ) -> Sequence[object]:
        with self.guard:
            self.active_calls += 1
            self.maximum_active_calls = max(
                self.maximum_active_calls,
                self.active_calls,
            )
        sleep(0.01)
        with self.guard:
            self.active_calls -= 1
        return super().run(output_names, input_feed)


def test_yolov9_serializes_session_inference(tmp_path: Path) -> None:
    session = _ConcurrentSession()
    detector = OnnxRuntimeYoloV9PlateDetector(
        _model_file(tmp_path),
        _factory=lambda _path, _providers: session,
    )
    frame = np.zeros((100, 200), dtype=np.uint8)
    context = DetectionContext(frame_index=0, timestamp_ms=0)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _index: detector.detect(frame, context),
                range(8),
            )
        )

    assert results == [[]] * 8
    assert session.maximum_active_calls == 1
    assert detector.max_concurrency == 1


def test_yolov9_serializes_cuda_session_inference(tmp_path: Path) -> None:
    session = _ConcurrentSession(
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )
    detector = OnnxRuntimeYoloV9PlateDetector(
        _model_file(tmp_path),
        _factory=lambda _path, _providers: session,
        _available_providers=lambda: (
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ),
    )
    frame = np.zeros((100, 200), dtype=np.uint8)
    context = DetectionContext(frame_index=0, timestamp_ms=0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _index: detector.detect(frame, context),
                range(12),
            )
        )

    assert results == [[]] * 12
    assert session.maximum_active_calls == 1
    assert detector.max_concurrency == 1


def _overlay_and_plate_detections() -> NDArray[np.float32]:
    """Two boxes in 384x384 model space for a 400x200 frame.

    Letterbox scale is 0.96 with pad_y 96, so these map back to a plate-shaped
    box (96x48 px, 2.0:1) and a banner-shaped box (360x40 px, 9.0:1) that
    stands in for a burned-in camera overlay.
    """
    return np.asarray(
        [
            [0, 96.0, 144.0, 188.16, 190.08, 0, 0.9],
            [0, 9.6, 100.8, 355.2, 139.2, 0, 0.9],
        ],
        dtype=np.float32,
    )


def test_yolov9_rejects_overlay_shaped_boxes(tmp_path: Path) -> None:
    session = _StubSession(_overlay_and_plate_detections())
    detector = OnnxRuntimeYoloV9PlateDetector(
        _model_file(tmp_path),
        confidence_threshold=0.5,
        min_plate_size_pixels=8,
        max_aspect_ratio=3.5,
        _factory=lambda _path, _providers: session,
        _available_providers=lambda: ("CPUExecutionProvider",),
    )
    frame = np.full((200, 400, 3), (10, 20, 30), dtype=np.uint8)

    proposals = detector.detect(frame, DetectionContext(frame_index=0, timestamp_ms=0))

    assert len(proposals) == 1
    box = proposals[0].bbox
    aspect = ((box.x2 - box.x1) * 400) / ((box.y2 - box.y1) * 200)
    assert aspect == pytest.approx(2.0, abs=0.05)


def test_yolov9_keeps_wide_boxes_without_an_aspect_limit(tmp_path: Path) -> None:
    session = _StubSession(_overlay_and_plate_detections())
    detector = OnnxRuntimeYoloV9PlateDetector(
        _model_file(tmp_path),
        confidence_threshold=0.5,
        min_plate_size_pixels=8,
        _factory=lambda _path, _providers: session,
        _available_providers=lambda: ("CPUExecutionProvider",),
    )
    frame = np.full((200, 400, 3), (10, 20, 30), dtype=np.uint8)

    proposals = detector.detect(frame, DetectionContext(frame_index=0, timestamp_ms=0))

    assert len(proposals) == 2


def test_yolov9_rejects_invalid_aspect_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_aspect_ratio must be positive"):
        OnnxRuntimeYoloV9PlateDetector(
            _model_file(tmp_path),
            max_aspect_ratio=0.0,
            _available_providers=lambda: ("CPUExecutionProvider",),
        )
