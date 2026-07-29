from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from clearframe.pipeline import (
    DetectionContext,
    DetectorUnavailableError,
    OnnxRuntimeYuNetFaceDetector,
    OpenCVYuNetFaceDetector,
)

_STRIDES = (8, 16, 32)
_OUTPUT_NAMES = (
    "cls_8",
    "cls_16",
    "cls_32",
    "obj_8",
    "obj_16",
    "obj_32",
    "bbox_8",
    "bbox_16",
    "bbox_32",
    "kps_8",
    "kps_16",
    "kps_32",
)


@dataclass(frozen=True)
class _StubNodeArgument:
    name: str
    shape: Sequence[int | str | None]
    type: str


def _output_shapes() -> dict[str, tuple[int, int, int]]:
    shapes: dict[str, tuple[int, int, int]] = {}
    for stride in _STRIDES:
        locations = (640 // stride) ** 2
        shapes[f"cls_{stride}"] = (1, locations, 1)
        shapes[f"obj_{stride}"] = (1, locations, 1)
        shapes[f"bbox_{stride}"] = (1, locations, 4)
        shapes[f"kps_{stride}"] = (1, locations, 10)
    return shapes


def _empty_outputs() -> list[NDArray[np.float32]]:
    shapes = _output_shapes()
    return [
        np.zeros(shapes[name], dtype=np.float32)
        for name in _OUTPUT_NAMES
    ]


def _encode_face(
    outputs: list[NDArray[np.float32]],
    *,
    stride: int,
    column: int,
    row: int,
    x1: float,
    y1: float,
    width: float,
    height: float,
    confidence: float,
) -> None:
    index = row * (640 // stride) + column
    stride_index = _STRIDES.index(stride)
    outputs[stride_index][0, index, 0] = confidence**2
    outputs[3 + stride_index][0, index, 0] = 1.0
    outputs[6 + stride_index][0, index] = [
        (x1 + width / 2.0) / stride - column,
        (y1 + height / 2.0) / stride - row,
        log(width / stride),
        log(height / stride),
    ]


class _StubSession:
    def __init__(
        self,
        outputs: Sequence[object],
        *,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self.outputs = outputs
        self.providers = providers
        self.input_spec = _StubNodeArgument(
            name="input",
            shape=(1, 3, 640, 640),
            type="tensor(float)",
        )
        shapes = _output_shapes()
        self.output_specs = [
            _StubNodeArgument(
                name=name,
                shape=shapes[name],
                type="tensor(float)",
            )
            for name in _OUTPUT_NAMES
        ]
        self.run_calls: list[
            tuple[Sequence[str] | None, Mapping[str, NDArray[np.float32]]]
        ] = []

    def get_inputs(self) -> Sequence[_StubNodeArgument]:
        return [self.input_spec]

    def get_outputs(self) -> Sequence[_StubNodeArgument]:
        return self.output_specs

    def get_providers(self) -> Sequence[str]:
        return self.providers

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, NDArray[np.float32]],
    ) -> Sequence[object]:
        copied_feed = {name: values.copy() for name, values in input_feed.items()}
        self.run_calls.append((output_names, copied_feed))
        return self.outputs


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "face_detection_yunet.onnx"
    path.write_bytes(b"controlled test model")
    return path


def test_yunet_onnx_matches_opencv_decode_nms_and_remap(tmp_path: Path) -> None:
    outputs = _empty_outputs()
    _encode_face(
        outputs,
        stride=8,
        column=15,
        row=30,
        x1=80,
        y1=200,
        width=80,
        height=80,
        confidence=0.95,
    )
    _encode_face(
        outputs,
        stride=8,
        column=16,
        row=31,
        x1=84,
        y1=204,
        width=80,
        height=80,
        confidence=0.90,
    )
    _encode_face(
        outputs,
        stride=16,
        column=22,
        row=17,
        x1=320,
        y1=240,
        width=64,
        height=64,
        confidence=0.95,
    )
    _encode_face(
        outputs,
        stride=32,
        column=2,
        row=1,
        x1=40,
        y1=20,
        width=40,
        height=40,
        confidence=0.99,
    )
    session = _StubSession(
        outputs,
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )
    factory_calls: list[tuple[str, tuple[str, ...]]] = []

    def factory(path: str, providers: tuple[str, ...]) -> _StubSession:
        factory_calls.append((path, providers))
        return session

    model_path = _model_file(tmp_path)
    detector = OnnxRuntimeYuNetFaceDetector(
        model_path,
        confidence_threshold=0.6,
        nms_threshold=0.3,
        min_face_size_pixels=16,
        _factory=factory,
        _available_providers=lambda: (
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ),
    )
    context = DetectionContext(frame_index=9, timestamp_ms=360)
    frame = np.full((320, 640, 3), (10, 20, 30), dtype=np.uint8)

    proposals = detector.detect(frame, context)

    assert detector.availability.available
    assert detector.provider == "CUDAExecutionProvider"
    assert detector.device == "cuda"
    assert factory_calls == [
        (
            str(model_path.resolve()),
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
        )
    ]
    output_names, input_feed = session.run_calls[0]
    assert output_names == _OUTPUT_NAMES
    tensor = input_feed["input"]
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor[0, :, 200, 100] == pytest.approx([10, 20, 30])
    assert tensor[0, :, 20, 100] == pytest.approx([0, 0, 0])

    assert [proposal.confidence for proposal in proposals] == pytest.approx(
        [0.95, 0.95]
    )
    expected_boxes = [
        [0.125, 0.125, 0.25, 0.375],
        [0.5, 0.25, 0.6, 0.45],
    ]
    for proposal, expected in zip(proposals, expected_boxes, strict=True):
        assert proposal.bbox.as_list() == pytest.approx(expected)
    assert all(proposal.class_name == "face" for proposal in proposals)
    assert all(proposal.frame_index == 9 for proposal in proposals)
    assert all(proposal.timestamp_ms == 360 for proposal in proposals)


def test_yunet_onnx_clamps_scores_before_geometric_mean(tmp_path: Path) -> None:
    outputs = _empty_outputs()
    stride = 8
    column = 20
    row = 20
    index = row * (640 // stride) + column
    outputs[0][0, index, 0] = 1.5
    outputs[3][0, index, 0] = 0.81
    outputs[6][0, index] = [0, 0, log(4), log(4)]
    session = _StubSession(outputs)
    detector = OnnxRuntimeYuNetFaceDetector(
        _model_file(tmp_path),
        confidence_threshold=0.6,
        min_face_size_pixels=8,
        _factory=lambda _path, _providers: session,
        _available_providers=lambda: ("CPUExecutionProvider",),
    )

    proposals = detector.detect(
        np.zeros((640, 640, 4), dtype=np.uint8),
        DetectionContext(frame_index=0, timestamp_ms=0),
    )

    assert len(proposals) == 1
    assert proposals[0].confidence == pytest.approx(0.9)
    assert proposals[0].bbox.as_list() == pytest.approx(
        [144 / 640, 144 / 640, 176 / 640, 176 / 640]
    )
    assert detector.provider == "CPUExecutionProvider"
    assert detector.device == "cpu"


def test_yunet_onnx_matches_opencv_on_bundled_graph() -> None:
    model_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "models"
        / "weights"
        / "face_detection_yunet_2023mar.onnx"
    )
    frame = np.random.default_rng(9).integers(
        0,
        256,
        (640, 640, 3),
        dtype=np.uint8,
    )
    context = DetectionContext(frame_index=0, timestamp_ms=0)
    onnx_detector = OnnxRuntimeYuNetFaceDetector(
        model_path,
        confidence_threshold=0.1,
        nms_threshold=0.3,
        top_k=100,
        min_face_size_pixels=1,
        _available_providers=lambda: ("CPUExecutionProvider",),
    )
    opencv_detector = OpenCVYuNetFaceDetector(
        model_path,
        confidence_threshold=0.1,
        nms_threshold=0.3,
        top_k=100,
        min_face_size_pixels=1,
    )

    onnx_proposals = onnx_detector.detect(frame, context)
    opencv_proposals = opencv_detector.detect(frame, context)

    assert len(onnx_proposals) == len(opencv_proposals)
    assert len(onnx_proposals) > 0
    for onnx_proposal, opencv_proposal in zip(
        onnx_proposals,
        opencv_proposals,
        strict=True,
    ):
        assert onnx_proposal.confidence == pytest.approx(
            opencv_proposal.confidence,
            abs=2e-5,
        )
        assert onnx_proposal.bbox.as_list() == pytest.approx(
            opencv_proposal.bbox.as_list(),
            abs=2e-5,
        )


def test_yunet_onnx_reports_unavailable_models_and_contracts(tmp_path: Path) -> None:
    missing = OnnxRuntimeYuNetFaceDetector(tmp_path / "missing.onnx")
    wrong_format = tmp_path / "model.pb"
    wrong_format.write_bytes(b"not ONNX")
    wrong = OnnxRuntimeYuNetFaceDetector(wrong_format)
    incompatible_session = _StubSession(_empty_outputs())
    incompatible_session.input_spec = _StubNodeArgument(
        name="input",
        shape=(1, 3, 320, 320),
        type="tensor(float)",
    )
    incompatible = OnnxRuntimeYuNetFaceDetector(
        _model_file(tmp_path),
        _factory=lambda _path, _providers: incompatible_session,
        _available_providers=lambda: ("CPUExecutionProvider",),
    )

    for detector in (missing, wrong, incompatible):
        assert not detector.availability.available
        assert detector.availability.reason is not None
        with pytest.raises(DetectorUnavailableError):
            detector.detect(
                np.zeros((100, 100, 3), dtype=np.uint8),
                DetectionContext(frame_index=0, timestamp_ms=0),
            )
    assert incompatible.availability.reason is not None
    assert "input shape" in incompatible.availability.reason


def test_yunet_onnx_rejects_invalid_configuration(tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)

    with pytest.raises(ValueError, match="confidence_threshold"):
        OnnxRuntimeYuNetFaceDetector(model_path, confidence_threshold=-0.1)
    with pytest.raises(ValueError, match="nms_threshold"):
        OnnxRuntimeYuNetFaceDetector(model_path, nms_threshold=1.1)
    with pytest.raises(ValueError, match="top_k"):
        OnnxRuntimeYuNetFaceDetector(model_path, top_k=0)
    with pytest.raises(ValueError, match="min_face_size_pixels"):
        OnnxRuntimeYuNetFaceDetector(model_path, min_face_size_pixels=0)


def test_yunet_onnx_rejects_malformed_runtime_output(tmp_path: Path) -> None:
    outputs: list[object] = list(_empty_outputs())
    outputs[0] = np.zeros((1, 5, 1), dtype=np.float32)
    detector = OnnxRuntimeYuNetFaceDetector(
        _model_file(tmp_path),
        _factory=lambda _path, _providers: _StubSession(outputs),
        _available_providers=lambda: ("CPUExecutionProvider",),
    )

    with pytest.raises(RuntimeError, match="malformed output cls_8"):
        detector.detect(
            np.zeros((100, 100), dtype=np.uint8),
            DetectionContext(frame_index=0, timestamp_ms=0),
        )
