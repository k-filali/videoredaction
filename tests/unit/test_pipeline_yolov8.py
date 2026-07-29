from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import pytest

from clearframe.pipeline import (
    DetectionContext,
    DetectorUnavailableError,
    OpenCVYoloV8PlateDetector,
)
from clearframe.pipeline.detection import Frame


class _StubNetwork:
    def __init__(self, output: object, *, is_empty: bool = False) -> None:
        self.output = output
        self.is_empty = is_empty
        self.backend: int | None = None
        self.target: int | None = None
        self.blobs: list[np.ndarray] = []

    def empty(self) -> bool:
        return self.is_empty

    def setPreferableBackend(self, backend_id: int) -> None:
        self.backend = backend_id

    def setPreferableTarget(self, target_id: int) -> None:
        self.target = target_id

    def setInput(self, blob: object) -> None:
        self.blobs.append(np.asarray(blob))

    def forward(self) -> object:
        return self.output


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "regional_plate_yolov8.onnx"
    path.write_bytes(b"controlled test model")
    return path


def _one_class_predictions() -> np.ndarray:
    return np.asarray(
        [
            [240.0, 252.5, 160.0, 45.0, 0.95],
            [245.0, 254.0, 160.0, 45.0, 0.90],
            [550.0, 415.0, 100.0, 50.0, 0.97],
            [400.0, 300.0, 80.0, 40.0, 0.20],
            [100.0, 200.0, 2.0, 2.0, 0.99],
            [np.nan, 200.0, 40.0, 20.0, 0.99],
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize("layout", ["boxes_first", "channels_first"])
def test_yolov8_decodes_common_exports_and_applies_nms(
    tmp_path: Path,
    layout: Literal["boxes_first", "channels_first"],
) -> None:
    predictions = _one_class_predictions()
    output = predictions[np.newaxis, :, :]
    if layout == "channels_first":
        output = predictions.T[np.newaxis, :, :]
    network = _StubNetwork(output)
    model_path = _model_file(tmp_path)
    factory_paths: list[str] = []

    def factory(path: str) -> _StubNetwork:
        factory_paths.append(path)
        return network

    detector = OpenCVYoloV8PlateDetector(
        model_path,
        confidence_threshold=0.5,
        nms_threshold=0.4,
        min_plate_size_pixels=10,
        _factory=factory,
    )
    context = DetectionContext(frame_index=12, timestamp_ms=400)

    proposals = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8), context)

    assert detector.availability.available
    assert factory_paths == [str(model_path.resolve())]
    assert network.backend == cv2.dnn.DNN_BACKEND_OPENCV
    assert network.target == cv2.dnn.DNN_TARGET_CPU
    assert network.blobs[0].shape == (1, 3, 640, 640)
    assert network.blobs[0].dtype == np.float32
    assert [proposal.confidence for proposal in proposals] == pytest.approx([0.97, 0.95])
    assert proposals[0].bbox.as_list() == pytest.approx(
        [1000 / 1280, 500 / 720, 1200 / 1280, 600 / 720]
    )
    assert proposals[1].bbox.as_list() == pytest.approx(
        [320 / 1280, 180 / 720, 640 / 1280, 270 / 720]
    )
    assert all(proposal.class_name == "license_plate" for proposal in proposals)
    assert all(proposal.frame_index == 12 for proposal in proposals)
    assert all(proposal.timestamp_ms == 400 for proposal in proposals)


def test_yolov8_selects_configured_class_and_clips_boxes(tmp_path: Path) -> None:
    predictions = np.asarray(
        [
            [10.0, 320.0, 80.0, 100.0, 0.95, 0.10],
            [630.0, 320.0, 80.0, 100.0, 0.10, 0.90],
        ],
        dtype=np.float32,
    )
    network = _StubNetwork(predictions[np.newaxis, :, :])
    detector = OpenCVYoloV8PlateDetector(
        _model_file(tmp_path),
        confidence_threshold=0.5,
        plate_class_index=1,
        _factory=lambda _path: network,
    )

    proposals = detector.detect(
        np.zeros((640, 640, 4), dtype=np.uint8),
        DetectionContext(frame_index=0, timestamp_ms=0),
    )

    assert len(proposals) == 1
    assert proposals[0].confidence == pytest.approx(0.90)
    assert proposals[0].bbox.as_list() == pytest.approx(
        [590 / 640, 270 / 640, 1.0, 370 / 640]
    )
    assert proposals[0].attributes["plate_class_index"] == 1


def test_yolov8_reports_unavailable_models(tmp_path: Path) -> None:
    detector = OpenCVYoloV8PlateDetector(tmp_path / "missing.onnx")

    assert not detector.availability.available
    assert detector.availability.reason is not None
    with pytest.raises(DetectorUnavailableError):
        detector.detect(
            np.zeros((100, 100, 3), dtype=np.uint8),
            DetectionContext(frame_index=0, timestamp_ms=0),
        )

    empty_network = _StubNetwork(np.empty((1, 0, 5), dtype=np.float32), is_empty=True)
    detector = OpenCVYoloV8PlateDetector(
        _model_file(tmp_path),
        _factory=lambda _path: empty_network,
    )
    assert not detector.availability.available
    assert detector.availability.reason is not None


def test_yolov8_rejects_invalid_configuration(tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)

    with pytest.raises(ValueError, match="confidence_threshold"):
        OpenCVYoloV8PlateDetector(model_path, confidence_threshold=-0.1)
    with pytest.raises(ValueError, match="nms_threshold"):
        OpenCVYoloV8PlateDetector(model_path, nms_threshold=1.1)
    with pytest.raises(ValueError, match="min_plate_size_pixels"):
        OpenCVYoloV8PlateDetector(model_path, min_plate_size_pixels=0)
    with pytest.raises(ValueError, match="plate_class_index"):
        OpenCVYoloV8PlateDetector(model_path, plate_class_index=-1)


def test_yolov8_rejects_incompatible_output(tmp_path: Path) -> None:
    network = _StubNetwork(np.zeros((2, 5, 10), dtype=np.float32))
    detector = OpenCVYoloV8PlateDetector(
        _model_file(tmp_path),
        _factory=lambda _path: network,
    )

    with pytest.raises(RuntimeError, match="batch size one"):
        detector.detect(
            cast(Frame, np.zeros((100, 100, 3), dtype=np.uint8)),
            DetectionContext(frame_index=0, timestamp_ms=0),
        )
