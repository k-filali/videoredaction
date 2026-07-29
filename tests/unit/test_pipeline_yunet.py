from pathlib import Path

import cv2
import numpy as np
import pytest

from clearframe.pipeline import (
    DetectionContext,
    DetectorUnavailableError,
    OpenCVYuNetFaceDetector,
)
from clearframe.pipeline.detection import Frame


class _StubYuNet:
    def __init__(self, faces: object | None) -> None:
        self.faces = faces
        self.input_sizes: list[tuple[int, int]] = []
        self.images: list[Frame] = []

    def setInputSize(self, input_size: tuple[int, int]) -> None:
        self.input_sizes.append(input_size)

    def detect(self, image: Frame) -> tuple[int, object]:
        self.images.append(image)
        return 1, self.faces


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "face_detection_yunet.onnx"
    path.write_bytes(b"controlled test model")
    return path


def test_yunet_normalizes_filters_and_sorts_detections(tmp_path: Path) -> None:
    faces = np.zeros((5, 15), dtype=np.float32)
    faces[0, [0, 1, 2, 3, 14]] = [120, 40, 90, 80, 0.95]
    faces[1, [0, 1, 2, 3, 14]] = [30, 20, 60, 50, 0.95]
    faces[2, [0, 1, 2, 3, 14]] = [-10, -5, 40, 45, 0.90]
    faces[3, [0, 1, 2, 3, 14]] = [60, 30, 80, 70, 0.50]
    faces[4, [0, 1, 2, 3, 14]] = [200, 100, 4, 4, 0.99]
    backend = _StubYuNet(faces)
    factory_calls: list[tuple[str, str, tuple[int, int], float, float, int]] = []

    def factory(
        model_path: str,
        config: str,
        input_size: tuple[int, int],
        confidence_threshold: float,
        nms_threshold: float,
        top_k: int,
    ) -> _StubYuNet:
        factory_calls.append(
            (
                model_path,
                config,
                input_size,
                confidence_threshold,
                nms_threshold,
                top_k,
            )
        )
        return backend

    model_path = _model_file(tmp_path)
    detector = OpenCVYuNetFaceDetector(
        model_path,
        confidence_threshold=0.85,
        nms_threshold=0.25,
        top_k=200,
        min_face_size_pixels=10,
        _factory=factory,
    )
    context = DetectionContext(frame_index=7, timestamp_ms=280)

    proposals = detector.detect(np.zeros((200, 300), dtype=np.uint8), context)

    assert detector.availability.available
    assert factory_calls == [
        (str(model_path.resolve()), "", (320, 320), 0.85, 0.25, 200)
    ]
    assert backend.input_sizes == [(300, 200)]
    assert backend.images[0].shape == (200, 300, 3)
    assert [proposal.confidence for proposal in proposals] == pytest.approx(
        [0.95, 0.95, 0.90]
    )
    expected_boxes = [
        [30 / 300, 20 / 200, 90 / 300, 70 / 200],
        [120 / 300, 40 / 200, 210 / 300, 120 / 200],
        [0.0, 0.0, 30 / 300, 40 / 200],
    ]
    for proposal, expected in zip(proposals, expected_boxes, strict=True):
        assert proposal.bbox.as_list() == pytest.approx(expected)
    assert all(proposal.class_name == "face" for proposal in proposals)
    assert all(proposal.frame_index == 7 for proposal in proposals)
    assert all(proposal.timestamp_ms == 280 for proposal in proposals)


def test_yunet_reuses_input_size_and_accepts_bgra_frames(tmp_path: Path) -> None:
    backend = _StubYuNet(None)
    detector = OpenCVYuNetFaceDetector(
        _model_file(tmp_path),
        _factory=lambda *_args: backend,
    )
    context = DetectionContext(frame_index=0, timestamp_ms=0)
    frame = np.zeros((80, 120, 4), dtype=np.uint8)

    assert detector.detect(frame, context) == []
    assert detector.detect(frame.copy(), context) == []
    assert backend.input_sizes == [(120, 80)]
    assert all(image.shape == (80, 120, 3) for image in backend.images)


def test_yunet_rejects_invalid_configuration(tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)

    with pytest.raises(ValueError, match="confidence_threshold"):
        OpenCVYuNetFaceDetector(model_path, confidence_threshold=-0.1)
    with pytest.raises(ValueError, match="nms_threshold"):
        OpenCVYuNetFaceDetector(model_path, nms_threshold=1.1)
    with pytest.raises(ValueError, match="top_k"):
        OpenCVYuNetFaceDetector(model_path, top_k=0)
    with pytest.raises(ValueError, match="min_face_size_pixels"):
        OpenCVYuNetFaceDetector(model_path, min_face_size_pixels=0)


@pytest.mark.parametrize("filename", ["missing.onnx", "model.pb"])
def test_yunet_reports_unavailable_model_paths(tmp_path: Path, filename: str) -> None:
    model_path = tmp_path / filename
    if model_path.suffix != ".onnx":
        model_path.write_bytes(b"not ONNX")
    detector = OpenCVYuNetFaceDetector(model_path)

    assert not detector.availability.available
    assert detector.availability.reason is not None
    with pytest.raises(DetectorUnavailableError):
        detector.detect(
            np.zeros((100, 100, 3), dtype=np.uint8),
            DetectionContext(frame_index=0, timestamp_ms=0),
        )


def test_yunet_reports_model_load_failure(tmp_path: Path) -> None:
    def failing_factory(
        _model_path: str,
        _config: str,
        _input_size: tuple[int, int],
        _confidence_threshold: float,
        _nms_threshold: float,
        _top_k: int,
    ) -> _StubYuNet:
        raise cv2.error("invalid ONNX")

    detector = OpenCVYuNetFaceDetector(
        _model_file(tmp_path),
        _factory=failing_factory,
    )

    assert not detector.availability.available
    assert detector.availability.reason is not None
    assert "could not be loaded" in detector.availability.reason
