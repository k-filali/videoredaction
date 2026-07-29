from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

import cv2
import numpy as np

from clearframe.domain.geometry import NormalizedBox
from clearframe.pipeline.detection import (
    DetectionContext,
    DetectionProposal,
    DetectorAvailability,
    DetectorUnavailableError,
    Frame,
)

_INPUT_SIZE = (640, 640)
_LETTERBOX_COLOR = (114, 114, 114)


class _DnnNetwork(Protocol):
    def empty(self) -> bool: ...

    def setPreferableBackend(self, backend_id: int) -> None: ...

    def setPreferableTarget(self, target_id: int) -> None: ...

    def setInput(self, blob: object) -> None: ...

    def forward(self) -> object: ...


_NetworkFactory = Callable[[str], _DnnNetwork]


def _load_network(model_path: str) -> _DnnNetwork:
    return cast(_DnnNetwork, cv2.dnn.readNetFromONNX(model_path))


def _as_bgr(frame: Frame) -> Frame:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError("frame must be a uint8 numpy array")
    if frame.size == 0:
        raise ValueError("frame cannot be empty")
    if frame.ndim == 2:
        return cast(Frame, cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    if frame.ndim != 3:
        raise ValueError("frame must have two or three dimensions")
    if frame.shape[2] == 3:
        return cast(Frame, np.ascontiguousarray(frame))
    if frame.shape[2] == 4:
        return cast(Frame, cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
    raise ValueError("frame must have one, three, or four channels")


@dataclass(frozen=True, slots=True)
class _Letterbox:
    image: Frame
    scale: float
    pad_x: int
    pad_y: int


def _letterbox(frame: Frame) -> _Letterbox:
    height, width = frame.shape[:2]
    target_width, target_height = _INPUT_SIZE
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, min(target_width, round(width * scale)))
    resized_height = max(1, min(target_height, round(height * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    canvas = np.full(
        (target_height, target_width, 3),
        _LETTERBOX_COLOR,
        dtype=np.uint8,
    )
    canvas[
        pad_y : pad_y + resized_height,
        pad_x : pad_x + resized_width,
    ] = resized
    return _Letterbox(
        image=cast(Frame, canvas),
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def _prediction_rows(raw_output: object, *, minimum_features: int) -> np.ndarray:
    output = raw_output
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes, np.ndarray)):
        if len(output) != 1:
            raise RuntimeError("YOLOv8 model must expose exactly one detection output")
        output = output[0]

    matrix = np.asarray(output)
    if matrix.ndim == 3:
        if matrix.shape[0] != 1:
            raise RuntimeError("YOLOv8 adapter only supports batch size one")
        matrix = matrix[0]
    if matrix.ndim != 2:
        raise RuntimeError("YOLOv8 returned malformed detection output")

    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return np.empty((0, 5), dtype=np.float32)
    if columns == minimum_features and rows != minimum_features:
        predictions = matrix
    elif rows == minimum_features and columns != minimum_features:
        predictions = matrix.T
    elif columns >= minimum_features and rows > columns:
        predictions = matrix
    elif rows >= minimum_features and columns > rows:
        predictions = matrix.T
    else:
        raise RuntimeError("YOLOv8 output layout is ambiguous")
    if predictions.shape[1] < 5:
        raise RuntimeError("YOLOv8 output does not contain class scores")
    return np.asarray(predictions, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class _Candidate:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


def _candidate_sort_key(candidate: _Candidate) -> tuple[float, float, float, float, float]:
    return (
        -candidate.confidence,
        candidate.x1,
        candidate.y1,
        candidate.x2,
        candidate.y2,
    )


def _iou(left: _Candidate, right: _Candidate) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _nms(candidates: list[_Candidate], threshold: float) -> list[_Candidate]:
    kept: list[_Candidate] = []
    for candidate in sorted(candidates, key=_candidate_sort_key):
        if all(_iou(candidate, previous) <= threshold for previous in kept):
            kept.append(candidate)
    return kept


class OpenCVYoloV8PlateDetector:
    """Ultralytics YOLOv8 ONNX adapter for license-plate localization."""

    name = "opencv_yolov8_plate"
    version = cv2.__version__
    supported_classes = frozenset({"license_plate"})

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        min_plate_size_pixels: int = 8,
        plate_class_index: int = 0,
        _factory: _NetworkFactory = _load_network,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between zero and one")
        if min_plate_size_pixels <= 0:
            raise ValueError("min_plate_size_pixels must be positive")
        if plate_class_index < 0:
            raise ValueError("plate_class_index cannot be negative")

        self._model_path = model_path.expanduser().resolve()
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._min_plate_size_pixels = min_plate_size_pixels
        self._plate_class_index = plate_class_index
        self._network: _DnnNetwork | None = None
        self._lock = Lock()

        if self._model_path.suffix.lower() != ".onnx":
            self._availability = DetectorAvailability(
                False,
                f"YOLOv8 model must be a local ONNX file: {self._model_path}",
            )
            return
        if not self._model_path.is_file():
            self._availability = DetectorAvailability(
                False,
                f"YOLOv8 model file not found: {self._model_path}",
            )
            return
        if self._model_path.stat().st_size == 0:
            self._availability = DetectorAvailability(
                False,
                f"YOLOv8 model file is empty: {self._model_path}",
            )
            return

        try:
            network = _factory(str(self._model_path))
            network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            if network.empty():
                raise RuntimeError("OpenCV returned an empty network")
        except (cv2.error, OSError, RuntimeError) as exc:
            self._availability = DetectorAvailability(
                False,
                f"YOLOv8 model could not be loaded: {exc}",
            )
            return

        self._network = network
        self._availability = DetectorAvailability(True)

    @property
    def availability(self) -> DetectorAvailability:
        return self._availability

    def detect(self, frame: Frame, context: DetectionContext) -> list[DetectionProposal]:
        if self._network is None:
            raise DetectorUnavailableError(self._availability.reason or "detector unavailable")

        image = _as_bgr(frame)
        height, width = image.shape[:2]
        letterbox = _letterbox(image)
        blob = cv2.dnn.blobFromImage(
            letterbox.image,
            scalefactor=1.0 / 255.0,
            size=_INPUT_SIZE,
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        with self._lock:
            self._network.setInput(blob)
            raw_output = self._network.forward()

        score_column = 4 + self._plate_class_index
        rows = _prediction_rows(raw_output, minimum_features=score_column + 1)
        if rows.shape[0] > 0 and score_column >= rows.shape[1]:
            raise RuntimeError("YOLOv8 output does not contain the configured plate class")

        candidates: list[_Candidate] = []
        for row in rows:
            values = np.asarray(
                [row[0], row[1], row[2], row[3], row[score_column]],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                continue
            center_x, center_y, box_width, box_height, confidence = (
                float(value) for value in values
            )
            if (
                confidence < self._confidence_threshold
                or confidence > 1.0
                or box_width <= 0.0
                or box_height <= 0.0
            ):
                continue

            x1 = (center_x - box_width / 2.0 - letterbox.pad_x) / letterbox.scale
            y1 = (center_y - box_height / 2.0 - letterbox.pad_y) / letterbox.scale
            x2 = (center_x + box_width / 2.0 - letterbox.pad_x) / letterbox.scale
            y2 = (center_y + box_height / 2.0 - letterbox.pad_y) / letterbox.scale
            x1 = min(float(width), max(0.0, x1))
            y1 = min(float(height), max(0.0, y1))
            x2 = min(float(width), max(0.0, x2))
            y2 = min(float(height), max(0.0, y2))
            if (
                x2 - x1 < self._min_plate_size_pixels
                or y2 - y1 < self._min_plate_size_pixels
            ):
                continue
            candidates.append(
                _Candidate(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                )
            )

        return [
            DetectionProposal(
                frame_index=context.frame_index,
                timestamp_ms=context.timestamp_ms,
                class_name="license_plate",
                bbox=NormalizedBox(
                    x1=candidate.x1 / width,
                    y1=candidate.y1 / height,
                    x2=candidate.x2 / width,
                    y2=candidate.y2 / height,
                ),
                confidence=candidate.confidence,
                detector_name=self.name,
                detector_version=self.version,
                attributes={"plate_class_index": self._plate_class_index},
            )
            for candidate in _nms(candidates, self._nms_threshold)
        ]
