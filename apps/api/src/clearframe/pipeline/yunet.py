from __future__ import annotations

from collections.abc import Callable
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

_BOOTSTRAP_INPUT_SIZE = (320, 320)


class _YuNetModel(Protocol):
    def setInputSize(self, input_size: tuple[int, int]) -> None: ...

    def detect(self, image: Frame) -> tuple[int, object]: ...


_YuNetFactory = Callable[
    [str, str, tuple[int, int], float, float, int],
    _YuNetModel,
]


def _create_yunet_model(
    model_path: str,
    config: str,
    input_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
    top_k: int,
) -> _YuNetModel:
    return cast(
        _YuNetModel,
        cv2.FaceDetectorYN.create(
            model_path,
            config,
            input_size,
            score_threshold,
            nms_threshold,
            top_k,
        ),
    )


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


def _sort_key(
    proposal: DetectionProposal,
) -> tuple[float, float, float, float, float]:
    return (
        -proposal.confidence,
        proposal.bbox.x1,
        proposal.bbox.y1,
        proposal.bbox.x2,
        proposal.bbox.y2,
    )


class OpenCVYuNetFaceDetector:
    """OpenCV YuNet adapter for face localization."""

    name = "opencv_yunet_face"
    version = cv2.__version__
    supported_classes = frozenset({"face"})

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5_000,
        min_face_size_pixels: int = 1,
        _factory: _YuNetFactory = _create_yunet_model,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between zero and one")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if min_face_size_pixels <= 0:
            raise ValueError("min_face_size_pixels must be positive")

        self._model_path = model_path.expanduser().resolve()
        self._confidence_threshold = confidence_threshold
        self._min_face_size_pixels = min_face_size_pixels
        self._model: _YuNetModel | None = None
        self._input_size: tuple[int, int] | None = None
        self._lock = Lock()

        if self._model_path.suffix.lower() != ".onnx":
            self._availability = DetectorAvailability(
                False,
                f"YuNet model must be a local ONNX file: {self._model_path}",
            )
            return
        if not self._model_path.is_file():
            self._availability = DetectorAvailability(
                False,
                f"YuNet model file not found: {self._model_path}",
            )
            return
        if self._model_path.stat().st_size == 0:
            self._availability = DetectorAvailability(
                False,
                f"YuNet model file is empty: {self._model_path}",
            )
            return

        try:
            self._model = _factory(
                str(self._model_path),
                "",
                _BOOTSTRAP_INPUT_SIZE,
                confidence_threshold,
                nms_threshold,
                top_k,
            )
        except (cv2.error, OSError, RuntimeError) as exc:
            self._availability = DetectorAvailability(
                False,
                f"YuNet model could not be loaded: {exc}",
            )
            return
        self._availability = DetectorAvailability(True)

    @property
    def availability(self) -> DetectorAvailability:
        return self._availability

    def detect(self, frame: Frame, context: DetectionContext) -> list[DetectionProposal]:
        if self._model is None:
            raise DetectorUnavailableError(self._availability.reason or "detector unavailable")

        image = _as_bgr(frame)
        height, width = image.shape[:2]
        input_size = (width, height)
        with self._lock:
            if self._input_size != input_size:
                self._model.setInputSize(input_size)
                self._input_size = input_size
            _, raw_faces = self._model.detect(image)

        if raw_faces is None:
            return []
        faces = np.asarray(raw_faces)
        if faces.size == 0:
            return []
        if faces.ndim == 1:
            faces = faces.reshape((1, -1))
        if faces.ndim != 2 or faces.shape[1] < 15:
            raise RuntimeError("YuNet returned malformed detection output")

        proposals: list[DetectionProposal] = []
        for row in faces:
            values = np.asarray(row[[0, 1, 2, 3, 14]], dtype=np.float64)
            if not np.isfinite(values).all():
                continue
            x, y, box_width, box_height, confidence = (float(value) for value in values)
            if (
                confidence < self._confidence_threshold
                or confidence > 1.0
                or box_width <= 0.0
                or box_height <= 0.0
            ):
                continue

            x1 = min(float(width), max(0.0, x))
            y1 = min(float(height), max(0.0, y))
            x2 = min(float(width), max(0.0, x + box_width))
            y2 = min(float(height), max(0.0, y + box_height))
            if (
                x2 - x1 < self._min_face_size_pixels
                or y2 - y1 < self._min_face_size_pixels
            ):
                continue

            proposals.append(
                DetectionProposal(
                    frame_index=context.frame_index,
                    timestamp_ms=context.timestamp_ms,
                    class_name="face",
                    bbox=NormalizedBox(
                        x1=x1 / width,
                        y1=y1 / height,
                        x2=x2 / width,
                        y2=y2 / height,
                    ),
                    confidence=confidence,
                    detector_name=self.name,
                    detector_version=self.version,
                )
            )
        return sorted(proposals, key=_sort_key)
