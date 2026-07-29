from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

import cv2
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from numpy.typing import NDArray

from clearframe.domain.geometry import NormalizedBox
from clearframe.pipeline.detection import (
    DetectionContext,
    DetectionProposal,
    DetectorAvailability,
    DetectorUnavailableError,
    Frame,
)

_INPUT_HEIGHT = 384
_INPUT_WIDTH = 384
_INPUT_SHAPE = (1, 3, _INPUT_HEIGHT, _INPUT_WIDTH)
_LETTERBOX_COLOR = (114, 114, 114)
_PLATE_CLASS_ID = 0
_CPU_PROVIDERS = ("CPUExecutionProvider",)


class _NodeArgument(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def shape(self) -> Sequence[int | str | None]: ...

    @property
    def type(self) -> str: ...


class _InferenceSession(Protocol):
    def get_inputs(self) -> Sequence[_NodeArgument]: ...

    def get_outputs(self) -> Sequence[_NodeArgument]: ...

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, NDArray[np.float32]],
    ) -> Sequence[object]: ...


_SessionFactory = Callable[[str, tuple[str, ...]], _InferenceSession]


def _create_session(
    model_path: str,
    providers: tuple[str, ...],
) -> _InferenceSession:
    return cast(
        _InferenceSession,
        ort.InferenceSession(model_path, providers=list(providers)),
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


@dataclass(frozen=True, slots=True)
class _Letterbox:
    image: Frame
    scale: float
    pad_x: int
    pad_y: int


def _letterbox(frame: Frame) -> _Letterbox:
    height, width = frame.shape[:2]
    scale = min(_INPUT_WIDTH / width, _INPUT_HEIGHT / height)
    resized_width = max(1, min(_INPUT_WIDTH, round(width * scale)))
    resized_height = max(1, min(_INPUT_HEIGHT, round(height * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_x = (_INPUT_WIDTH - resized_width) // 2
    pad_y = (_INPUT_HEIGHT - resized_height) // 2
    canvas = np.full(
        (_INPUT_HEIGHT, _INPUT_WIDTH, 3),
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


def _input_tensor(image: Frame) -> NDArray[np.float32]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    channels_first = np.transpose(rgb, (2, 0, 1))
    return np.ascontiguousarray(
        channels_first[np.newaxis],
        dtype=np.float32,
    ) / np.float32(255.0)


def _validate_session(session: _InferenceSession) -> str:
    inputs = tuple(session.get_inputs())
    if len(inputs) != 1:
        raise RuntimeError("YOLOv9 model must expose exactly one input")
    input_spec = inputs[0]
    if tuple(input_spec.shape) != _INPUT_SHAPE:
        raise RuntimeError(f"YOLOv9 input shape must be {_INPUT_SHAPE}")
    if input_spec.type != "tensor(float)":
        raise RuntimeError("YOLOv9 input must contain float32 values")

    outputs = tuple(session.get_outputs())
    if len(outputs) != 1:
        raise RuntimeError("YOLOv9 model must expose exactly one output")
    output_shape = tuple(outputs[0].shape)
    if len(output_shape) != 2 or output_shape[1] != 7:
        raise RuntimeError("YOLOv9 output shape must be N x 7")
    return input_spec.name


def _detection_rows(raw_output: object) -> NDArray[np.float32]:
    try:
        rows = np.asarray(raw_output, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("YOLOv9 returned malformed detection output") from exc
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise RuntimeError("YOLOv9 returned malformed detection output")
    return rows


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


class OnnxRuntimeYoloV9PlateDetector:
    """ONNX Runtime adapter for an end-to-end YOLOv9 plate detector."""

    name = "onnxruntime_yolov9_plate"
    version = ort.__version__
    supported_classes = frozenset({"license_plate"})

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float = 0.5,
        min_plate_size_pixels: int = 8,
        _factory: _SessionFactory = _create_session,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if min_plate_size_pixels <= 0:
            raise ValueError("min_plate_size_pixels must be positive")

        self._model_path = model_path.expanduser().resolve()
        self._confidence_threshold = confidence_threshold
        self._min_plate_size_pixels = min_plate_size_pixels
        self._session: _InferenceSession | None = None
        self._input_name: str | None = None
        self._lock = Lock()

        if self._model_path.suffix.lower() != ".onnx":
            self._availability = DetectorAvailability(
                False,
                f"YOLOv9 model must be a local ONNX file: {self._model_path}",
            )
            return
        if not self._model_path.is_file():
            self._availability = DetectorAvailability(
                False,
                f"YOLOv9 model file not found: {self._model_path}",
            )
            return
        if self._model_path.stat().st_size == 0:
            self._availability = DetectorAvailability(
                False,
                f"YOLOv9 model file is empty: {self._model_path}",
            )
            return

        try:
            session = _factory(str(self._model_path), _CPU_PROVIDERS)
            input_name = _validate_session(session)
        except Exception as exc:
            self._availability = DetectorAvailability(
                False,
                f"YOLOv9 model could not be loaded: {exc}",
            )
            return

        self._session = session
        self._input_name = input_name
        self._availability = DetectorAvailability(True)

    @property
    def availability(self) -> DetectorAvailability:
        return self._availability

    def detect(self, frame: Frame, context: DetectionContext) -> list[DetectionProposal]:
        if self._session is None or self._input_name is None:
            raise DetectorUnavailableError(self._availability.reason or "detector unavailable")

        image = _as_bgr(frame)
        height, width = image.shape[:2]
        letterbox = _letterbox(image)
        tensor = _input_tensor(letterbox.image)
        with self._lock:
            outputs = self._session.run(None, {self._input_name: tensor})
        if len(outputs) != 1:
            raise RuntimeError("YOLOv9 returned an unexpected number of outputs")
        rows = _detection_rows(outputs[0])

        proposals: list[DetectionProposal] = []
        for row in rows:
            values = np.asarray(row, dtype=np.float64)
            if not np.isfinite(values).all():
                continue
            batch_index, x1, y1, x2, y2, class_id, confidence = (
                float(value) for value in values
            )
            if (
                batch_index != 0.0
                or class_id != float(_PLATE_CLASS_ID)
                or confidence < self._confidence_threshold
                or confidence > 1.0
                or x2 <= x1
                or y2 <= y1
            ):
                continue

            frame_x1 = min(
                float(width),
                max(0.0, (x1 - letterbox.pad_x) / letterbox.scale),
            )
            frame_y1 = min(
                float(height),
                max(0.0, (y1 - letterbox.pad_y) / letterbox.scale),
            )
            frame_x2 = min(
                float(width),
                max(0.0, (x2 - letterbox.pad_x) / letterbox.scale),
            )
            frame_y2 = min(
                float(height),
                max(0.0, (y2 - letterbox.pad_y) / letterbox.scale),
            )
            if (
                frame_x2 - frame_x1 < self._min_plate_size_pixels
                or frame_y2 - frame_y1 < self._min_plate_size_pixels
            ):
                continue

            proposals.append(
                DetectionProposal(
                    frame_index=context.frame_index,
                    timestamp_ms=context.timestamp_ms,
                    class_name="license_plate",
                    bbox=NormalizedBox(
                        x1=frame_x1 / width,
                        y1=frame_y1 / height,
                        x2=frame_x2 / width,
                        y2=frame_y2 / height,
                    ),
                    confidence=confidence,
                    detector_name=self.name,
                    detector_version=self.version,
                )
            )
        return sorted(proposals, key=_sort_key)
