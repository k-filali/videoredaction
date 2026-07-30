from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from clearframe.domain.geometry import NormalizedBox
from clearframe.pipeline.detection import (
    DetectionContext,
    DetectionProposal,
    DetectorAvailability,
    DetectorUnavailableError,
    Frame,
)
from clearframe.pipeline.onnx_runtime import (
    ONNX_RUNTIME_VERSION,
    OnnxSession,
    ProviderDiscovery,
    SessionFactory,
    active_execution_provider,
    create_onnx_session,
    create_session_with_cpu_fallback,
    discover_execution_providers,
    provider_device,
    select_execution_providers,
)

_INPUT_HEIGHT = 384
_INPUT_WIDTH = 384
_INPUT_SHAPE = (1, 3, _INPUT_HEIGHT, _INPUT_WIDTH)
_LETTERBOX_COLOR = (114, 114, 114)
_PLATE_CLASS_ID = 0


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


def _validate_session(session: OnnxSession) -> str:
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
    version = ONNX_RUNTIME_VERSION
    supported_classes = frozenset({"license_plate"})

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float = 0.5,
        tracking_confidence: float | None = None,
        min_plate_size_pixels: int = 8,
        max_aspect_ratio: float | None = None,
        _factory: SessionFactory = create_onnx_session,
        _available_providers: ProviderDiscovery = discover_execution_providers,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if tracking_confidence is not None and not (
            0.0 <= tracking_confidence <= confidence_threshold
        ):
            raise ValueError(
                "tracking_confidence must be between zero and confidence_threshold"
            )
        if min_plate_size_pixels <= 0:
            raise ValueError("min_plate_size_pixels must be positive")
        if max_aspect_ratio is not None and max_aspect_ratio <= 0.0:
            raise ValueError("max_aspect_ratio must be positive")

        self._model_path = model_path.expanduser().resolve()
        self._confidence_threshold = confidence_threshold
        # Weak detections are emitted so the tracker can use them for
        # association, flagged provisional so they never start a track.
        self._emit_threshold = (
            confidence_threshold if tracking_confidence is None else tracking_confidence
        )
        self._min_plate_size_pixels = min_plate_size_pixels
        self._max_aspect_ratio = max_aspect_ratio
        self._session: OnnxSession | None = None
        self._input_name: str | None = None
        self._provider: str | None = None
        self._device: str | None = None
        self._max_concurrency = 1
        self._inference_gate = Lock()

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
            requested_providers = select_execution_providers(
                _available_providers()
            )
            session, requested_providers = create_session_with_cpu_fallback(
                str(self._model_path),
                requested_providers,
                _factory,
            )
            input_name = _validate_session(session)
            provider = active_execution_provider(session, requested_providers)
        except Exception as exc:
            self._availability = DetectorAvailability(
                False,
                f"YOLOv9 model could not be loaded: {exc}",
            )
            return

        self._session = session
        self._input_name = input_name
        self._provider = provider
        self._device = provider_device(provider)
        self._availability = DetectorAvailability(True)

    @property
    def availability(self) -> DetectorAvailability:
        return self._availability

    @property
    def provider(self) -> str | None:
        return self._provider

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    def detect(self, frame: Frame, context: DetectionContext) -> list[DetectionProposal]:
        if self._session is None or self._input_name is None:
            raise DetectorUnavailableError(self._availability.reason or "detector unavailable")

        image = _as_bgr(frame)
        height, width = image.shape[:2]
        letterbox = _letterbox(image)
        tensor = _input_tensor(letterbox.image)
        with self._inference_gate:
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
                or confidence < self._emit_threshold
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
            box_width = frame_x2 - frame_x1
            box_height = frame_y2 - frame_y1
            if (
                box_width < self._min_plate_size_pixels
                or box_height < self._min_plate_size_pixels
            ):
                continue
            if (
                self._max_aspect_ratio is not None
                and box_width / box_height > self._max_aspect_ratio
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
                    provisional=confidence < self._confidence_threshold,
                )
            )
        return sorted(proposals, key=_sort_key)
