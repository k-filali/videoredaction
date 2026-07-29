from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
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

_INPUT_HEIGHT = 640
_INPUT_WIDTH = 640
_INPUT_SHAPE = (1, 3, _INPUT_HEIGHT, _INPUT_WIDTH)
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
    canvas = np.zeros((_INPUT_HEIGHT, _INPUT_WIDTH, 3), dtype=np.uint8)
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
    channels_first = np.transpose(image, (2, 0, 1))
    return np.ascontiguousarray(
        channels_first[np.newaxis],
        dtype=np.float32,
    )


def _expected_output_shapes() -> dict[str, tuple[int, int, int]]:
    shapes: dict[str, tuple[int, int, int]] = {}
    for stride in _STRIDES:
        locations = (_INPUT_WIDTH // stride) * (_INPUT_HEIGHT // stride)
        shapes[f"cls_{stride}"] = (1, locations, 1)
        shapes[f"obj_{stride}"] = (1, locations, 1)
        shapes[f"bbox_{stride}"] = (1, locations, 4)
        shapes[f"kps_{stride}"] = (1, locations, 10)
    return shapes


def _validate_session(session: OnnxSession) -> str:
    inputs = tuple(session.get_inputs())
    if len(inputs) != 1:
        raise RuntimeError("YuNet model must expose exactly one input")
    input_spec = inputs[0]
    if tuple(input_spec.shape) != _INPUT_SHAPE:
        raise RuntimeError(f"YuNet input shape must be {_INPUT_SHAPE}")
    if input_spec.type != "tensor(float)":
        raise RuntimeError("YuNet input must contain float32 values")

    output_specs = {output.name: output for output in session.get_outputs()}
    expected_shapes = _expected_output_shapes()
    if output_specs.keys() != expected_shapes.keys():
        raise RuntimeError("YuNet model exposes incompatible output names")
    for name, expected_shape in expected_shapes.items():
        output = output_specs[name]
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(f"YuNet output {name} must have shape {expected_shape}")
        if output.type != "tensor(float)":
            raise RuntimeError(f"YuNet output {name} must contain float32 values")
    return input_spec.name


def _output_array(
    raw_output: object,
    *,
    name: str,
    expected_shape: tuple[int, int, int],
) -> NDArray[np.float32]:
    try:
        output = np.asarray(raw_output, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"YuNet returned malformed output {name}") from exc
    if output.shape != expected_shape:
        raise RuntimeError(f"YuNet returned malformed output {name}")
    return output


@dataclass(frozen=True, slots=True)
class _FaceCandidate:
    x1: float
    y1: float
    width: float
    height: float
    confidence: float


def _decode_candidates(
    raw_outputs: tuple[object, ...],
    *,
    confidence_threshold: float,
) -> list[_FaceCandidate]:
    if len(raw_outputs) != len(_OUTPUT_NAMES):
        raise RuntimeError("YuNet returned an unexpected number of outputs")
    expected_shapes = _expected_output_shapes()
    outputs = {
        name: _output_array(
            raw,
            name=name,
            expected_shape=expected_shapes[name],
        )
        for name, raw in zip(_OUTPUT_NAMES, raw_outputs, strict=True)
    }

    candidates: list[_FaceCandidate] = []
    for stride in _STRIDES:
        columns = _INPUT_WIDTH // stride
        class_scores = np.clip(outputs[f"cls_{stride}"][0, :, 0], 0.0, 1.0)
        object_scores = np.clip(outputs[f"obj_{stride}"][0, :, 0], 0.0, 1.0)
        scores = np.sqrt(class_scores * object_scores)
        boxes = outputs[f"bbox_{stride}"][0]

        for index in np.flatnonzero(
            np.isfinite(scores) & (scores >= confidence_threshold)
        ):
            box = boxes[index]
            if not np.isfinite(box).all():
                continue
            row, column = divmod(int(index), columns)
            try:
                width = exp(float(box[2])) * stride
                height = exp(float(box[3])) * stride
            except OverflowError:
                continue
            center_x = (column + float(box[0])) * stride
            center_y = (row + float(box[1])) * stride
            x1 = center_x - width / 2.0
            y1 = center_y - height / 2.0
            confidence = float(scores[index])
            if not all(
                isfinite(value)
                for value in (x1, y1, width, height, confidence)
            ):
                continue
            candidates.append(
                _FaceCandidate(
                    x1=x1,
                    y1=y1,
                    width=width,
                    height=height,
                    confidence=confidence,
                )
            )
    return candidates


def _nms(
    candidates: list[_FaceCandidate],
    *,
    confidence_threshold: float,
    nms_threshold: float,
    top_k: int,
) -> list[_FaceCandidate]:
    if not candidates:
        return []
    boxes = [
        [
            int(candidate.x1),
            int(candidate.y1),
            int(candidate.width),
            int(candidate.height),
        ]
        for candidate in candidates
    ]
    scores = [candidate.confidence for candidate in candidates]
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        confidence_threshold,
        nms_threshold,
        eta=1.0,
        top_k=top_k,
    )
    return [
        candidates[int(index)]
        for index in np.asarray(indices, dtype=np.int64).reshape(-1)
    ]


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


class OnnxRuntimeYuNetFaceDetector:
    """ONNX Runtime adapter for fixed-input OpenCV YuNet face detection."""

    name = "onnxruntime_yunet_face"
    version = ONNX_RUNTIME_VERSION
    supported_classes = frozenset({"face"})

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5_000,
        min_face_size_pixels: int = 16,
        _factory: SessionFactory = create_onnx_session,
        _available_providers: ProviderDiscovery = discover_execution_providers,
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
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._min_face_size_pixels = min_face_size_pixels
        self._session: OnnxSession | None = None
        self._input_name: str | None = None
        self._provider: str | None = None
        self._device: str | None = None
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
                f"YuNet model could not be loaded: {exc}",
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

    def detect(self, frame: Frame, context: DetectionContext) -> list[DetectionProposal]:
        if self._session is None or self._input_name is None:
            raise DetectorUnavailableError(self._availability.reason or "detector unavailable")

        image = _as_bgr(frame)
        height, width = image.shape[:2]
        letterbox = _letterbox(image)
        tensor = _input_tensor(letterbox.image)
        with self._lock:
            raw_outputs = tuple(
                self._session.run(
                    _OUTPUT_NAMES,
                    {self._input_name: tensor},
                )
            )
        candidates = _decode_candidates(
            raw_outputs,
            confidence_threshold=self._confidence_threshold,
        )
        candidates = _nms(
            candidates,
            confidence_threshold=self._confidence_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
        )

        proposals: list[DetectionProposal] = []
        for candidate in candidates:
            frame_x1 = min(
                float(width),
                max(0.0, (candidate.x1 - letterbox.pad_x) / letterbox.scale),
            )
            frame_y1 = min(
                float(height),
                max(0.0, (candidate.y1 - letterbox.pad_y) / letterbox.scale),
            )
            frame_x2 = min(
                float(width),
                max(
                    0.0,
                    (
                        candidate.x1
                        + candidate.width
                        - letterbox.pad_x
                    )
                    / letterbox.scale,
                ),
            )
            frame_y2 = min(
                float(height),
                max(
                    0.0,
                    (
                        candidate.y1
                        + candidate.height
                        - letterbox.pad_y
                    )
                    / letterbox.scale,
                ),
            )
            if (
                frame_x2 - frame_x1 < self._min_face_size_pixels
                or frame_y2 - frame_y1 < self._min_face_size_pixels
            ):
                continue
            proposals.append(
                DetectionProposal(
                    frame_index=context.frame_index,
                    timestamp_ms=context.timestamp_ms,
                    class_name="face",
                    bbox=NormalizedBox(
                        x1=frame_x1 / width,
                        y1=frame_y1 / height,
                        x2=frame_x2 / width,
                        y2=frame_y2 / height,
                    ),
                    confidence=candidate.confidence,
                    detector_name=self.name,
                    detector_version=self.version,
                )
            )
        return sorted(proposals, key=_sort_key)
