from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from clearframe.domain.geometry import NormalizedBox

Frame: TypeAlias = NDArray[np.uint8]
AttributeValue: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class DetectionContext:
    frame_index: int
    timestamp_ms: int

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index cannot be negative")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms cannot be negative")


@dataclass(frozen=True, slots=True)
class DetectionProposal:
    frame_index: int
    timestamp_ms: int
    class_name: str
    bbox: NormalizedBox
    confidence: float
    detector_name: str
    detector_version: str
    # Scored below the detector's spawn threshold. Provisional detections may
    # extend an existing track but never start one, so weak evidence improves
    # continuity without inventing objects.
    provisional: bool = False
    attributes: Mapping[str, AttributeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index cannot be negative")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms cannot be negative")
        if not self.class_name.strip():
            raise ValueError("class_name cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if not self.detector_name.strip() or not self.detector_version.strip():
            raise ValueError("detector identity cannot be empty")


@dataclass(frozen=True, slots=True)
class DetectorAvailability:
    available: bool
    reason: str | None = None


class DetectorUnavailableError(RuntimeError):
    pass


class Detector(Protocol):
    name: str
    version: str
    supported_classes: frozenset[str]

    @property
    def availability(self) -> DetectorAvailability: ...

    def detect(self, frame: Frame, context: DetectionContext) -> list[DetectionProposal]: ...


def _grayscale(frame: Frame) -> Frame:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError("frame must be a uint8 numpy array")
    if frame.size == 0:
        raise ValueError("frame cannot be empty")
    if frame.ndim == 2:
        return frame
    if frame.ndim != 3:
        raise ValueError("frame must have two or three dimensions")
    channels = frame.shape[2]
    if channels == 3:
        return cast(Frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if channels == 4:
        return cast(Frame, cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY))
    raise ValueError("frame must have one, three, or four channels")


class MockPlateDetector:
    name = "mock_plate"
    version = "1.0"
    supported_classes = frozenset({"license_plate"})

    def __init__(self, *, brightness_threshold: int = 180) -> None:
        if not 0 <= brightness_threshold <= 255:
            raise ValueError("brightness_threshold must be between zero and 255")
        self._brightness_threshold = brightness_threshold

    @property
    def availability(self) -> DetectorAvailability:
        return DetectorAvailability(available=True)

    def detect(self, frame: Frame, context: DetectionContext) -> list[DetectionProposal]:
        gray = _grayscale(frame)
        height, width = gray.shape
        mask = np.where(gray >= self._brightness_threshold, 255, 0).astype(np.uint8)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        candidates: list[tuple[int, int, int, int, int]] = []
        frame_area = width * height
        for component in range(1, component_count):
            x = int(stats[component, cv2.CC_STAT_LEFT])
            y = int(stats[component, cv2.CC_STAT_TOP])
            box_width = int(stats[component, cv2.CC_STAT_WIDTH])
            box_height = int(stats[component, cv2.CC_STAT_HEIGHT])
            pixel_area = int(stats[component, cv2.CC_STAT_AREA])
            if box_width <= 0 or box_height <= 0:
                continue
            aspect_ratio = box_width / box_height
            box_fraction = (box_width * box_height) / frame_area
            fill_ratio = pixel_area / (box_width * box_height)
            if (
                2.0 <= aspect_ratio <= 6.5
                and 0.001 <= box_fraction <= 0.2
                and fill_ratio >= 0.2
            ):
                candidates.append((box_width * box_height, x, y, box_width, box_height))

        if not candidates:
            return []

        _, x, y, box_width, box_height = max(candidates)
        bbox = NormalizedBox(
            x1=max(0.0, x / width),
            y1=max(0.0, y / height),
            x2=min(1.0, (x + box_width) / width),
            y2=min(1.0, (y + box_height) / height),
        )
        return [
            DetectionProposal(
                frame_index=context.frame_index,
                timestamp_ms=context.timestamp_ms,
                class_name="license_plate",
                bbox=bbox,
                confidence=0.99,
                detector_name=self.name,
                detector_version=self.version,
                attributes={"deterministic": True},
            )
        ]


class _OpenCVCascadeDetector:
    name: str
    version = cv2.__version__

    def __init__(
        self,
        *,
        class_name: str,
        cascade_filename: str,
        cascade_path: Path | None,
        confidence: float,
        scale_factor: float,
        min_neighbors: int,
        min_size: tuple[int, int],
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than one")
        if min_neighbors < 0:
            raise ValueError("min_neighbors cannot be negative")
        if min_size[0] <= 0 or min_size[1] <= 0:
            raise ValueError("min_size values must be positive")

        self.supported_classes = frozenset({class_name})
        self._class_name = class_name
        self._confidence = confidence
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        cv2_file = Path(cv2.__file__) if cv2.__file__ else None
        default_path = (
            cv2_file.parent / "data" / cascade_filename if cv2_file is not None else None
        )
        self._cascade_path = cascade_path or default_path
        self._classifier: cv2.CascadeClassifier | None = None

        if self._cascade_path is None:
            self._availability = DetectorAvailability(False, "OpenCV data path is unavailable")
        elif not self._cascade_path.is_file():
            self._availability = DetectorAvailability(
                False,
                f"cascade file not found: {self._cascade_path}",
            )
        else:
            classifier = cv2.CascadeClassifier(str(self._cascade_path))
            if classifier.empty():
                self._availability = DetectorAvailability(
                    False,
                    f"cascade could not be loaded: {self._cascade_path}",
                )
            else:
                self._classifier = classifier
                self._availability = DetectorAvailability(True)

    @property
    def availability(self) -> DetectorAvailability:
        return self._availability

    def detect(self, frame: Frame, context: DetectionContext) -> list[DetectionProposal]:
        if self._classifier is None:
            raise DetectorUnavailableError(self._availability.reason or "detector unavailable")

        gray = _grayscale(frame)
        height, width = gray.shape
        raw_boxes = self._classifier.detectMultiScale(
            cv2.equalizeHist(gray),
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )
        boxes = np.asarray(raw_boxes, dtype=np.int64).reshape((-1, 4))
        proposals = [
            DetectionProposal(
                frame_index=context.frame_index,
                timestamp_ms=context.timestamp_ms,
                class_name=self._class_name,
                bbox=NormalizedBox(
                    x1=max(0.0, int(row[0]) / width),
                    y1=max(0.0, int(row[1]) / height),
                    x2=min(1.0, (int(row[0]) + int(row[2])) / width),
                    y2=min(1.0, (int(row[1]) + int(row[3])) / height),
                ),
                confidence=self._confidence,
                detector_name=self.name,
                detector_version=self.version,
            )
            for row in boxes
            if int(row[2]) > 0 and int(row[3]) > 0
        ]
        return sorted(proposals, key=_proposal_sort_key)


class OpenCVPlateCascadeDetector(_OpenCVCascadeDetector):
    name = "opencv_plate_cascade"

    def __init__(self, cascade_path: Path | None = None) -> None:
        super().__init__(
            class_name="license_plate",
            cascade_filename="haarcascade_russian_plate_number.xml",
            cascade_path=cascade_path,
            confidence=0.65,
            scale_factor=1.1,
            min_neighbors=3,
            min_size=(30, 10),
        )


class OpenCVFaceCascadeDetector(_OpenCVCascadeDetector):
    name = "opencv_face_cascade"

    def __init__(self, cascade_path: Path | None = None) -> None:
        super().__init__(
            class_name="face",
            cascade_filename="haarcascade_frontalface_default.xml",
            cascade_path=cascade_path,
            confidence=0.7,
            scale_factor=1.1,
            min_neighbors=5,
            min_size=(24, 24),
        )


def _proposal_sort_key(
    proposal: DetectionProposal,
) -> tuple[int, float, str, float, float, float, float, str, str]:
    return (
        proposal.frame_index,
        -proposal.confidence,
        proposal.class_name,
        proposal.bbox.x1,
        proposal.bbox.y1,
        proposal.bbox.x2,
        proposal.bbox.y2,
        proposal.detector_name,
        proposal.detector_version,
    )


def class_aware_nms(
    proposals: Iterable[DetectionProposal],
    *,
    iou_threshold: float = 0.5,
) -> list[DetectionProposal]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between zero and one")

    kept: list[DetectionProposal] = []
    for candidate in sorted(proposals, key=_proposal_sort_key):
        overlaps_kept = any(
            previous.frame_index == candidate.frame_index
            and previous.class_name == candidate.class_name
            and previous.bbox.iou(candidate.bbox) > iou_threshold
            for previous in kept
        )
        if not overlaps_kept:
            kept.append(candidate)
    return kept
