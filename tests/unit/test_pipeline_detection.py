from pathlib import Path

import numpy as np
import pytest

from clearframe.domain.geometry import NormalizedBox
from clearframe.pipeline import (
    DetectionContext,
    DetectionProposal,
    DetectorUnavailableError,
    MockPlateDetector,
    OpenCVPlateCascadeDetector,
    class_aware_nms,
)


def proposal(
    class_name: str,
    confidence: float,
    bbox: NormalizedBox,
    *,
    frame_index: int = 0,
) -> DetectionProposal:
    return DetectionProposal(
        frame_index=frame_index,
        timestamp_ms=frame_index * 100,
        class_name=class_name,
        bbox=bbox,
        confidence=confidence,
        detector_name="test",
        detector_version="1",
    )


def test_mock_detector_is_deterministic_and_finds_demo_plate() -> None:
    frame = np.full((360, 640, 3), (58, 35, 23), dtype=np.uint8)
    frame[220:274, 160:340] = 255
    frame[232:262, 172:328] = 255
    frame[232:235, 172:328] = 0
    frame[259:262, 172:328] = 0
    frame[232:262, 172:175] = 0
    frame[232:262, 325:328] = 0
    detector = MockPlateDetector()
    context = DetectionContext(frame_index=4, timestamp_ms=267)

    first = detector.detect(frame, context)
    second = detector.detect(frame.copy(), context)

    assert first == second
    assert len(first) == 1
    assert first[0].bbox.as_list() == pytest.approx(
        [160 / 640, 220 / 360, 340 / 640, 274 / 360]
    )
    assert first[0].class_name == "license_plate"


def test_class_aware_nms_keeps_other_classes_and_frames() -> None:
    high = proposal(
        "license_plate",
        0.9,
        NormalizedBox(x1=0.1, y1=0.1, x2=0.4, y2=0.3),
    )
    overlapping = proposal(
        "license_plate",
        0.7,
        NormalizedBox(x1=0.12, y1=0.1, x2=0.42, y2=0.3),
    )
    face = proposal("face", 0.6, overlapping.bbox)
    next_frame = proposal("license_plate", 0.6, overlapping.bbox, frame_index=1)

    kept = class_aware_nms(
        [overlapping, next_frame, face, high],
        iou_threshold=0.5,
    )

    assert kept == [high, face, next_frame]


def test_detector_availability_is_explicit(tmp_path: Path) -> None:
    detector = OpenCVPlateCascadeDetector(tmp_path / "missing.xml")

    assert not detector.availability.available
    assert detector.availability.reason is not None
    with pytest.raises(DetectorUnavailableError):
        detector.detect(
            np.zeros((100, 100, 3), dtype=np.uint8),
            DetectionContext(frame_index=0, timestamp_ms=0),
        )
