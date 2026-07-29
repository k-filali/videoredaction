import pytest

from clearframe.domain.geometry import NormalizedBox
from clearframe.pipeline import (
    DetectionProposal,
    DetectionTrack,
    IoUTracker,
    TrackPoint,
    validate_continuity,
)


def detection(
    frame_index: int,
    class_name: str = "license_plate",
    *,
    x: float = 0.1,
) -> DetectionProposal:
    return DetectionProposal(
        frame_index=frame_index,
        timestamp_ms=frame_index * 100,
        class_name=class_name,
        bbox=NormalizedBox(x1=x, y1=0.2, x2=x + 0.2, y2=0.4),
        confidence=0.9,
        detector_name="test",
        detector_version="1",
    )


def test_tracker_is_deterministic_and_interpolates_short_occlusion() -> None:
    proposals = [detection(0, x=0.1), detection(2, x=0.2)]

    first = IoUTracker(iou_threshold=0.2, max_gap=1).track(proposals)
    second = IoUTracker(iou_threshold=0.2, max_gap=1).track(reversed(proposals))

    assert first == second
    assert len(first) == 1
    assert [point.frame_index for point in first[0].points] == [0, 1, 2]
    assert first[0].points[1].is_interpolated
    assert first[0].points[1].bbox.as_list() == pytest.approx([0.15, 0.2, 0.35, 0.4])


def test_tracker_never_associates_across_classes() -> None:
    tracks = IoUTracker(iou_threshold=0.3, max_gap=1).track(
        [detection(0, "license_plate"), detection(1, "face")]
    )

    assert len(tracks) == 2
    assert [track.class_name for track in tracks] == ["license_plate", "face"]


def test_interpolated_boxes_remain_normalized() -> None:
    tracks = IoUTracker(iou_threshold=0.01, max_gap=2).track(
        [detection(0, x=0.7), detection(3, x=0.79)]
    )

    assert len(tracks[0].points) == 4
    for point in tracks[0].points:
        assert all(0.0 <= value <= 1.0 for value in point.bbox.as_list())
        assert point.bbox.x1 < point.bbox.x2
        assert point.bbox.y1 < point.bbox.y2


def test_long_gap_splits_track_and_emits_continuity_warning() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_gap=1)

    tracks = tracker.track([detection(0), detection(3)])

    assert len(tracks) == 2
    assert len(tracker.warnings) == 1
    assert tracker.warnings[0].code == "possible_fragment"
    assert (tracker.warnings[0].start_frame, tracker.warnings[0].end_frame) == (1, 2)


def test_continuity_validator_flags_missing_span() -> None:
    track = DetectionTrack(
        track_id="manual-1",
        class_name="license_plate",
        points=(
            TrackPoint(0, 0, detection(0).bbox, 1.0),
            TrackPoint(3, 300, detection(3).bbox, 1.0),
        ),
    )

    warnings = validate_continuity(track)

    assert len(warnings) == 1
    assert warnings[0].code == "missing_span"
    assert (warnings[0].start_frame, warnings[0].end_frame) == (1, 2)
