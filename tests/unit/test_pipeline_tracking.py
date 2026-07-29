import pytest

from clearframe.domain.geometry import NormalizedBox
from clearframe.pipeline import (
    DetectionProposal,
    DetectionTrack,
    IoUTracker,
    TrackPoint,
    validate_continuity,
)
from clearframe.pipeline.tracking import smooth_track_points


def detection(
    frame_index: int,
    class_name: str = "license_plate",
    *,
    x: float = 0.1,
    y: float = 0.2,
    width: float = 0.2,
    height: float = 0.2,
    confidence: float = 0.9,
) -> DetectionProposal:
    return DetectionProposal(
        frame_index=frame_index,
        timestamp_ms=frame_index * 100,
        class_name=class_name,
        bbox=NormalizedBox(x1=x, y1=y, x2=x + width, y2=y + height),
        confidence=confidence,
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


def test_tracker_uses_constant_velocity_for_non_overlapping_motion() -> None:
    tracks = IoUTracker(iou_threshold=0.3, max_gap=5).track(
        [
            detection(0, x=0.10),
            detection(1, x=0.25),
            detection(2, x=0.40),
        ]
    )

    assert len(tracks) == 1
    assert [point.bbox.x1 for point in tracks[0].points] == pytest.approx(
        [0.10, 0.25, 0.40]
    )


def test_tracker_reconnects_after_sampled_occlusion() -> None:
    track = IoUTracker(
        iou_threshold=0.3,
        max_gap=30,
        materialize_interpolated=False,
    ).track(
        [
            detection(0, x=0.10),
            detection(5, x=0.15),
            detection(20, x=0.30),
        ]
    )[0]

    assert [point.frame_index for point in track.points] == [0, 5, 20]
    assert track.coverage_point_count == 21


def test_motion_gate_rejects_implausible_jump_and_scale_change() -> None:
    jump_tracks = IoUTracker(iou_threshold=0.3, max_gap=30).track(
        [
            detection(0, x=0.05),
            detection(5, x=0.75),
        ]
    )
    scale_tracks = IoUTracker(iou_threshold=0.3, max_gap=30).track(
        [
            detection(0, x=0.40, y=0.40, width=0.04, height=0.04),
            detection(5, x=0.32, y=0.32, width=0.20, height=0.20),
        ]
    )

    assert len(jump_tracks) == 2
    assert len(scale_tracks) == 2


def test_motion_association_is_deterministic_with_multiple_tracks() -> None:
    proposals = [
        detection(0, x=0.10, confidence=0.90),
        detection(0, x=0.70, confidence=0.80),
        detection(1, x=0.25, confidence=0.90),
        detection(1, x=0.60, confidence=0.80),
        detection(2, x=0.40, confidence=0.90),
        detection(2, x=0.50, confidence=0.80),
    ]

    first = IoUTracker(iou_threshold=0.3, max_gap=5).track(proposals)
    second = IoUTracker(iou_threshold=0.3, max_gap=5).track(reversed(proposals))

    assert first == second
    assert len(first) == 2
    assert [point.bbox.x1 for point in first[0].points] == pytest.approx(
        [0.10, 0.25, 0.40]
    )
    assert [point.bbox.x1 for point in first[1].points] == pytest.approx(
        [0.70, 0.60, 0.50]
    )


def test_interpolated_boxes_remain_normalized() -> None:
    tracks = IoUTracker(iou_threshold=0.01, max_gap=2).track(
        [detection(0, x=0.7), detection(3, x=0.79)]
    )

    assert len(tracks[0].points) == 4
    for point in tracks[0].points:
        assert all(0.0 <= value <= 1.0 for value in point.bbox.as_list())
        assert point.bbox.x1 < point.bbox.x2
        assert point.bbox.y1 < point.bbox.y2


def test_sparse_tracker_preserves_virtual_interpolation_metrics() -> None:
    track = IoUTracker(
        iou_threshold=0.01,
        max_gap=3,
        materialize_interpolated=False,
    ).track(
        [
            detection(0, x=0.1, confidence=0.5),
            detection(3, x=0.2, confidence=1.0),
        ]
    )[0]

    assert [point.frame_index for point in track.points] == [0, 3]
    assert track.interpolates_gaps
    assert track.observed_point_count == 2
    assert track.interpolated_point_count == 2
    assert track.coverage_point_count == 4
    assert track.mean_confidence == pytest.approx(0.75)
    assert validate_continuity(track) == ()


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


def test_smoothing_removes_jitter_from_stationary_boxes() -> None:
    jitter = [0.0, 0.004, -0.003, 0.005, -0.004, 0.003, -0.005, 0.004, -0.002, 0.001]
    points = tuple(
        TrackPoint(
            frame_index=index * 5,
            timestamp_ms=index * 200,
            bbox=NormalizedBox(
                x1=0.4 + offset,
                y1=0.4 + offset,
                x2=0.6 + offset,
                y2=0.6 + offset,
            ),
            confidence=0.9,
        )
        for index, offset in enumerate(jitter)
    )

    smoothed = smooth_track_points(points)

    def center_x(box: NormalizedBox) -> float:
        return (box.x1 + box.x2) / 2

    raw_deviation = max(abs(center_x(point.bbox) - 0.5) for point in points)
    smoothed_deviation = max(
        abs(center_x(point.bbox) - 0.5) for point in smoothed[2:-2]
    )
    assert smoothed_deviation < raw_deviation / 2
    assert all(
        original.frame_index == result.frame_index
        and original.confidence == result.confidence
        for original, result in zip(points, smoothed, strict=True)
    )


def test_smoothing_preserves_linear_motion() -> None:
    points = tuple(
        TrackPoint(
            frame_index=index * 5,
            timestamp_ms=index * 200,
            bbox=NormalizedBox(
                x1=0.1 + index * 0.05,
                y1=0.3,
                x2=0.2 + index * 0.05,
                y2=0.4,
            ),
            confidence=0.9,
        )
        for index in range(10)
    )

    smoothed = smooth_track_points(points)

    for original, result in zip(points[2:-2], smoothed[2:-2], strict=True):
        assert result.bbox.x1 == pytest.approx(original.bbox.x1, abs=1e-9)
        assert result.bbox.x2 == pytest.approx(original.bbox.x2, abs=1e-9)


def test_smoothing_passes_short_tracks_through() -> None:
    points = tuple(
        TrackPoint(
            frame_index=index,
            timestamp_ms=index * 40,
            bbox=NormalizedBox(x1=0.1, y1=0.1, x2=0.2, y2=0.2),
            confidence=0.5,
        )
        for index in range(2)
    )

    assert smooth_track_points(points) == points
