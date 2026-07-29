import numpy as np
import pytest

from clearframe.domain.enums import RedactionStyle, TrackSource
from clearframe.domain.geometry import NormalizedBox
from clearframe.domain.review import (
    ReviewKeyframe,
    ReviewSnapshot,
    TrackReviewState,
)
from clearframe.rendering import (
    SequentialRedactionLookup,
    box_at_frame,
    redact_frame,
    redactions_at_frame,
)


def build_track(*, redacted: bool = True, active: bool = True) -> TrackReviewState:
    return TrackReviewState(
        track_id="plate-track",
        class_name="license_plate",
        source=TrackSource.MODEL,
        active=active,
        redacted=redacted,
        start_frame=0,
        end_frame=10,
        start_ms=0,
        end_ms=1000,
        keyframes=[
            ReviewKeyframe(
                frame_index=0,
                timestamp_ms=0,
                bbox=NormalizedBox(x1=0.1, y1=0.2, x2=0.3, y2=0.4),
            ),
            ReviewKeyframe(
                frame_index=10,
                timestamp_ms=1000,
                bbox=NormalizedBox(x1=0.3, y1=0.4, x2=0.5, y2=0.6),
            ),
        ],
    )


def test_box_interpolation_respects_track_state_and_span() -> None:
    track = build_track()

    midpoint = box_at_frame(track, 5)

    assert midpoint is not None
    assert midpoint.as_list() == pytest.approx([0.2, 0.3, 0.4, 0.5])
    assert box_at_frame(track, 11) is None
    assert box_at_frame(build_track(redacted=False), 5) is None
    assert box_at_frame(build_track(active=False), 5) is None


def test_black_box_changes_every_pixel_inside_and_none_outside() -> None:
    frame = np.full((100, 100, 3), 180, dtype=np.uint8)
    snapshot = ReviewSnapshot(
        video_id="video",
        revision=0,
        tracks={"plate-track": build_track()},
    )
    redactions = redactions_at_frame(snapshot, 5, padding={"license_plate": 0.0})

    rendered = redact_frame(frame, redactions, RedactionStyle.BLACK_BOX)

    assert np.all(rendered[30:50, 20:40] == 0)
    assert np.array_equal(rendered[:20, :20], frame[:20, :20])


def test_sequential_lookup_compiles_keyframes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    track = build_track()
    track.keyframes.reverse()
    snapshot = ReviewSnapshot(
        video_id="video",
        revision=0,
        tracks={"plate-track": track},
    )
    expected = [
        redactions_at_frame(snapshot, frame_index, padding={"license_plate": 0.0})
        for frame_index in range(11)
    ]
    lookup = SequentialRedactionLookup(
        snapshot,
        padding={"license_plate": 0.0},
    )

    def fail_if_sorted_again(_: TrackReviewState) -> list[ReviewKeyframe]:
        raise AssertionError("compiled keyframes were sorted again")

    monkeypatch.setattr("clearframe.rendering._visible_keyframes", fail_if_sorted_again)

    assert [lookup.at_frame(frame_index) for frame_index in range(11)] == expected
    with pytest.raises(ValueError, match="cannot move backward"):
        lookup.at_frame(9)


def test_sequential_lookup_tracks_active_spans_in_snapshot_order() -> None:
    late = build_track()
    late.track_id = "late"
    late.start_frame = 5
    late.start_ms = 500
    late.keyframes = [keyframe for keyframe in late.keyframes if keyframe.frame_index >= 5]
    early = build_track()
    early.track_id = "early"
    snapshot = ReviewSnapshot(
        video_id="video",
        revision=0,
        tracks={"late": late, "early": early},
    )
    lookup = SequentialRedactionLookup(snapshot)

    assert [item.track_id for item in lookup.at_frame(0)] == ["early"]
    assert [item.track_id for item in lookup.at_frame(5)] == ["late", "early"]
    assert lookup.at_frame(11) == []


def test_all_rendering_styles_modify_the_target_region() -> None:
    gradient = np.indices((120, 160)).sum(axis=0).astype(np.uint8)
    frame = np.repeat(gradient[:, :, None], 3, axis=2)
    snapshot = ReviewSnapshot(
        video_id="video",
        revision=0,
        tracks={"plate-track": build_track()},
    )
    redactions = redactions_at_frame(snapshot, 5)

    for style in RedactionStyle:
        rendered = redact_frame(frame, redactions, style)
        assert not np.array_equal(rendered, frame)
        assert rendered.shape == frame.shape
        assert rendered.dtype == frame.dtype
