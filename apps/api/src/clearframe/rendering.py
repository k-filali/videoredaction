from bisect import bisect_left
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from clearframe.domain.enums import RedactionShape, RedactionStyle
from clearframe.domain.geometry import NormalizedBox, PixelBox
from clearframe.domain.review import ReviewKeyframe, ReviewSnapshot, TrackReviewState

Frame = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ActiveRedaction:
    track_id: str
    class_name: str
    bbox: NormalizedBox


@dataclass(frozen=True, slots=True)
class ClassTreatment:
    style: RedactionStyle | None = None
    shape: RedactionShape | None = None


DEFAULT_PADDING: dict[str, float] = {
    "license_plate": 0.15,
    "face": 0.25,
    "scene_text": 0.12,
}

DEFAULT_SHAPES: dict[str, RedactionShape] = {
    "face": RedactionShape.ELLIPSE,
}


def resolve_treatment(
    class_name: str,
    default_style: RedactionStyle,
    treatments: dict[str, ClassTreatment] | None = None,
) -> tuple[RedactionStyle, RedactionShape]:
    treatment = (treatments or {}).get(class_name)
    style = treatment.style if treatment and treatment.style else default_style
    shape = (
        treatment.shape
        if treatment and treatment.shape
        else DEFAULT_SHAPES.get(class_name, RedactionShape.RECTANGLE)
    )
    return style, shape


def parse_class_treatments(raw: dict[str, object]) -> dict[str, ClassTreatment]:
    treatments: dict[str, ClassTreatment] = {}
    for class_name, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError("class treatment entries must be objects")
        style_value = value.get("style")
        shape_value = value.get("shape")
        treatments[str(class_name)] = ClassTreatment(
            style=RedactionStyle(style_value) if style_value is not None else None,
            shape=RedactionShape(shape_value) if shape_value is not None else None,
        )
    return treatments


@dataclass(slots=True)
class _SequentialTrack:
    ordinal: int
    track_id: str
    class_name: str
    start_frame: int
    end_frame: int
    keyframes: tuple[ReviewKeyframe, ...]
    padding: float
    keyframe_index: int = 0

    def box_at(self, frame_index: int) -> NormalizedBox | None:
        if frame_index < self.start_frame or frame_index > self.end_frame:
            return None
        if frame_index <= self.keyframes[0].frame_index:
            return self.keyframes[0].bbox
        if frame_index >= self.keyframes[-1].frame_index:
            return self.keyframes[-1].bbox

        while (
            self.keyframe_index + 1 < len(self.keyframes)
            and self.keyframes[self.keyframe_index + 1].frame_index < frame_index
        ):
            self.keyframe_index += 1
        left = self.keyframes[self.keyframe_index]
        right = self.keyframes[self.keyframe_index + 1]
        distance = right.frame_index - left.frame_index
        if distance <= 0:
            return right.bbox
        progress = (frame_index - left.frame_index) / distance
        return left.bbox.interpolate(right.bbox, progress)


class SequentialRedactionLookup:
    def __init__(
        self,
        snapshot: ReviewSnapshot,
        padding: dict[str, float] | None = None,
    ) -> None:
        padding_by_class = {**DEFAULT_PADDING, **(padding or {})}
        tracks = tuple(
            _SequentialTrack(
                ordinal=ordinal,
                track_id=track.track_id,
                class_name=track.class_name,
                start_frame=track.start_frame,
                end_frame=track.end_frame,
                keyframes=tuple(_visible_keyframes(track)),
                padding=padding_by_class.get(track.class_name, 0.12),
            )
            for ordinal, track in enumerate(snapshot.tracks.values())
            if track.active and track.redacted and track.keyframes
        )
        self._starts = tuple(
            sorted(tracks, key=lambda track: (track.start_frame, track.ordinal))
        )
        self._ends = tuple(
            sorted(tracks, key=lambda track: (track.end_frame, track.ordinal))
        )
        self._next_start = 0
        self._next_end = 0
        self._active_ordinals: list[int] = []
        self._active_tracks: list[_SequentialTrack] = []
        self._last_frame_index: int | None = None

    def at_frame(self, frame_index: int) -> list[ActiveRedaction]:
        if self._last_frame_index is not None and frame_index < self._last_frame_index:
            raise ValueError("sequential redaction frames cannot move backward")
        self._last_frame_index = frame_index

        while (
            self._next_start < len(self._starts)
            and self._starts[self._next_start].start_frame <= frame_index
        ):
            self._add_active(self._starts[self._next_start])
            self._next_start += 1
        while (
            self._next_end < len(self._ends)
            and self._ends[self._next_end].end_frame < frame_index
        ):
            self._remove_active(self._ends[self._next_end])
            self._next_end += 1

        active: list[ActiveRedaction] = []
        for track in self._active_tracks:
            bbox = track.box_at(frame_index)
            if bbox is None:
                continue
            active.append(
                ActiveRedaction(
                    track_id=track.track_id,
                    class_name=track.class_name,
                    bbox=bbox.padded(track.padding),
                )
            )
        return active

    def _add_active(self, track: _SequentialTrack) -> None:
        index = bisect_left(self._active_ordinals, track.ordinal)
        self._active_ordinals.insert(index, track.ordinal)
        self._active_tracks.insert(index, track)

    def _remove_active(self, track: _SequentialTrack) -> None:
        index = bisect_left(self._active_ordinals, track.ordinal)
        if (
            index >= len(self._active_ordinals)
            or self._active_ordinals[index] != track.ordinal
        ):
            return
        self._active_ordinals.pop(index)
        self._active_tracks.pop(index)


def _visible_keyframes(track: TrackReviewState) -> list[ReviewKeyframe]:
    return sorted(track.keyframes, key=lambda item: item.frame_index)


def box_at_frame(track: TrackReviewState, frame_index: int) -> NormalizedBox | None:
    if not track.active or not track.redacted:
        return None
    if frame_index < track.start_frame or frame_index > track.end_frame:
        return None

    keyframes = _visible_keyframes(track)
    if not keyframes:
        return None
    if len(keyframes) == 1 or frame_index <= keyframes[0].frame_index:
        return keyframes[0].bbox
    if frame_index >= keyframes[-1].frame_index:
        return keyframes[-1].bbox

    for left, right in pairwise(keyframes):
        if left.frame_index <= frame_index <= right.frame_index:
            distance = right.frame_index - left.frame_index
            if distance <= 0:
                return right.bbox
            progress = (frame_index - left.frame_index) / distance
            return left.bbox.interpolate(right.bbox, progress)
    return None


def redactions_at_frame(
    snapshot: ReviewSnapshot,
    frame_index: int,
    padding: dict[str, float] | None = None,
) -> list[ActiveRedaction]:
    padding_by_class = {**DEFAULT_PADDING, **(padding or {})}
    active: list[ActiveRedaction] = []
    for track in snapshot.tracks.values():
        box = box_at_frame(track, frame_index)
        if box is None:
            continue
        active.append(
            ActiveRedaction(
                track_id=track.track_id,
                class_name=track.class_name,
                bbox=box.padded(padding_by_class.get(track.class_name, 0.12)),
            )
        )
    return active


def _styled_region(region: Frame, style: RedactionStyle, pixels: PixelBox) -> Frame:
    if style == RedactionStyle.BLACK_BOX:
        return np.zeros_like(region)
    if style == RedactionStyle.PIXELATE:
        reduced_width = max(1, pixels.width // 12)
        reduced_height = max(1, pixels.height // 12)
        reduced = cv2.resize(
            region,
            (reduced_width, reduced_height),
            interpolation=cv2.INTER_AREA,
        )
        return cast(
            Frame,
            cv2.resize(
                reduced,
                (pixels.width, pixels.height),
                interpolation=cv2.INTER_NEAREST,
            ),
        )
    if style == RedactionStyle.GAUSSIAN_BLUR:
        kernel = max(3, min(pixels.width, pixels.height) // 3)
        if kernel % 2 == 0:
            kernel += 1
        return cast(Frame, cv2.GaussianBlur(region, (kernel, kernel), 0))
    raise ValueError(f"unsupported redaction style: {style}")


def redact_frame(
    frame: Frame,
    redactions: list[ActiveRedaction],
    style: RedactionStyle,
    treatments: dict[str, ClassTreatment] | None = None,
) -> Frame:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("renderer expects a three-channel BGR frame")
    output = frame.copy()
    height, width = output.shape[:2]

    for redaction in redactions:
        pixels = redaction.bbox.to_pixels(width, height)
        region = output[pixels.y1 : pixels.y2, pixels.x1 : pixels.x2]
        if region.size == 0:
            continue

        region_style, region_shape = resolve_treatment(
            redaction.class_name,
            style,
            treatments,
        )
        styled = _styled_region(region, region_style, pixels)
        if region_shape == RedactionShape.ELLIPSE:
            mask = np.zeros((pixels.height, pixels.width), dtype=np.uint8)
            cv2.ellipse(
                mask,
                (pixels.width // 2, pixels.height // 2),
                (max(1, pixels.width // 2), max(1, pixels.height // 2)),
                0,
                0,
                360,
                255,
                -1,
            )
            region[mask > 0] = styled[mask > 0]
        else:
            region[:] = styled
    return output
