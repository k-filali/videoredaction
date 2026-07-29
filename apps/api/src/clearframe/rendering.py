from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np
from numpy.typing import NDArray

from clearframe.domain.enums import RedactionStyle
from clearframe.domain.geometry import NormalizedBox
from clearframe.domain.review import ReviewKeyframe, ReviewSnapshot, TrackReviewState

Frame = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ActiveRedaction:
    track_id: str
    class_name: str
    bbox: NormalizedBox


DEFAULT_PADDING: dict[str, float] = {
    "license_plate": 0.15,
    "face": 0.25,
    "scene_text": 0.12,
}


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


def redact_frame(
    frame: Frame,
    redactions: list[ActiveRedaction],
    style: RedactionStyle,
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

        if style == RedactionStyle.BLACK_BOX:
            region[:] = 0
        elif style == RedactionStyle.PIXELATE:
            reduced_width = max(1, pixels.width // 12)
            reduced_height = max(1, pixels.height // 12)
            reduced = cv2.resize(
                region,
                (reduced_width, reduced_height),
                interpolation=cv2.INTER_AREA,
            )
            region[:] = cv2.resize(
                reduced,
                (pixels.width, pixels.height),
                interpolation=cv2.INTER_NEAREST,
            )
        elif style == RedactionStyle.GAUSSIAN_BLUR:
            kernel = max(3, min(pixels.width, pixels.height) // 3)
            if kernel % 2 == 0:
                kernel += 1
            region[:] = cv2.GaussianBlur(region, (kernel, kernel), 0)
        else:
            raise ValueError(f"unsupported redaction style: {style}")
    return output
