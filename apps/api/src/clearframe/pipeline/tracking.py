from collections.abc import Iterable
from dataclasses import dataclass, field

from clearframe.domain.geometry import NormalizedBox
from clearframe.pipeline.detection import DetectionProposal


@dataclass(frozen=True, slots=True)
class TrackPoint:
    frame_index: int
    timestamp_ms: int
    bbox: NormalizedBox
    confidence: float
    is_interpolated: bool = False


@dataclass(frozen=True, slots=True)
class ContinuityWarning:
    code: str
    class_name: str
    start_frame: int
    end_frame: int
    track_ids: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class DetectionTrack:
    track_id: str
    class_name: str
    points: tuple[TrackPoint, ...]
    warnings: tuple[ContinuityWarning, ...] = ()

    @property
    def start_frame(self) -> int:
        return self.points[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.points[-1].frame_index

    @property
    def mean_confidence(self) -> float:
        return sum(point.confidence for point in self.points) / len(self.points)


@dataclass(slots=True)
class _TrackState:
    track_id: str
    class_name: str
    points: list[TrackPoint] = field(default_factory=list)
    active: bool = True

    @property
    def last_point(self) -> TrackPoint:
        return self.points[-1]


def _detection_sort_key(
    detection: DetectionProposal,
) -> tuple[str, float, float, float, float, float, str, str]:
    return (
        detection.class_name,
        -detection.confidence,
        detection.bbox.x1,
        detection.bbox.y1,
        detection.bbox.x2,
        detection.bbox.y2,
        detection.detector_name,
        detection.detector_version,
    )


class IoUTracker:
    name = "deterministic_iou"
    version = "1.0"

    def __init__(self, *, iou_threshold: float = 0.3, max_gap: int = 2) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be greater than zero and at most one")
        if max_gap < 0:
            raise ValueError("max_gap cannot be negative")
        self.iou_threshold = iou_threshold
        self.max_gap = max_gap
        self.reset()

    def reset(self) -> None:
        self._states: list[_TrackState] = []
        self._warnings: list[ContinuityWarning] = []
        self._next_track_number = 1
        self._last_frame_index: int | None = None

    @property
    def tracks(self) -> tuple[DetectionTrack, ...]:
        return tuple(self._snapshot(state) for state in self._states)

    @property
    def warnings(self) -> tuple[ContinuityWarning, ...]:
        return tuple(self._warnings)

    def track(self, detections: Iterable[DetectionProposal]) -> tuple[DetectionTrack, ...]:
        self.reset()
        grouped: dict[int, list[DetectionProposal]] = {}
        for detection in detections:
            grouped.setdefault(detection.frame_index, []).append(detection)
        for frame_index in sorted(grouped):
            self.update(frame_index, grouped[frame_index])
        return self.tracks

    def update(
        self,
        frame_index: int,
        detections: Iterable[DetectionProposal],
    ) -> list[TrackPoint]:
        if frame_index < 0:
            raise ValueError("frame_index cannot be negative")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frames must be supplied in strictly increasing order")

        ordered = sorted(detections, key=_detection_sort_key)
        if any(detection.frame_index != frame_index for detection in ordered):
            raise ValueError("all detections must belong to frame_index")

        for state in self._states:
            if (
                state.active
                and frame_index - state.last_point.frame_index - 1 > self.max_gap
            ):
                state.active = False

        candidates: list[tuple[float, str, int, _TrackState]] = []
        for detection_index, detection in enumerate(ordered):
            for state in self._states:
                if not state.active or state.class_name != detection.class_name:
                    continue
                overlap = state.last_point.bbox.iou(detection.bbox)
                if overlap >= self.iou_threshold:
                    candidates.append((-overlap, state.track_id, detection_index, state))

        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()
        assignments: list[tuple[_TrackState, int]] = []
        for _, _, detection_index, state in sorted(candidates):
            if state.track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(state.track_id)
            matched_detections.add(detection_index)
            assignments.append((state, detection_index))

        updates: list[TrackPoint] = []
        for state, detection_index in assignments:
            detection = ordered[detection_index]
            updates.extend(self._append_detection(state, detection))

        for detection_index, detection in enumerate(ordered):
            if detection_index in matched_detections:
                continue
            state = self._new_track(detection)
            updates.append(state.last_point)
            self._warn_if_fragmented(state)

        self._last_frame_index = frame_index
        return sorted(updates, key=lambda point: point.frame_index)

    def _new_track(self, detection: DetectionProposal) -> _TrackState:
        track_id = f"track-{self._next_track_number:06d}"
        self._next_track_number += 1
        state = _TrackState(
            track_id=track_id,
            class_name=detection.class_name,
            points=[
                TrackPoint(
                    frame_index=detection.frame_index,
                    timestamp_ms=detection.timestamp_ms,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                )
            ],
        )
        self._states.append(state)
        return state

    def _append_detection(
        self,
        state: _TrackState,
        detection: DetectionProposal,
    ) -> list[TrackPoint]:
        previous = state.last_point
        frame_distance = detection.frame_index - previous.frame_index
        appended: list[TrackPoint] = []
        for offset in range(1, frame_distance):
            progress = offset / frame_distance
            point = TrackPoint(
                frame_index=previous.frame_index + offset,
                timestamp_ms=round(
                    previous.timestamp_ms
                    + (detection.timestamp_ms - previous.timestamp_ms) * progress
                ),
                bbox=previous.bbox.interpolate(detection.bbox, progress),
                confidence=previous.confidence
                + (detection.confidence - previous.confidence) * progress,
                is_interpolated=True,
            )
            state.points.append(point)
            appended.append(point)
        observed = TrackPoint(
            frame_index=detection.frame_index,
            timestamp_ms=detection.timestamp_ms,
            bbox=detection.bbox,
            confidence=detection.confidence,
        )
        state.points.append(observed)
        appended.append(observed)
        return appended

    def _warn_if_fragmented(self, new_state: _TrackState) -> None:
        point = new_state.last_point
        possible_fragments = [
            state
            for state in self._states
            if state is not new_state
            and not state.active
            and state.class_name == new_state.class_name
            and state.last_point.bbox.iou(point.bbox) >= self.iou_threshold
        ]
        if not possible_fragments:
            return
        previous = max(
            possible_fragments,
            key=lambda state: (state.last_point.frame_index, state.track_id),
        )
        missing_start = previous.last_point.frame_index + 1
        missing_end = point.frame_index - 1
        self._warnings.append(
            ContinuityWarning(
                code="possible_fragment",
                class_name=new_state.class_name,
                start_frame=missing_start,
                end_frame=missing_end,
                track_ids=(previous.track_id, new_state.track_id),
                message=(
                    f"similar tracks are separated by frames "
                    f"{missing_start}-{missing_end}"
                ),
            )
        )

    def _snapshot(self, state: _TrackState) -> DetectionTrack:
        warnings = tuple(
            warning for warning in self._warnings if state.track_id in warning.track_ids
        )
        return DetectionTrack(
            track_id=state.track_id,
            class_name=state.class_name,
            points=tuple(state.points),
            warnings=warnings,
        )


def validate_continuity(
    track: DetectionTrack,
    *,
    allowed_gap: int = 0,
) -> tuple[ContinuityWarning, ...]:
    if allowed_gap < 0:
        raise ValueError("allowed_gap cannot be negative")
    warnings: list[ContinuityWarning] = []
    for previous, current in zip(track.points, track.points[1:], strict=False):
        missing = current.frame_index - previous.frame_index - 1
        if missing > allowed_gap:
            warnings.append(
                ContinuityWarning(
                    code="missing_span",
                    class_name=track.class_name,
                    start_frame=previous.frame_index + 1,
                    end_frame=current.frame_index - 1,
                    track_ids=(track.track_id,),
                    message=(
                        f"track {track.track_id} is missing frames "
                        f"{previous.frame_index + 1}-{current.frame_index - 1}"
                    ),
                )
            )
    return tuple(warnings)
