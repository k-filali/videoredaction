from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from math import exp, hypot, log

from clearframe.domain.geometry import NormalizedBox
from clearframe.pipeline.detection import DetectionProposal

_MIN_BOX_SIZE = 0.000001
_MAX_PREDICTED_SCALE = 2.0


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
    interpolates_gaps: bool = False

    @property
    def start_frame(self) -> int:
        return self.points[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.points[-1].frame_index

    @property
    def mean_confidence(self) -> float:
        if not self.interpolates_gaps:
            return sum(point.confidence for point in self.points) / len(self.points)

        confidence_sum = self.points[0].confidence
        coverage = 1
        for previous, current in zip(self.points, self.points[1:], strict=False):
            distance = current.frame_index - previous.frame_index
            confidence_sum += (
                distance * previous.confidence
                + (current.confidence - previous.confidence) * (distance + 1) / 2
            )
            coverage += distance
        return confidence_sum / coverage

    @property
    def observed_point_count(self) -> int:
        return sum(not point.is_interpolated for point in self.points)

    @property
    def interpolated_point_count(self) -> int:
        materialized = sum(point.is_interpolated for point in self.points)
        if not self.interpolates_gaps:
            return materialized
        virtual = sum(
            max(0, current.frame_index - previous.frame_index - 1)
            for previous, current in zip(self.points, self.points[1:], strict=False)
        )
        return materialized + virtual

    @property
    def coverage_point_count(self) -> int:
        return self.observed_point_count + self.interpolated_point_count


@dataclass(slots=True)
class _TrackState:
    track_id: str
    class_name: str
    points: list[TrackPoint] = field(default_factory=list)
    active: bool = True

    @property
    def last_point(self) -> TrackPoint:
        return self.points[-1]

    @property
    def observed_points(self) -> tuple[TrackPoint, ...]:
        observed: list[TrackPoint] = []
        for point in reversed(self.points):
            if not point.is_interpolated:
                observed.append(point)
                if len(observed) == 2:
                    break
        return tuple(reversed(observed))


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


def _box_geometry(box: NormalizedBox) -> tuple[float, float, float, float]:
    width = box.x2 - box.x1
    height = box.y2 - box.y1
    return (
        (box.x1 + box.x2) / 2,
        (box.y1 + box.y2) / 2,
        width,
        height,
    )


def _box_from_geometry(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> NormalizedBox:
    width = min(1.0, max(_MIN_BOX_SIZE, width))
    height = min(1.0, max(_MIN_BOX_SIZE, height))
    center_x = min(1.0 - width / 2, max(width / 2, center_x))
    center_y = min(1.0 - height / 2, max(height / 2, center_y))
    return NormalizedBox(
        x1=center_x - width / 2,
        y1=center_y - height / 2,
        x2=center_x + width / 2,
        y2=center_y + height / 2,
    )


def smooth_track_points(
    points: Sequence[TrackPoint],
    *,
    window: int = 5,
) -> tuple[TrackPoint, ...]:
    """Suppress detector jitter with a centered moving average over box geometry.

    A centered window preserves linear motion exactly, so smoothing only
    removes high-frequency oscillation rather than introducing tracking lag.
    """
    if window < 3 or len(points) < 3:
        return tuple(points)
    half = window // 2
    geometries = [_box_geometry(point.bbox) for point in points]
    smoothed: list[TrackPoint] = []
    for index, point in enumerate(points):
        neighborhood = geometries[max(0, index - half) : index + half + 1]
        count = len(neighborhood)
        smoothed.append(
            replace(
                point,
                bbox=_box_from_geometry(
                    sum(geometry[0] for geometry in neighborhood) / count,
                    sum(geometry[1] for geometry in neighborhood) / count,
                    sum(geometry[2] for geometry in neighborhood) / count,
                    sum(geometry[3] for geometry in neighborhood) / count,
                ),
            )
        )
    return tuple(smoothed)


def _predict_box(state: _TrackState, frame_index: int) -> NormalizedBox:
    observed = state.observed_points
    latest = observed[-1]
    if len(observed) < 2:
        return latest.bbox

    previous = observed[-2]
    history_frames = latest.frame_index - previous.frame_index
    prediction_frames = frame_index - latest.frame_index
    if history_frames <= 0 or prediction_frames <= 0:
        return latest.bbox

    previous_x, previous_y, previous_width, previous_height = _box_geometry(
        previous.bbox
    )
    latest_x, latest_y, latest_width, latest_height = _box_geometry(latest.bbox)
    horizon = prediction_frames / history_frames
    scale_limit = log(_MAX_PREDICTED_SCALE)
    width_change = max(
        -scale_limit,
        min(scale_limit, log(latest_width / previous_width) * horizon),
    )
    height_change = max(
        -scale_limit,
        min(scale_limit, log(latest_height / previous_height) * horizon),
    )
    return _box_from_geometry(
        latest_x + (latest_x - previous_x) * horizon,
        latest_y + (latest_y - previous_y) * horizon,
        latest_width * exp(width_change),
        latest_height * exp(height_change),
    )


def _scale_ratio(first: NormalizedBox, second: NormalizedBox) -> float:
    _, _, first_width, first_height = _box_geometry(first)
    _, _, second_width, second_height = _box_geometry(second)
    return max(
        first_width / second_width,
        second_width / first_width,
        first_height / second_height,
        second_height / first_height,
    )


class IoUTracker:
    name = "deterministic_motion_iou"
    version = "2.0"

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_gap: int = 2,
        materialize_interpolated: bool = True,
        motion_center_scale: float = 1.5,
        motion_center_floor: float = 0.01,
        motion_center_ceiling: float = 0.18,
        motion_max_scale_ratio: float = 2.25,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be greater than zero and at most one")
        if max_gap < 0:
            raise ValueError("max_gap cannot be negative")
        if motion_center_scale <= 0:
            raise ValueError("motion_center_scale must be positive")
        if not 0.0 <= motion_center_floor <= motion_center_ceiling:
            raise ValueError("motion center limits are invalid")
        if motion_center_ceiling > 1.0:
            raise ValueError("motion_center_ceiling cannot exceed one")
        if motion_max_scale_ratio <= 1.0:
            raise ValueError("motion_max_scale_ratio must be greater than one")
        self.iou_threshold = iou_threshold
        self.max_gap = max_gap
        self.materialize_interpolated = materialize_interpolated
        self.motion_center_scale = motion_center_scale
        self.motion_center_floor = motion_center_floor
        self.motion_center_ceiling = motion_center_ceiling
        self.motion_max_scale_ratio = motion_max_scale_ratio
        self.reset()

    def reset(self) -> None:
        self._states: list[_TrackState] = []
        self._active_states: dict[str, _TrackState] = {}
        self._warnings: list[ContinuityWarning] = []
        self._warnings_by_track: dict[str, list[ContinuityWarning]] = {}
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

        for state in tuple(self._active_states.values()):
            if (
                frame_index - state.last_point.frame_index - 1 > self.max_gap
            ):
                state.active = False
                self._active_states.pop(state.track_id)

        # Two-stage association. Confident detections claim tracks first, then
        # provisional ones are offered to whatever is still unmatched. This
        # keeps a flickering object attached to its track without letting weak
        # evidence spawn spurious tracks.
        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()
        assignments: list[tuple[_TrackState, int]] = []
        confident = [
            index for index, item in enumerate(ordered) if not item.provisional
        ]
        provisional = [
            index for index, item in enumerate(ordered) if item.provisional
        ]
        for tier in (confident, provisional):
            candidates: list[
                tuple[int, float, float, float, float, str, int, _TrackState]
            ] = []
            for detection_index in tier:
                detection = ordered[detection_index]
                for state in self._active_states.values():
                    if state.class_name != detection.class_name:
                        continue
                    if state.track_id in matched_tracks:
                        continue
                    candidate = self._association_candidate(
                        state,
                        detection,
                        detection_index=detection_index,
                        frame_index=frame_index,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

            for *_, detection_index, state in sorted(candidates):
                if (
                    state.track_id in matched_tracks
                    or detection_index in matched_detections
                ):
                    continue
                matched_tracks.add(state.track_id)
                matched_detections.add(detection_index)
                assignments.append((state, detection_index))

        updates: list[TrackPoint] = []
        for state, detection_index in assignments:
            detection = ordered[detection_index]
            updates.extend(self._append_detection(state, detection))

        for detection_index in confident:
            if detection_index in matched_detections:
                continue
            state = self._new_track(ordered[detection_index])
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
        self._active_states[state.track_id] = state
        return state

    def _association_candidate(
        self,
        state: _TrackState,
        detection: DetectionProposal,
        *,
        detection_index: int,
        frame_index: int,
    ) -> tuple[int, float, float, float, float, str, int, _TrackState] | None:
        predicted = _predict_box(state, frame_index)
        previous_overlap = state.last_point.bbox.iou(detection.bbox)
        predicted_overlap = predicted.iou(detection.bbox)
        best_overlap = max(previous_overlap, predicted_overlap)

        predicted_x, predicted_y, predicted_width, predicted_height = _box_geometry(
            predicted
        )
        detection_x, detection_y, detection_width, detection_height = _box_geometry(
            detection.bbox
        )
        center_distance = hypot(
            predicted_x - detection_x,
            predicted_y - detection_y,
        )
        reference_diagonal = max(
            hypot(predicted_width, predicted_height),
            hypot(detection_width, detection_height),
        )
        center_gate = min(
            self.motion_center_ceiling,
            self.motion_center_floor + self.motion_center_scale * reference_diagonal,
        )
        scale_ratio = _scale_ratio(predicted, detection.bbox)
        strong_overlap = best_overlap >= self.iou_threshold
        if not strong_overlap and (
            center_distance > center_gate
            or scale_ratio > self.motion_max_scale_ratio
        ):
            return None

        normalized_distance = center_distance / max(center_gate, _MIN_BOX_SIZE)
        normalized_scale = min(
            1.0,
            log(scale_ratio) / log(self.motion_max_scale_ratio),
        )
        normalized_age = min(
            1.0,
            (frame_index - state.last_point.frame_index) / max(1, self.max_gap + 1),
        )
        cost = (
            0.55 * (1.0 - best_overlap)
            + 0.30 * normalized_distance
            + 0.10 * normalized_scale
            + 0.05 * normalized_age
        )
        return (
            0 if strong_overlap else 1,
            cost,
            -best_overlap,
            normalized_distance,
            scale_ratio,
            state.track_id,
            detection_index,
            state,
        )

    def _append_detection(
        self,
        state: _TrackState,
        detection: DetectionProposal,
    ) -> list[TrackPoint]:
        previous = state.last_point
        frame_distance = detection.frame_index - previous.frame_index
        appended: list[TrackPoint] = []
        if self.materialize_interpolated:
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
            warning := ContinuityWarning(
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
        for track_id in warning.track_ids:
            self._warnings_by_track.setdefault(track_id, []).append(warning)

    def _snapshot(self, state: _TrackState) -> DetectionTrack:
        return DetectionTrack(
            track_id=state.track_id,
            class_name=state.class_name,
            points=tuple(state.points),
            warnings=tuple(self._warnings_by_track.get(state.track_id, ())),
            interpolates_gaps=not self.materialize_interpolated,
        )


def validate_continuity(
    track: DetectionTrack,
    *,
    allowed_gap: int = 0,
) -> tuple[ContinuityWarning, ...]:
    if allowed_gap < 0:
        raise ValueError("allowed_gap cannot be negative")
    if track.interpolates_gaps:
        return ()
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
