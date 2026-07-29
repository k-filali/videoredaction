from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import cast

import cv2
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult

from clearframe.database import Database
from clearframe.domain.enums import (
    InterpolationMode,
    JobStatus,
    JobType,
    ProposalSource,
    RunStatus,
    TrackSource,
    TrackStatus,
    VideoStatus,
)
from clearframe.jobs import JobContext, LocalJobRunner
from clearframe.model_registry import (
    DEFAULT_REGISTRY_PATH,
    AdapterKind,
    DetectorClass,
    ModelEntry,
    ModelRegistry,
    load_model_registry,
    resolve_weight_path,
    validate_enabled_weights,
)
from clearframe.models import (
    Detection,
    ModelRun,
    ProcessingJob,
    Track,
    TrackKeyframe,
    VideoAsset,
    new_id,
)
from clearframe.pipeline import (
    DetectionContext,
    DetectionProposal,
    Detector,
    IoUTracker,
    MockPlateDetector,
    OpenCVFaceCascadeDetector,
    OpenCVPlateCascadeDetector,
    class_aware_nms,
)
from clearframe.pipeline.detection import Frame
from clearframe.storage import LocalStorage


def utc_now() -> datetime:
    return datetime.now(UTC)


class ProcessingError(ValueError):
    pass


class ProcessingNotFoundError(ProcessingError):
    pass


class ModelRunNotFoundError(ProcessingNotFoundError):
    pass


class ProcessingConflictError(ProcessingError):
    pass


class ProcessingValidationError(ProcessingError):
    pass


class DetectorSelectionError(ProcessingValidationError):
    pass


@dataclass(frozen=True, slots=True)
class RequestedProcessing:
    run: ModelRun
    job: ProcessingJob


@dataclass(frozen=True, slots=True)
class _DetectorBinding:
    entry: ModelEntry
    detector: Detector


@dataclass(frozen=True, slots=True)
class _ObservedDetection:
    proposal: DetectionProposal
    registry_id: str
    inference_ms: float
    proposal_source: ProposalSource
    nms_suppressed: bool = False


class ProcessingService:
    def __init__(
        self,
        database: Database,
        storage: LocalStorage,
        runner: LocalJobRunner,
        *,
        registry: ModelRegistry | None = None,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
        weights_root: Path | None = None,
    ) -> None:
        self.database = database
        self.storage = storage
        self.runner = runner
        self.registry_path = registry_path.resolve()
        self.weights_root = weights_root
        self.registry = registry or load_model_registry(
            self.registry_path,
            weights_root=weights_root,
        )
        if registry is not None:
            validate_enabled_weights(
                registry,
                registry_path=self.registry_path,
                weights_root=weights_root,
            )

    def request(
        self,
        video_id: str,
        *,
        model_ids: list[str] | None = None,
        sample_every_frames: int = 5,
    ) -> RequestedProcessing:
        if sample_every_frames <= 0:
            raise ProcessingValidationError("sample_every_frames must be positive")
        bindings = self._select_bindings(model_ids)

        with self.database.session() as session:
            video = session.get(VideoAsset, video_id)
            if video is None:
                raise ProcessingNotFoundError("video not found")
            if not video.proxy_uri or not self.storage.exists(video.proxy_uri):
                raise ProcessingValidationError("review proxy is not ready")
            if video.status != VideoStatus.READY_FOR_REVIEW:
                raise ProcessingConflictError("video is not available for detection")
            if video.review_revision > 0:
                raise ProcessingConflictError(
                    "full detection cannot be repeated after review has started"
                )
            active_job = session.scalar(
                select(ProcessingJob.id).where(
                    ProcessingJob.video_id == video_id,
                    ProcessingJob.job_type == JobType.DETECT,
                    ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
                )
            )
            if active_job is not None:
                raise ProcessingConflictError("detection is already running for this video")
            starting_review_revision = video.review_revision
            reservation = cast(
                CursorResult[object],
                session.execute(
                    update(VideoAsset)
                    .where(
                        VideoAsset.id == video_id,
                        VideoAsset.review_revision == starting_review_revision,
                        VideoAsset.status == VideoStatus.READY_FOR_REVIEW,
                    )
                    .values(status=VideoStatus.PROCESSING, error_message=None)
                    .execution_options(synchronize_session="fetch")
                ),
            )
            if reservation.rowcount != 1:
                session.rollback()
                raise ProcessingConflictError(
                    "video processing was claimed by another request"
                )

            run = ModelRun(
                video_id=video_id,
                detector_versions={
                    binding.entry.id: {
                        "adapter": binding.entry.adapter,
                        "adapter_version": binding.entry.adapter_version,
                        "model_version": binding.entry.model_version,
                        "runtime_version": binding.detector.version,
                    }
                    for binding in bindings
                },
                tracker_name=IoUTracker.name,
                tracker_version=IoUTracker.version,
                thresholds={
                    binding.entry.id: binding.entry.thresholds.model_dump(mode="json")
                    for binding in bindings
                },
                config_hash=self.registry.config_fingerprint,
                device="cpu",
                status=RunStatus.QUEUED,
                metrics={},
            )
            session.add(run)
            session.flush()
            job = ProcessingJob(
                video_id=video_id,
                job_type=JobType.DETECT,
                payload={
                    "model_run_id": run.id,
                    "model_ids": [binding.entry.id for binding in bindings],
                    "sample_every_frames": sample_every_frames,
                    "starting_review_revision": starting_review_revision,
                },
            )
            session.add(job)
            session.commit()

        self.runner.submit(
            job.id,
            lambda context: self._process_guarded(
                context,
                video_id=video_id,
                run_id=run.id,
                bindings=bindings,
                sample_every_frames=sample_every_frames,
                starting_review_revision=starting_review_revision,
            ),
        )
        return RequestedProcessing(run=run, job=job)

    def latest_run(self, video_id: str) -> ModelRun:
        with self.database.session() as session:
            if session.get(VideoAsset, video_id) is None:
                raise ProcessingNotFoundError("video not found")
            run = session.scalar(
                select(ModelRun)
                .where(ModelRun.video_id == video_id)
                .order_by(ModelRun.created_at.desc(), ModelRun.id.desc())
            )
            if run is None:
                raise ModelRunNotFoundError("model run not found")
            session.expunge(run)
            return run

    def _select_bindings(self, requested_ids: list[str] | None) -> tuple[_DetectorBinding, ...]:
        enabled = {entry.id: entry for entry in self.registry.enabled_models}
        if requested_ids is None:
            selected_ids = sorted(enabled)
        else:
            selected_ids = sorted(set(requested_ids))
            if len(selected_ids) != len(requested_ids):
                raise DetectorSelectionError("model_ids cannot contain duplicates")
            unknown = sorted(set(selected_ids) - enabled.keys())
            if unknown:
                raise DetectorSelectionError(
                    f"models are unavailable or disabled: {', '.join(unknown)}"
                )
        if not selected_ids:
            raise DetectorSelectionError("no enabled detectors were selected")

        bindings = tuple(
            _DetectorBinding(
                entry=enabled[model_id],
                detector=self._build_detector(enabled[model_id]),
            )
            for model_id in selected_ids
        )
        unavailable = [
            f"{binding.entry.id}: {binding.detector.availability.reason or 'unavailable'}"
            for binding in bindings
            if not binding.detector.availability.available
        ]
        if unavailable:
            raise DetectorSelectionError("; ".join(unavailable))
        return bindings

    def _build_detector(self, entry: ModelEntry) -> Detector:
        if entry.adapter is AdapterKind.DETERMINISTIC_MOCK:
            return MockPlateDetector()
        if entry.adapter is not AdapterKind.OPENCV_CASCADE:
            raise DetectorSelectionError(f"{entry.id}: adapter is not implemented")

        cascade_path = resolve_weight_path(
            entry,
            registry_path=self.registry_path,
            weights_root=self.weights_root,
        )
        classes = set(entry.supported_classes)
        if classes == {DetectorClass.FACE}:
            return OpenCVFaceCascadeDetector(cascade_path)
        if classes == {DetectorClass.LICENSE_PLATE}:
            return OpenCVPlateCascadeDetector(cascade_path)
        raise DetectorSelectionError(
            f"{entry.id}: OpenCV cascade must support exactly one known class"
        )

    def _process_guarded(
        self,
        context: JobContext,
        *,
        video_id: str,
        run_id: str,
        bindings: tuple[_DetectorBinding, ...],
        sample_every_frames: int,
        starting_review_revision: int,
    ) -> None:
        try:
            self._process(
                context,
                video_id=video_id,
                run_id=run_id,
                bindings=bindings,
                sample_every_frames=sample_every_frames,
                starting_review_revision=starting_review_revision,
            )
        except Exception as exc:
            with self.database.session() as session:
                run = session.get(ModelRun, run_id)
                if run is not None:
                    run.status = RunStatus.FAILED
                    run.completed_at = utc_now()
                    run.metrics = {"error_type": type(exc).__name__}
                    session.commit()
            raise

    def _process(
        self,
        context: JobContext,
        *,
        video_id: str,
        run_id: str,
        bindings: tuple[_DetectorBinding, ...],
        sample_every_frames: int,
        starting_review_revision: int,
    ) -> None:
        with self.database.session() as session:
            video = session.get(VideoAsset, video_id)
            run = session.get(ModelRun, run_id)
            if video is None or run is None or not video.proxy_uri:
                raise ProcessingNotFoundError("processing input disappeared")
            proxy_uri = video.proxy_uri
            fallback_fps = video.fps or 0.0
            run.status = RunStatus.RUNNING
            run.started_at = utc_now()
            session.commit()

        started = perf_counter()
        context.update(0.03, "sampling proxy frames")
        capture = cv2.VideoCapture(str(self.storage.path_for(proxy_uri)))
        if not capture.isOpened():
            raise ProcessingValidationError("review proxy could not be opened")

        capture_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = capture_fps if capture_fps > 0 else fallback_fps
        if fps <= 0:
            capture.release()
            raise ProcessingValidationError("proxy frame rate is unavailable")
        estimated_frames = max(1, round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        observations: list[_ObservedDetection] = []
        tracking_proposals: list[DetectionProposal] = []
        inference_totals: dict[str, float] = defaultdict(float)
        frames_decoded = 0
        frames_sampled = 0
        nms_threshold = min(binding.entry.thresholds.nms_iou for binding in bindings)

        try:
            while True:
                success, raw_frame = capture.read()
                if not success:
                    break
                frame_index = frames_decoded
                frames_decoded += 1
                if frame_index % sample_every_frames != 0:
                    continue

                frames_sampled += 1
                timestamp_ms = round(frame_index * 1000 / fps)
                detection_context = DetectionContext(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
                frame_observations: list[_ObservedDetection] = []
                for binding in bindings:
                    inference_started = perf_counter()
                    proposals = binding.detector.detect(
                        cast(Frame, raw_frame),
                        detection_context,
                    )
                    inference_ms = (perf_counter() - inference_started) * 1000
                    inference_totals[binding.entry.id] += inference_ms
                    allowed_classes = {value.value for value in binding.entry.supported_classes}
                    for proposal in proposals:
                        if proposal.class_name not in allowed_classes:
                            raise ProcessingValidationError(
                                f"{binding.entry.id}: detector returned an unsupported class"
                            )
                        if proposal.confidence < binding.entry.thresholds.confidence:
                            continue
                        frame_observations.append(
                            _ObservedDetection(
                                proposal=proposal,
                                registry_id=binding.entry.id,
                                inference_ms=inference_ms,
                                proposal_source=(
                                    ProposalSource.MOCK
                                    if binding.entry.adapter
                                    is AdapterKind.DETERMINISTIC_MOCK
                                    else ProposalSource.MODEL
                                ),
                            )
                        )

                kept = {
                    id(proposal)
                    for proposal in class_aware_nms(
                        (item.proposal for item in frame_observations),
                        iou_threshold=nms_threshold,
                    )
                }
                observations.extend(
                    _ObservedDetection(
                        proposal=item.proposal,
                        registry_id=item.registry_id,
                        inference_ms=item.inference_ms,
                        proposal_source=item.proposal_source,
                        nms_suppressed=id(item.proposal) not in kept,
                    )
                    for item in frame_observations
                )
                tracking_proposals.extend(
                    item.proposal
                    for item in frame_observations
                    if id(item.proposal) in kept
                )
                if frames_sampled % max(1, estimated_frames // sample_every_frames // 20) == 0:
                    progress = min(0.72, 0.05 + 0.67 * frames_decoded / estimated_frames)
                    context.update(progress, "running detectors")
        finally:
            capture.release()

        if frames_decoded == 0:
            raise ProcessingValidationError("review proxy contained no decodable frames")

        context.update(0.76, "linking detections")
        tracker = IoUTracker(
            iou_threshold=0.3,
            max_gap=max(2, sample_every_frames * 2),
        )
        tracks = tracker.track(tracking_proposals)
        elapsed_ms = (perf_counter() - started) * 1000
        detection_classes = Counter(item.proposal.class_name for item in observations)
        detector_counts = Counter(item.registry_id for item in observations)
        keyframe_count = sum(len(track.points) for track in tracks)
        metrics: dict[str, object] = {
            "frames_decoded": frames_decoded,
            "frames_sampled": frames_sampled,
            "sample_every_frames": sample_every_frames,
            "fps": round(fps, 6),
            "detections": len(observations),
            "tracker_detections": len(tracking_proposals),
            "nms_suppressed": sum(item.nms_suppressed for item in observations),
            "tracks": len(tracks),
            "keyframes": keyframe_count,
            "continuity_warnings": len(tracker.warnings),
            "detection_classes": dict(sorted(detection_classes.items())),
            "detector_counts": dict(sorted(detector_counts.items())),
            "detector_inference_ms": {
                detector_id: round(duration, 3)
                for detector_id, duration in sorted(inference_totals.items())
            },
            "processing_ms": round(elapsed_ms, 3),
        }

        context.update(0.84, "persisting proposals")
        with self.database.session() as session:
            stored_run = session.get(ModelRun, run_id)
            stored_video = session.get(VideoAsset, video_id)
            if stored_run is None or stored_video is None:
                raise ProcessingNotFoundError("processing records disappeared")
            if (
                stored_video.review_revision != starting_review_revision
                or stored_video.status != VideoStatus.PROCESSING
            ):
                raise ProcessingConflictError(
                    "review state changed while detection was running"
                )

            for observation in observations:
                proposal = observation.proposal
                session.add(
                    Detection(
                        video_id=video_id,
                        model_run_id=run_id,
                        frame_index=proposal.frame_index,
                        timestamp_ms=proposal.timestamp_ms,
                        class_name=proposal.class_name,
                        x1=proposal.bbox.x1,
                        y1=proposal.bbox.y1,
                        x2=proposal.bbox.x2,
                        y2=proposal.bbox.y2,
                        confidence=proposal.confidence,
                        detector_name=proposal.detector_name,
                        detector_version=proposal.detector_version,
                        proposal_source=observation.proposal_source,
                        attributes={
                            **dict(proposal.attributes),
                            "nms_suppressed": observation.nms_suppressed,
                        },
                        inference_ms=observation.inference_ms,
                    )
                )

            for detection_track in tracks:
                track_id = new_id()
                points = detection_track.points
                warning = (
                    "; ".join(item.message for item in detection_track.warnings) or None
                )
                confidences = [point.confidence for point in points]
                track = Track(
                    id=track_id,
                    video_id=video_id,
                    model_run_id=run_id,
                    class_name=detection_track.class_name,
                    start_frame=detection_track.start_frame,
                    end_frame=detection_track.end_frame,
                    start_ms=points[0].timestamp_ms,
                    end_ms=points[-1].timestamp_ms,
                    status=TrackStatus.PROPOSED,
                    default_redacted=True,
                    source=TrackSource.MODEL,
                    confidence_summary={
                        "min": min(confidences),
                        "mean": detection_track.mean_confidence,
                        "max": max(confidences),
                        "observed_points": sum(
                            not point.is_interpolated for point in points
                        ),
                        "interpolated_points": sum(
                            point.is_interpolated for point in points
                        ),
                    },
                    warning=warning,
                )
                session.add(track)
                session.add_all(
                    TrackKeyframe(
                        track_id=track_id,
                        frame_index=point.frame_index,
                        timestamp_ms=point.timestamp_ms,
                        x1=point.bbox.x1,
                        y1=point.bbox.y1,
                        x2=point.bbox.x2,
                        y2=point.bbox.y2,
                        interpolation_mode=InterpolationMode.LINEAR,
                        visibility=True,
                        confidence=point.confidence,
                        source=TrackSource.MODEL,
                    )
                    for point in points
                )

            stored_run.status = RunStatus.COMPLETED
            stored_run.metrics = metrics
            stored_run.completed_at = utc_now()
            stored_video.active_model_run_id = run_id
            stored_video.status = VideoStatus.READY_FOR_REVIEW
            stored_video.error_message = None
            session.commit()
        context.update(0.98, "proposals ready for review")
