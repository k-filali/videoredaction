from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
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
from clearframe.jobs import JobContext, JobDispatcher
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
    OnnxRuntimeYoloV9PlateDetector,
    OnnxRuntimeYuNetFaceDetector,
    OpenCVFaceCascadeDetector,
    OpenCVPlateCascadeDetector,
    OpenCVYoloV8PlateDetector,
    OpenCVYuNetFaceDetector,
    class_aware_nms,
)
from clearframe.pipeline.detection import Frame
from clearframe.pipeline.tracking import smooth_track_points
from clearframe.storage import ArtifactStorage


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


_DetectorResult = tuple[
    _DetectorBinding,
    list[DetectionProposal],
    float,
]


@dataclass(frozen=True, slots=True)
class _PendingFrame:
    frame_index: int
    futures: tuple[Future[_DetectorResult], ...]


def _run_detector(
    binding: _DetectorBinding,
    frame: Frame,
    context: DetectionContext,
) -> _DetectorResult:
    started = perf_counter()
    proposals = binding.detector.detect(frame, context)
    inference_ms = (perf_counter() - started) * 1000
    return binding, proposals, inference_ms


def _detector_device(detector: Detector) -> str:
    device = getattr(detector, "device", None)
    return device if isinstance(device, str) else "cpu"


def _detector_max_concurrency(detector: Detector) -> int:
    value = getattr(detector, "max_concurrency", 1)
    return value if isinstance(value, int) and value > 0 else 1


def _run_device(bindings: tuple[_DetectorBinding, ...]) -> str:
    devices = sorted({_detector_device(binding.detector) for binding in bindings})
    return "+".join(devices)


def _detector_version(binding: _DetectorBinding) -> dict[str, str]:
    metadata = {
        "adapter": binding.entry.adapter,
        "adapter_version": binding.entry.adapter_version,
        "model_version": binding.entry.model_version,
        "runtime_version": binding.detector.version,
        "device": _detector_device(binding.detector),
    }
    provider = getattr(binding.detector, "provider", None)
    if isinstance(provider, str):
        metadata["execution_provider"] = provider
    return metadata


def _pending_detector_version(entry: ModelEntry) -> dict[str, str]:
    return {
        "adapter": entry.adapter,
        "adapter_version": entry.adapter_version,
        "model_version": entry.model_version,
        "runtime_version": "pending",
        "device": "pending",
    }


class ProcessingService:
    def __init__(
        self,
        database: Database,
        storage: ArtifactStorage,
        runner: JobDispatcher,
        *,
        registry: ModelRegistry | None = None,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
        weights_root: Path | None = None,
        require_cuda: bool = False,
    ) -> None:
        self.database = database
        self.storage = storage
        self.runner = runner
        self.registry_path = registry_path.resolve()
        self.weights_root = weights_root
        self.require_cuda = require_cuda
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
        entries = self._select_entries(model_ids)

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
                    entry.id: _pending_detector_version(entry)
                    for entry in entries
                },
                tracker_name=IoUTracker.name,
                tracker_version=IoUTracker.version,
                thresholds={
                    entry.id: entry.thresholds.model_dump(mode="json")
                    for entry in entries
                },
                config_hash=self.registry.config_fingerprint,
                device="pending",
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
                    "model_ids": [entry.id for entry in entries],
                    "sample_every_frames": sample_every_frames,
                    "starting_review_revision": starting_review_revision,
                },
            )
            session.add(job)
            session.commit()

        self.runner.enqueue(job.id)
        return RequestedProcessing(run=run, job=job)

    def execute(self, context: JobContext, job_id: str) -> None:
        with self.database.session() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None or job.job_type != JobType.DETECT or not job.video_id:
                raise ProcessingValidationError("detection job is invalid")
            run_id = job.payload.get("model_run_id")
            model_ids = job.payload.get("model_ids")
            sample_every_frames = job.payload.get("sample_every_frames")
            starting_review_revision = job.payload.get("starting_review_revision")
            video_id = job.video_id

        if (
            not isinstance(run_id, str)
            or not isinstance(model_ids, list)
            or not all(isinstance(model_id, str) for model_id in model_ids)
            or isinstance(sample_every_frames, bool)
            or not isinstance(sample_every_frames, int)
            or isinstance(starting_review_revision, bool)
            or not isinstance(starting_review_revision, int)
        ):
            raise ProcessingValidationError("detection job payload is invalid")

        self._process_guarded(
            context,
            video_id=video_id,
            run_id=run_id,
            model_ids=cast(list[str], model_ids),
            sample_every_frames=sample_every_frames,
            starting_review_revision=starting_review_revision,
        )

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

    def _select_entries(self, requested_ids: list[str] | None) -> tuple[ModelEntry, ...]:
        enabled = {
            entry.id: entry
            for entry in self.registry.enabled_models
            if entry.deployment_allowed and not entry.research_only
        }
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

        return tuple(enabled[model_id] for model_id in selected_ids)

    def _select_bindings(self, requested_ids: list[str] | None) -> tuple[_DetectorBinding, ...]:
        bindings = tuple(
            _DetectorBinding(
                entry=entry,
                detector=self._build_detector(entry),
            )
            for entry in self._select_entries(requested_ids)
        )
        unavailable = [
            f"{binding.entry.id}: {binding.detector.availability.reason or 'unavailable'}"
            for binding in bindings
            if not binding.detector.availability.available
        ]
        if unavailable:
            raise DetectorSelectionError("; ".join(unavailable))
        if self.require_cuda:
            non_cuda = [
                binding.entry.id
                for binding in bindings
                if _detector_device(binding.detector) != "cuda"
            ]
            if non_cuda:
                raise DetectorSelectionError(
                    "CUDA inference is required but unavailable for: "
                    + ", ".join(non_cuda)
                )
        return bindings

    def _build_detector(self, entry: ModelEntry) -> Detector:
        if entry.adapter is AdapterKind.DETERMINISTIC_MOCK:
            return MockPlateDetector()

        model_path = resolve_weight_path(
            entry,
            registry_path=self.registry_path,
            weights_root=self.weights_root,
        )
        if model_path is None:
            raise DetectorSelectionError(f"{entry.id}: model weights are not configured")

        classes = set(entry.supported_classes)
        if entry.adapter is AdapterKind.ONNXRUNTIME_YUNET:
            if classes == {DetectorClass.FACE}:
                return OnnxRuntimeYuNetFaceDetector(
                    model_path,
                    confidence_threshold=entry.thresholds.confidence,
                    nms_threshold=entry.thresholds.nms_iou,
                    min_face_size_pixels=entry.thresholds.min_size_pixels,
                )
            raise DetectorSelectionError(
                f"{entry.id}: YuNet adapter must support exactly one known class"
            )

        if entry.adapter is AdapterKind.OPENCV_YUNET:
            if classes == {DetectorClass.FACE}:
                return OpenCVYuNetFaceDetector(
                    model_path,
                    confidence_threshold=entry.thresholds.confidence,
                    nms_threshold=entry.thresholds.nms_iou,
                    min_face_size_pixels=entry.thresholds.min_size_pixels,
                )
            raise DetectorSelectionError(
                f"{entry.id}: YuNet adapter must support exactly one known class"
            )

        if entry.adapter is AdapterKind.ONNXRUNTIME_YOLOV9:
            if classes == {DetectorClass.LICENSE_PLATE}:
                return OnnxRuntimeYoloV9PlateDetector(
                    model_path,
                    confidence_threshold=entry.thresholds.confidence,
                    tracking_confidence=entry.thresholds.tracking_confidence,
                    min_plate_size_pixels=entry.thresholds.min_size_pixels,
                    max_aspect_ratio=entry.thresholds.max_aspect_ratio,
                )
            raise DetectorSelectionError(
                f"{entry.id}: YOLOv9 adapter must support exactly one known class"
            )

        if entry.adapter is AdapterKind.OPENCV_YOLOV8:
            if classes == {DetectorClass.LICENSE_PLATE}:
                return OpenCVYoloV8PlateDetector(
                    model_path,
                    confidence_threshold=entry.thresholds.confidence,
                    nms_threshold=entry.thresholds.nms_iou,
                    min_plate_size_pixels=entry.thresholds.min_size_pixels,
                )
            raise DetectorSelectionError(
                f"{entry.id}: YOLOv8 adapter must support exactly one known class"
            )

        if entry.adapter is not AdapterKind.OPENCV_CASCADE:
            raise DetectorSelectionError(f"{entry.id}: adapter is not implemented")
        if classes == {DetectorClass.FACE}:
            return OpenCVFaceCascadeDetector(model_path)
        if classes == {DetectorClass.LICENSE_PLATE}:
            return OpenCVPlateCascadeDetector(model_path)
        raise DetectorSelectionError(
            f"{entry.id}: OpenCV cascade must support exactly one known class"
        )

    def _process_guarded(
        self,
        context: JobContext,
        *,
        video_id: str,
        run_id: str,
        model_ids: list[str],
        sample_every_frames: int,
        starting_review_revision: int,
    ) -> None:
        try:
            bindings = self._select_bindings(model_ids)
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
            if run.config_hash != self.registry.config_fingerprint:
                raise ProcessingValidationError(
                    "model registry changed after detection was queued"
                )
            proxy_uri = video.proxy_uri
            fallback_fps = video.fps or 0.0
            run.detector_versions = {
                binding.entry.id: _detector_version(binding)
                for binding in bindings
            }
            run.device = _run_device(bindings)
            run.status = RunStatus.RUNNING
            run.started_at = utc_now()
            session.commit()

        started = perf_counter()
        context.update(0.03, "sampling proxy frames")
        materialized_input = ExitStack()
        proxy_path = materialized_input.enter_context(
            self.storage.materialize_input(proxy_uri)
        )
        capture = cv2.VideoCapture(str(proxy_path))
        if not capture.isOpened():
            materialized_input.close()
            raise ProcessingValidationError("review proxy could not be opened")

        capture_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = capture_fps if capture_fps > 0 else fallback_fps
        if fps <= 0:
            capture.release()
            materialized_input.close()
            raise ProcessingValidationError("proxy frame rate is unavailable")
        estimated_frames = max(1, round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        observations: list[_ObservedDetection] = []
        tracking_proposals: list[DetectionProposal] = []
        inference_totals: dict[str, float] = defaultdict(float)
        frames_decoded = 0
        frames_sampled = 0
        nms_thresholds = {
            class_name.value: min(
                binding.entry.thresholds.nms_iou
                for binding in bindings
                if class_name in binding.entry.supported_classes
            )
            for class_name in DetectorClass
            if any(class_name in binding.entry.supported_classes for binding in bindings)
        }

        detector_concurrency = tuple(
            _detector_max_concurrency(binding.detector)
            for binding in bindings
        )
        detector_pools = tuple(
            ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f"clearframe-{binding.entry.id}",
            )
            for binding, max_workers in zip(
                bindings,
                detector_concurrency,
                strict=True,
            )
        )
        max_inflight_frames = max(detector_concurrency)
        pending_frames: deque[_PendingFrame] = deque()
        frames_completed = 0
        progress_interval = max(
            1,
            estimated_frames // sample_every_frames // 20,
        )
        allowed_classes = {
            binding.entry.id: {
                value.value
                for value in binding.entry.supported_classes
            }
            for binding in bindings
        }

        def consume_frame(pending: _PendingFrame) -> None:
            nonlocal frames_completed
            detector_results = tuple(
                future.result()
                for future in pending.futures
            )
            # Confident detections are persisted and may start tracks.
            # Provisional detections (tracking_confidence <= score <
            # confidence) are held back from persistence but still offered to
            # the tracker so a flickering plate keeps its existing track alive.
            frame_observations: list[_ObservedDetection] = []
            frame_candidates: list[DetectionProposal] = []
            for binding, proposals, inference_ms in detector_results:
                inference_totals[binding.entry.id] += inference_ms
                for proposal in proposals:
                    if (
                        proposal.class_name
                        not in allowed_classes[binding.entry.id]
                    ):
                        raise ProcessingValidationError(
                            f"{binding.entry.id}: detector returned an unsupported class"
                        )
                    if proposal.provisional:
                        frame_candidates.append(proposal)
                        continue
                    if proposal.confidence < binding.entry.thresholds.confidence:
                        continue
                    frame_candidates.append(proposal)
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

            kept: set[int] = set()
            for class_name in sorted(
                {proposal.class_name for proposal in frame_candidates}
            ):
                kept.update(
                    id(proposal)
                    for proposal in class_aware_nms(
                        (
                            proposal
                            for proposal in frame_candidates
                            if proposal.class_name == class_name
                        ),
                        iou_threshold=nms_thresholds[class_name],
                    )
                )
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
                proposal
                for proposal in frame_candidates
                if id(proposal) in kept
            )
            frames_completed += 1
            if frames_completed % progress_interval == 0:
                progress = min(
                    0.72,
                    0.05
                    + 0.67
                    * (pending.frame_index + 1)
                    / estimated_frames,
                )
                context.update(progress, "running detectors")

        try:
            while capture.grab():
                frame_index = frames_decoded
                frames_decoded += 1
                is_tail_frame = frame_index >= max(
                    0,
                    estimated_frames - sample_every_frames,
                )
                if frame_index % sample_every_frames != 0 and not is_tail_frame:
                    continue

                success, raw_frame = capture.retrieve()
                if not success:
                    break
                frames_sampled += 1
                timestamp_ms = round(frame_index * 1000 / fps)
                detection_context = DetectionContext(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
                frame = cast(Frame, raw_frame)
                pending_frames.append(
                    _PendingFrame(
                        frame_index=frame_index,
                        futures=tuple(
                            detector_pool.submit(
                                _run_detector,
                                binding,
                                frame,
                                detection_context,
                            )
                            for binding, detector_pool in zip(
                                bindings,
                                detector_pools,
                                strict=True,
                            )
                        ),
                    )
                )
                if len(pending_frames) >= max_inflight_frames:
                    consume_frame(pending_frames[0])
                    pending_frames.popleft()

            while pending_frames:
                consume_frame(pending_frames[0])
                pending_frames.popleft()
        finally:
            for pending in pending_frames:
                for future in pending.futures:
                    future.cancel()
            for detector_pool in detector_pools:
                detector_pool.shutdown(
                    wait=True,
                    cancel_futures=True,
                )
            capture.release()
            materialized_input.close()

        if frames_decoded == 0:
            raise ProcessingValidationError("review proxy contained no decodable frames")

        context.update(0.76, "linking detections")
        # Small plates flicker in and out of detection, so allow a track to
        # bridge roughly two seconds of dropout. Association still has to pass
        # the motion gate, and gaps are interpolated between real observations
        # rather than extrapolated past the last one.
        tracker = IoUTracker(
            iou_threshold=0.3,
            max_gap=max(sample_every_frames * 6, round(fps * 2)),
            materialize_interpolated=False,
        )
        tracks = tracker.track(tracking_proposals)
        elapsed_ms = (perf_counter() - started) * 1000
        detection_classes = Counter(item.proposal.class_name for item in observations)
        detector_counts = Counter(item.registry_id for item in observations)
        keyframe_count = sum(len(track.points) for track in tracks)
        interpolated_track_points = sum(
            track.interpolated_point_count for track in tracks
        )
        track_coverage_points = sum(track.coverage_point_count for track in tracks)
        metrics: dict[str, object] = {
            "frames_decoded": frames_decoded,
            "frames_sampled": frames_sampled,
            "sample_every_frames": sample_every_frames,
            "max_inflight_frames": max_inflight_frames,
            "detector_concurrency": {
                binding.entry.id: max_workers
                for binding, max_workers in zip(
                    bindings,
                    detector_concurrency,
                    strict=True,
                )
            },
            "fps": round(fps, 6),
            "detections": len(observations),
            "tracker_detections": len(tracking_proposals),
            "provisional_detections": sum(
                1 for proposal in tracking_proposals if proposal.provisional
            ),
            "nms_suppressed": sum(item.nms_suppressed for item in observations),
            "tracks": len(tracks),
            "keyframes": keyframe_count,
            "observed_track_points": keyframe_count,
            "interpolated_track_points": interpolated_track_points,
            "track_coverage_points": track_coverage_points,
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
                points = smooth_track_points(
                    tuple(
                        point
                        for point in detection_track.points
                        if not point.is_interpolated
                    )
                )
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
                        "observed_points": detection_track.observed_point_count,
                        "interpolated_points": (
                            detection_track.interpolated_point_count
                        ),
                        "coverage_points": detection_track.coverage_point_count,
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
