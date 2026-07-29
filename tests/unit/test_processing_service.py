from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from threading import Lock
from time import sleep

import pytest
from sqlalchemy import func, select
from tests.helpers import MOCK_MODEL_REGISTRY_PATH, generate_test_video

from clearframe.database import Database
from clearframe.domain.enums import JobStatus, RunStatus, VideoStatus
from clearframe.domain.geometry import NormalizedBox
from clearframe.jobs import LocalJobRunner
from clearframe.media import MediaProcessor
from clearframe.model_registry import ModelEntry
from clearframe.models import Detection, ModelRun, Track, TrackKeyframe, VideoAsset
from clearframe.pipeline import (
    DetectionContext,
    DetectionProposal,
    Detector,
    DetectorAvailability,
    OnnxRuntimeYoloV9PlateDetector,
    OnnxRuntimeYuNetFaceDetector,
    class_aware_nms,
)
from clearframe.pipeline.detection import Frame
from clearframe.rendering import box_at_frame
from clearframe.services.processing import (
    DetectorSelectionError,
    ProcessingConflictError,
    ProcessingService,
)
from clearframe.services.review import build_review_snapshot
from clearframe.storage import LocalStorage


def _build_processing(
    tmp_path: Path,
) -> tuple[Database, LocalStorage, LocalJobRunner, ProcessingService, str]:
    database = Database(f"sqlite:///{(tmp_path / 'processing.db').as_posix()}")
    database.create_schema()
    storage = LocalStorage(tmp_path / "storage")
    runner = LocalJobRunner(database, max_workers=1)
    media = MediaProcessor()
    video_id = "processing-video"
    proxy_uri = storage.proxy_uri(video_id)
    proxy_path = generate_test_video(
        storage.prepare(proxy_uri),
        media,
        duration_seconds=0.8,
        audio=False,
    )
    metadata = media.probe(proxy_path)
    with database.session() as session:
        session.add(
            VideoAsset(
                id=video_id,
                original_filename="sample.mp4",
                safe_filename="sample.mp4",
                content_type="video/mp4",
                duration_ms=metadata.duration_ms,
                fps=metadata.fps,
                width=metadata.width,
                height=metadata.height,
                proxy_uri=proxy_uri,
                status=VideoStatus.READY_FOR_REVIEW,
            )
        )
        session.commit()
    service = ProcessingService(
        database,
        storage,
        runner,
        registry_path=MOCK_MODEL_REGISTRY_PATH,
    )
    return database, storage, runner, service, video_id


def test_mock_processing_persists_run_detections_and_tracks(tmp_path: Path) -> None:
    database, _, runner, service, video_id = _build_processing(tmp_path)
    try:
        requested = service.request(video_id, sample_every_frames=3)
        runner.wait(requested.job.id, timeout=60)

        with database.session() as session:
            run = session.get(ModelRun, requested.run.id)
            video = session.get(VideoAsset, video_id)
            assert run is not None
            assert video is not None
            assert run.status == RunStatus.COMPLETED
            assert video.status == VideoStatus.READY_FOR_REVIEW
            assert video.active_model_run_id == run.id
            assert run.config_hash == service.registry.config_fingerprint
            assert run.metrics["frames_sampled"] > 0
            assert run.metrics["detections"] > 0
            assert run.metrics["tracks"] == 1
            track = session.scalar(
                select(Track).where(Track.model_run_id == run.id)
            )
            assert track is not None
            assert track.end_frame == run.metrics["frames_decoded"] - 1
            assert session.scalar(
                select(func.count(Detection.id)).where(Detection.model_run_id == run.id)
            ) == run.metrics["detections"]
            assert session.scalar(
                select(func.count(Track.id)).where(Track.model_run_id == run.id)
            ) == 1
            keyframes = list(
                session.scalars(
                    select(TrackKeyframe)
                    .where(TrackKeyframe.track_id == track.id)
                    .order_by(TrackKeyframe.frame_index)
                )
            )
            assert len(keyframes) > 1
            assert len(keyframes) == run.metrics["keyframes"]
            assert len(keyframes) == run.metrics["observed_track_points"]
            virtual_interpolated = sum(
                current.frame_index - previous.frame_index - 1
                for previous, current in pairwise(keyframes)
            )
            assert virtual_interpolated > 0
            assert virtual_interpolated == run.metrics["interpolated_track_points"]
            assert (
                len(keyframes) + virtual_interpolated
                == run.metrics["track_coverage_points"]
            )
            assert track.confidence_summary["observed_points"] == len(keyframes)
            assert (
                track.confidence_summary["interpolated_points"]
                == virtual_interpolated
            )
            gap_start, _ = next(
                (previous, current)
                for previous, current in pairwise(keyframes)
                if current.frame_index - previous.frame_index > 1
            )
            virtual_frame = gap_start.frame_index + 1
            assert all(item.frame_index != virtual_frame for item in keyframes)
            snapshot = build_review_snapshot(session, video_id)
            assert box_at_frame(snapshot.tracks[track.id], virtual_frame) is not None
            assert requested.job.status == JobStatus.QUEUED
    finally:
        runner.shutdown()


def test_processing_rejects_disabled_detector_selection(tmp_path: Path) -> None:
    _, _, runner, service, video_id = _build_processing(tmp_path)
    try:
        with pytest.raises(DetectorSelectionError, match="disabled"):
            service.request(video_id, model_ids=["unavailable-model"])
    finally:
        runner.shutdown()


def test_default_processing_loads_real_model_adapters(tmp_path: Path) -> None:
    database, storage, runner, _, _ = _build_processing(tmp_path)
    service = ProcessingService(database, storage, runner)
    try:
        bindings = service._select_bindings(None)

        assert [binding.entry.id for binding in bindings] == [
            "yolov9t-plate",
            "yunet-face",
        ]
        assert isinstance(bindings[0].detector, OnnxRuntimeYoloV9PlateDetector)
        assert isinstance(bindings[1].detector, OnnxRuntimeYuNetFaceDetector)
        assert all(binding.detector.availability.available for binding in bindings)
    finally:
        runner.shutdown()


class _ConcurrentDetector:
    name = "concurrent_test"
    version = "1.0"
    supported_classes = frozenset({"license_plate"})
    device = "cuda"
    max_concurrency = 4

    def __init__(self, *, fail_frame: int | None = None) -> None:
        self.fail_frame = fail_frame
        self.active_calls = 0
        self.maximum_active_calls = 0
        self.started_frames: list[int] = []
        self.completed_frames: list[int] = []
        self.guard = Lock()

    @property
    def availability(self) -> DetectorAvailability:
        return DetectorAvailability(available=True)

    def detect(
        self,
        frame: Frame,
        context: DetectionContext,
    ) -> list[DetectionProposal]:
        del frame
        with self.guard:
            self.active_calls += 1
            self.maximum_active_calls = max(
                self.maximum_active_calls,
                self.active_calls,
            )
            self.started_frames.append(context.frame_index)
        try:
            sleep(0.01 * (4 - context.frame_index % 4))
            if context.frame_index == self.fail_frame:
                raise RuntimeError("controlled detector failure")
            return [
                DetectionProposal(
                    frame_index=context.frame_index,
                    timestamp_ms=context.timestamp_ms,
                    class_name="license_plate",
                    bbox=NormalizedBox(
                        x1=0.2,
                        y1=0.6,
                        x2=0.5,
                        y2=0.75,
                    ),
                    confidence=0.99,
                    detector_name=self.name,
                    detector_version=self.version,
                )
            ]
        finally:
            with self.guard:
                self.active_calls -= 1
                self.completed_frames.append(context.frame_index)


class _ConcurrentProcessingService(ProcessingService):
    def __init__(
        self,
        database: Database,
        storage: LocalStorage,
        runner: LocalJobRunner,
        detector: _ConcurrentDetector,
    ) -> None:
        super().__init__(
            database,
            storage,
            runner,
            registry_path=MOCK_MODEL_REGISTRY_PATH,
        )
        self.detector = detector

    def _build_detector(self, entry: ModelEntry) -> Detector:
        del entry
        return self.detector


def test_processing_bounds_frames_and_consumes_results_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, storage, runner, _, video_id = _build_processing(tmp_path)
    detector = _ConcurrentDetector()
    service = _ConcurrentProcessingService(
        database,
        storage,
        runner,
        detector,
    )
    consumed_frames: list[int] = []
    def recording_nms(
        proposals: Iterable[DetectionProposal],
        *,
        iou_threshold: float = 0.5,
    ) -> list[DetectionProposal]:
        materialized = list(proposals)
        consumed_frames.extend(
            proposal.frame_index
            for proposal in materialized
        )
        return class_aware_nms(
            materialized,
            iou_threshold=iou_threshold,
        )

    monkeypatch.setattr(
        "clearframe.services.processing.class_aware_nms",
        recording_nms,
    )
    try:
        requested = service.request(video_id, sample_every_frames=1)
        runner.wait(requested.job.id, timeout=60)

        with database.session() as session:
            run = session.get(ModelRun, requested.run.id)
            assert run is not None
            assert run.status == RunStatus.COMPLETED
            assert consumed_frames == sorted(consumed_frames)
            assert len(consumed_frames) == run.metrics["frames_sampled"]
            assert run.metrics["max_inflight_frames"] == 4
            assert run.metrics["detector_concurrency"] == {
                "mock-plate-v1": 4
            }
        assert detector.maximum_active_calls == 4
        assert detector.completed_frames != sorted(detector.completed_frames)
        assert detector.active_calls == 0
    finally:
        runner.shutdown()


def test_processing_drains_detector_workers_after_failure(tmp_path: Path) -> None:
    database, storage, runner, _, video_id = _build_processing(tmp_path)
    detector = _ConcurrentDetector(fail_frame=2)
    service = _ConcurrentProcessingService(
        database,
        storage,
        runner,
        detector,
    )
    try:
        requested = service.request(video_id, sample_every_frames=1)
        runner.wait(requested.job.id, timeout=60)

        with database.session() as session:
            run = session.get(ModelRun, requested.run.id)
            video = session.get(VideoAsset, video_id)
            assert run is not None
            assert video is not None
            assert run.status == RunStatus.FAILED
            assert video.status == VideoStatus.FAILED
        assert detector.active_calls == 0
        assert detector.maximum_active_calls <= detector.max_concurrency
    finally:
        runner.shutdown()


class _DuplicateDetector:
    name = "duplicate_test"
    version = "1.0"
    supported_classes = frozenset({"license_plate"})

    @property
    def availability(self) -> DetectorAvailability:
        return DetectorAvailability(available=True)

    def detect(
        self,
        frame: Frame,
        context: DetectionContext,
    ) -> list[DetectionProposal]:
        del frame
        return [
            DetectionProposal(
                frame_index=context.frame_index,
                timestamp_ms=context.timestamp_ms,
                class_name="license_plate",
                bbox=NormalizedBox(x1=0.2, y1=0.6, x2=0.5, y2=0.75),
                confidence=0.99,
                detector_name=self.name,
                detector_version=self.version,
                attributes={"candidate": index},
            )
            for index in range(2)
        ]


class _DuplicateProcessingService(ProcessingService):
    def _build_detector(self, entry: ModelEntry) -> Detector:
        del entry
        return _DuplicateDetector()


def test_nms_preserves_raw_detections_and_only_filters_tracking(tmp_path: Path) -> None:
    database, storage, runner, _, video_id = _build_processing(tmp_path)
    service = _DuplicateProcessingService(
        database,
        storage,
        runner,
        registry_path=MOCK_MODEL_REGISTRY_PATH,
    )
    try:
        requested = service.request(video_id, sample_every_frames=3)
        runner.wait(requested.job.id, timeout=60)

        with database.session() as session:
            run = session.get(ModelRun, requested.run.id)
            assert run is not None
            detections = list(
                session.scalars(
                    select(Detection)
                    .where(Detection.model_run_id == run.id)
                    .order_by(Detection.frame_index, Detection.id)
                )
            )
            assert run.metrics["detections"] == 2 * run.metrics["frames_sampled"]
            assert run.metrics["tracker_detections"] == run.metrics["frames_sampled"]
            assert run.metrics["nms_suppressed"] == run.metrics["frames_sampled"]
            assert len(detections) == run.metrics["detections"]
            assert sum(
                bool(detection.attributes["nms_suppressed"])
                for detection in detections
            ) == run.metrics["nms_suppressed"]
            assert session.scalar(
                select(func.count(Track.id)).where(Track.model_run_id == run.id)
            ) == 1
    finally:
        runner.shutdown()


def test_repeat_run_replaces_review_scope_until_review_starts(tmp_path: Path) -> None:
    database, _, runner, service, video_id = _build_processing(tmp_path)
    try:
        first = service.request(video_id, sample_every_frames=3)
        runner.wait(first.job.id, timeout=60)
        second = service.request(video_id, sample_every_frames=3)
        runner.wait(second.job.id, timeout=60)

        with database.session() as session:
            video = session.get(VideoAsset, video_id)
            assert video is not None
            assert video.active_model_run_id == second.run.id
            assert session.scalar(
                select(func.count(ModelRun.id)).where(ModelRun.video_id == video_id)
            ) == 2
            assert session.scalar(
                select(func.count(Track.id)).where(Track.video_id == video_id)
            ) == 2
            active_track_ids = set(
                session.scalars(
                    select(Track.id).where(Track.model_run_id == second.run.id)
                )
            )
            snapshot = build_review_snapshot(session, video_id)
            assert set(snapshot.tracks) == active_track_ids
            video.review_revision = 1
            session.commit()

        with pytest.raises(ProcessingConflictError, match="review has started"):
            service.request(video_id, sample_every_frames=3)
    finally:
        runner.shutdown()
