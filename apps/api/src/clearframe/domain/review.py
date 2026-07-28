from typing import Any

from pydantic import BaseModel, Field, model_validator

from clearframe.domain.enums import ReviewActionType, TrackSource
from clearframe.domain.geometry import NormalizedBox


class ReviewKeyframe(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    bbox: NormalizedBox
    locked: bool = True


class TrackReviewState(BaseModel):
    track_id: str
    class_name: str
    source: TrackSource
    active: bool = True
    redacted: bool = True
    accepted: bool = False
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    warning: str | None = None
    keyframes: list[ReviewKeyframe] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_span(self) -> "TrackReviewState":
        if self.end_frame < self.start_frame or self.end_ms < self.start_ms:
            raise ValueError("track end cannot precede track start")
        return self

    def upsert_keyframe(self, keyframe: ReviewKeyframe) -> None:
        self.keyframes = [
            existing for existing in self.keyframes if existing.frame_index != keyframe.frame_index
        ]
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda item: item.frame_index)


class ReviewSnapshot(BaseModel):
    video_id: str
    revision: int = Field(ge=0)
    tracks: dict[str, TrackReviewState] = Field(default_factory=dict)


class ReviewCommand(BaseModel):
    action_type: ReviewActionType
    expected_revision: int = Field(ge=0)
    track_id: str | None = None
    frame_index: int | None = Field(default=None, ge=0)
    timestamp_ms: int | None = Field(default=None, ge=0)
    reason_code: str | None = Field(default=None, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


class ReviewActionRead(BaseModel):
    id: str
    video_id: str
    track_id: str | None
    frame_index: int | None
    timestamp_ms: int | None
    action_type: ReviewActionType
    before_state: dict[str, Any]
    after_state: dict[str, Any]
    reason_code: str | None
    reviewer_session_id: str
    revision: int
    inverse_of_action_id: str | None
