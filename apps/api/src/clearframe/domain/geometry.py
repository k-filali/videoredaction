from dataclasses import dataclass
from math import ceil, floor
from typing import Self

from pydantic import BaseModel, Field, model_validator


@dataclass(frozen=True, slots=True)
class PixelBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class NormalizedBox(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_area(self) -> Self:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive area")
        return self

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, other: Self) -> float:
        intersection_x1 = max(self.x1, other.x1)
        intersection_y1 = max(self.y1, other.y1)
        intersection_x2 = min(self.x2, other.x2)
        intersection_y2 = min(self.y2, other.y2)
        intersection_width = max(0.0, intersection_x2 - intersection_x1)
        intersection_height = max(0.0, intersection_y2 - intersection_y1)
        intersection = intersection_width * intersection_height
        union = self.area() + other.area() - intersection
        return intersection / union if union > 0 else 0.0

    def padded(self, fraction: float) -> Self:
        if fraction < 0:
            raise ValueError("padding cannot be negative")
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return type(self)(
            x1=max(0.0, self.x1 - width * fraction),
            y1=max(0.0, self.y1 - height * fraction),
            x2=min(1.0, self.x2 + width * fraction),
            y2=min(1.0, self.y2 + height * fraction),
        )

    def interpolate(self, other: Self, progress: float) -> Self:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be between zero and one")
        return type(self)(
            x1=self.x1 + (other.x1 - self.x1) * progress,
            y1=self.y1 + (other.y1 - self.y1) * progress,
            x2=self.x2 + (other.x2 - self.x2) * progress,
            y2=self.y2 + (other.y2 - self.y2) * progress,
        )

    def to_pixels(self, width: int, height: int) -> PixelBox:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        x1 = max(0, min(width - 1, floor(self.x1 * width)))
        y1 = max(0, min(height - 1, floor(self.y1 * height)))
        x2 = max(x1 + 1, min(width, ceil(self.x2 * width)))
        y2 = max(y1 + 1, min(height, ceil(self.y2 * height)))
        return PixelBox(x1=x1, y1=y1, x2=x2, y2=y2)
