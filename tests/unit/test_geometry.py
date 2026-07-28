import pytest
from pydantic import ValidationError

from clearframe.domain.geometry import NormalizedBox, PixelBox


def test_normalized_box_converts_to_bounded_pixels() -> None:
    box = NormalizedBox(x1=0.1, y1=0.2, x2=0.9, y2=0.8)

    assert box.to_pixels(1920, 1080) == PixelBox(192, 216, 1728, 864)
    assert box.padded(0.25).to_pixels(100, 100) == PixelBox(0, 4, 100, 95)


def test_interpolation_and_iou_are_deterministic() -> None:
    start = NormalizedBox(x1=0.1, y1=0.1, x2=0.3, y2=0.3)
    end = NormalizedBox(x1=0.3, y1=0.3, x2=0.5, y2=0.5)

    midpoint = start.interpolate(end, 0.5)

    assert midpoint.as_list() == pytest.approx([0.2, 0.2, 0.4, 0.4])
    assert start.iou(start) == pytest.approx(1.0)
    assert start.iou(end) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"x1": -0.1, "y1": 0.0, "x2": 0.5, "y2": 0.5}, "greater than or equal"),
        ({"x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 0.5}, "positive area"),
        ({"x1": 0.0, "y1": 0.8, "x2": 0.5, "y2": 0.2}, "positive area"),
    ],
)
def test_invalid_boxes_are_rejected(values: dict[str, float], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        NormalizedBox.model_validate(values)
