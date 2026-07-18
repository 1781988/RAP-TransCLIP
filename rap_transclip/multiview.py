from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class ViewSpec:
    scale: float
    position: str

    @property
    def name(self) -> str:
        value = str(self.scale).replace(".", "p")
        return f"s{value}_{self.position}"


_VALID_POSITIONS = {
    "center",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
}


def build_view_specs(
    scales: Iterable[float],
    positions: Iterable[str],
) -> list[ViewSpec]:
    specs: list[ViewSpec] = []
    for scale in scales:
        scale = float(scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"Crop scale must be in (0, 1], received {scale}")
        for position in positions:
            if position not in _VALID_POSITIONS:
                raise ValueError(
                    f"Unsupported crop position: {position}. "
                    f"Expected one of {sorted(_VALID_POSITIONS)}"
                )
            specs.append(ViewSpec(scale=scale, position=position))
    if not specs:
        raise ValueError("At least one local view must be configured")
    return specs


def _crop_box(
    width: int,
    height: int,
    scale: float,
    position: str,
) -> tuple[int, int, int, int]:
    crop_w = max(1, min(width, int(round(width * scale))))
    crop_h = max(1, min(height, int(round(height * scale))))

    if position == "center":
        left = (width - crop_w) // 2
        top = (height - crop_h) // 2
    elif position == "top_left":
        left, top = 0, 0
    elif position == "top_right":
        left, top = width - crop_w, 0
    elif position == "bottom_left":
        left, top = 0, height - crop_h
    elif position == "bottom_right":
        left, top = width - crop_w, height - crop_h
    else:
        raise ValueError(f"Unsupported crop position: {position}")

    return left, top, left + crop_w, top + crop_h


def deterministic_local_crops(
    image: Image.Image,
    specs: Iterable[ViewSpec],
) -> list[Image.Image]:
    width, height = image.size
    return [
        image.crop(_crop_box(width, height, spec.scale, spec.position))
        for spec in specs
    ]


def apply_resolution_degradation(
    image: Image.Image,
    downsample_factor: int,
) -> Image.Image:
    factor = max(1, int(downsample_factor))
    if factor == 1:
        return image
    width, height = image.size
    reduced = image.resize(
        (max(1, width // factor), max(1, height // factor)),
        resample=Image.Resampling.BICUBIC,
    )
    return reduced.resize(
        (width, height),
        resample=Image.Resampling.BICUBIC,
    )
