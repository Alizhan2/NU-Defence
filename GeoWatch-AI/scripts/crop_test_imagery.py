"""Create a small, georeferenced before/after pair from local GeoTIFF sources."""

from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "test_images" / "real_pair"
OUTPUT = SOURCE / "cropped"
WINDOW = Window(col_off=6656, row_off=6656, width=4096, height=4096)


def crop(source_name: str, output_name: str) -> None:
    source = SOURCE / source_name
    target = OUTPUT / output_name
    with rasterio.open(source) as dataset:
        profile = dataset.profile.copy()
        profile.update(
            height=int(WINDOW.height),
            width=int(WINDOW.width),
            transform=dataset.window_transform(WINDOW),
        )
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with rasterio.open(target, "w", **profile) as output:
            output.write(dataset.read(window=WINDOW))


if __name__ == "__main__":
    crop("before_2020-11-18_visual.tif", "before_crop_2020-11-18.tif")
    crop("after_2021-01-16_visual.tif", "after_crop_2021-01-16.tif")
