from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    image: np.ndarray
    x: int
    y: int
    valid_width: int
    valid_height: int


def _origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def make_tiles(image: np.ndarray, tile_size: int = 1024, overlap: float = 0.2) -> list[Tile]:
    if not 0 <= overlap < 1:
        raise ValueError("overlap должен быть в диапазоне [0, 1).")
    h, w = image.shape[:2]
    stride = max(1, round(tile_size * (1 - overlap)))
    tiles: list[Tile] = []
    for y in _origins(h, tile_size, stride):
        for x in _origins(w, tile_size, stride):
            crop = image[y : min(y + tile_size, h), x : min(x + tile_size, w)]
            tiles.append(Tile(crop, x, y, crop.shape[1], crop.shape[0]))
    return tiles
