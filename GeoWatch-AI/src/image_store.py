from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from .config import settings


class AnalysisImageStore:
    """Stores display previews locally; it deliberately keeps no external uploads."""

    def __init__(self, root: Path | None = None):
        self.root = (root or settings.runs_dir) / "assets"
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, analysis_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", analysis_id):
            raise KeyError("Некорректный ID анализа.")
        return self.root / f"{analysis_id}.png"

    def save(self, image: np.ndarray, analysis_id: str) -> Path:
        target = self.path(analysis_id)
        preview = Image.fromarray(image.astype(np.uint8))
        preview.thumbnail((1600, 1600))
        preview.save(target, format="PNG", optimize=True)
        return target

    def get(self, analysis_id: str) -> Path | None:
        target = self.path(analysis_id)
        return target if target.exists() else None


image_store = AnalysisImageStore()
