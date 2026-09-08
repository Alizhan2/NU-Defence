from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


QualityStatus = Literal["ready", "warning", "reject"]


@dataclass(frozen=True)
class ImageQualityReport:
    status: QualityStatus
    brightness: float
    contrast: float
    sharpness: float
    dark_clipping: float
    bright_clipping: float
    issues: tuple[str, ...]

    @property
    def safe_for_inference(self) -> bool:
        return self.status != "reject"

    def to_dict(self) -> dict:
        return asdict(self)


def assess_image_quality(image: np.ndarray) -> ImageQualityReport:
    """Estimate basic optical quality without pretending to detect clouds.

    Scores are deterministic input diagnostics, not learned accuracy estimates.
    They are intended to catch empty, clipped, very dark and visibly blurred
    inputs before inference.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3 or array.size == 0:
        raise ValueError("Ожидается непустое RGB-изображение.")

    rgb = array[..., :3].astype(np.float32)
    if rgb.max(initial=0) > 1.0:
        rgb /= 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    brightness = float(gray.mean())
    contrast = float(gray.std())
    dark_clipping = float((gray <= 0.02).mean())
    bright_clipping = float((gray >= 0.98).mean())

    if min(gray.shape) < 3:
        sharpness = 0.0
    else:
        center = gray[1:-1, 1:-1]
        laplacian = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * center
        )
        sharpness = float(laplacian.var())

    rejects: list[str] = []
    warnings: list[str] = []
    if brightness < 0.05:
        rejects.append("Снимок почти полностью тёмный.")
    elif brightness < 0.15:
        warnings.append("Низкая средняя яркость может снизить полноту детекции.")
    if brightness > 0.97:
        rejects.append("Снимок почти полностью пересвечен.")
    elif brightness > 0.88:
        warnings.append("Высокая средняя яркость может скрывать детали.")
    if contrast < 0.015:
        rejects.append("На снимке недостаточно различимых деталей (очень низкий контраст).")
    elif contrast < 0.05:
        warnings.append("Низкий контраст: результат потребует особенно тщательной проверки.")
    if sharpness < 0.00001:
        rejects.append("Снимок однородный или слишком размытый для надёжного анализа.")
    elif sharpness < 0.00015:
        warnings.append("Низкая резкость может ухудшить локализацию объектов.")
    if dark_clipping > 0.70:
        rejects.append("Более 70% пикселей потеряно в тенях.")
    elif dark_clipping > 0.35:
        warnings.append("Значительная часть снимка потеряна в тенях.")
    if bright_clipping > 0.70:
        rejects.append("Более 70% пикселей потеряно в светах.")
    elif bright_clipping > 0.35:
        warnings.append("Значительная часть снимка потеряна в светах.")

    status: QualityStatus = "reject" if rejects else "warning" if warnings else "ready"
    return ImageQualityReport(
        status=status,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        dark_clipping=dark_clipping,
        bright_clipping=bright_clipping,
        issues=tuple(rejects + warnings),
    )
