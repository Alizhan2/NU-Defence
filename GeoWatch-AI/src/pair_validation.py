from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .models import ImageMetadata


PairStatus = Literal["ready", "warning", "reject"]


@dataclass(frozen=True)
class PairValidationReport:
    status: PairStatus
    overlap_ratio: float | None
    issues: tuple[str, ...]

    @property
    def safe_to_compare(self) -> bool:
        return self.status != "reject"

    def to_dict(self) -> dict:
        return asdict(self)


def _overlap_ratio(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    bottom = max(first[1], second[1])
    right = min(first[2], second[2])
    top = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    denominator = min(first_area, second_area)
    return intersection / denominator if denominator else 0.0


def validate_image_pair(before: ImageMetadata, after: ImageMetadata) -> PairValidationReport:
    rejects: list[str] = []
    warnings: list[str] = []
    overlap: float | None = None

    before_ratio = before.width / before.height
    after_ratio = after.width / after.height
    aspect_delta = abs(before_ratio - after_ratio) / max(before_ratio, after_ratio)
    if aspect_delta > 0.02:
        rejects.append("Пропорции кадров различаются более чем на 2%; нормализованные координаты несопоставимы.")
    elif (before.width, before.height) != (after.width, after.height):
        warnings.append("Размеры файлов различаются; проверьте одинаковый масштаб и совмещение.")

    before_geo = bool(before.crs and before.bounds)
    after_geo = bool(after.crs and after.bounds)
    if before_geo != after_geo:
        rejects.append("Геопривязка есть только у одного снимка; автоматическое сравнение небезопасно.")
    elif before_geo and after_geo:
        if before.crs != after.crs:
            rejects.append("Системы координат различаются; перед сравнением требуется репроекция.")
        else:
            overlap = _overlap_ratio(before.bounds or [], after.bounds or [])
            if overlap < 0.70:
                rejects.append("Географическое перекрытие снимков меньше 70%.")
            elif overlap < 0.95:
                warnings.append("Географическое перекрытие неполное; изменения у границ могут быть ложными.")
    else:
        warnings.append("В файлах нет геопривязки: совпадение территории и масштаба должен подтвердить аналитик.")

    status: PairStatus = "reject" if rejects else "warning" if warnings else "ready"
    return PairValidationReport(status=status, overlap_ratio=overlap, issues=tuple(rejects + warnings))
