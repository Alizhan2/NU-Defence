from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np

from .models import ImageMetadata


AlignmentStatus = Literal["ready", "warning", "reject"]
AlignmentMode = Literal["geospatial", "pixel_translation", "unregistered"]


class AlignmentError(ValueError):
    """Raised when a pair cannot be aligned without unsafe assumptions."""


@dataclass(frozen=True)
class GridParameters:
    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    nodata: float | None


@dataclass(frozen=True)
class AlignmentReport:
    status: AlignmentStatus
    mode: AlignmentMode
    aligned: bool
    safe_for_change_detection: bool
    overlap_ratio: float
    valid_data_fraction: float | None
    estimated_shift_px: tuple[float, float] | None
    shift_to_apply_px: tuple[float, float] | None
    residual_shift_px: tuple[float, float] | None
    registration_score: float | None
    common_grid: GridParameters | None
    assumptions: tuple[str, ...]
    issues: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentResult:
    before: np.ndarray
    after: np.ndarray
    valid_mask: np.ndarray
    report: AlignmentReport


def normalize_aligned_pair(before: np.ndarray, after: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply one shared percentile stretch and return RGB uint8 model inputs."""
    first, second = np.asarray(before, dtype=np.float32), np.asarray(after, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 3:
        raise AlignmentError("Совмещённые кадры должны иметь одинаковую HWC-форму.")
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != first.shape[:2] or not mask.any():
        raise AlignmentError("Нет общей валидной области для нормализации.")
    output = [np.zeros(first.shape, dtype=np.uint8), np.zeros(second.shape, dtype=np.uint8)]
    for channel in range(first.shape[2]):
        values = np.concatenate((first[..., channel][mask], second[..., channel][mask]))
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        low, high = np.percentile(values, [2, 98])
        scale = max(float(high - low), 1e-6)
        for destination, source in zip(output, (first, second)):
            normalized = np.clip((np.nan_to_num(source[..., channel], nan=low) - low) / scale * 255, 0, 255)
            destination[..., channel] = normalized.astype(np.uint8)
    return output[0], output[1]


def _gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float64)
    if array.ndim != 3 or array.shape[2] < 1:
        raise AlignmentError("Ожидается двумерное или многоканальное изображение.")
    rgb = array[..., : min(array.shape[2], 3)].astype(np.float64)
    if rgb.shape[2] == 1:
        return rgb[..., 0]
    if rgb.shape[2] == 2:
        return rgb.mean(axis=2)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _normalized_correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if int(valid.sum()) < 16:
        return 0.0
    a = a[valid]
    b = b[valid]
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)) if denominator else 0.0


def estimate_translation(before: np.ndarray, after: np.ndarray) -> tuple[tuple[float, float], float]:
    """Estimate the integer (dy, dx) translation to apply to ``after``.

    This is phase correlation with a Hann window and a conservative quality
    score. It does not estimate rotation, scale, perspective or geolocation.
    """
    first = _gray(before)
    second = _gray(after)
    if first.shape != second.shape or first.ndim != 2:
        raise AlignmentError("Translation-регистрация требует одинакового размера кадров.")
    if min(first.shape) < 8:
        raise AlignmentError("Кадры слишком малы для устойчивой регистрации.")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise AlignmentError("Для регистрации нужны конечные значения пикселей.")
    if float(first.std()) < 1e-8 or float(second.std()) < 1e-8:
        raise AlignmentError("Однородные кадры нельзя надёжно зарегистрировать.")

    window = np.outer(np.hanning(first.shape[0]), np.hanning(first.shape[1]))
    first = (first - first.mean()) * window
    second = (second - second.mean()) * window
    cross = np.fft.fft2(first) * np.conj(np.fft.fft2(second))
    magnitude = np.abs(cross)
    cross /= np.maximum(magnitude, np.finfo(np.float64).eps)
    surface = np.abs(np.fft.ifft2(cross))
    peak_index = np.unravel_index(int(np.argmax(surface)), surface.shape)
    dy, dx = (int(peak_index[0]), int(peak_index[1]))
    if dy > first.shape[0] // 2:
        dy -= first.shape[0]
    if dx > first.shape[1] // 2:
        dx -= first.shape[1]

    # The peak ratio rejects ambiguous repetitive textures. Correlation after
    # cropping rejects pairs which merely happen to have a sharp FFT peak.
    peak = float(surface[peak_index])
    excluded = surface.copy()
    y0, y1 = max(0, peak_index[0] - 2), min(surface.shape[0], peak_index[0] + 3)
    x0, x1 = max(0, peak_index[1] - 2), min(surface.shape[1], peak_index[1] + 3)
    excluded[y0:y1, x0:x1] = 0
    second_peak = float(excluded.max(initial=0.0))
    peak_quality = peak / max(peak + second_peak, np.finfo(np.float64).eps)
    before_crop, after_crop = _translation_overlap(before, after, dy, dx)
    correlation = max(0.0, _normalized_correlation(_gray(before_crop), _gray(after_crop)))
    return (float(dy), float(dx)), float(np.clip(peak_quality * correlation, 0.0, 1.0))


def _translation_overlap(
    before: np.ndarray, after: np.ndarray, dy: int, dx: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = before.shape[:2]
    by0, by1 = max(0, dy), min(height, height + dy)
    bx0, bx1 = max(0, dx), min(width, width + dx)
    if by1 <= by0 or bx1 <= bx0:
        raise AlignmentError("Оценённый сдвиг не оставляет общей области.")
    ay0, ay1 = by0 - dy, by1 - dy
    ax0, ax1 = bx0 - dx, bx1 - dx
    return before[by0:by1, bx0:bx1], after[ay0:ay1, ax0:ax1]


def register_translation_pair(
    before: np.ndarray,
    after: np.ndarray,
    *,
    max_shift_fraction: float = 0.15,
    min_registration_score: float = 0.35,
) -> AlignmentResult:
    """Register non-georeferenced images under a translation-only model."""
    first = np.asarray(before)
    second = np.asarray(after)
    issues: list[str] = []
    assumptions = (
        "Кадры показывают одну территорию в одинаковом масштабе и ориентации.",
        "Компенсируется только целочисленный сдвиг; поворот, масштаб и перспектива не исправляются.",
        "Пиксельное совмещение не является географической привязкой.",
    )
    if first.shape != second.shape:
        issues.append("Размеры или число каналов различаются; translation-регистрация остановлена.")
        report = AlignmentReport(
            status="reject", mode="unregistered", aligned=False, safe_for_change_detection=False,
            overlap_ratio=0.0, valid_data_fraction=None, estimated_shift_px=None,
            shift_to_apply_px=None, residual_shift_px=None, registration_score=None,
            common_grid=None, assumptions=assumptions, issues=tuple(issues),
        )
        return AlignmentResult(first, second, np.zeros(first.shape[:2], dtype=bool), report)

    try:
        shift, score = estimate_translation(first, second)
    except AlignmentError as exc:
        report = AlignmentReport(
            status="reject", mode="unregistered", aligned=False, safe_for_change_detection=False,
            overlap_ratio=0.0, valid_data_fraction=None, estimated_shift_px=None,
            shift_to_apply_px=None, residual_shift_px=None, registration_score=None,
            common_grid=None, assumptions=assumptions, issues=(str(exc),),
        )
        return AlignmentResult(first, second, np.zeros(first.shape[:2], dtype=bool), report)

    dy, dx = int(round(shift[0])), int(round(shift[1]))
    before_crop, after_crop = _translation_overlap(first, second, dy, dx)
    overlap = float(before_crop.shape[0] * before_crop.shape[1] / (first.shape[0] * first.shape[1]))
    too_large = abs(dy) > first.shape[0] * max_shift_fraction or abs(dx) > first.shape[1] * max_shift_fraction
    if too_large:
        issues.append("Сдвиг превышает допустимую долю кадра; требуется ручная контрольная привязка.")
    if score < min_registration_score:
        issues.append("Регистрационная оценка недостаточна для автоматического сравнения.")
    safe = not issues
    valid_mask = np.ones(before_crop.shape[:2], dtype=bool)
    residual: tuple[float, float] | None = None
    if safe:
        try:
            residual, _ = estimate_translation(before_crop, after_crop)
        except AlignmentError:
            residual = None
    report = AlignmentReport(
        status="warning" if safe else "reject",
        mode="pixel_translation",
        aligned=safe,
        safe_for_change_detection=safe,
        overlap_ratio=overlap,
        valid_data_fraction=1.0,
        estimated_shift_px=shift,
        shift_to_apply_px=shift,
        residual_shift_px=residual,
        registration_score=score,
        common_grid=None,
        assumptions=assumptions,
        issues=tuple(issues),
    )
    return AlignmentResult(before_crop, after_crop, valid_mask, report)


def _metadata_overlap(before: ImageMetadata, after: ImageMetadata) -> float:
    if not before.bounds or not after.bounds:
        return 0.0
    left = max(before.bounds[0], after.bounds[0])
    bottom = max(before.bounds[1], after.bounds[1])
    right = min(before.bounds[2], after.bounds[2])
    top = min(before.bounds[3], after.bounds[3])
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    area = min(
        max(0.0, before.bounds[2] - before.bounds[0]) * max(0.0, before.bounds[3] - before.bounds[1]),
        max(0.0, after.bounds[2] - after.bounds[0]) * max(0.0, after.bounds[3] - after.bounds[1]),
    )
    return intersection / area if area else 0.0


def plan_geotiff_alignment(before: ImageMetadata, after: ImageMetadata) -> AlignmentReport:
    """Validate whether GeoTIFF bytes may be passed to raster alignment.

    The exact common grid is intentionally computed from the datasets by
    :func:`align_geotiff_pair`; metadata alone is not enough across CRSs.
    """
    issues: list[str] = []
    if not before.crs or not before.bounds or not before.transform:
        issues.append("Первый GeoTIFF не содержит полной CRS/bounds/transform геопривязки.")
    if not after.crs or not after.bounds or not after.transform:
        issues.append("Второй GeoTIFF не содержит полной CRS/bounds/transform геопривязки.")
    overlap = _metadata_overlap(before, after) if before.crs == after.crs else 0.0
    if before.crs and after.crs and before.crs == after.crs and overlap <= 0:
        issues.append("GeoTIFF не имеют общей географической области.")
    safe = not issues
    return AlignmentReport(
        status="warning" if safe else "reject",
        mode="geospatial" if safe else "unregistered",
        aligned=False,
        safe_for_change_detection=False,
        overlap_ratio=overlap,
        valid_data_fraction=None,
        estimated_shift_px=None,
        shift_to_apply_px=None,
        residual_shift_px=None,
        registration_score=None,
        common_grid=None,
        assumptions=("Репроекция ещё не выполнена; это только предварительная проверка метаданных.",),
        issues=tuple(issues),
    )


def align_geotiff_pair(
    before_data: bytes,
    after_data: bytes,
    *,
    minimum_valid_fraction: float = 0.50,
    maximum_residual_shift_px: float = 2.0,
) -> AlignmentResult:
    """Reproject two GeoTIFFs to their intersection on a common grid.

    Rasterio is optional. If it is unavailable, an actionable ``AlignmentError``
    is raised instead of silently treating pixel coordinates as geospatial.
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AlignmentError("Для безопасного совмещения GeoTIFF установите optional dependency rasterio>=1.4.") from exc

    if not before_data or not after_data:
        raise AlignmentError("Оба GeoTIFF должны быть непустыми.")

    with MemoryFile(before_data) as before_mem, MemoryFile(after_data) as after_mem:
        with before_mem.open() as before_ds, after_mem.open() as after_ds:
            if before_ds.crs is None or after_ds.crs is None:
                raise AlignmentError("Оба GeoTIFF должны содержать CRS.")
            target_crs = before_ds.crs
            after_bounds = transform_bounds(after_ds.crs, target_crs, *after_ds.bounds, densify_pts=21)
            left = max(before_ds.bounds.left, after_bounds[0])
            bottom = max(before_ds.bounds.bottom, after_bounds[1])
            right = min(before_ds.bounds.right, after_bounds[2])
            top = min(before_ds.bounds.top, after_bounds[3])
            if right <= left or top <= bottom:
                raise AlignmentError("GeoTIFF не имеют общей области после приведения CRS.")

            before_rx, before_ry = abs(before_ds.transform.a), abs(before_ds.transform.e)
            after_transform, _, _ = calculate_default_transform(
                after_ds.crs, target_crs, after_ds.width, after_ds.height, *after_ds.bounds
            )
            # Use the coarser native resolution: upsampling must not fabricate detail.
            resolution_x = max(before_rx, abs(after_transform.a))
            resolution_y = max(before_ry, abs(after_transform.e))
            width = int(math.floor((right - left) / resolution_x))
            height = int(math.floor((top - bottom) / resolution_y))
            if width < 2 or height < 2:
                raise AlignmentError("Общая область слишком мала для выбранного пространственного разрешения.")
            right = left + width * resolution_x
            bottom = top - height * resolution_y
            target_transform = from_origin(left, top, resolution_x, resolution_y)
            band_count = min(before_ds.count, after_ds.count, 3)
            if band_count < 1:
                raise AlignmentError("GeoTIFF не содержит растровых каналов.")

            before_array = np.full((band_count, height, width), np.nan, dtype=np.float32)
            after_array = np.full_like(before_array, np.nan)
            before_mask = np.zeros((height, width), dtype=np.uint8)
            after_mask = np.zeros_like(before_mask)
            for band in range(1, band_count + 1):
                reproject(
                    source=rasterio.band(before_ds, band), destination=before_array[band - 1],
                    src_transform=before_ds.transform, src_crs=before_ds.crs, src_nodata=before_ds.nodata,
                    dst_transform=target_transform, dst_crs=target_crs, dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                reproject(
                    source=rasterio.band(after_ds, band), destination=after_array[band - 1],
                    src_transform=after_ds.transform, src_crs=after_ds.crs, src_nodata=after_ds.nodata,
                    dst_transform=target_transform, dst_crs=target_crs, dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
            reproject(
                source=before_ds.dataset_mask(), destination=before_mask,
                src_transform=before_ds.transform, src_crs=before_ds.crs,
                dst_transform=target_transform, dst_crs=target_crs,
                src_nodata=0, dst_nodata=0, resampling=Resampling.nearest,
            )
            reproject(
                source=after_ds.dataset_mask(), destination=after_mask,
                src_transform=after_ds.transform, src_crs=after_ds.crs,
                dst_transform=target_transform, dst_crs=target_crs,
                src_nodata=0, dst_nodata=0, resampling=Resampling.nearest,
            )

            before_hwc = np.moveaxis(before_array, 0, -1)
            after_hwc = np.moveaxis(after_array, 0, -1)
            valid = (before_mask > 0) & (after_mask > 0)
            valid &= np.isfinite(before_hwc).all(axis=2) & np.isfinite(after_hwc).all(axis=2)
            valid_fraction = float(valid.mean())
            issues: list[str] = []
            if valid_fraction < minimum_valid_fraction:
                issues.append("После учёта NoData совместно валидно менее требуемой доли общей сетки.")

            residual: tuple[float, float] | None = None
            score: float | None = None
            if int(valid.sum()) >= 64:
                first_gray = np.where(valid, _gray(before_hwc), 0.0)
                second_gray = np.where(valid, _gray(after_hwc), 0.0)
                try:
                    residual, score = estimate_translation(first_gray, second_gray)
                    if math.hypot(*residual) > maximum_residual_shift_px and score >= 0.35:
                        issues.append("После геопривязки обнаружен значимый остаточный пиксельный сдвиг.")
                except AlignmentError:
                    issues.append("Недостаточно текстуры для независимой оценки остаточного сдвига.")
            else:
                issues.append("Недостаточно совместно валидных пикселей для оценки остаточного сдвига.")

            safe = not issues
            common_grid = GridParameters(
                crs=str(target_crs),
                transform=tuple(float(v) for v in tuple(target_transform)[:6]),
                width=width,
                height=height,
                bounds=(float(left), float(bottom), float(right), float(top)),
                resolution=(float(resolution_x), float(resolution_y)),
                nodata=None,
            )
            before_area = abs((before_ds.bounds.right - before_ds.bounds.left) * (before_ds.bounds.top - before_ds.bounds.bottom))
            after_area = abs((after_bounds[2] - after_bounds[0]) * (after_bounds[3] - after_bounds[1]))
            overlap_area = (right - left) * (top - bottom)
            overlap_ratio = float(overlap_area / min(before_area, after_area)) if min(before_area, after_area) else 0.0
            report = AlignmentReport(
                status="ready" if safe else "reject",
                mode="geospatial",
                aligned=True,
                safe_for_change_detection=safe,
                overlap_ratio=float(np.clip(overlap_ratio, 0.0, 1.0)),
                valid_data_fraction=valid_fraction,
                estimated_shift_px=residual,
                shift_to_apply_px=(0.0, 0.0),
                residual_shift_px=residual,
                registration_score=score,
                common_grid=common_grid,
                assumptions=(
                    "Первый GeoTIFF задаёт целевую CRS.",
                    "Используется более грубое из двух пространственных разрешений.",
                    "Спектральные/радиометрические различия этой операцией не нормализуются.",
                ),
                issues=tuple(issues),
            )
            return AlignmentResult(before_hwc, after_hwc, valid, report)


def persist_alignment_report(report: AlignmentReport, destination: str | Path) -> Path:
    """Atomically persist the exact grid and transform parameters as JSON."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return path
