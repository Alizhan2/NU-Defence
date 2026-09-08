import json
import builtins

import numpy as np
import pytest

from src.geospatial_alignment import (
    AlignmentError,
    AlignmentReport,
    GridParameters,
    align_geotiff_pair,
    estimate_translation,
    normalize_aligned_pair,
    persist_alignment_report,
    plan_geotiff_alignment,
    register_translation_pair,
)
from src.models import ImageMetadata


def _metadata(*, crs="EPSG:4326", bounds=None, transform=None):
    return ImageMetadata(
        filename="scene.tif", format="GeoTIFF", width=100, height=100,
        sha256="0" * 64, crs=crs, bounds=bounds or [0, 0, 1, 1],
        transform=transform or [0.01, 0, 0, 0, -0.01, 1],
    )


def _textured_image(height=96, width=96):
    rng = np.random.default_rng(8)
    image = rng.normal(120, 35, (height, width)).clip(0, 255).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=2)


def _shift_with_zeros(image, dy, dx):
    shifted = np.zeros_like(image)
    sy0, sy1 = max(0, -dy), min(image.shape[0], image.shape[0] - dy)
    sx0, sx1 = max(0, -dx), min(image.shape[1], image.shape[1] - dx)
    dy0, dy1 = sy0 + dy, sy1 + dy
    dx0, dx1 = sx0 + dx, sx1 + dx
    shifted[dy0:dy1, dx0:dx1] = image[sy0:sy1, sx0:sx1]
    return shifted


def test_translation_estimator_returns_shift_to_apply():
    before = _textured_image()
    after = _shift_with_zeros(before, 5, -4)
    shift, score = estimate_translation(before, after)
    assert shift == (-5.0, 4.0)
    assert score > 0.35


def test_translation_registration_crops_to_valid_overlap_and_warns():
    before = _textured_image()
    after = _shift_with_zeros(before, 3, 6)
    result = register_translation_pair(before, after)
    assert result.report.status == "warning"
    assert result.report.safe_for_change_detection
    assert result.before.shape == result.after.shape == (93, 90, 3)
    assert result.report.shift_to_apply_px == (-3.0, -6.0)
    assert np.array_equal(result.before, result.after)
    assert "географической привязкой" in result.report.assumptions[2]


def test_translation_rejects_different_shapes():
    result = register_translation_pair(np.zeros((20, 20, 3)), np.zeros((21, 20, 3)))
    assert result.report.status == "reject"
    assert not result.report.safe_for_change_detection


def test_geotiff_plan_requires_complete_georeferencing():
    missing = _metadata(crs=None)
    missing.crs = None
    report = plan_geotiff_alignment(missing, _metadata())
    assert report.status == "reject"
    assert not report.aligned


def test_geotiff_plan_accepts_reprojection_candidate():
    first = _metadata(crs="EPSG:4326")
    second = _metadata(crs="EPSG:3857", bounds=[0, 0, 1000, 1000], transform=[10, 0, 0, 0, -10, 1000])
    report = plan_geotiff_alignment(first, second)
    assert report.mode == "geospatial"
    assert report.status == "warning"
    assert not report.safe_for_change_detection  # planning is not alignment


def test_report_persistence_contains_transform(tmp_path):
    grid = GridParameters(
        crs="EPSG:4326", transform=(0.01, 0.0, 10.0, 0.0, -0.01, 20.0),
        width=100, height=80, bounds=(10.0, 19.2, 11.0, 20.0),
        resolution=(0.01, 0.01), nodata=None,
    )
    report = AlignmentReport(
        status="ready", mode="geospatial", aligned=True, safe_for_change_detection=True,
        overlap_ratio=1.0, valid_data_fraction=0.9, estimated_shift_px=(0.0, 0.0),
        shift_to_apply_px=(0.0, 0.0), residual_shift_px=(0.0, 0.0), registration_score=0.8,
        common_grid=grid, assumptions=(), issues=(),
    )
    path = persist_alignment_report(report, tmp_path / "alignment.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["safe_for_change_detection"] is True
    assert payload["mode"] == "geospatial"
    assert payload["common_grid"]["transform"] == [0.01, 0.0, 10.0, 0.0, -0.01, 20.0]


def test_shared_normalization_returns_uint8_pair():
    before = np.arange(48, dtype=np.float32).reshape(4, 4, 3)
    after = before + 10
    first, second = normalize_aligned_pair(before, after, np.ones((4, 4), dtype=bool))
    assert first.dtype == second.dtype == np.uint8
    assert first.shape == second.shape == (4, 4, 3)
    assert float(second.mean()) > float(first.mean())


def test_geotiff_alignment_fails_explicitly_without_rasterio(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "rasterio" or name.startswith("rasterio."):
            raise ImportError("simulated optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(AlignmentError, match="rasterio"):
        align_geotiff_pair(b"before", b"after")
