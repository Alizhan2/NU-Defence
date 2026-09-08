from src.models import ImageMetadata
from src.pair_validation import validate_image_pair


def metadata(width=100, height=100, crs=None, bounds=None):
    return ImageMetadata(filename="x.tif", format="GeoTIFF", width=width, height=height, sha256="0" * 64, crs=crs, bounds=bounds)


def test_matching_georeferenced_pair_is_ready():
    report = validate_image_pair(metadata(crs="EPSG:4326", bounds=[0, 0, 1, 1]), metadata(crs="EPSG:4326", bounds=[0, 0, 1, 1]))
    assert report.status == "ready"
    assert report.overlap_ratio == 1.0


def test_non_overlapping_pair_is_rejected():
    report = validate_image_pair(metadata(crs="EPSG:4326", bounds=[0, 0, 1, 1]), metadata(crs="EPSG:4326", bounds=[2, 2, 3, 3]))
    assert report.status == "reject"
    assert not report.safe_to_compare


def test_ungeoreferenced_pair_requires_manual_confirmation():
    report = validate_image_pair(metadata(), metadata())
    assert report.status == "warning"
    assert report.safe_to_compare


def test_different_aspect_ratios_are_rejected():
    report = validate_image_pair(metadata(width=100, height=100), metadata(width=160, height=100))
    assert report.status == "reject"
