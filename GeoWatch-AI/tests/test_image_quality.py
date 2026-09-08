import numpy as np

from src.image_quality import assess_image_quality


def test_textured_balanced_image_passes_quality_gate():
    grid = np.indices((64, 64)).sum(axis=0) % 2
    image = np.repeat((grid * 180 + 35)[..., None], 3, axis=2).astype(np.uint8)
    report = assess_image_quality(image)
    assert report.status == "ready"
    assert report.safe_for_inference
    assert report.sharpness > 0


def test_uniform_dark_image_is_rejected():
    report = assess_image_quality(np.zeros((32, 32, 3), dtype=np.uint8))
    assert report.status == "reject"
    assert not report.safe_for_inference
    assert report.issues
