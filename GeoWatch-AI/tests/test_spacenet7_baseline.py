import numpy as np

from src.spacenet7_baseline import _grid_positions, derive_target, metrics_from_confusion


def test_directional_change_target():
    before = np.array([[0, 1], [1, 0]], dtype=bool)
    after = np.array([[1, 0], [1, 0]], dtype=bool)
    assert derive_target(before, after).tolist() == [[1, 2], [0, 0]]


def test_metrics_do_not_invent_score_for_absent_class():
    confusion = np.array([[8, 0, 0], [0, 2, 0], [0, 0, 0]])
    result = metrics_from_confusion(confusion)
    assert result["classes"]["appeared"]["f1"] == 1.0
    assert result["classes"]["disappeared"]["f1"] is None
    assert result["macro_change_f1"] == 1.0


def test_eval_grid_covers_pixels_once_and_leaves_only_padding():
    positions = _grid_positions(1000, 256)
    assert positions == [0, 256, 512, 768]
    assert positions[-1] + 256 >= 1000
    assert all(right - left == 256 for left, right in zip(positions, positions[1:]))
