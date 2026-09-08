import numpy as np

from src.image_store import AnalysisImageStore


def test_analysis_preview_is_saved_and_loaded(tmp_path):
    store = AnalysisImageStore(tmp_path)
    path = store.save(np.zeros((12, 16, 3), dtype=np.uint8), "analysis-1")
    assert path.exists()
    assert store.get("analysis-1") == path
