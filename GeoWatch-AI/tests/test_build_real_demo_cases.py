import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from src.demo_cases import load_demo_case_catalog


SPEC = importlib.util.spec_from_file_location(
    "build_real_demo_cases",
    Path(__file__).parents[1] / "scripts" / "build_real_demo_cases.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_directional_change_respects_tolerance():
    before = np.zeros((20, 20), dtype=np.uint8)
    after = np.zeros_like(before)
    before[4:8, 4:8] = 1
    after[5:9, 5:9] = 1
    appeared, disappeared = MODULE.directional_change(before, after, tolerance=1)
    assert not appeared.any()
    assert not disappeared.any()


def test_case_selector_returns_three_distinct_event_types():
    appeared = np.zeros((12, 12), dtype=bool)
    disappeared = np.zeros_like(appeared)
    appeared[0:3, 0:3] = True
    disappeared[9:12, 9:12] = True
    candidates = MODULE.window_candidates(appeared, disappeared, pair_index=0, crop_size=4, stride=4)
    result = MODULE.choose_case_candidates(candidates)
    identities = {(item.pair_index, item.row, item.col) for item in result.values()}
    assert set(result) == {"appeared", "disappeared_or_changed", "negative_or_uncertain"}
    assert len(identities) == 3
    assert result["appeared"].appeared_pixels == 9
    assert result["disappeared_or_changed"].disappeared_pixels == 9
    assert result["negative_or_uncertain"].changed_pixels == 0


def test_generated_catalog_schema_loads_in_ui(tmp_path):
    catalog_dir = tmp_path / "data" / "demo_cases"
    case_dir = catalog_dir / "cases" / "sn7-case"
    case_dir.mkdir(parents=True)
    (case_dir / "before.png").write_bytes(b"png")
    (case_dir / "after.png").write_bytes(b"png")
    (catalog_dir / "catalog.json").write_text(json.dumps({"records": [{
        "id": "sn7-case", "title": "Appeared", "event_type": "appeared",
        "before_image": "cases/sn7-case/before.png", "after_image": "cases/sn7-case/after.png",
        "before_date": "2019-01-01", "after_date": "2019-02-01", "aoi": "AOI",
        "source_url": MODULE.SOURCE_URL, "license": MODULE.LICENSE,
        "evidence_status": "ground_truth", "note": "annotation-derived",
        "attribution": MODULE.ATTRIBUTION,
    }]}), encoding="utf-8")

    case = load_demo_case_catalog(tmp_path)[0]
    assert case.is_ready
    assert case.event_type == "appeared"
    assert case.evidence_status == "ground_truth"
