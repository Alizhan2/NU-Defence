import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_case_selector_refuses_to_invent_missing_event_types():
    empty = np.zeros((8, 8), dtype=bool)
    candidates = MODULE.window_candidates(empty, empty, pair_index=0, crop_size=4, stride=4)
    with pytest.raises(ValueError, match="появлением"):
        MODULE.choose_case_candidates(candidates)


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


def test_builder_materializes_three_loadable_cases(tmp_path, monkeypatch):
    before_mask = np.zeros((12, 12), dtype=np.uint8)
    after_mask = np.zeros_like(before_mask)
    before_mask[8:11, 8:11] = 1
    after_mask[0:3, 0:3] = 1

    monkeypatch.setattr(
        MODULE,
        "_read_mask",
        lambda path: before_mask if "before_mask" in str(path) else after_mask,
    )
    monkeypatch.setattr(
        MODULE,
        "_read_rgb_window",
        lambda path, item: np.full((item.height, item.width, 3), 96, dtype=np.uint8),
    )
    source = tmp_path / "source"
    source.mkdir()
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps({
        "schema": "geowatch-spacenet7-pairs-v1",
        "valid": True,
        "root": str(source),
        "pairs": [{
            "pair_id": "fixture-pair", "split": "test", "aoi": "fixture-aoi",
            "before_date": "2019-01", "after_date": "2019-02",
            "before_image": "before.tif", "after_image": "after.tif",
            "before_footprints": "before_mask.tif", "after_footprints": "after_mask.tif",
        }],
    }), encoding="utf-8")

    output = tmp_path / "data" / "demo_cases"
    catalog = MODULE.build_cases(
        pairs, output, crop_size=4, stride=4, tolerance=0,
    )
    loaded = load_demo_case_catalog(tmp_path)

    assert len(catalog["records"]) == len(loaded) == 3
    assert {case.event_type for case in loaded} == {"appeared", "disappeared", "uncertain"}
    assert all(case.evidence_status == "ground_truth" and case.is_ready for case in loaded)
    for record in catalog["records"]:
        assert set(record["sha256"]) == {"before_image", "after_image"}
        persisted = json.loads((output / "cases" / record["id"] / "record.json").read_text(encoding="utf-8"))
        assert persisted["sha256"] == record["sha256"]
