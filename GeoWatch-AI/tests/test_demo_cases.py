import json

from src.demo_cases import load_demo_case_catalog, load_spacenet_baseline_evidence


def test_catalog_loads_records_and_does_not_call_missing_images_ready(tmp_path):
    catalog = tmp_path / "data" / "demo_cases" / "catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({"records": [{
        "id": "sn7-case-1", "title": "New building", "event_type": "appeared",
        "before_image": "assets/demo/before.png", "after_image": "assets/demo/after.png",
        "before_date": "2020-01-01", "after_date": "2020-02-01", "aoi": "AOI-1",
        "source_url": "https://example.test", "license": "CC BY-SA 4.0",
        "evidence_status": "ground_truth", "note": "footprint labels", "attribution": "SpaceNet",
    }]}), encoding="utf-8")

    cases = load_demo_case_catalog(tmp_path)

    assert len(cases) == 1
    assert cases[0].event_label == "Объект появился"
    assert cases[0].evidence_label == "Разметка источника"
    assert cases[0].is_ready is False


def test_catalog_accepts_generator_event_and_evidence_names(tmp_path):
    catalog = tmp_path / "data" / "demo_cases" / "catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({"cases": [{
        "id": "generated", "title": "No confirmed event", "event_type": "negative_or_uncertain",
        "before_image": "cases/generated/before.png", "after_image": "cases/generated/after.png",
        "before_date": "2020-01-01", "after_date": "2020-02-01",
        "evidence_status": "ground_truth_annotation_derived",
    }]}), encoding="utf-8")

    case = load_demo_case_catalog(tmp_path)[0]

    assert case.event_type == "uncertain"
    assert case.evidence_status == "ground_truth"
    assert case.before_image == catalog.parent / "cases" / "generated" / "before.png"


def test_catalog_rejects_paths_outside_project(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"cases": [{
        "id": "bad", "title": "Bad", "event_type": "uncertain",
        "before_image": "../before.png", "after_image": "after.png",
        "before_date": "2020-01-01", "after_date": "2020-02-01",
    }]}), encoding="utf-8")

    try:
        load_demo_case_catalog(tmp_path, catalog)
    except ValueError as exc:
        assert "пределы проекта" in str(exc)
    else:
        raise AssertionError("unsafe catalog path was accepted")


def test_missing_baseline_is_explicitly_not_trained(tmp_path):
    evidence = load_spacenet_baseline_evidence(tmp_path)
    assert evidence.status == "not_trained"
    assert evidence.metrics == {}


def test_metrics_require_test_split_and_hashes_to_be_verified(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"verified": True, "split": "val", "metrics": {"iou": 0.72}}), encoding="utf-8")
    assert load_spacenet_baseline_evidence(tmp_path, metrics).status == "unverified"

    metrics.write_text(json.dumps({
        "verified": True, "split": "test", "checkpoint_sha256": "a" * 64,
        "manifest_sha256": "b" * 64, "metrics": {"iou": 0.72, "f1": 0.81},
    }), encoding="utf-8")
    evidence = load_spacenet_baseline_evidence(tmp_path, metrics)
    assert evidence.status == "verified_test"
    assert evidence.metrics["iou"] == 0.72
