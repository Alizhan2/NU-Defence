import json
from pathlib import Path

from PIL import Image

from src.model_audit import audit_dataset, audit_metrics, probe_checkpoint, source_scene


def make_dataset(root: Path, test_classes=(0, 1, 2, 3)) -> Path:
    for index, (split, scene) in enumerate((("train", "A"), ("val", "B"), ("test", "C"))):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8), (10 + index * 30, 20, 30)).save(image_dir / f"{scene}__tile.jpg")
        classes = test_classes if split == "test" else (0, 1, 2, 3)
        (label_dir / f"{scene}__tile.txt").write_text("".join(f"{item} .1 .1 .2 .1 .2 .2 .1 .2\n" for item in classes), encoding="utf-8")
    config = root / "data.yaml"
    config.write_text(json.dumps({"path": str(root), "train": "images/train", "val": "images/val", "test": "images/test", "names": {"0": "aircraft", "1": "ship", "2": "small vehicle", "3": "large vehicle"}}), encoding="utf-8")
    return config


def test_source_scene_groups_dota_tiles():
    assert source_scene("P1000__1024__0___0") == "P1000"


def test_dataset_audit_accepts_scene_separated_complete_test(tmp_path):
    report = audit_dataset(make_dataset(tmp_path))
    assert report["valid_for_reproducible_training"] is True
    assert report["valid_for_independent_evaluation"] is True
    assert not any(report["scene_leakage"].values())


def test_dataset_audit_rejects_missing_test_class(tmp_path):
    report = audit_dataset(make_dataset(tmp_path, test_classes=(0,)))
    assert report["valid_for_independent_evaluation"] is False
    assert report["splits"]["test"]["missing_required_classes"] == ["large vehicle", "ship", "small vehicle"]


def test_dataset_audit_detects_scene_leakage(tmp_path):
    config = make_dataset(tmp_path)
    source = tmp_path / "images" / "train" / "A__tile.jpg"
    target = tmp_path / "images" / "test" / "A__another.jpg"
    target.write_bytes(source.read_bytes())
    (tmp_path / "labels" / "test" / "A__another.txt").write_text("0 .1 .1 .2 .1 .2 .2 .1 .2\n", encoding="utf-8")
    report = audit_dataset(config)
    assert report["scene_leakage"]["train_test"] == ["A"]
    assert report["content_hash_leakage_counts"]["train_test"] == 1


def test_metrics_require_explicit_test_and_fingerprint(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"dataset": "DOTA4", "model": "x", "seed": 42, "precision": .8, "recall": .7, "f1": .75, "map50": .8, "map50_95": .6}), encoding="utf-8")
    report = audit_metrics(metrics)
    assert report["status"] == "unverified"
    assert "explicit split=test is missing" in report["issues"]


def test_metrics_reject_mismatched_dataset_fingerprint(tmp_path):
    data = audit_dataset(make_dataset(tmp_path / "dataset"))
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"dataset": "DOTA4", "model": "x", "seed": 42, "precision": .8, "recall": .7, "f1": .75, "map50": .8, "map50_95": .6, "split": "test", "dataset_fingerprint": "sha256:" + "0" * 64}), encoding="utf-8")
    report = audit_metrics(metrics, data)
    assert "dataset_fingerprint does not match the audited dataset" in report["issues"]


def test_probe_distinguishes_missing_checkpoint(tmp_path):
    report = probe_checkpoint(tmp_path / "missing.pt")
    assert report["file_found"] is False
    assert report["file_loaded"] is False
    assert report["probe_completed"] is False
