import json

from src.config import model_evidence


def metrics():
    return {"dataset": "DOTA4 test", "model": "yolov8n-obb", "seed": 42, "precision": .8, "recall": .7, "f1": .75, "map50": .8, "map50_95": .6}


def test_smoke_metrics_are_not_promoted(tmp_path):
    path = tmp_path / "data" / "runs"
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps({"dataset": "DOTA8", "purpose": "pipeline smoke test"}), encoding="utf-8")
    assert model_evidence(tmp_path)["state"] == "smoke"


def test_model_card_is_promoted_evidence(tmp_path):
    path = tmp_path / "models"
    path.mkdir()
    (path / "model_card.json").write_text(json.dumps({"model_version": "dota4-v1", "evidence_status": "verified", "metrics": metrics()}), encoding="utf-8")
    evidence = model_evidence(tmp_path)
    assert evidence["state"] == "promoted"
    assert evidence["model_version"] == "dota4-v1"


def test_unverified_model_card_is_not_promoted(tmp_path):
    path = tmp_path / "models"
    path.mkdir()
    (path / "model_card.json").write_text(json.dumps({"model_version": "candidate-v1", "metrics": metrics()}), encoding="utf-8")
    assert model_evidence(tmp_path)["state"] == "unverified"
