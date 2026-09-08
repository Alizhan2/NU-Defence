import hashlib
import json

from scripts.promote_model import promote


def metrics(weights):
    return {"dataset":"DOTA4","dataset_fingerprint":"sha256:" + "a" * 64,"split":"test","model":"yolov8n-obb","checkpoint_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),"seed":42,"precision":.8,"recall":.7,"f1":.75,"map50":.8,"map50_95":.6}


def test_promote_dry_run_validates_without_replacing(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"0" * 1_000_001)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics(weights)), encoding="utf-8")
    destination = tmp_path / "models" / "best.pt"
    card = promote(weights, metrics_path, "dota4-v1", destination=destination, project_root=tmp_path)
    assert card["model_version"] == "dota4-v1"
    assert not destination.exists()


def test_promote_apply_backs_up_existing_weight(tmp_path):
    weights = tmp_path / "incoming.pt"
    weights.write_bytes(b"1" * 1_000_001)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics(weights)), encoding="utf-8")
    destination = tmp_path / "models" / "best.pt"
    destination.parent.mkdir()
    destination.write_bytes(b"old")
    promote(weights, metrics_path, "dota4-v1", destination=destination, apply=True, project_root=tmp_path)
    assert destination.read_bytes() == weights.read_bytes()
    assert list(destination.parent.glob("best.backup-*.pt"))


def test_promote_rejects_metrics_for_other_checkpoint(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"1" * 1_000_001)
    metrics_path = tmp_path / "metrics.json"
    payload = metrics(weights)
    payload["checkpoint_sha256"] = "0" * 64
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        promote(weights, metrics_path, "wrong-v1", destination=tmp_path / "models" / "best.pt", project_root=tmp_path)
    except ValueError as exc:
        assert "не совпадает" in str(exc)
    else:
        raise AssertionError("promotion must reject metrics from a different checkpoint")
