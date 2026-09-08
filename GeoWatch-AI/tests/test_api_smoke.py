from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from src import api
from src.models import AnalysisResult, BoundingBox, Detection, ImageMetadata, ModelStatus


class PassingQuality:
    safe_for_inference = True
    issues = ()
    def to_dict(self): return {"status": "ready", "issues": []}


class PassingPair:
    safe_to_compare = True
    issues = ()
    def to_dict(self): return {"status": "ready", "issues": []}


class TemporalStub:
    def save(self, before, after, changes, **kwargs):
        events = [SimpleNamespace(**change.__dict__, event_id=f"event-{index}", comparison_id="timeline-test", review_status="needs_review", comment="", reviewed_at=None) for index, change in enumerate(changes)]
        return SimpleNamespace(comparison_id="timeline-test", events=events)


class PassingAlignment:
    before = object()
    after = object()
    valid_mask = object()
    report = SimpleNamespace(safe_for_change_detection=True, common_grid=None, to_dict=lambda: {"status": "warning", "mode": "pixel_translation"})


def sample_result() -> AnalysisResult:
    return AnalysisResult(
        analysis_id="analysis-smoke",
        image_id="a" * 16,
        metadata=ImageMetadata(
            filename="scene.png", format="PNG", width=16, height=12, sha256="a" * 64
        ),
        detections=[
            Detection(
                id="det_000001",
                class_name="ship",
                confidence=0.8,
                bbox=BoundingBox(x1=1, y1=2, x2=9, y2=10),
                model_version="test-obb-v1",
            )
        ],
        model_version="test-obb-v1",
        confidence_threshold=0.4,
        inference_ms=12.5,
        tile_count=1,
    )


def test_health_has_stable_status_and_detector_schema(monkeypatch):
    class ReadyDetector:
        def status(self):
            return ModelStatus(state="ready", model_version="test-obb-v1", message_ru="ready")

    monkeypatch.setattr(api, "detector", ReadyDetector())
    response = TestClient(api.app).get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["detector"] == {
        "state": "ready",
        "model_version": "test-obb-v1",
        "message_ru": "ready",
    }
    assert payload["model_evidence"]["state"] in {"promoted", "smoke", "unverified"}
    assert isinstance(payload["qwen_enabled"], bool)


def test_inference_returns_analysis_result_schema_and_persists(monkeypatch):
    result = sample_result()
    saved = []
    prepared = SimpleNamespace(array=object(), metadata=result.metadata)

    monkeypatch.setattr(api, "validate_and_prepare", lambda data, filename: prepared)
    monkeypatch.setattr(api, "assess_image_quality", lambda image: PassingQuality())
    monkeypatch.setattr(api, "run_inference", lambda image, metadata, detector, confidence: result)
    monkeypatch.setattr(api.store, "save", lambda item: saved.append(item))

    response = TestClient(api.app).post(
        "/api/v1/inference?confidence=0.4",
        files={"file": ("scene.png", b"not-used-by-mock", "image/png")},
    )

    assert response.status_code == 200
    returned = AnalysisResult.model_validate(response.json())
    assert returned.analysis_id == result.analysis_id
    assert returned.confidence_threshold == 0.4
    assert returned.detections[0].class_name == "ship"
    assert saved == [result]


def test_change_compare_returns_reviewable_temporal_summary(monkeypatch):
    before = sample_result()
    after = sample_result().model_copy(deep=True)
    after.analysis_id = "analysis-after"
    after.detections[0].bbox = BoundingBox(x1=12, y1=2, x2=15, y2=10)
    prepared = SimpleNamespace(array=object(), metadata=before.metadata)
    outputs = iter([before, after])

    monkeypatch.setattr(api, "validate_and_prepare", lambda data, filename: prepared)
    monkeypatch.setattr(api, "assess_image_quality", lambda image: PassingQuality())
    monkeypatch.setattr(api, "validate_image_pair", lambda before, after: PassingPair())
    monkeypatch.setattr(api, "register_translation_pair", lambda before, after: PassingAlignment())
    aligned_array = np.zeros((12, 16, 3), dtype=np.uint8)
    monkeypatch.setattr(api, "normalize_aligned_pair", lambda before, after, mask: (aligned_array, aligned_array))
    monkeypatch.setattr(api, "persist_alignment_report", lambda report, path: path)
    monkeypatch.setattr(api, "temporal_store", TemporalStub())
    monkeypatch.setattr(api, "evaluate_alert_rules", lambda *args, **kwargs: [])
    monkeypatch.setattr(api.alert_store, "save_many", lambda items: [])
    monkeypatch.setattr(api, "run_inference", lambda image, metadata, detector, confidence: next(outputs))
    monkeypatch.setattr(api.store, "save", lambda item: None)
    monkeypatch.setattr(api.image_store, "save", lambda image, analysis_id: None)

    response = TestClient(api.app).post(
        "/api/v1/changes/compare?confidence=0.4",
        files={
            "before_file": ("before.png", b"before", "image/png"),
            "after_file": ("after.png", b"after", "image/png"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == {"appeared": 1, "disappeared": 1, "stable": 0, "needs_review": 2}
    assert payload["comparison_id"] == "timeline-test"
    assert all("event_id" in item for item in payload["changes"])
    assert payload["warnings"]


def test_operational_resource_endpoints_have_stable_contract():
    client = TestClient(api.app)

    source = client.get("/api/v1/sources/earth-engine")
    zones = client.get("/api/v1/watch-zones")
    alerts = client.get("/api/v1/alerts?review_status=needs_review")

    assert source.status_code == 200
    assert {"dependency_available", "project_configured", "initialized", "ready", "collection_id"} <= source.json().keys()
    assert zones.status_code == 200 and isinstance(zones.json(), list)
    assert alerts.status_code == 200 and isinstance(alerts.json(), list)
