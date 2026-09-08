from pathlib import Path

from src.change_detection import ChangeEvent
from src.models import AnalysisResult, ImageMetadata
from src.temporal_store import TemporalStore


def result(name: str, image_id: str) -> AnalysisResult:
    return AnalysisResult(
        analysis_id=name, image_id=image_id,
        metadata=ImageMetadata(filename=f"{name}.png", format="PNG", width=32, height=32, sha256=image_id * 4),
        detections=[], model_version="test", confidence_threshold=.25, inference_ms=1, tile_count=1,
    )


def test_temporal_comparison_is_idempotent_and_survives_new_store(tmp_path: Path):
    path = tmp_path / "timeline.sqlite3"
    before, after = result("before", "a" * 16), result("after", "b" * 16)
    event = ChangeEvent("appeared", "ship", .8, None, "det_1", 0.0)
    first = TemporalStore(path).save(before, after, [event], before_date="2026-01-01", after_date="2026-01-02")
    second = TemporalStore(path).save(before, after, [event], before_date="2026-01-01", after_date="2026-01-02")
    assert first.comparison_id == second.comparison_id
    assert len(TemporalStore(path).list()) == 1
    assert len(second.events) == 1


def test_event_review_history_and_case_link_are_persisted(tmp_path: Path):
    store = TemporalStore(tmp_path / "timeline.sqlite3")
    record = store.save(result("before", "a" * 16), result("after", "b" * 16), [ChangeEvent("disappeared", "aircraft", .7, "det_1", None, 0.0)])
    reviewed = store.review(record.comparison_id, record.events[0].event_id, "confirmed", "Проверено по фрагментам")
    assert reviewed.review_status == "confirmed"
    assert reviewed.reviewed_at
    store.attach_to_case("case-1", record.comparison_id)
    store.attach_to_case("case-1", record.comparison_id)
    assert store.comparisons_for_case("case-1") == [record.comparison_id]
