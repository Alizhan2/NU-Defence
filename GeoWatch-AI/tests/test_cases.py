from src.case_service import CaseStore
from src.models import (
    AnalysisResult,
    BoundingBox,
    CaseCreateRequest,
    Detection,
    ImageMetadata,
)
from src.review_service import ReviewStore


def sample(analysis_id: str = "a"):
    return AnalysisResult(
        analysis_id=analysis_id,
        image_id="0" * 16,
        metadata=ImageMetadata(filename="x.png", format="PNG", width=10, height=10, sha256="0" * 64),
        detections=[Detection(id="det_1", class_name="ship", confidence=.8, bbox=BoundingBox(x1=1, y1=1, x2=2, y2=2), model_version="m")],
        model_version="m", confidence_threshold=.2, inference_ms=1, tile_count=1,
    )


def test_cases_can_attach_an_immutable_analysis(tmp_path):
    reviews = ReviewStore(tmp_path)
    result = sample()
    reviews.save(result)
    cases = CaseStore(tmp_path)
    record = cases.create(CaseCreateRequest(title="Harbor review", priority="high"))
    attached = cases.attach_analysis(record.case_id, result.analysis_id)
    assert attached.analysis_ids == [result.analysis_id]
    assert cases.list()[0].title == "Harbor review"


def test_case_status_can_be_updated(tmp_path):
    cases = CaseStore(tmp_path)
    record = cases.create(CaseCreateRequest(title="Review"))
    assert cases.update_status(record.case_id, "closed").status == "closed"


def test_multiple_analyses_of_one_image_are_not_overwritten(tmp_path):
    reviews = ReviewStore(tmp_path)
    reviews.save(sample("a"))
    reviews.save(sample("another"))
    assert reviews.load_by_analysis("a").analysis_id == "a"
    assert reviews.load_by_analysis("another").analysis_id == "another"
