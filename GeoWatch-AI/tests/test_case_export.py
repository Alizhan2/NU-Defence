import io
import zipfile

import numpy as np

from src.case_export import export_case_bundle
from src.case_service import CaseStore
from src.image_store import AnalysisImageStore
from src.models import AnalysisResult, CaseCreateRequest, ImageMetadata
from src.review_service import ReviewStore
from src.temporal_store import TemporalStore


def result(name: str, image: str) -> AnalysisResult:
    return AnalysisResult(
        analysis_id=name, image_id=image,
        metadata=ImageMetadata(filename=f"{name}.png", format="PNG", width=8, height=8, sha256=image * 4),
        detections=[], model_version="test", confidence_threshold=.25, inference_ms=1, tile_count=1,
    )


def test_case_bundle_contains_linked_temporal_evidence(tmp_path):
    cases, analyses = CaseStore(tmp_path), ReviewStore(tmp_path)
    images, timelines = AnalysisImageStore(tmp_path), TemporalStore(tmp_path / "timeline.sqlite3")
    before, after = result("before", "a" * 16), result("after", "b" * 16)
    analyses.save(before); analyses.save(after)
    images.save(np.zeros((8, 8, 3), dtype=np.uint8), before.analysis_id)
    comparison = timelines.save(before, after, [])
    case = cases.create(CaseCreateRequest(title="Проверяемый кейс"))
    timelines.attach_to_case(case.case_id, comparison.comparison_id)

    archive = zipfile.ZipFile(io.BytesIO(export_case_bundle(
        case.case_id, cases=cases, analyses=analyses, timelines=timelines, images=images,
    )))
    names = set(archive.namelist())
    assert "manifest.json" in names and "case.json" in names
    assert f"comparisons/{comparison.comparison_id}.json" in names
    assert "analyses/before.json" in names and "analyses/after.json" in names
    assert "evidence/before.png" in names
