from src.change_detection import compare_results
from src.models import AnalysisResult, BoundingBox, Detection, ImageMetadata


def result(name: str, boxes: list[tuple[str, tuple[int, int, int, int]]]) -> AnalysisResult:
    metadata = ImageMetadata(filename=name, format="PNG", width=100, height=100, sha256="0" * 64)
    detections = [
        Detection(id=f"{name}_{index}", class_name=label, confidence=0.8, bbox=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]), model_version="test")
        for index, (label, box) in enumerate(boxes)
    ]
    return AnalysisResult(analysis_id=name, image_id=name, metadata=metadata, detections=detections, model_version="test", confidence_threshold=0.25, inference_ms=1, tile_count=1)


def test_compare_results_reports_appeared_stable_and_disappeared():
    before = result("before", [("aircraft", (10, 10, 30, 30)), ("ship", (60, 60, 80, 80))])
    after = result("after", [("aircraft", (11, 11, 31, 31)), ("small vehicle", (40, 40, 50, 50))])

    events = compare_results(before, after)

    assert [event.status for event in events] == ["stable", "disappeared", "appeared"]
    assert events[0].overlap > 0.8
