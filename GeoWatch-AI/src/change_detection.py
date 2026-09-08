from __future__ import annotations

from dataclasses import dataclass

from .models import AnalysisResult, Detection


@dataclass(frozen=True)
class ChangeEvent:
    status: str
    class_name: str
    confidence: float
    before_id: str | None
    after_id: str | None
    overlap: float


def _normalized_box(detection: Detection, result: AnalysisResult) -> tuple[float, float, float, float]:
    width = max(1, result.metadata.width)
    height = max(1, result.metadata.height)
    box = detection.bbox
    return box.x1 / width, box.y1 / height, box.x2 / width, box.y2 / height


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def compare_results(before: AnalysisResult, after: AnalysisResult, match_iou: float = 0.25) -> list[ChangeEvent]:
    """Match same-class detections and expose candidates for human review.

    Coordinates are normalized, so equal areas may be compared at different image
    resolutions. The caller remains responsible for spatial co-registration.
    """
    unmatched_after = set(range(len(after.detections)))
    events: list[ChangeEvent] = []

    for old in before.detections:
        old_box = _normalized_box(old, before)
        candidates = [
            (index, _iou(old_box, _normalized_box(new, after)))
            for index, new in enumerate(after.detections)
            if index in unmatched_after and new.class_name == old.class_name
        ]
        best_index, best_overlap = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        if best_overlap >= match_iou:
            new = after.detections[best_index]
            unmatched_after.remove(best_index)
            events.append(ChangeEvent("stable", old.class_name, max(old.confidence, new.confidence), old.id, new.id, best_overlap))
        else:
            events.append(ChangeEvent("disappeared", old.class_name, old.confidence, old.id, None, best_overlap))

    for index in sorted(unmatched_after):
        new = after.detections[index]
        events.append(ChangeEvent("appeared", new.class_name, new.confidence, None, new.id, 0.0))
    return events
