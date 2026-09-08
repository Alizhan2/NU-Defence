from __future__ import annotations

from .models import BoundingBox, Detection


def iou(a: BoundingBox, b: BoundingBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def class_aware_nms(detections: list[Detection], threshold: float = 0.45) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda d: (-d.confidence, d.id)):
        if candidate.bbox.area <= 0:
            continue
        if any(candidate.class_name == chosen.class_name and iou(candidate.bbox, chosen.bbox) > threshold for chosen in kept):
            continue
        kept.append(candidate)
    return kept


def clip_box(box: BoundingBox, width: int, height: int) -> BoundingBox:
    return BoundingBox(
        x1=max(0, min(width, box.x1)), y1=max(0, min(height, box.y1)),
        x2=max(0, min(width, box.x2)), y2=max(0, min(height, box.y2)),
    )
