from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class DemoEvidence:
    status: str
    class_name: str
    confidence: float
    before_id: str | None
    after_id: str | None
    overlap: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class DemoTemporalCase:
    title: str
    area_name: str
    before_date: date
    after_date: date
    before: np.ndarray
    after: np.ndarray
    events: tuple[DemoEvidence, ...]
    disclaimer: str


def _base_scene(width: int = 960, height: int = 600) -> Image.Image:
    """Build a deterministic synthetic scene for an honest UI walkthrough."""
    rng = np.random.default_rng(17)
    yy, xx = np.mgrid[0:height, 0:width]
    terrain = 48 + 14 * np.sin(xx / 61) + 9 * np.cos(yy / 43)
    noise = rng.normal(0, 5, (height, width))
    image = np.stack(
        [terrain + noise, terrain * 1.08 + noise, terrain * .92 + noise], axis=-1
    )
    canvas = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB").filter(
        ImageFilter.GaussianBlur(.7)
    )
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(0, 425), (245, 375), (545, 430), (960, 360), (960, 600), (0, 600)], fill=(59, 71, 66))
    draw.line([(0, 470), (260, 418), (545, 472), (960, 400)], fill=(132, 139, 129), width=22)
    draw.line([(0, 470), (260, 418), (545, 472), (960, 400)], fill=(84, 92, 87), width=14)
    for x, y, w, h in ((95, 105, 132, 92), (276, 119, 108, 74), (692, 96, 146, 101), (705, 258, 94, 62)):
        draw.rectangle((x, y, x + w, y + h), fill=(104, 111, 104), outline=(154, 162, 152), width=3)
        draw.line((x + 8, y + h // 2, x + w - 8, y + h // 2), fill=(83, 91, 86), width=2)
    return canvas


def load_demo_temporal_case() -> DemoTemporalCase:
    before = _base_scene()
    after = before.copy()
    draw = ImageDraw.Draw(after)

    # Synthetic construction footprint. This is ground-truth demo evidence, not ML output.
    bbox = (452, 208, 594, 322)
    draw.rectangle(bbox, fill=(119, 126, 118), outline=(180, 188, 176), width=4)
    draw.line((460, 217, 586, 313), fill=(88, 96, 91), width=3)
    draw.line((586, 217, 460, 313), fill=(88, 96, 91), width=3)

    event = DemoEvidence(
        status="appeared",
        class_name="structure",
        confidence=.91,
        before_id=None,
        after_id="demo-structure-01",
        overlap=0.0,
        bbox=bbox,
    )
    return DemoTemporalCase(
        title="Появление нового объекта",
        area_name="Учебная зона A-17",
        before_date=date(2026, 5, 12),
        after_date=date(2026, 6, 3),
        before=np.asarray(before),
        after=np.asarray(after),
        events=(event,),
        disclaimer=(
            "Синтетический сценарий для демонстрации интерфейса. Confidence и событие заданы "
            "как контрольный пример и не являются метрикой обученной модели."
        ),
    )


def annotate_demo(image: np.ndarray, events: tuple[DemoEvidence, ...]) -> Image.Image:
    output = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(output)
    for event in events:
        x1, y1, x2, y2 = event.bbox
        draw.rectangle((x1, y1, x2, y2), outline="#45e6cf", width=4)
        draw.rectangle((x1, max(0, y1 - 28), x1 + 205, y1), fill="#07151a")
        draw.text((x1 + 7, max(2, y1 - 22)), f"candidate · {event.confidence:.0%}", fill="#45e6cf")
    return output
