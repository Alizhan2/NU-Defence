"""Materialize three honest SpaceNet 7 temporal demo cases.

The generated evidence is derived from the official per-date building masks. It is
reference evidence for a product demonstration, not a prediction made by GeoWatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE_URL = "https://www.spacenet.ai/sn7-challenge/"
LICENSE = "CC BY-SA 4.0"
ATTRIBUTION = "SpaceNet 7 / SpaceNet Partners; imagery courtesy of Planet"


@dataclass(frozen=True)
class WindowCandidate:
    pair_index: int
    row: int
    col: int
    height: int
    width: int
    appeared_pixels: int
    disappeared_pixels: int

    @property
    def changed_pixels(self) -> int:
        return self.appeared_pixels + self.disappeared_pixels


def directional_change(before: np.ndarray, after: np.ndarray, tolerance: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return new/lost footprint pixels with a small registration tolerance."""
    before = np.asarray(before) > 0
    after = np.asarray(after) > 0
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    def dilate(mask: np.ndarray) -> np.ndarray:
        if tolerance == 0:
            return mask
        padded = np.pad(mask, tolerance, mode="constant", constant_values=False)
        expanded = np.zeros_like(mask, dtype=bool)
        height, width = mask.shape
        for row_offset in range(tolerance * 2 + 1):
            for col_offset in range(tolerance * 2 + 1):
                expanded |= padded[row_offset : row_offset + height, col_offset : col_offset + width]
        return expanded

    before_near = dilate(before)
    after_near = dilate(after)
    appeared = after & ~before_near
    disappeared = before & ~after_near
    return appeared, disappeared


def window_candidates(
    appeared: np.ndarray,
    disappeared: np.ndarray,
    *,
    pair_index: int,
    crop_size: int,
    stride: int,
) -> list[WindowCandidate]:
    height, width = appeared.shape
    crop_h, crop_w = min(crop_size, height), min(crop_size, width)
    rows = sorted(set(range(0, max(1, height - crop_h + 1), stride)) | {height - crop_h})
    cols = sorted(set(range(0, max(1, width - crop_w + 1), stride)) | {width - crop_w})
    return [
        WindowCandidate(
            pair_index=pair_index,
            row=row,
            col=col,
            height=crop_h,
            width=crop_w,
            appeared_pixels=int(appeared[row : row + crop_h, col : col + crop_w].sum()),
            disappeared_pixels=int(disappeared[row : row + crop_h, col : col + crop_w].sum()),
        )
        for row in rows
        for col in cols
    ]


def choose_case_candidates(candidates: list[WindowCandidate]) -> dict[str, WindowCandidate]:
    """Choose distinct, deterministic appeared/lost/negative windows."""
    if len(candidates) < 3:
        raise ValueError("Нужно минимум три окна-кандидата.")

    def identity(item: WindowCandidate) -> tuple[int, int, int]:
        return item.pair_index, item.row, item.col

    appeared = max(candidates, key=lambda item: (item.appeared_pixels, -item.disappeared_pixels, -item.pair_index, -item.row, -item.col))
    remaining = [item for item in candidates if identity(item) != identity(appeared)]
    disappeared = max(remaining, key=lambda item: (item.disappeared_pixels, item.changed_pixels, -item.pair_index, -item.row, -item.col))
    remaining = [item for item in remaining if identity(item) != identity(disappeared)]
    negative = min(remaining, key=lambda item: (item.changed_pixels, item.pair_index, item.row, item.col))
    return {"appeared": appeared, "disappeared_or_changed": disappeared, "negative_or_uncertain": negative}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mask(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as error:
        raise RuntimeError("Для SpaceNet GeoTIFF установите rasterio>=1.4.") from error
    with rasterio.open(path) as dataset:
        return dataset.read(1)


def _read_rgb_window(path: Path, item: WindowCandidate) -> np.ndarray:
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as dataset:
        count = min(3, dataset.count)
        data = dataset.read(
            list(range(1, count + 1)),
            window=Window(item.col, item.row, item.width, item.height),
            boundless=True,
            fill_value=0,
        )
    if count == 1:
        data = np.repeat(data, 3, axis=0)
    rgb = np.moveaxis(data[:3], 0, -1).astype(np.float32)
    valid = rgb[np.any(rgb != 0, axis=2)]
    if valid.size:
        low, high = np.percentile(valid, (2, 98))
        rgb = (rgb - low) * 255.0 / max(1.0, high - low)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _save_binary(path: Path, mask: np.ndarray, item: WindowCandidate) -> None:
    crop = mask[item.row : item.row + item.height, item.col : item.col + item.width]
    Image.fromarray((crop.astype(np.uint8) * 255), "L").save(path)


def build_cases(
    pairs_manifest: Path,
    output: Path,
    *,
    split: str = "test",
    crop_size: int = 384,
    stride: int = 192,
    tolerance: int = 2,
    max_pairs: int = 0,
) -> dict:
    manifest = json.loads(pairs_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "geowatch-spacenet7-pairs-v1" or not manifest.get("valid"):
        raise ValueError("Нужен валидный manifest от prepare_spacenet7_pairs.py.")
    source_root = Path(manifest["root"])
    pairs = [pair for pair in manifest["pairs"] if pair["split"] == split]
    if max_pairs:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError(f"В manifest нет пар split={split!r}.")

    all_candidates: list[WindowCandidate] = []
    masks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for pair_index, pair in enumerate(pairs):
        before_mask = _read_mask(source_root / pair["before_footprints"]) > 0
        after_mask = _read_mask(source_root / pair["after_footprints"]) > 0
        if before_mask.shape != after_mask.shape:
            raise ValueError(f"Размеры масок не совпадают: {pair['pair_id']}")
        appeared, disappeared = directional_change(before_mask, after_mask, tolerance)
        masks.append((before_mask, after_mask, appeared, disappeared))
        all_candidates.extend(window_candidates(appeared, disappeared, pair_index=pair_index, crop_size=crop_size, stride=stride))

    selected = choose_case_candidates(all_candidates)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    titles = {
        "appeared": "Здание появилось",
        "disappeared_or_changed": "Здание исчезло или изменило footprint",
        "negative_or_uncertain": "Нет подтверждённого события в footprint-разметке",
    }
    catalog_event_types = {
        "appeared": "appeared",
        "disappeared_or_changed": "disappeared",
        "negative_or_uncertain": "uncertain",
    }
    for event_type, item in selected.items():
        pair = pairs[item.pair_index]
        before_mask, after_mask, appeared_mask, disappeared_mask = masks[item.pair_index]
        case_id = f"sn7-{event_type}-{pair['pair_id']}"
        case_dir = output / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        before_path, after_path = case_dir / "before.png", case_dir / "after.png"
        Image.fromarray(_read_rgb_window(source_root / pair["before_image"], item), "RGB").save(before_path)
        Image.fromarray(_read_rgb_window(source_root / pair["after_image"], item), "RGB").save(after_path)
        _save_binary(case_dir / "before_footprints.png", before_mask, item)
        _save_binary(case_dir / "after_footprints.png", after_mask, item)
        _save_binary(case_dir / "appeared_evidence.png", appeared_mask, item)
        _save_binary(case_dir / "disappeared_evidence.png", disappeared_mask, item)
        relative_before = before_path.relative_to(output).as_posix()
        relative_after = after_path.relative_to(output).as_posix()
        record = {
            "id": case_id,
            "title": titles[event_type],
            "event_type": catalog_event_types[event_type],
            "before_image": relative_before,
            "after_image": relative_after,
            "before_date": pair["before_date"],
            "after_date": pair["after_date"],
            "aoi": pair["aoi"],
            "source_url": SOURCE_URL,
            "license": LICENSE,
            "evidence_status": "ground_truth",
            "note": (
                "Тип события вычислен из официальных per-date footprint masks с допуском "
                f"регистрации {tolerance}px; это не предсказание модели. "
                if event_type != "negative_or_uncertain"
                else "В этом crop нет подтверждённого directional footprint-события; это не доказывает отсутствие иных изменений. "
            ),
            "attribution": ATTRIBUTION,
            "source_pair_id": pair["pair_id"],
            "source_files": {
                key: pair[key]
                for key in ("before_image", "after_image", "before_footprints", "after_footprints")
            },
            "crop_window": {"row": item.row, "col": item.col, "height": item.height, "width": item.width},
            "evidence_pixels": {
                "appeared": item.appeared_pixels,
                "disappeared": item.disappeared_pixels,
            },
        }
        (case_dir / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        record["sha256"] = {"before_image": _sha256(before_path), "after_image": _sha256(after_path)}
        records.append(record)

    catalog = {
        "schema": "geowatch-real-demo-catalog-v1",
        "generated_from": str(pairs_manifest.resolve()),
        "selection_split": split,
        "selection_policy": "deterministic annotation-derived extrema; distinct crop windows",
        "human_review_required": True,
        "records": records,
    }
    (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="pairs.json from prepare_spacenet7_pairs.py")
    parser.add_argument("--output", type=Path, default=Path("data/demo_cases"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()
    catalog = build_cases(
        args.pairs,
        args.output,
        split=args.split,
        crop_size=args.crop_size,
        stride=args.stride,
        tolerance=args.tolerance,
        max_pairs=args.max_pairs,
    )
    print(json.dumps({"catalog": str(args.output / "catalog.json"), "cases": len(catalog["records"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
