"""Build a leakage-safe pair manifest from an extracted official SpaceNet 7 train archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


DATE_PATTERN = re.compile(r"(20\d{2})[_-](0[1-9]|1[0-2])")


def date_key(path: Path) -> str | None:
    match = DATE_PATTERN.search(path.stem)
    return "-".join(match.groups()) if match else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocate(aois: list[str], val_fraction: float, test_fraction: float, seed: int) -> dict[str, str]:
    if len(aois) < 3:
        raise ValueError("Нужно минимум три AOI для непересекающихся train/val/test.")
    shuffled = sorted(aois)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_fraction))
    val_count = max(1, round(len(shuffled) * val_fraction))
    if test_count + val_count >= len(shuffled):
        test_count = val_count = 1
    result = {item: "test" for item in shuffled[:test_count]}
    result.update({item: "val" for item in shuffled[test_count:test_count + val_count]})
    result.update({item: "train" for item in shuffled[test_count + val_count:]})
    return result


def build_manifest(source: Path, *, seed: int = 42, val_fraction: float = 0.15, test_fraction: float = 0.15, hash_files: bool = False) -> dict:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val/test fractions должны быть положительными и в сумме меньше 1.")
    aoi_images: dict[str, dict[str, Path]] = defaultdict(dict)
    aoi_labels: dict[str, dict[str, Path]] = defaultdict(dict)
    for image in source.rglob("images_masked/*.tif"):
        key = date_key(image)
        if key:
            aoi_images[image.parent.parent.name][key] = image
    for label in source.rglob("labels_match_pix/*.tif"):
        key = date_key(label)
        if key:
            aoi_labels[label.parent.parent.name][key] = label
    aois = sorted(set(aoi_images) & set(aoi_labels))
    split_by_aoi = _allocate(aois, val_fraction, test_fraction, seed)

    pairs = []
    missing_labels = []
    for aoi in aois:
        dates = sorted(aoi_images[aoi])
        for first, second in zip(dates, dates[1:]):
            if first not in aoi_labels[aoi] or second not in aoi_labels[aoi]:
                missing_labels.append({"aoi": aoi, "before": first, "after": second})
                continue
            paths = {
                "before_image": aoi_images[aoi][first], "after_image": aoi_images[aoi][second],
                "before_footprints": aoi_labels[aoi][first], "after_footprints": aoi_labels[aoi][second],
            }
            record = {
                "pair_id": hashlib.sha256(f"{aoi}|{first}|{second}".encode()).hexdigest()[:20],
                "aoi": aoi, "split": split_by_aoi[aoi], "before_date": first, "after_date": second,
                **{name: str(path.relative_to(source)).replace("\\", "/") for name, path in paths.items()},
            }
            if hash_files:
                record["sha256"] = {name: _sha256(path) for name, path in paths.items()}
            pairs.append(record)
    split_aois = {split: sorted(aoi for aoi, assigned in split_by_aoi.items() if assigned == split) for split in ("train", "val", "test")}
    counts = {split: sum(item["split"] == split for item in pairs) for split in split_aois}
    valid = bool(pairs) and not missing_labels and all(counts.values()) and not (
        set(split_aois["train"]) & set(split_aois["val"]) or set(split_aois["train"]) & set(split_aois["test"]) or set(split_aois["val"]) & set(split_aois["test"])
    )
    return {
        "schema": "geowatch-spacenet7-pairs-v1",
        "source": "SpaceNet 7 Multi-Temporal Urban Development Challenge",
        "upstream": "https://www.spacenet.ai/sn7-challenge/",
        "license": "CC BY-SA 4.0; retain SpaceNet and Planet attribution",
        "root": str(source.resolve()), "seed": seed, "split_unit": "AOI",
        "direction_labels": "derived from per-date building footprint masks, not from a binary XOR alone",
        "split_aois": split_aois, "counts": counts, "missing_labels": missing_labels,
        "pairs": pairs, "valid": valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--hash-files", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args.source, seed=args.seed, val_fraction=args.val_fraction, test_fraction=args.test_fraction, hash_files=args.hash_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("valid", "counts", "split_aois", "missing_labels")}, ensure_ascii=False, indent=2))
    if not manifest["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
