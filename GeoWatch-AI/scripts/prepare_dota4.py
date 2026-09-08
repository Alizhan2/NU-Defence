"""Build a four-class YOLO-OBB dataset from an already converted DOTA dataset.

Input must contain images/{train,val} and labels/{train,val}.  It never
downloads DOTA and never changes the source data.  A scene-level subset of
the source train split becomes the internal test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


# Ultralytics DOTA class IDs: plane, ship, large vehicle, small vehicle.
REMAP = {0: 0, 1: 1, 9: 3, 10: 2}
NAMES = {0: "aircraft", 1: "ship", 2: "small vehicle", 3: "large vehicle"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def source_scene(stem: str) -> str:
    return stem.split("__", 1)[0]


def source_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def remap_label(source: Path, destination: Path) -> tuple[Counter[int], int]:
    counts: Counter[int] = Counter()
    clipped = 0
    lines: list[str] = []
    if source.exists():
        for raw in source.read_text(encoding="utf-8").splitlines():
            values = raw.split()
            if not values:
                continue
            class_id = int(float(values[0]))
            if class_id not in REMAP:
                continue
            # OBB format is class + 8 normalized polygon coordinates.
            if len(values) != 9:
                raise ValueError(f"Invalid OBB label in {source}: {raw}")
            coords = [float(value) for value in values[1:]]
            # DOTA tiles legitimately contain cut-off polygons at image edges.
            # Clip them and report the count instead of silently discarding them.
            if not all(0 <= value <= 1 for value in coords):
                clipped += 1
                coords = [min(1.0, max(0.0, value)) for value in coords]
            mapped = REMAP[class_id]
            lines.append(" ".join([str(mapped), *(f"{value:.8f}" for value in coords)]))
            counts[mapped] += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return counts, clipped


def copy_record(image: Path, source_labels: Path, output: Path, split: str) -> tuple[Counter[int], int]:
    image_target = output / "images" / split / image.name
    label_target = output / "labels" / split / f"{image.stem}.txt"
    image_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_target)
    return remap_label(source_labels / f"{image.stem}.txt", label_target)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Converted DOTA root: images/train, labels/train, images/val, labels/val")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 0.5:
        raise ValueError("--test-fraction must be between 0 and 0.5")
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")

    train_images = source_images(source / "images" / "train")
    val_images = source_images(source / "images" / "val")
    if not train_images or not val_images:
        raise FileNotFoundError("Expected non-empty images/train and images/val in source.")
    scenes = sorted({source_scene(image.stem) for image in train_images})
    rng = random.Random(args.seed); rng.shuffle(scenes)
    test_scenes = set(scenes[:max(1, round(len(scenes) * args.test_fraction))])
    splits = {
        "train": [image for image in train_images if source_scene(image.stem) not in test_scenes],
        "test": [image for image in train_images if source_scene(image.stem) in test_scenes],
        "val": val_images,
    }
    if not splits["train"] or not splits["test"]:
        raise ValueError("Scene split left train or test empty.")

    class_counts: dict[str, Counter[int]] = {}
    clipped_labels: dict[str, int] = {}
    hashes: dict[str, dict[str, str]] = {}
    for split, images in splits.items():
        class_counts[split] = Counter()
        clipped_labels[split] = 0
        label_root = source / "labels" / ("val" if split == "val" else "train")
        hashes[split] = {}
        for image in images:
            counts, clipped = copy_record(image, label_root, output, split)
            class_counts[split].update(counts)
            clipped_labels[split] += clipped
            hashes[split][image.name] = file_hash(image)

    config = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": NAMES}
    (output / "dota4.yaml").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"source": str(source), "seed": args.seed, "test_fraction": args.test_fraction,
                "class_map": {str(key): value for key, value in NAMES.items()},
                "images": {split: len(items) for split, items in splits.items()},
                "instances": {split: {NAMES[key]: value for key, value in counts.items()} for split, counts in class_counts.items()},
                "edge_clipped_polygons": clipped_labels,
                "sha256": hashes}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Dataset YAML: {output / 'dota4.yaml'}")


if __name__ == "__main__":
    main()
