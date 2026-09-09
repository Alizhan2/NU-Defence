"""Download (optionally) and prepare the DOTA4 YOLO-OBB dataset for GeoWatch.

The source archive is the Ultralytics DOTAv1 conversion.  It is retained outside
Git because it is large and subject to the original DOTA academic-use terms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip"
REMAP = {0: 0, 1: 1, 9: 3, 10: 2}
NAMES = {0: "aircraft", 1: "ship", 2: "small vehicle", 3: "large vehicle"}


def find_source(root: Path) -> Path:
    candidates = [p for p in root.rglob("DOTAv1") if (p / "images" / "train").is_dir() and (p / "labels" / "train").is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one extracted DOTAv1 directory below {root}; found {len(candidates)}.")
    return candidates[0]


def scene_id(image: Path) -> str:
    return image.stem.split("__", 1)[0]


def build(source: Path, output: Path, seed: int, test_fraction: float, overwrite: bool) -> dict:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite only after checking it.")
        shutil.rmtree(output)
    train_images = sorted((source / "images" / "train").glob("*"))
    val_images = sorted((source / "images" / "val").glob("*"))
    if not train_images or not val_images:
        raise FileNotFoundError("DOTAv1 images/train or images/val is empty.")

    scenes = sorted({scene_id(image) for image in train_images})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    test_scenes = set(scenes[:max(1, round(len(scenes) * test_fraction))])
    splits = {
        "train": [image for image in train_images if scene_id(image) not in test_scenes],
        "test": [image for image in train_images if scene_id(image) in test_scenes],
        "val": val_images,
    }
    counts: dict[str, Counter] = {}
    clipped: dict[str, int] = {}
    for split, images in splits.items():
        counts[split] = Counter()
        clipped[split] = 0
        source_split = "val" if split == "val" else "train"
        for image in images:
            target_image = output / "images" / split / image.name
            target_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target_image)
            source_label = source / "labels" / source_split / f"{image.stem}.txt"
            if not source_label.is_file():
                raise FileNotFoundError(f"Missing label: {source_label}")
            records = []
            for line in source_label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                old_class = int(float(parts[0]))
                if old_class not in REMAP:
                    continue
                polygon = [float(value) for value in parts[1:]]
                if len(polygon) != 8:
                    raise ValueError(f"Expected 8 OBB coordinates in {source_label}")
                if any(value < 0 or value > 1 for value in polygon):
                    clipped[split] += 1
                    polygon = [min(1.0, max(0.0, value)) for value in polygon]
                new_class = REMAP[old_class]
                records.append(str(new_class) + " " + " ".join(f"{value:.8f}" for value in polygon))
                counts[split][new_class] += 1
            (output / "labels" / split / f"{image.stem}.txt").parent.mkdir(parents=True, exist_ok=True)
            (output / "labels" / split / f"{image.stem}.txt").write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")

    if {scene_id(image) for image in splits["train"]} & {scene_id(image) for image in splits["test"]}:
        raise AssertionError("Scene leakage between train and test.")
    dataset_yaml = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": NAMES}
    (output / "dota4.yaml").write_text(json.dumps(dataset_yaml, indent=2), encoding="utf-8")
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        for label in sorted((output / "labels" / split).glob("*.txt")):
            digest.update(f"{split}/{label.name}\0".encode())
            digest.update(label.read_bytes())
    manifest = {
        "seed": seed,
        "split": "scene-separated",
        "images": {name: len(items) for name, items in splits.items()},
        "instances": {name: {NAMES[class_id]: count for class_id, count in counter.items()} for name, counter in counts.items()},
        "edge_clipped_polygons": clipped,
        "dataset_fingerprint": "sha256:" + digest.hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "downloads" / "DOTAv1.zip")
    parser.add_argument("--source", type=Path, help="Existing extracted DOTAv1 folder; skips archive extraction.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "dota4")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--download", action="store_true", help="Download archive when it is missing.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be between 0 and 1.")
    if args.source:
        source = args.source
    else:
        if not args.archive.is_file():
            if not args.download:
                raise FileNotFoundError(f"Archive missing: {args.archive}. Add --download to fetch it.")
            args.archive.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {args.url} ...")
            urllib.request.urlretrieve(args.url, args.archive)
        extract_root = args.archive.parent / "extracted"
        if not extract_root.exists():
            print(f"Extracting {args.archive} ...")
            with zipfile.ZipFile(args.archive) as archive:
                archive.extractall(extract_root)
        source = find_source(extract_root)
    manifest = build(source, args.output, args.seed, args.test_fraction, args.overwrite)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"READY: {args.output / 'dota4.yaml'}")


if __name__ == "__main__":
    main()
