"""Validate YOLO dataset structure and guard against filename leakage across splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def scene_id(image_id: str) -> str:
    """Keep all DOTA tiles from one original scene in the same split."""
    return image_id.split("__", 1)[0]


def resolve_split(config_dir: Path, root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = root / path
    return candidate if candidate.exists() else config_dir / path


def image_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split directory not found: {path}")
    return {scene_id(item.stem) for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES}


def present_class_ids(image_dir: Path) -> set[int]:
    """Read YOLO labels from the standard sibling labels/<split> directory."""
    labels_dir = image_dir.parent.parent / "labels" / image_dir.name
    if not labels_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")
    found: set[int] = set()
    for label in labels_dir.rglob("*.txt"):
        for line in label.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if values:
                found.add(int(float(values[0])))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--required-classes", nargs="*", default=["aircraft", "ship", "small vehicle", "large vehicle"])
    args = parser.parse_args()

    config_path = Path(args.data).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root_value = Path(config.get("path", config_path.parent))
    root = root_value if root_value.is_absolute() else (config_path.parent / root_value).resolve()
    names_value = config.get("names", {})
    names_by_id = {int(key): value for key, value in names_value.items()} if isinstance(names_value, dict) else dict(enumerate(names_value))
    names = list(names_by_id.values())
    missing_classes = sorted(set(args.required_classes) - set(names))

    splits, split_paths = {}, {}
    for name in ("train", "val", "test"):
        if name not in config:
            raise ValueError(f"Dataset YAML must define '{name}'")
        split_paths[name] = resolve_split(config_path.parent, root, str(config[name]))
        splits[name] = image_ids(split_paths[name])

    test_class_ids = present_class_ids(split_paths["test"])
    test_classes = {names_by_id[class_id] for class_id in test_class_ids if class_id in names_by_id}
    missing_test_classes = sorted(set(args.required_classes) - test_classes)

    leakage = {
        "train_val": sorted(splits["train"] & splits["val"]),
        "train_test": sorted(splits["train"] & splits["test"]),
        "val_test": sorted(splits["val"] & splits["test"]),
    }
    report = {
        "dataset_yaml": str(config_path),
        "counts": {key: len(value) for key, value in splits.items()},
        "missing_required_classes": missing_classes,
        "test_classes": sorted(test_classes),
        "missing_required_test_classes": missing_test_classes,
        "leakage": leakage,
        "valid": not missing_classes and not missing_test_classes and not any(leakage.values()) and all(splits.values()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
