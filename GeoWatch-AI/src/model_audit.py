"""Reproducible, conservative evidence checks for GeoWatch ML artifacts.

The audit deliberately separates checkpoint presence, deserialization and a probe
prediction.  A successful probe is an operability check, never an accuracy claim.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_CLASSES = ("aircraft", "ship", "small vehicle", "large vehicle")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_scene(stem: str) -> str:
    """Return the original DOTA scene id for a tiled image name."""
    return stem.split("__", 1)[0]


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read non-JSON dataset YAML") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _split_path(config_path: Path, config: dict[str, Any], split: str) -> Path:
    root_value = Path(str(config.get("path", config_path.parent)))
    root = root_value if root_value.is_absolute() else (config_path.parent / root_value).resolve()
    value = Path(str(config[split]))
    return value if value.is_absolute() else (root / value).resolve()


def _names(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("names", {})
    if isinstance(raw, list):
        return dict(enumerate(str(item) for item in raw))
    return {int(key): str(value) for key, value in raw.items()}


def audit_dataset(config_path: Path, required_classes: Iterable[str] = DEFAULT_CLASSES) -> dict[str, Any]:
    """Audit scene/content leakage, labels and class coverage for all splits."""
    config_path = config_path.resolve()
    config = _load_mapping(config_path)
    names = _names(config)
    required = set(required_classes)
    missing_schema_classes = sorted(required - set(names.values()))
    split_data: dict[str, dict[str, Any]] = {}
    scene_sets: dict[str, set[str]] = {}
    hash_sets: dict[str, set[str]] = {}
    fingerprint_records: list[str] = []
    problems: list[str] = []

    for split in ("train", "val", "test"):
        if split not in config:
            problems.append(f"missing split in config: {split}")
            split_data[split] = {"images": 0, "scenes": 0, "instances": {}, "classes": [], "empty_labels": 0}
            scene_sets[split], hash_sets[split] = set(), set()
            continue
        image_dir = _split_path(config_path, config, split)
        images = sorted(item for item in image_dir.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES) if image_dir.exists() else []
        labels_dir = image_dir.parent.parent / "labels" / image_dir.name
        scenes = {source_scene(image.stem) for image in images}
        hashes = {sha256_file(image) for image in images}
        counts: Counter[str] = Counter()
        empty_labels = 0
        missing_labels = 0
        invalid_labels = 0
        unknown_class_ids: set[int] = set()
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            image_hash = sha256_file(image)
            label_hash = sha256_file(label) if label.exists() else "missing"
            fingerprint_records.append(f"{split}/{image.name}:{image_hash}:{label_hash}")
            if not label.exists():
                missing_labels += 1
                continue
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            for line in lines:
                values = line.split()
                try:
                    class_id = int(float(values[0]))
                    if len(values) not in {5, 6, 9}:
                        invalid_labels += 1
                    if class_id not in names:
                        unknown_class_ids.add(class_id)
                    else:
                        counts[names[class_id]] += 1
                except (ValueError, IndexError):
                    invalid_labels += 1
        if not images:
            problems.append(f"empty or missing image split: {split}")
        if missing_labels:
            problems.append(f"{split}: {missing_labels} images without labels")
        if invalid_labels:
            problems.append(f"{split}: {invalid_labels} invalid label rows")
        if unknown_class_ids:
            problems.append(f"{split}: unknown class ids {sorted(unknown_class_ids)}")
        split_data[split] = {
            "images": len(images),
            "scenes": len(scenes),
            "instances": dict(sorted(counts.items())),
            "classes": sorted(counts),
            "missing_required_classes": sorted(required - set(counts)),
            "empty_labels": empty_labels,
            "missing_labels": missing_labels,
            "invalid_label_rows": invalid_labels,
        }
        scene_sets[split], hash_sets[split] = scenes, hashes

    scene_leakage = {
        "train_val": sorted(scene_sets["train"] & scene_sets["val"]),
        "train_test": sorted(scene_sets["train"] & scene_sets["test"]),
        "val_test": sorted(scene_sets["val"] & scene_sets["test"]),
    }
    content_leakage = {
        "train_val": len(hash_sets["train"] & hash_sets["val"]),
        "train_test": len(hash_sets["train"] & hash_sets["test"]),
        "val_test": len(hash_sets["val"] & hash_sets["test"]),
    }
    no_leakage = not any(scene_leakage.values()) and not any(content_leakage.values())
    test_has_all_classes = not split_data["test"]["missing_required_classes"]
    train_val_have_all_classes = not split_data["train"]["missing_required_classes"] and not split_data["val"]["missing_required_classes"]
    valid_structure = not missing_schema_classes and not problems and no_leakage
    fingerprint = hashlib.sha256("\n".join(sorted(fingerprint_records)).encode("utf-8")).hexdigest()
    return {
        "dataset_config": str(config_path),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "required_classes": sorted(required),
        "dataset_fingerprint": f"sha256:{fingerprint}",
        "missing_schema_classes": missing_schema_classes,
        "splits": split_data,
        "scene_leakage": scene_leakage,
        "content_hash_leakage_counts": content_leakage,
        "problems": problems,
        "valid_structure": valid_structure,
        "valid_for_reproducible_training": valid_structure and train_val_have_all_classes,
        "valid_for_independent_evaluation": valid_structure and test_has_all_classes,
    }


def audit_metrics(metrics_path: Path, dataset_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check whether a metrics JSON can support a test-set quality claim."""
    issues: list[str] = []
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(metrics_path), "status": "invalid", "issues": [str(exc)], "metrics": {}}
    required = {"dataset", "model", "seed", "precision", "recall", "f1", "map50", "map50_95"}
    missing = sorted(required - metrics.keys())
    if missing:
        issues.append(f"missing fields: {missing}")
    purpose = str(metrics.get("purpose", "")).lower()
    if "smoke" in purpose or "smoke" in str(metrics.get("dataset", "")).lower():
        issues.append("smoke metrics are not independent evaluation evidence")
    if metrics.get("split") not in {"test", "independent_test"}:
        issues.append("explicit split=test is missing")
    if not metrics.get("dataset_fingerprint"):
        issues.append("dataset_fingerprint is missing")
    elif dataset_report and metrics.get("dataset_fingerprint") != dataset_report.get("dataset_fingerprint"):
        issues.append("dataset_fingerprint does not match the audited dataset")
    if dataset_report and not dataset_report.get("valid_for_independent_evaluation"):
        issues.append("local test split lacks required class coverage or integrity")
    return {
        "path": str(metrics_path.resolve()),
        "status": "verified" if not issues else "unverified",
        "issues": issues,
        "metrics": metrics,
    }


def probe_checkpoint(weights: Path, sample_image: Path | None = None, confidence: float = 0.25) -> dict[str, Any]:
    """Load a checkpoint and run one prediction when dependencies and a sample exist."""
    weights = weights.resolve()
    report: dict[str, Any] = {
        "path": str(weights),
        "file_found": weights.is_file(),
        "file_loaded": False,
        "probe_completed": False,
        "probe_is_accuracy_evidence": False,
    }
    if not weights.is_file():
        report["status"] = "missing"
        report["error"] = "checkpoint file not found"
        return report
    report.update({"weights_bytes": weights.stat().st_size, "weights_sha256": sha256_file(weights)})
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        report.update(status="dependency_missing", error=f"ultralytics import failed: {exc}")
        return report
    try:
        started = time.perf_counter()
        model = YOLO(str(weights))
        report["file_loaded"] = True
        report["load_ms"] = round((time.perf_counter() - started) * 1000, 2)
        report["task"] = getattr(model, "task", None)
        names = getattr(model, "names", {})
        report["checkpoint_classes"] = list(names.values()) if isinstance(names, dict) else list(names or [])
        if sample_image is None or not sample_image.is_file():
            report.update(status="loaded_not_probed", error="sample image not found")
            return report
        started = time.perf_counter()
        result = model.predict(str(sample_image), conf=confidence, verbose=False)[0]
        predictions = result.obb if getattr(result, "obb", None) is not None else result.boxes
        report.update(
            status="probe_passed",
            probe_completed=True,
            sample_image=str(sample_image.resolve()),
            probe_ms=round((time.perf_counter() - started) * 1000, 2),
            prediction_count=len(predictions),
        )
    except Exception as exc:  # model backends expose heterogeneous errors
        report.update(status="load_or_probe_error", error=f"{type(exc).__name__}: {exc}")
    return report


def build_audit_report(weights: Path, data: Path, metrics: Path, sample_image: Path | None = None) -> dict[str, Any]:
    dataset = audit_dataset(data)
    checkpoint = probe_checkpoint(weights, sample_image)
    metric_report = audit_metrics(metrics, dataset)
    smoke_hash = weights.parents[1] / "runs" / "obb" / "models" / "dota8_smoke" / "weights" / "best.pt"
    matches_smoke = smoke_hash.is_file() and weights.is_file() and sha256_file(smoke_hash) == sha256_file(weights)
    blockers = list(metric_report["issues"])
    if matches_smoke:
        blockers.append("active checkpoint is byte-identical to the local DOTA8 smoke checkpoint")
    if not checkpoint.get("probe_completed"):
        blockers.append("checkpoint probe inference has not completed")
    verified = dataset["valid_for_reproducible_training"] and dataset["valid_for_independent_evaluation"] and metric_report["status"] == "verified" and checkpoint.get("probe_completed") and not matches_smoke
    if not dataset["valid_for_reproducible_training"]:
        blockers.append("train/validation splits lack complete class coverage or integrity")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "evidence_status": "verified" if verified else "candidate_unverified",
        "checkpoint": checkpoint,
        "dataset": dataset,
        "metrics": metric_report,
        "active_checkpoint_matches_smoke": matches_smoke,
        "blockers": blockers,
        "human_review_required": True,
    }
