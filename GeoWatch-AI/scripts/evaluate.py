"""Evaluate a checkpoint once on an integrity-checked independent test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_audit import audit_dataset, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "runs" / "metrics_test.json")
    args = parser.parse_args()

    dataset = audit_dataset(args.data)
    if not dataset["valid_for_independent_evaluation"]:
        raise SystemExit(
            "Evaluation blocked: test split is empty, leaking, malformed, or lacks required classes. "
            "Run scripts/audit_model.py for details."
        )
    if not args.model.is_file():
        raise SystemExit(f"Evaluation blocked: checkpoint not found: {args.model}")

    from ultralytics import YOLO, __version__ as ultralytics_version

    model = YOLO(str(args.model))
    result = model.val(data=str(args.data), conf=args.threshold, imgsz=1024, split="test")
    precision, recall = float(result.box.mp), float(result.box.mr)
    class_names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))
    class_maps = list(getattr(result.box, "maps", []))
    report = {
        "dataset": "DOTA4 scene-separated internal test",
        "dataset_config": str(args.data.resolve()),
        "dataset_fingerprint": dataset["dataset_fingerprint"],
        "split": "test",
        "model": "yolo-obb",
        "checkpoint_sha256": sha256_file(args.model),
        "ultralytics_version": ultralytics_version,
        "seed": 42,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "per_class_map50_95": {
            str(class_names[index]): float(value)
            for index, value in enumerate(class_maps)
            if index in class_names
        },
        "confidence_threshold": args.threshold,
        "note": "Independent test evaluation. Model selection and threshold tuning must use validation only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
