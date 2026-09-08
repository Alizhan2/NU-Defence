"""Safely promote a completed Colab artifact into the local GeoWatch app."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METRICS = {
    "dataset", "dataset_fingerprint", "split", "model", "checkpoint_sha256",
    "seed", "precision", "recall", "f1", "map50", "map50_95",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(weights: Path, metrics_path: Path) -> dict:
    if not weights.is_file() or weights.stat().st_size < 1_000_000:
        raise ValueError("best.pt не найден или слишком мал: нужен реальный обученный checkpoint.")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("metrics_test.json отсутствует или повреждён.") from exc
    missing = REQUIRED_METRICS - metrics.keys()
    if missing:
        raise ValueError(f"В metrics_test.json не хватает полей: {sorted(missing)}")
    if metrics["split"] != "test":
        raise ValueError("Promotion разрешён только для метрик отдельного test split.")
    actual_hash = sha256(weights)
    if metrics["checkpoint_sha256"].lower() != actual_hash.lower():
        raise ValueError("checkpoint_sha256 в метриках не совпадает с выбранными весами.")
    if not str(metrics["dataset_fingerprint"]).startswith("sha256:"):
        raise ValueError("dataset_fingerprint должен быть SHA-256 fingerprint от audit_dataset.")
    return metrics


def promote(weights: Path, metrics_path: Path, version: str, destination: Path = ROOT / "models" / "best.pt", apply: bool = False, project_root: Path = ROOT) -> dict:
    if not version.strip():
        raise ValueError("Укажите непустую версию модели.")
    metrics = validate_artifacts(weights, metrics_path)
    card = {
        "model_version": version,
        "weights_sha256": sha256(weights),
        "weights_bytes": weights.stat().st_size,
        "evidence_status": "verified",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "human_review_required": True,
    }
    if apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            backup = destination.with_name(f"best.backup-{datetime.now():%Y%m%d-%H%M%S}.pt")
            shutil.copy2(destination, backup)
            card["backup"] = str(backup)
        staged = destination.with_suffix(".staged")
        shutil.copy2(weights, staged)
        staged.replace(destination)
        (destination.parent / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics_target = project_root / "data" / "runs" / "metrics.json"
        metrics_target.parent.mkdir(parents=True, exist_ok=True)
        metrics_target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--apply", action="store_true", help="Выполнить замену; без флага доступен только dry run.")
    args = parser.parse_args()
    card = promote(args.weights, args.metrics, args.version, apply=args.apply)
    print(json.dumps(card, ensure_ascii=False, indent=2))
    print("PROMOTED" if args.apply else "DRY RUN: artifacts validated, no files changed.")


if __name__ == "__main__":
    main()
