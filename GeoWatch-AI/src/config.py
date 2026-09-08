from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("GEOWATCH_MODEL_PATH", ROOT / "models" / "best.pt"))
    model_version: str = os.getenv("GEOWATCH_MODEL_VERSION", "yolov8n-obb-dota8-smoke-v1")
    tile_size: int = int(os.getenv("GEOWATCH_TILE_SIZE", "1024"))
    overlap: float = float(os.getenv("GEOWATCH_TILE_OVERLAP", "0.20"))
    confidence: float = float(os.getenv("GEOWATCH_CONFIDENCE", "0.25"))
    nms_iou: float = float(os.getenv("GEOWATCH_NMS_IOU", "0.45"))
    max_file_mb: int = int(os.getenv("GEOWATCH_MAX_FILE_MB", "80"))
    max_pixels: int = int(os.getenv("GEOWATCH_MAX_PIXELS", "120000000"))
    qwen_enabled: bool = _bool("QWEN_ENABLED")
    qwen_model_id: str = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
    qwen_device: str = os.getenv("QWEN_DEVICE", "auto")
    qwen_max_new_tokens: int = int(os.getenv("QWEN_MAX_NEW_TOKENS", "450"))
    runs_dir: Path = ROOT / "data" / "runs"


settings = Settings()


def active_model_version() -> str:
    """Prefer a promoted model card over the smoke fallback, unless env overrides it."""
    override = os.getenv("GEOWATCH_MODEL_VERSION")
    if override:
        return override
    card = ROOT / "models" / "model_card.json"
    try:
        return str(json.loads(card.read_text(encoding="utf-8"))["model_version"])
    except (OSError, ValueError, KeyError, TypeError):
        return settings.model_version


def model_evidence(project_root: Path = ROOT) -> dict:
    """Describe evidence behind the active model without treating a smoke run as validation."""
    card_path = project_root / "models" / "model_card.json"
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
        metrics = card["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError("metrics must be an object")
        if card.get("evidence_status") != "verified":
            return {
                "state": "unverified",
                "label_ru": "Checkpoint-кандидат: доказательства ещё не верифицированы",
                "model_version": str(card.get("model_version", active_model_version())),
                "metrics": metrics,
                "human_review_required": True,
            }
        return {
            "state": "promoted",
            "label_ru": "Проверенная DOTA4-модель установлена",
            "model_version": str(card["model_version"]),
            "metrics": metrics,
            "promoted_at": card.get("promoted_at"),
            "human_review_required": bool(card.get("human_review_required", True)),
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        pass

    audit_path = project_root / "data" / "runs" / "model_audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("evidence_status") != "verified":
            checkpoint = audit.get("checkpoint", {})
            label = "Активные веса не прошли проверку загрузки и inference"
            if audit.get("active_checkpoint_matches_smoke"):
                label = "Только smoke-checkpoint: не конкурсная модель"
            return {
                "state": "unverified",
                "label_ru": label,
                "model_version": active_model_version(),
                "metrics": {},
                "checkpoint_status": checkpoint.get("status", "unknown"),
                "human_review_required": True,
            }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass

    metrics_path = project_root / "data" / "runs" / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        purpose = str(metrics.get("purpose", "")).lower()
        if "smoke" in purpose or "smoke" in str(metrics.get("dataset", "")).lower():
            return {
                "state": "smoke",
                "label_ru": "Только smoke-baseline: не конкурсная оценка",
                "model_version": "yolov8n-obb-dota8-smoke-v1",
                "metrics": metrics,
                "human_review_required": True,
            }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return {
        "state": "unverified",
        "label_ru": "Финальная DOTA4-модель ещё не установлена",
        "model_version": active_model_version(),
        "metrics": {},
        "human_review_required": True,
    }
