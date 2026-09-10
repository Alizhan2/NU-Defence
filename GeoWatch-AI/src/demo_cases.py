from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal


EvidenceStatus = Literal["ground_truth", "unverified"]
EventType = Literal["appeared", "disappeared", "uncertain"]


EVIDENCE_LABELS = {
    "ground_truth": "Разметка источника",
    "unverified": "Не проверено",
}

_EVIDENCE_ALIASES = {"ground_truth_annotation_derived": "ground_truth"}

EVENT_LABELS = {
    "appeared": "Объект появился",
    "disappeared": "Объект исчез или изменился",
    "uncertain": "Событие не подтверждено",
}

_EVENT_ALIASES = {
    "disappeared_or_changed": "disappeared",
    "negative_or_uncertain": "uncertain",
}


@dataclass(frozen=True)
class RealDemoCase:
    case_id: str
    title: str
    event_type: EventType
    before_image: Path
    after_image: Path
    before_date: date
    after_date: date
    aoi: str
    source_url: str
    license: str
    evidence_status: EvidenceStatus
    note: str
    attribution: str

    @property
    def is_ready(self) -> bool:
        return self.before_image.is_file() and self.after_image.is_file()

    @property
    def evidence_label(self) -> str:
        return EVIDENCE_LABELS[self.evidence_status]

    @property
    def event_label(self) -> str:
        return EVENT_LABELS[self.event_type]


@dataclass(frozen=True)
class BaselineEvidence:
    status: Literal["not_trained", "unverified", "verified_test"]
    label: str
    detail: str
    metrics: dict[str, float]
    checkpoint_sha256: str | None = None
    manifest_sha256: str | None = None


def _safe_project_path(project_root: Path, value: Any, *, catalog_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Путь изображения отсутствует.")
    root = project_root.resolve()
    # Catalog assets are portable as a directory and resolve next to catalog.json.
    candidate = (catalog_dir / value).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Путь изображения выходит за пределы проекта.")
    return candidate


def load_demo_case_catalog(project_root: Path, catalog_path: Path | None = None) -> tuple[RealDemoCase, ...]:
    """Load real-data demo metadata without claiming that missing evidence exists."""
    project_root = project_root.resolve()
    target = catalog_path or project_root / "data" / "demo_cases" / "catalog.json"
    if not target.is_file():
        return ()
    payload = json.loads(target.read_text(encoding="utf-8"))
    records = payload.get("records", payload.get("cases", []))
    if not isinstance(records, list):
        raise ValueError("catalog.json: records/cases должен быть списком.")

    cases: list[RealDemoCase] = []
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("catalog.json: запись кейса должна быть объектом.")
        event_type = _EVENT_ALIASES.get(raw.get("event_type"), raw.get("event_type"))
        evidence_status = _EVIDENCE_ALIASES.get(raw.get("evidence_status"), raw.get("evidence_status", "unverified"))
        if event_type not in EVENT_LABELS:
            raise ValueError(f"Неизвестный event_type: {event_type!r}")
        if evidence_status not in EVIDENCE_LABELS:
            raise ValueError(f"Неизвестный evidence_status: {evidence_status!r}")
        before_date = date.fromisoformat(str(raw["before_date"]))
        after_date = date.fromisoformat(str(raw["after_date"]))
        if after_date <= before_date:
            raise ValueError("Дата after должна быть позже before.")
        cases.append(RealDemoCase(
            case_id=str(raw.get("id") or raw.get("case_id") or "").strip(),
            title=str(raw["title"]).strip(),
            event_type=event_type,
            before_image=_safe_project_path(project_root, raw["before_image"], catalog_dir=target.parent),
            after_image=_safe_project_path(project_root, raw["after_image"], catalog_dir=target.parent),
            before_date=before_date,
            after_date=after_date,
            aoi=str(raw.get("aoi", "Не указана")),
            source_url=str(raw.get("source_url", "")),
            license=str(raw.get("license", "Не указана")),
            evidence_status=evidence_status,
            note=str(raw.get("note", "")),
            attribution=str(raw.get("attribution", "")),
        ))
    return tuple(cases)


def load_spacenet_baseline_evidence(project_root: Path, metrics_path: Path | None = None) -> BaselineEvidence:
    """Classify baseline evidence conservatively; metric values alone are never verification."""
    target = metrics_path or project_root / "data" / "runs" / "spacenet7_metrics.json"
    if not target.is_file():
        return BaselineEvidence("not_trained", "Не обучено", "Файл результатов SpaceNet 7 отсутствует.", {})
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return BaselineEvidence("unverified", "Результат не проверен", f"Не удалось прочитать файл метрик: {exc}", {})

    raw_metrics = payload.get("metrics", {})
    metrics = {
        str(key): float(value)
        for key, value in raw_metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    } if isinstance(raw_metrics, dict) else {}
    checkpoint_hash = payload.get("checkpoint_sha256")
    manifest_hash = payload.get("manifest_sha256") or payload.get("dataset_manifest_sha256")
    split = str(payload.get("split", "")).lower()
    verified = bool(payload.get("verified")) and split == "test" and bool(checkpoint_hash) and bool(manifest_hash)
    if verified:
        return BaselineEvidence(
            "verified_test", "Проверено на test split", "Метрики связаны с checkpoint и manifest по SHA-256.",
            metrics, str(checkpoint_hash), str(manifest_hash),
        )
    return BaselineEvidence(
        "unverified", "Результат не проверен",
        "Есть файл результатов, но нет полного test-evidence: verified=true, split=test и двух SHA-256.",
        metrics, str(checkpoint_hash) if checkpoint_hash else None, str(manifest_hash) if manifest_hash else None,
    )
