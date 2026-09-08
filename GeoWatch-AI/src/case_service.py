from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .models import CaseCreateRequest, CaseRecord, CaseStatus


class CaseStore:
    def __init__(self, root: Path | None = None):
        self.root = (root or settings.runs_dir) / "cases"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", case_id):
            raise KeyError("Некорректный ID кейса.")
        return self.root / f"case_{case_id}.json"

    def save(self, record: CaseRecord) -> CaseRecord:
        target = self._path(record.case_id)
        record.updated_at = datetime.now(timezone.utc)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(target)
        return record

    def create(self, request: CaseCreateRequest) -> CaseRecord:
        return self.save(CaseRecord(**request.model_dump()))

    def list(self) -> list[CaseRecord]:
        records = []
        for path in self.root.glob("case_*.json"):
            try:
                records.append(CaseRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def load(self, case_id: str) -> CaseRecord:
        target = self._path(case_id)
        if not target.exists():
            raise KeyError("Кейс не найден.")
        return CaseRecord.model_validate_json(target.read_text(encoding="utf-8"))

    def attach_analysis(self, case_id: str, analysis_id: str) -> CaseRecord:
        record = self.load(case_id)
        if analysis_id not in record.analysis_ids:
            record.analysis_ids.append(analysis_id)
        return self.save(record)

    def update_status(self, case_id: str, status: CaseStatus) -> CaseRecord:
        record = self.load(case_id)
        record.status = status
        return self.save(record)


case_store = CaseStore()
