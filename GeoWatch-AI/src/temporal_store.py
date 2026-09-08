from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

from .change_detection import ChangeEvent
from .config import settings
from .models import AnalysisResult


TemporalReviewStatus = Literal["needs_review", "confirmed", "rejected"]


@dataclass(frozen=True)
class StoredChangeEvent:
    event_id: str
    comparison_id: str
    status: str
    class_name: str
    confidence: float
    before_id: str | None
    after_id: str | None
    overlap: float
    review_status: TemporalReviewStatus
    comment: str
    reviewed_at: str | None


@dataclass(frozen=True)
class TemporalComparison:
    comparison_id: str
    before_analysis_id: str
    after_analysis_id: str
    before_image_id: str
    after_image_id: str
    before_date: str | None
    after_date: str | None
    match_iou: float
    pair_validation: dict
    quality: dict
    created_at: str
    events: tuple[StoredChangeEvent, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["events"] = [asdict(item) for item in self.events]
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _comparison_id(before: AnalysisResult, after: AnalysisResult, before_date: str | None, after_date: str | None, match_iou: float) -> str:
    source = "|".join((before.image_id, after.image_id, before_date or "-", after_date or "-", f"{match_iou:.6f}", before.model_version, after.model_version))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]


def _event_id(comparison_id: str, event: ChangeEvent) -> str:
    source = "|".join((comparison_id, event.status, event.class_name, event.before_id or "-", event.after_id or "-"))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]


class TemporalStore:
    """SQLite persistence for reproducible temporal comparisons and reviews."""

    def __init__(self, path: Path | None = None):
        self.path = path or (settings.runs_dir / "geowatch.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS temporal_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    before_analysis_id TEXT NOT NULL,
                    after_analysis_id TEXT NOT NULL,
                    before_image_id TEXT NOT NULL,
                    after_image_id TEXT NOT NULL,
                    before_date TEXT,
                    after_date TEXT,
                    match_iou REAL NOT NULL,
                    pair_validation_json TEXT NOT NULL,
                    quality_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS temporal_events (
                    event_id TEXT PRIMARY KEY,
                    comparison_id TEXT NOT NULL REFERENCES temporal_comparisons(comparison_id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    before_detection_id TEXT,
                    after_detection_id TEXT,
                    overlap REAL NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'needs_review',
                    comment TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS temporal_review_history (
                    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL REFERENCES temporal_events(event_id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    comment TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS case_comparisons (
                    case_id TEXT NOT NULL,
                    comparison_id TEXT NOT NULL REFERENCES temporal_comparisons(comparison_id) ON DELETE CASCADE,
                    attached_at TEXT NOT NULL,
                    PRIMARY KEY(case_id, comparison_id)
                );
                INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (1, CURRENT_TIMESTAMP);
                """
            )

    def save(
        self,
        before: AnalysisResult,
        after: AnalysisResult,
        events: Sequence[ChangeEvent],
        *,
        before_date: str | None = None,
        after_date: str | None = None,
        match_iou: float = 0.25,
        pair_validation: dict | None = None,
        quality: dict | None = None,
    ) -> TemporalComparison:
        comparison_id = _comparison_id(before, after, before_date, after_date, match_iou)
        created_at = _utc_now()
        with self._connect() as db:
            db.execute(
                """INSERT OR IGNORE INTO temporal_comparisons
                (comparison_id, before_analysis_id, after_analysis_id, before_image_id, after_image_id,
                 before_date, after_date, match_iou, pair_validation_json, quality_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (comparison_id, before.analysis_id, after.analysis_id, before.image_id, after.image_id,
                 before_date, after_date, match_iou, json.dumps(pair_validation or {}, ensure_ascii=False),
                 json.dumps(quality or {}, ensure_ascii=False), created_at),
            )
            for event in events:
                db.execute(
                    """INSERT OR IGNORE INTO temporal_events
                    (event_id, comparison_id, status, class_name, confidence, before_detection_id,
                     after_detection_id, overlap) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (_event_id(comparison_id, event), comparison_id, event.status, event.class_name,
                     event.confidence, event.before_id, event.after_id, event.overlap),
                )
        return self.load(comparison_id)

    def _event(self, row: sqlite3.Row) -> StoredChangeEvent:
        return StoredChangeEvent(
            event_id=row["event_id"], comparison_id=row["comparison_id"], status=row["status"],
            class_name=row["class_name"], confidence=row["confidence"], before_id=row["before_detection_id"],
            after_id=row["after_detection_id"], overlap=row["overlap"], review_status=row["review_status"],
            comment=row["comment"], reviewed_at=row["reviewed_at"],
        )

    def load(self, comparison_id: str) -> TemporalComparison:
        with self._connect() as db:
            row = db.execute("SELECT * FROM temporal_comparisons WHERE comparison_id = ?", (comparison_id,)).fetchone()
            if row is None:
                raise KeyError("Сравнение не найдено.")
            events = tuple(self._event(item) for item in db.execute(
                "SELECT * FROM temporal_events WHERE comparison_id = ? ORDER BY event_id", (comparison_id,)
            ).fetchall())
        return TemporalComparison(
            comparison_id=row["comparison_id"], before_analysis_id=row["before_analysis_id"],
            after_analysis_id=row["after_analysis_id"], before_image_id=row["before_image_id"],
            after_image_id=row["after_image_id"], before_date=row["before_date"], after_date=row["after_date"],
            match_iou=row["match_iou"], pair_validation=json.loads(row["pair_validation_json"]),
            quality=json.loads(row["quality_json"]), created_at=row["created_at"], events=events,
        )

    def list(self, limit: int = 100) -> list[TemporalComparison]:
        with self._connect() as db:
            ids = [row[0] for row in db.execute(
                "SELECT comparison_id FROM temporal_comparisons ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)
            ).fetchall()]
        return [self.load(item) for item in ids]

    def review(self, comparison_id: str, event_id: str, status: TemporalReviewStatus, comment: str = "") -> StoredChangeEvent:
        if status not in {"needs_review", "confirmed", "rejected"}:
            raise ValueError("Неподдерживаемый статус решения.")
        timestamp = _utc_now()
        with self._connect() as db:
            current = db.execute(
                "SELECT event_id FROM temporal_events WHERE comparison_id = ? AND event_id = ?", (comparison_id, event_id)
            ).fetchone()
            if current is None:
                raise KeyError("Событие изменения не найдено.")
            db.execute(
                "UPDATE temporal_events SET review_status = ?, comment = ?, reviewed_at = ? WHERE event_id = ?",
                (status, comment[:2000], timestamp, event_id),
            )
            db.execute(
                "INSERT INTO temporal_review_history(event_id, status, comment, created_at) VALUES (?, ?, ?, ?)",
                (event_id, status, comment[:2000], timestamp),
            )
            row = db.execute("SELECT * FROM temporal_events WHERE event_id = ?", (event_id,)).fetchone()
        return self._event(row)

    def attach_to_case(self, case_id: str, comparison_id: str) -> None:
        self.load(comparison_id)
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO case_comparisons(case_id, comparison_id, attached_at) VALUES (?, ?, ?)",
                (case_id, comparison_id, _utc_now()),
            )

    def comparisons_for_case(self, case_id: str) -> list[str]:
        with self._connect() as db:
            return [row[0] for row in db.execute(
                "SELECT comparison_id FROM case_comparisons WHERE case_id = ? ORDER BY attached_at", (case_id,)
            ).fetchall()]


temporal_store = TemporalStore()
