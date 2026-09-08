from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .config import settings


JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


@dataclass(frozen=True)
class ProcessingJob:
    job_id: str
    kind: str
    payload: dict
    status: JobStatus
    attempts: int
    max_attempts: int
    dedupe_key: str | None
    result: dict | None
    error: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobQueue:
    def __init__(self, path: Path | None = None):
        self.path = path or (settings.runs_dir / "geowatch.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    dedupe_key TEXT,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_processing_jobs_dedupe
                ON processing_jobs(kind, dedupe_key) WHERE dedupe_key IS NOT NULL;
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def _record(self, row: sqlite3.Row) -> ProcessingJob:
        return ProcessingJob(
            job_id=row["job_id"], kind=row["kind"], payload=json.loads(row["payload_json"]),
            status=row["status"], attempts=row["attempts"], max_attempts=row["max_attempts"],
            dedupe_key=row["dedupe_key"], result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"], created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def enqueue(self, kind: str, payload: dict, *, dedupe_key: str | None = None, max_attempts: int = 3) -> ProcessingJob:
        if not kind or not kind.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Некорректный тип задания.")
        if not 1 <= max_attempts <= 10:
            raise ValueError("max_attempts должен быть от 1 до 10.")
        identity = f"{kind}|{dedupe_key}" if dedupe_key else uuid4().hex
        job_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        timestamp = _now()
        with self._connect() as db:
            db.execute(
                """INSERT OR IGNORE INTO processing_jobs
                (job_id, kind, payload_json, status, max_attempts, dedupe_key, created_at, updated_at)
                VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)""",
                (job_id, kind, json.dumps(payload, ensure_ascii=False), max_attempts, dedupe_key, timestamp, timestamp),
            )
        return self.load(job_id)

    def load(self, job_id: str) -> ProcessingJob:
        with self._connect() as db:
            row = db.execute("SELECT * FROM processing_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("Задание не найдено.")
        return self._record(row)

    def list(self, limit: int = 100) -> list[ProcessingJob]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM processing_jobs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)).fetchall()
        return [self._record(row) for row in rows]

    def claim_next(self, kind: str | None = None) -> ProcessingJob | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            query = "SELECT job_id FROM processing_jobs WHERE status = 'queued' AND attempts < max_attempts"
            params: tuple = ()
            if kind:
                query += " AND kind = ?"
                params = (kind,)
            query += " ORDER BY created_at LIMIT 1"
            row = db.execute(query, params).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE processing_jobs SET status = 'running', attempts = attempts + 1, updated_at = ? WHERE job_id = ?",
                (_now(), row["job_id"]),
            )
        return self.load(row["job_id"])

    def complete(self, job_id: str, result: dict) -> ProcessingJob:
        with self._connect() as db:
            changed = db.execute(
                "UPDATE processing_jobs SET status = 'completed', result_json = ?, error = NULL, updated_at = ? WHERE job_id = ? AND status = 'running'",
                (json.dumps(result, ensure_ascii=False), _now(), job_id),
            ).rowcount
        if not changed:
            raise ValueError("Завершить можно только выполняющееся задание.")
        return self.load(job_id)

    def fail(self, job_id: str, error: str) -> ProcessingJob:
        job = self.load(job_id)
        if job.status != "running":
            raise ValueError("Ошибка может быть записана только для выполняющегося задания.")
        status = "queued" if job.attempts < job.max_attempts else "failed"
        with self._connect() as db:
            db.execute(
                "UPDATE processing_jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?",
                (status, error[:2000], _now(), job_id),
            )
        return self.load(job_id)

    def cancel(self, job_id: str) -> ProcessingJob:
        with self._connect() as db:
            changed = db.execute(
                "UPDATE processing_jobs SET status = 'cancelled', updated_at = ? WHERE job_id = ? AND status IN ('queued','running')",
                (_now(), job_id),
            ).rowcount
        if not changed:
            raise ValueError("Отменить можно только ожидающее или выполняющееся задание.")
        return self.load(job_id)


job_queue = JobQueue()
