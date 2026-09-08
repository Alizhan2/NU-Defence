from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

from .change_detection import ChangeEvent


AlertPriority = Literal["low", "normal", "high"]
AlertReviewStatus = Literal["needs_review", "confirmed", "rejected"]
SUPPORTED_CHANGE_STATUSES = frozenset({"appeared", "disappeared", "stable"})


@dataclass(frozen=True)
class AlertRule:
    """Deterministic filter for observable change events.

    ``priority`` is operator-defined queue priority. It is not a model
    confidence, threat probability, or autonomous conclusion.
    """

    rule_id: str
    name: str
    event_statuses: frozenset[str]
    class_names: frozenset[str] | None = None
    min_confidence: float = 0.0
    priority: AlertPriority = "normal"
    enabled: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.rule_id):
            raise ValueError("rule_id must contain only letters, numbers, '_' or '-'.")
        if not self.name.strip():
            raise ValueError("Rule name must not be empty.")
        if not self.event_statuses or not self.event_statuses <= SUPPORTED_CHANGE_STATUSES:
            raise ValueError("Rule must contain only supported change statuses.")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1.")
        if self.class_names is not None and not self.class_names:
            raise ValueError("class_names must be None or a non-empty set.")


@dataclass(frozen=True)
class AlertCandidate:
    alert_id: str
    timeline_id: str
    rule_id: str
    rule_name: str
    event_status: str
    class_name: str
    confidence: float
    priority: AlertPriority
    triage_score: int
    evidence: str
    before_id: str | None
    after_id: str | None
    overlap: float
    review_status: AlertReviewStatus = "needs_review"
    created_at: str = ""


def _event_identity(event: ChangeEvent) -> str:
    return "|".join(
        (event.status, event.class_name, event.before_id or "-", event.after_id or "-")
    )


def _alert_id(timeline_id: str, rule_id: str, event: ChangeEvent) -> str:
    source = f"{timeline_id}|{rule_id}|{_event_identity(event)}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]


def _triage_score(confidence: float, priority: AlertPriority) -> int:
    """Return a transparent queue-ordering score, not a risk probability."""
    priority_bonus = {"low": 0, "normal": 5, "high": 10}[priority]
    return min(100, round(confidence * 90) + priority_bonus)


def evaluate_alert_rules(
    timeline_id: str,
    events: Sequence[ChangeEvent],
    rules: Sequence[AlertRule],
    *,
    created_at: datetime | None = None,
) -> list[AlertCandidate]:
    """Apply rules to change evidence and return a deduplicated review queue.

    Every candidate starts as ``needs_review``. Matching rules do not confirm an
    event and do not infer intent, ownership, or threat level.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", timeline_id):
        raise ValueError("timeline_id must be a safe non-empty identifier.")

    timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    candidates: dict[str, AlertCandidate] = {}
    for rule in rules:
        if not rule.enabled:
            continue
        for event in events:
            if event.status not in rule.event_statuses:
                continue
            if rule.class_names is not None and event.class_name not in rule.class_names:
                continue
            if event.confidence < rule.min_confidence:
                continue

            alert_id = _alert_id(timeline_id, rule.rule_id, event)
            candidates.setdefault(
                alert_id,
                AlertCandidate(
                    alert_id=alert_id,
                    timeline_id=timeline_id,
                    rule_id=rule.rule_id,
                    rule_name=rule.name.strip(),
                    event_status=event.status,
                    class_name=event.class_name,
                    confidence=round(event.confidence, 6),
                    priority=rule.priority,
                    triage_score=_triage_score(event.confidence, rule.priority),
                    evidence=(
                        f"change_status={event.status}; class={event.class_name}; "
                        f"model_confidence={event.confidence:.3f}; overlap={event.overlap:.3f}"
                    ),
                    before_id=event.before_id,
                    after_id=event.after_id,
                    overlap=round(event.overlap, 6),
                    created_at=timestamp,
                ),
            )

    return sorted(
        candidates.values(),
        key=lambda item: (-item.triage_score, item.rule_id, item.alert_id),
    )


class AlertStore:
    """Small filesystem store with idempotent alert persistence."""

    def __init__(self, root: Path):
        self.root = root / "alerts"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, alert_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{24}", alert_id):
            raise KeyError("Invalid alert ID.")
        return self.root / f"alert_{alert_id}.json"

    def save_many(self, alerts: Sequence[AlertCandidate]) -> list[AlertCandidate]:
        saved: list[AlertCandidate] = []
        for alert in alerts:
            target = self._path(alert.alert_id)
            if target.exists():
                saved.append(self.load(alert.alert_id))
                continue
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(asdict(alert), ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(target)
            saved.append(alert)
        return saved

    def load(self, alert_id: str) -> AlertCandidate:
        target = self._path(alert_id)
        if not target.exists():
            raise KeyError("Alert not found.")
        return AlertCandidate(**json.loads(target.read_text(encoding="utf-8")))

    def list(self, *, review_status: AlertReviewStatus | None = None) -> list[AlertCandidate]:
        records: list[AlertCandidate] = []
        for target in self.root.glob("alert_*.json"):
            try:
                record = AlertCandidate(**json.loads(target.read_text(encoding="utf-8")))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if review_status is None or record.review_status == review_status:
                records.append(record)
        return sorted(records, key=lambda item: (-item.triage_score, item.created_at, item.alert_id))
