from datetime import datetime, timezone

import pytest

from src.alerting import AlertRule, AlertStore, evaluate_alert_rules
from src.change_detection import ChangeEvent


NOW = datetime(2026, 9, 8, 8, 30, tzinfo=timezone.utc)


def event(status: str, class_name: str, confidence: float = 0.8) -> ChangeEvent:
    return ChangeEvent(
        status=status,
        class_name=class_name,
        confidence=confidence,
        before_id="before_1" if status != "appeared" else None,
        after_id="after_1" if status != "disappeared" else None,
        overlap=0.0,
    )


def test_rule_filters_by_status_class_and_confidence():
    rule = AlertRule(
        rule_id="new-vehicles",
        name="New vehicle candidate",
        event_statuses=frozenset({"appeared"}),
        class_names=frozenset({"small vehicle", "large vehicle"}),
        min_confidence=0.7,
        priority="high",
    )
    alerts = evaluate_alert_rules(
        "timeline_01",
        [
            event("appeared", "small vehicle", 0.82),
            event("appeared", "ship", 0.95),
            event("appeared", "large vehicle", 0.4),
            event("stable", "small vehicle", 0.99),
        ],
        [rule],
        created_at=NOW,
    )
    assert len(alerts) == 1
    assert alerts[0].class_name == "small vehicle"
    assert alerts[0].review_status == "needs_review"
    assert alerts[0].triage_score == 84
    assert "model_confidence=0.820" in alerts[0].evidence


def test_ids_are_stable_and_duplicate_input_is_deduplicated():
    rule = AlertRule("change", "Any change", frozenset({"appeared", "disappeared"}))
    change = event("disappeared", "ship")
    first = evaluate_alert_rules("timeline_01", [change, change], [rule], created_at=NOW)
    second = evaluate_alert_rules("timeline_01", [change], [rule], created_at=NOW)
    assert len(first) == 1
    assert first[0].alert_id == second[0].alert_id


def test_alert_store_is_idempotent(tmp_path):
    rule = AlertRule("change", "Any change", frozenset({"appeared"}))
    alerts = evaluate_alert_rules("timeline_01", [event("appeared", "aircraft")], [rule], created_at=NOW)
    store = AlertStore(tmp_path)
    assert store.save_many(alerts) == alerts
    assert store.save_many(alerts) == alerts
    assert store.list(review_status="needs_review") == alerts
    assert len(list((tmp_path / "alerts").glob("*.json"))) == 1


def test_disabled_rule_never_creates_an_alert():
    rule = AlertRule("off", "Disabled", frozenset({"appeared"}), enabled=False)
    assert evaluate_alert_rules("timeline_01", [event("appeared", "ship")], [rule]) == []


def test_priority_score_is_not_described_as_probability():
    rule = AlertRule("valid", "Valid", frozenset({"appeared"}))
    alert = evaluate_alert_rules("timeline_01", [event("appeared", "ship", 1.0)], [rule])[0]
    assert alert.triage_score <= 100
    assert "threat" not in alert.evidence.lower()


def test_invalid_rules_are_rejected():
    with pytest.raises(ValueError):
        AlertRule("bad id", "Bad", frozenset({"appeared"}))
    with pytest.raises(ValueError):
        AlertRule("bad-status", "Bad", frozenset({"unknown"}))
    with pytest.raises(ValueError):
        AlertRule("bad-confidence", "Bad", frozenset({"appeared"}), min_confidence=1.1)
