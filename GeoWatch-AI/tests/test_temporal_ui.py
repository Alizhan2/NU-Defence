from src.temporal_ui import REVIEW_STATUS_LABELS, review_decision_hint, review_decision_ready


def test_confirmed_and_rejected_decisions_require_a_reason():
    assert review_decision_ready("confirmed", "") is False
    assert review_decision_ready("rejected", "   ") is False
    assert review_decision_ready("confirmed", "Сверено с footprint-разметкой") is True
    assert "основание" in review_decision_hint("rejected", "").lower()


def test_needs_review_can_be_preserved_without_a_reason():
    assert review_decision_ready("needs_review", "") is True
    assert REVIEW_STATUS_LABELS["needs_review"] == "Требует проверки"


def test_unknown_review_state_is_never_saved():
    assert review_decision_ready("approved", "ok") is False
    assert "допустимое" in review_decision_hint("approved", "ok")
