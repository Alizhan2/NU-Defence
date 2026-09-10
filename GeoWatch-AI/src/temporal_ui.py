from __future__ import annotations


REVIEW_STATUS_LABELS = {
    "needs_review": "Требует проверки",
    "confirmed": "Подтверждено",
    "rejected": "Отклонено",
}


def review_decision_ready(status: str, comment: str) -> bool:
    """A consequential human decision must carry a reviewable rationale."""
    if status not in REVIEW_STATUS_LABELS:
        return False
    return status == "needs_review" or bool(comment.strip())


def review_decision_hint(status: str, comment: str) -> str:
    if status not in REVIEW_STATUS_LABELS:
        return "Выберите допустимое решение."
    if status != "needs_review" and not comment.strip():
        return "Добавьте основание, чтобы решение можно было проверить позднее."
    return "Решение и основание будут сохранены в журнале temporal-события."
