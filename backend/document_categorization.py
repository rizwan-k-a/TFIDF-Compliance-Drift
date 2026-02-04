"""Keyword-based document categorization."""

from __future__ import annotations

from typing import Dict

from .config import CATEGORIES


def categorize_document(text: str, filename: str) -> str:
    """Categorize a document by keyword frequency.

    Args:
        text: Document text.
        filename: Original filename (used as a weak signal).

    Returns:
        Category name or 'Uncategorized'.
    """

    text_lower = (text or "").lower()
    filename_lower = (filename or "").lower()

    scores: Dict[str, int] = {}
    for category, info in CATEGORIES.items():
        keywords = info.get("keywords", [])
        score = 0
        for kw in keywords:
            score += text_lower.count(str(kw))
            score += filename_lower.count(str(kw)) * 2
        scores[category] = score

    if not scores:
        return "Uncategorized"

    best = max(scores, key=scores.get)
    return best if scores.get(best, 0) > 0 else "Uncategorized"
