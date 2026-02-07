"""Keyword-based document categorization with proper stemming."""

from __future__ import annotations

import logging
import re
from typing import Dict

from .config import CATEGORIES

logger = logging.getLogger(__name__)

_stemmer = None

def _get_stemmer():
    """Lazy-load NLTK stemmer (optional fallback)."""
    global _stemmer
    if _stemmer is None:
        try:
            from nltk.stem import PorterStemmer
            _stemmer = PorterStemmer()
        except ImportError:
            logger.debug("NLTK not available; using simple keyword matching")
            _stemmer = False
    return _stemmer if _stemmer else None


def _tokenize_and_stem(text: str) -> set[str]:
    """Tokenize text and optionally stem words."""
    tokens = set(re.findall(r'\b\w+\b', text.lower()))
    filtered = {t for t in tokens if len(t) > 2}
    stemmer = _get_stemmer()
    if stemmer:
        return {stemmer.stem(t) for t in filtered}
    else:
        return filtered


def categorize_document(text: str, filename: str) -> str:
    """Categorize document using stemmed keyword matching with Jaccard similarity."""
    
    text = text or ""
    filename = filename or ""
    
    text_tokens = _tokenize_and_stem(text)
    filename_tokens = _tokenize_and_stem(filename)
    
    if not text_tokens and not filename_tokens:
        return "Uncategorized"
    
    scores: Dict[str, float] = {}
    
    for category, info in CATEGORIES.items():
        keywords = info.get("keywords", [])
        if not keywords:
            continue
        
        keyword_stems = _tokenize_and_stem(" ".join(str(kw) for kw in keywords))
        if not keyword_stems:
            scores[category] = 0
            continue
        
        text_intersection = len(text_tokens & keyword_stems)
        text_union = len(text_tokens | keyword_stems)
        filename_intersection = len(filename_tokens & keyword_stems)
        filename_union = len(filename_tokens | keyword_stems)
        
        text_jaccard = text_intersection / text_union if text_union > 0 else 0
        filename_jaccard = filename_intersection / filename_union if filename_union > 0 else 0
        combined_score = (text_jaccard * 3 + filename_jaccard * 1) / 4
        scores[category] = combined_score
    
    if not scores:
        return "Uncategorized"
    
    best_category = max(scores, key=scores.get)
    best_score = scores.get(best_category, 0)
    
    if best_score < 0.1:
        return "Uncategorized"
    
    logger.info("Categorized as '%s' (score: %.2f)", best_category, best_score)
    return best_category
