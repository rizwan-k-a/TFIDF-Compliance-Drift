"""Similarity computations for compliance comparisons."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import CATEGORIES
from .tfidf_engine import build_tfidf_vectors

logger = logging.getLogger(__name__)


def _error_result(message: str, details: str | None = None) -> Dict[str, object]:
    logger.error("Similarity error: %s", message)
    return {"error": message, "details": details}


def compute_similarity_scores_by_category_from_vectors(
    internal_names_by_category: Dict[str, list[str]],
    internal_indices_by_category: Dict[str, list[int]],
    guideline_names_by_category: Dict[str, list[str]],
    guideline_indices_by_category: Dict[str, list[int]],
    ref_vectors,
    int_vectors,
) -> pd.DataFrame:
    """Compute similarity per category using a shared TF-IDF matrix.

    This avoids duplicate preprocessing/vectorization across tabs by reusing the
    already-computed TF-IDF vectors.

    Args:
        internal_names_by_category: category -> list of internal document names
        internal_indices_by_category: category -> list of row indices into int_vectors
        guideline_names_by_category: category -> list of guideline document names
        guideline_indices_by_category: category -> list of row indices into ref_vectors
        ref_vectors: TF-IDF matrix for guideline/reference docs
        int_vectors: TF-IDF matrix for internal docs
    """

    if not isinstance(internal_names_by_category, dict) or not isinstance(internal_indices_by_category, dict):
        return _error_result("Invalid internal category inputs", "Expected dicts for internal names/indices")

    if not isinstance(guideline_names_by_category, dict) or not isinstance(guideline_indices_by_category, dict):
        return _error_result("Invalid guideline category inputs", "Expected dicts for guideline names/indices")

    if ref_vectors is None or int_vectors is None:
        return _error_result("Missing TF-IDF vectors", "ref_vectors and int_vectors must be provided")

    all_results = []

    for category in CATEGORIES.keys():
        i_idx = internal_indices_by_category.get(category) or []
        g_idx = guideline_indices_by_category.get(category) or []
        if not i_idx or not g_idx:
            continue

        internal_names = internal_names_by_category.get(category) or []
        guideline_names = guideline_names_by_category.get(category) or []

        if not internal_names or not guideline_names:
            continue

        int_vectors_cat = int_vectors[i_idx]
        ref_vectors_cat = ref_vectors[g_idx]

        batch_size = 100
        for batch_start in range(0, len(internal_names), batch_size):
            batch_end = min(batch_start + batch_size, len(internal_names))
            batch_vectors = int_vectors_cat[batch_start:batch_end]
            try:
                sim = cosine_similarity(batch_vectors, ref_vectors_cat)
            except Exception as e:
                return _error_result("Similarity computation failed", str(e))

            for row_offset, doc_name in enumerate(internal_names[batch_start:batch_end]):
                row = sim[row_offset]
                max_similarity = float(np.max(row))
                best_match_local_idx = int(np.argmax(row))
                matched_name = (
                    guideline_names[best_match_local_idx]
                    if best_match_local_idx < len(guideline_names)
                    else "(unknown)"
                )
                all_results.append(
                    {
                        "category": category,
                        "internal_document": doc_name,
                        "matched_guideline": matched_name,
                        "compliance_score": max_similarity,
                        "similarity_percent": round(max_similarity * 100.0, 1),
                        "divergence_percent": round((1.0 - max_similarity) * 100.0, 1),
                    }
                )

    return pd.DataFrame(all_results)


def compute_similarity_scores_by_category(
    categorized_docs: Dict,
    categorized_guidelines: Dict,
    keep_numbers: bool = True,
    use_lemma: bool = False,
    max_features=None,
    min_df=None,
    max_df=None,
) -> pd.DataFrame:
    """Compute cosine similarity per category between internal docs and guidelines.

    Expected inputs:
      categorized_docs[category] = {'docs': [...], 'names': [...]}
      categorized_guidelines[category] = {'docs': [...], 'names': [...]}

    Returns:
        DataFrame with:
          category, internal_document, matched_guideline, compliance_score,
          similarity_percent, divergence_percent
    """

    if not isinstance(categorized_docs, dict) or not isinstance(categorized_guidelines, dict):
        return _error_result("Invalid categorized inputs", "Expected dicts for categorized docs and guidelines")

    if not categorized_docs or not categorized_guidelines:
        return _error_result("No categorized documents provided", "Both categorized_docs and categorized_guidelines must be non-empty")

    all_results = []

    for category in CATEGORIES.keys():
        if category not in categorized_docs or category not in categorized_guidelines:
            continue

        internal_docs = categorized_docs[category].get("docs", [])
        internal_names = categorized_docs[category].get("names", [])
        guideline_docs = categorized_guidelines[category].get("docs", [])
        guideline_names = categorized_guidelines[category].get("names", [])

        if not internal_docs or not guideline_docs:
            continue

        total_docs = len(internal_docs) + len(guideline_docs)
        if total_docs < 2:
            continue

        try:
            _vectorizer, ref_vecs, int_vecs = build_tfidf_vectors(
                reference_docs=guideline_docs,
                internal_docs=internal_docs,
                keep_numbers=keep_numbers,
                use_lemma=use_lemma,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
            )
        except Exception as e:
            return _error_result("TF-IDF vectorization failed", str(e))

        batch_size = 100
        for batch_start in range(0, len(internal_names), batch_size):
            batch_end = min(batch_start + batch_size, len(internal_names))
            batch_vectors = int_vecs[batch_start:batch_end]
            try:
                sim = cosine_similarity(batch_vectors, ref_vecs)
            except Exception as e:
                return _error_result("Similarity computation failed", str(e))

            for row_offset, doc_name in enumerate(internal_names[batch_start:batch_end]):
                row = sim[row_offset]
                max_similarity = float(np.max(row)) if len(row) > 0 else 0.0
                best_match_idx = int(np.argmax(row)) if len(row) > 0 else 0
                all_results.append(
                    {
                        "category": category,
                        "internal_document": doc_name,
                        "matched_guideline": guideline_names[best_match_idx] if best_match_idx < len(guideline_names) else "(unknown)",
                        "compliance_score": max_similarity,
                        "similarity_percent": round(max_similarity * 100.0, 1),
                        "divergence_percent": round((1.0 - max_similarity) * 100.0, 1),
                    }
                )

    return pd.DataFrame(all_results)
