"""Similarity computations for compliance comparisons."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import CATEGORIES
from .tfidf_engine import build_tfidf_vectors


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

        vectorizer, ref_vecs, int_vecs = build_tfidf_vectors(
            reference_docs=guideline_docs,
            internal_docs=internal_docs,
            keep_numbers=keep_numbers,
            use_lemma=use_lemma,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        )

        sim = cosine_similarity(int_vecs, ref_vecs)

        for i, doc_name in enumerate(internal_names):
            row = sim[i]
            max_similarity = float(np.max(row))
            best_match_idx = int(np.argmax(row))
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
