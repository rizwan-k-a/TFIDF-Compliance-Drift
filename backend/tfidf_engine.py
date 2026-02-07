"""TF-IDF engine: manual variants + sklearn TF-IDF vectorization.

Public API:
  - compute_tf_variants
  - compute_idf_variants
  - compute_manual_tfidf_complete
  - build_tfidf_vectors
  - vectorize_documents (convenience)
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import CONFIG
from .text_processing import preprocess_text


logger = logging.getLogger(__name__)


def compute_tf_variants(term_count: int, doc_length: int, max_term_count: int) -> Dict[str, float]:
    """Compute standard TF variants.

    Variants:
      - binary
      - raw
      - normalized
      - log_norm
      - double_norm
    """

    if doc_length <= 0:
        return {k: 0.0 for k in ("binary", "raw", "normalized", "log_norm", "double_norm")}

    return {
        "binary": 1.0 if term_count > 0 else 0.0,
        "raw": float(term_count),
        "normalized": float(term_count) / float(doc_length),
        "log_norm": 1.0 + math.log(term_count) if term_count > 0 else 0.0,
        "double_norm": 0.5 + 0.5 * (float(term_count) / float(max_term_count)) if max_term_count > 0 else 0.0,
    }


def compute_idf_variants(df: int, n_docs: int) -> Dict[str, float]:
    """Compute standard IDF variants.

    Variants:
      - standard
      - smooth
      - sklearn_smooth
      - probabilistic
    """

    if df <= 0 or n_docs <= 0:
        return {k: 0.0 for k in ("standard", "smooth", "sklearn_smooth", "probabilistic")}

    return {
        "standard": math.log(n_docs / df) if df > 0 else 0.0,
        "smooth": math.log(n_docs / df) + 1.0 if df > 0 else 0.0,
        "sklearn_smooth": math.log((1.0 + n_docs) / (1.0 + df)) + 1.0,
        "probabilistic": math.log((n_docs - df) / df) if df < n_docs else 0.0,
    }


def compute_manual_tfidf_complete(
    documents: Sequence[str],
    sample_words: Sequence[str],
    keep_numbers: bool = True,
    use_lemma: bool = False,
) -> Dict[str, object]:
    """Compute manual TF/IDF variants for selected words.

    Returns a nested dict designed for direct rendering in the UI.
    """

    if documents is None or sample_words is None:
        logger.error("documents and sample_words must not be None")
        raise ValueError("documents and sample_words must not be None")

    if not isinstance(documents, (list, tuple)):
        logger.error("documents must be a list of strings")
        raise ValueError("documents must be a list of strings")

    if not isinstance(sample_words, (list, tuple)):
        logger.error("sample_words must be a list of strings")
        raise ValueError("sample_words must be a list of strings")

    docs = list(documents)
    if not docs:
        logger.error("documents must be non-empty")
        raise ValueError("documents must be non-empty")

    if any(not isinstance(d, str) for d in docs):
        logger.error("All documents must be strings")
        raise ValueError("All documents must be strings")

    if not sample_words:
        logger.error("sample_words must be non-empty")
        raise ValueError("sample_words must be non-empty")
    n_docs = len(docs)

    processed_docs = [preprocess_text(d, keep_numbers=keep_numbers, use_lemmatization=use_lemma).split() for d in docs]
    max_term_counts = [max(Counter(doc).values()) if doc else 1 for doc in processed_docs]

    results: Dict[str, object] = {}

    for word in sample_words:
        word_lower = (word or "").lower().strip()
        if not word_lower:
            continue

        word_data = {
            "tf_variants_per_doc": [],
            "df": 0,
            "idf_variants": {},
            "tfidf_variants_per_doc": [],
        }

        for idx, doc_words in enumerate(processed_docs):
            word_count = doc_words.count(word_lower)
            total_words = len(doc_words)

            tf_variants = compute_tf_variants(word_count, total_words, max_term_counts[idx])

            word_data["tf_variants_per_doc"].append(
                {
                    "doc_id": idx + 1,
                    "count": word_count,
                    "total_words": total_words,
                    **{f"tf_{k}": round(v, 6) for k, v in tf_variants.items()},
                }
            )

            if word_count > 0:
                word_data["df"] += 1

        df = int(word_data["df"])
        word_data["idf_variants"] = {k: round(v, 6) for k, v in compute_idf_variants(df, n_docs).items()}

        idf_sklearn = float(word_data["idf_variants"]["sklearn_smooth"])
        for tf_data in word_data["tf_variants_per_doc"]:
            tfidf = float(tf_data["tf_normalized"]) * idf_sklearn
            word_data["tfidf_variants_per_doc"].append({"doc_id": tf_data["doc_id"], "tfidf": round(tfidf, 6)})

        results[word_lower] = word_data

    return results


def _adjust_min_df(n_docs: int, min_df: Optional[float]) -> float | int:
    if n_docs <= 10:
        return 1

    if min_df is None:
        config_min_df = float(getattr(CONFIG, "min_df", 0.01))
        return max(1, int(n_docs * config_min_df))

    # If min_df provided as fraction in (0,1), convert to count
    if 0 < min_df < 1:
        return max(1, int(n_docs * min_df))

    return int(min_df)


def _adjust_max_df(n_docs: int, max_df: Optional[float]) -> float:
    if n_docs <= 10:
        return 1.0

    if max_df is None:
        return float(getattr(CONFIG, "max_df", 1.0))

    return float(max_df)


def build_tfidf_vectors(
    reference_docs: Sequence[str],
    internal_docs: Sequence[str],
    keep_numbers: bool = True,
    use_lemma: bool = False,
    max_features: Optional[int] = None,
    min_df: Optional[float] = None,
    max_df: Optional[float] = None,
) -> Tuple[TfidfVectorizer, "np.ndarray | object", "np.ndarray | object"]:
    """Build TF-IDF vectors for reference and internal documents.

    Args:
        reference_docs: Guideline/reference documents.
        internal_docs: Internal policy documents.

    Returns:
        (vectorizer, ref_vectors, int_vectors) where vectors are sparse matrices.

    Raises:
        ValueError: If there is insufficient text for vectorization.
    """

    if reference_docs is None or internal_docs is None:
        logger.error("reference_docs and internal_docs must not be None")
        raise ValueError("reference_docs and internal_docs must not be None")

    if not isinstance(reference_docs, (list, tuple)):
        logger.error("reference_docs must be a list of strings")
        raise ValueError("reference_docs must be a list of strings")

    if not isinstance(internal_docs, (list, tuple)):
        logger.error("internal_docs must be a list of strings")
        raise ValueError("internal_docs must be a list of strings")

    ref = list(reference_docs)
    internal = list(internal_docs)
    if not ref and not internal:
        logger.error("At least one document is required for TF-IDF")
        raise ValueError("At least one document is required for TF-IDF")

    if any(not isinstance(d, str) for d in ref + internal):
        logger.error("All documents must be strings")
        raise ValueError("All documents must be strings")

    if max_features is not None and (not isinstance(max_features, int) or max_features <= 0):
        logger.error("max_features must be a positive integer, got %s", max_features)
        raise ValueError(f"max_features must be a positive integer, got {max_features}")

    if min_df is not None and (not isinstance(min_df, (int, float)) or min_df <= 0):
        logger.error("min_df must be > 0, got %s", min_df)
        raise ValueError(f"min_df must be > 0, got {min_df}")

    if max_df is not None and (not isinstance(max_df, (int, float)) or max_df <= 0):
        logger.error("max_df must be > 0, got %s", max_df)
        raise ValueError(f"max_df must be > 0, got {max_df}")

    if isinstance(min_df, float) and min_df > 1.0:
        logger.error("min_df as a fraction must be <= 1.0, got %s", min_df)
        raise ValueError(f"min_df as a fraction must be <= 1.0, got {min_df}")

    if isinstance(max_df, float) and max_df > 1.0:
        logger.error("max_df as a fraction must be <= 1.0, got %s", max_df)
        raise ValueError(f"max_df as a fraction must be <= 1.0, got {max_df}")

    if min_df is not None and max_df is not None and min_df > max_df:
        logger.error("min_df must be <= max_df (got %s > %s)", min_df, max_df)
        raise ValueError(f"min_df must be <= max_df (got {min_df} > {max_df})")
    all_docs = ref + internal

    processed_docs = [preprocess_text(d, keep_numbers=keep_numbers, use_lemmatization=use_lemma) for d in all_docs]
    n_docs = len(processed_docs)

    meaningful = [d for d in processed_docs if d and d.strip()]
    if len(meaningful) < 2:
        raise ValueError("Not enough meaningful text for TF-IDF computation.")

    adjusted_min_df = _adjust_min_df(n_docs, min_df)
    adjusted_max_df = _adjust_max_df(n_docs, max_df)

    effective_max_features = max_features or int(getattr(CONFIG, "tfidf_max_features", 5000))

    primary = TfidfVectorizer(
        max_features=effective_max_features,
        ngram_range=CONFIG.ngram_range,
        min_df=adjusted_min_df,
        max_df=adjusted_max_df,
        stop_words="english",
        sublinear_tf=True,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
    )

    try:
        all_vectors = primary.fit_transform(processed_docs)
        vectorizer = primary
    except ValueError as e:
        logger.info("Primary TF-IDF failed (%s). Falling back to relaxed settings.", e)
        relaxed = TfidfVectorizer(
            max_features=min(1000, effective_max_features),
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0,
            stop_words=None,
            sublinear_tf=True,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
        )
        all_vectors = relaxed.fit_transform(processed_docs)
        vectorizer = relaxed

    ref_vectors = all_vectors[: len(ref)]
    int_vectors = all_vectors[len(ref) :]

    return vectorizer, ref_vectors, int_vectors


def vectorize_documents(
    documents: Sequence[str],
    keep_numbers: bool = True,
    use_lemma: bool = False,
    max_features: Optional[int] = None,
    min_df: Optional[float] = None,
    max_df: Optional[float] = None,
) -> Dict[str, object]:
    """Convenience wrapper to vectorize a single corpus."""

    try:
        vectorizer, _ref, X = build_tfidf_vectors(
            reference_docs=[],
            internal_docs=list(documents),
            keep_numbers=keep_numbers,
            use_lemma=use_lemma,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        )
        return {"vectorizer": vectorizer, "matrix": X}
    except Exception as e:
        logger.error("Vectorization failed: %s", e)
        return {"error": "Vectorization failed", "details": str(e)}
