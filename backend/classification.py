"""Supervised classification backend.

Implements:
- Multinomial Naive Bayes
- Logistic Regression
- Optional 5-fold cross validation

This module is UI-agnostic; it returns structured results instead of calling Streamlit.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

from .config import CONFIG
from .text_processing import preprocess_text


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CVSummary:
    mean: float
    std: float


def perform_classification(
    documents: Sequence[str],
    categories: Sequence[str],
    test_size: float = 0.3,
    keep_numbers: bool = True,
    use_lemma: bool = False,
    max_features: Optional[int] = None,
    min_df: Optional[float] = None,
    max_df: Optional[float] = None,
    use_cv: bool = False,
    precomputed_vectorizer: Optional[TfidfVectorizer] = None,
    precomputed_matrix: Optional[object] = None,
) -> Optional[Dict[str, object]]:
    """Train NB + LR classifiers on TF-IDF vectors.

        Filtering rules:
            - Do not split when dataset is small or class counts are too low.
            - Train/evaluate on full set with warnings when needed.

    Returns:
        Dict of results, or error dict if validation fails.
    """

    # INPUT VALIDATION (prevents silent ML failures)
    if not isinstance(test_size, (int, float)) or not (0 < test_size < 1):
        logger.error("test_size must be in (0, 1), got %s", test_size)
        return {"error": f"test_size must be in (0, 1), got {test_size}", "vectorizer": None}

    if documents is None or categories is None:
        logger.error("documents and categories must not be None")
        return {"error": "documents and categories must not be None", "vectorizer": None}

    if not isinstance(documents, (list, tuple)):
        logger.error("documents must be a list of strings")
        return {"error": "documents must be a list of strings", "vectorizer": None}

    if not isinstance(categories, (list, tuple)):
        logger.error("categories must be a list of strings")
        return {"error": "categories must be a list of strings", "vectorizer": None}

    docs = list(documents)
    cats = list(categories)

    if not docs or not cats:
        logger.error("documents and categories must be non-empty")
        return {"error": "documents and categories must be non-empty", "vectorizer": None}

    if len(docs) != len(cats):
        logger.error("Length mismatch: %d documents vs %d categories", len(docs), len(cats))
        return {
            "error": f"Length mismatch: {len(docs)} documents vs {len(cats)} categories",
            "vectorizer": None,
        }

    if any(not isinstance(d, str) for d in docs):
        logger.error("All documents must be strings")
        return {"error": "All documents must be strings", "vectorizer": None}

    if any(not isinstance(c, str) for c in cats):
        logger.error("All categories must be strings")
        return {"error": "All categories must be strings", "vectorizer": None}

    if max_features is not None and (not isinstance(max_features, int) or max_features <= 0):
        logger.error("max_features must be a positive integer, got %s", max_features)
        return {"error": f"max_features must be a positive integer, got {max_features}", "vectorizer": None}

    if min_df is not None and (not isinstance(min_df, (int, float)) or min_df <= 0):
        logger.error("min_df must be > 0, got %s", min_df)
        return {"error": f"min_df must be > 0, got {min_df}", "vectorizer": None}

    if max_df is not None and (not isinstance(max_df, (int, float)) or max_df <= 0):
        logger.error("max_df must be > 0, got %s", max_df)
        return {"error": f"max_df must be > 0, got {max_df}", "vectorizer": None}

    if isinstance(min_df, float) and min_df > 1.0:
        logger.error("min_df as a fraction must be <= 1.0, got %s", min_df)
        return {"error": f"min_df as a fraction must be <= 1.0, got {min_df}", "vectorizer": None}

    if isinstance(max_df, float) and max_df > 1.0:
        logger.error("max_df as a fraction must be <= 1.0, got %s", max_df)
        return {"error": f"max_df as a fraction must be <= 1.0, got {max_df}", "vectorizer": None}

    if min_df is not None and max_df is not None and min_df > max_df:
        logger.error("min_df must be <= max_df (got %s > %s)", min_df, max_df)
        return {"error": f"min_df must be <= max_df (got {min_df} > {max_df})", "vectorizer": None}

    category_counts = Counter(cats)
    filtered_docs = docs
    filtered_categories = cats
    filtered_indices = list(range(len(docs)))
    excluded_count = 0

    if precomputed_vectorizer is not None and precomputed_matrix is not None:
        vectorizer = precomputed_vectorizer
        try:
            X = precomputed_matrix[filtered_indices]
        except Exception:
            logger.error("Precomputed matrix indexing failed")
            return {"error": "Precomputed matrix indexing failed", "vectorizer": None}

        if getattr(X, "nnz", 0) <= 0:
            logger.error("Precomputed matrix has no non-zero entries")
            return {"error": "Precomputed matrix has no non-zero entries", "vectorizer": None}
    else:
        processed_docs = [preprocess_text(d, keep_numbers=keep_numbers, use_lemmatization=use_lemma) for d in filtered_docs]
        meaningful = [d for d in processed_docs if d and d.strip()]
        if len(meaningful) < 2:
            logger.error("Not enough meaningful text for classification")
            return {"error": "Not enough meaningful text for classification", "vectorizer": None}

        primary = TfidfVectorizer(
            max_features=max_features or CONFIG.tfidf_max_features,
            ngram_range=CONFIG.ngram_range,
            stop_words="english",
            min_df=1 if (min_df is None or min_df <= 0 or len(processed_docs) <= 10) else min_df,
            max_df=1.0 if (max_df is None or max_df >= 1.0 or len(processed_docs) <= 10) else max_df,
        )

        try:
            X = primary.fit_transform(processed_docs)
            vectorizer = primary
        except ValueError:
            relaxed = TfidfVectorizer(
                max_features=min(1000, max_features or CONFIG.tfidf_max_features),
                ngram_range=(1, 1),
                stop_words=None,
                min_df=1,
                max_df=1.0,
            )
            try:
                X = relaxed.fit_transform(processed_docs)
                vectorizer = relaxed
            except Exception:
                logger.error("TF-IDF vectorization failed (relaxed settings)")
                return {"error": "TF-IDF vectorization failed", "vectorizer": None}
    y = filtered_categories
    warnings: List[str] = []
    debug: Dict[str, object] = {
        "doc_count": len(y),
        "class_counts": dict(Counter(y)),
    }

    # CRITICAL: Check minimum samples per class (MUST have >= 3)
    class_counter = Counter(y)
    if len(class_counter) < 2:
        error_msg = "At least 2 categories are required for classification"
        logger.warning("Classification rejected: %s", error_msg)
        return {"error": error_msg, "vectorizer": None}

    min_class_count = min(class_counter.values()) if y else 0
    
    if min_class_count < 3:
        min_class_name = min(y, key=lambda c: class_counter[c]) if y else "unknown"
        error_msg = f"Insufficient samples for class '{min_class_name}': {min_class_count} < 3 required"
        logger.warning("Classification rejected: %s", error_msg)
        return {"error": error_msg, "vectorizer": None}
    
    # Check for severe class imbalance (>10:1 ratio)
    max_class_count = max(class_counter.values()) if y else 0
    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else 0
    
    if imbalance_ratio > 10:
        imbalance_msg = f"Severe class imbalance detected (ratio {imbalance_ratio:.1f}:1). Results may be unreliable."
        logger.warning("Class imbalance warning: %s", imbalance_msg)
        warnings.append(imbalance_msg)

    if len(y) < 10:
        warnings.append("Dataset has fewer than 10 labeled documents; training on full set")

    use_split = len(warnings) == 0

    if use_split:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=CONFIG.random_state,
                stratify=y,
            )
            train_classes = set(y_train)
            test_classes = set(y_test)
            if train_classes != set(y) or test_classes != set(y):
                warnings.append("Train/test split missing classes; training on full set")
                use_split = False
        except Exception as e:
            logger.info("Train/test split failed: %s", e)
            warnings.append("Train/test split failed; training on full set")
            use_split = False

    if not use_split:
        X_train, y_train = X, y
        X_test, y_test = X, y

    debug["train_size"] = len(y_train)
    debug["test_size"] = len(y_test)

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_acc = float(accuracy_score(y_test, nb_pred))

    lr_model = LogisticRegression(max_iter=1000, random_state=CONFIG.random_state)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = float(accuracy_score(y_test, lr_pred))

    nb_cv_scores = None
    lr_cv_scores = None
    nb_cv_summary = None
    lr_cv_summary = None

    if use_cv:
        try:
            nb_cv_scores = cross_val_score(nb_model, X, y, cv=5, scoring="accuracy")
            lr_cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring="accuracy")
            nb_cv_summary = CVSummary(mean=float(nb_cv_scores.mean()), std=float(nb_cv_scores.std()))
            lr_cv_summary = CVSummary(mean=float(lr_cv_scores.mean()), std=float(lr_cv_scores.std()))
        except Exception as e:
            logger.info("Cross-validation failed: %s", e)

    feature_names = vectorizer.get_feature_names_out()

    top_features_per_class: Dict[str, List[tuple]] = {}
    classes = [str(c) for c in lr_model.classes_]

    # Binary LogisticRegression stores one coefficient vector (for classes_[1]).
    if lr_model.coef_.ndim == 2 and lr_model.coef_.shape[0] == 1 and len(classes) == 2:
        coef = lr_model.coef_[0]
        pos_idx = np.argsort(coef)[-10:][::-1]
        neg_idx = np.argsort(coef)[:10]
        top_features_per_class[classes[1]] = [(str(feature_names[i]), float(coef[i])) for i in pos_idx]
        top_features_per_class[classes[0]] = [(str(feature_names[i]), float(coef[i])) for i in neg_idx]
    else:
        for idx, category in enumerate(classes):
            coef = lr_model.coef_[idx] if lr_model.coef_.ndim > 1 else lr_model.coef_
            top_idx = np.argsort(np.abs(coef))[-10:][::-1]
            top_features_per_class[str(category)] = [(str(feature_names[i]), float(coef[i])) for i in top_idx]

    return {
        "vectorizer": vectorizer,
        "nb_model": nb_model,
        "nb_accuracy": nb_acc,
        "nb_predictions": nb_pred,
        "nb_cv_scores": nb_cv_scores,
        "nb_cv_mean": nb_cv_summary.mean if nb_cv_summary else None,
        "nb_cv_std": nb_cv_summary.std if nb_cv_summary else None,
        "lr_model": lr_model,
        "lr_accuracy": lr_acc,
        "lr_predictions": lr_pred,
        "lr_cv_scores": lr_cv_scores,
        "lr_cv_mean": lr_cv_summary.mean if lr_cv_summary else None,
        "lr_cv_std": lr_cv_summary.std if lr_cv_summary else None,
        "y_test": y_test,
        "top_features": top_features_per_class,
        "confusion_matrix_nb": confusion_matrix(y_test, nb_pred, labels=nb_model.classes_),
        "confusion_matrix_lr": confusion_matrix(y_test, lr_pred, labels=lr_model.classes_),
        "classification_report_nb": classification_report(y_test, nb_pred, zero_division=0),
        "classification_report_lr": classification_report(y_test, lr_pred, zero_division=0),
        "filtered_count": len(filtered_docs),
        "excluded_count": excluded_count,
        "category_distribution": dict(Counter(filtered_categories)),
        "warnings": warnings,
        "debug": debug,
    }

