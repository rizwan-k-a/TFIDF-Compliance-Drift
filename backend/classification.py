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
      - Remove categories with < 2 samples.
      - Require at least 6 documents after filtering (matches UI expectation).

    Returns:
        Dict of results, or None if insufficient data.
    """

    docs = list(documents)
    cats = list(categories)

    if not docs or not cats or len(docs) != len(cats):
        return None

    category_counts = Counter(cats)
    valid_categories = {cat for cat, count in category_counts.items() if count >= 2}
    if len(valid_categories) < 2:
        return None

    filtered_docs: List[str] = []
    filtered_categories: List[str] = []
    filtered_indices: List[int] = []

    for idx, (doc, cat) in enumerate(zip(docs, cats)):
        if cat in valid_categories:
            filtered_docs.append(doc)
            filtered_categories.append(cat)
            filtered_indices.append(idx)

    excluded_count = len(docs) - len(filtered_docs)
    if len(filtered_docs) < 6:
        return None

    if precomputed_vectorizer is not None and precomputed_matrix is not None:
        vectorizer = precomputed_vectorizer
        try:
            X = precomputed_matrix[filtered_indices]
        except Exception:
            return None

        if getattr(X, "nnz", 0) <= 0:
            return None
    else:
        processed_docs = [preprocess_text(d, keep_numbers=keep_numbers, use_lemmatization=use_lemma) for d in filtered_docs]
        meaningful = [d for d in processed_docs if d and d.strip()]
        if len(meaningful) < 2:
            return None

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
                return None
    y = filtered_categories

    min_class_count = min(Counter(y).values())
    use_stratify = min_class_count >= 2 and (test_size * len(y) >= len(set(y)))

    try:
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=CONFIG.random_state,
                stratify=y,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=CONFIG.random_state,
            )
    except Exception as e:
        logger.info("Train/test split failed: %s", e)
        return None

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
    }
