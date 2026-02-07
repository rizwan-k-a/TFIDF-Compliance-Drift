"""Unsupervised clustering backend with quality metrics and top terms."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score

from .config import CONFIG
from .text_processing import preprocess_text


logger = logging.getLogger(__name__)


def _error_result(message: str, details: str | None = None) -> Dict[str, object]:
    logger.error("Clustering error: %s", message)
    return {"error": message, "details": details}


def perform_enhanced_clustering(
    documents: Sequence[str],
    names: Sequence[str],
    n_clusters: int = 3,
    keep_numbers: bool = True,
    use_lemma: bool = False,
    max_features: Optional[int] = None,
    min_df: Optional[float] = None,
    max_df: Optional[float] = None,
    precomputed_vectorizer: Optional[TfidfVectorizer] = None,
    precomputed_matrix: Optional[object] = None,
) -> Optional[Dict[str, object]]:
    """Cluster documents using KMeans with metrics and PCA visualization."""

    if documents is None or names is None:
        return _error_result("documents and names must not be None")

    if not isinstance(documents, (list, tuple)):
        return _error_result("documents must be a list of strings")

    if not isinstance(names, (list, tuple)):
        return _error_result("names must be a list of strings")

    docs = list(documents)
    name_list = list(names)

    if not docs:
        return _error_result("documents must be non-empty")

    if any(not isinstance(d, str) for d in docs):
        return _error_result("All documents must be strings")

    if len(name_list) != len(docs):
        return _error_result("Length mismatch between documents and names")

    if not isinstance(n_clusters, int) or n_clusters < 2:
        return _error_result("n_clusters must be an integer >= 2")

    if max_features is not None and (not isinstance(max_features, int) or max_features <= 0):
        return _error_result(f"max_features must be a positive integer, got {max_features}")

    if min_df is not None and (not isinstance(min_df, (int, float)) or min_df <= 0):
        return _error_result(f"min_df must be > 0, got {min_df}")

    if max_df is not None and (not isinstance(max_df, (int, float)) or max_df <= 0):
        return _error_result(f"max_df must be > 0, got {max_df}")

    if isinstance(min_df, float) and min_df > 1.0:
        return _error_result(f"min_df as a fraction must be <= 1.0, got {min_df}")

    if isinstance(max_df, float) and max_df > 1.0:
        return _error_result(f"max_df as a fraction must be <= 1.0, got {max_df}")

    if min_df is not None and max_df is not None and min_df > max_df:
        return _error_result(f"min_df must be <= max_df (got {min_df} > {max_df})")

    if precomputed_vectorizer is not None and precomputed_matrix is not None:
        vectorizer = precomputed_vectorizer
        X = precomputed_matrix
        n_docs = X.shape[0]
        if n_docs < 2 or getattr(X, "nnz", 0) <= 0:
            return _error_result("Precomputed matrix is empty or too small")
        if len(name_list) != n_docs:
            return _error_result("Name count does not match precomputed matrix rows")
    else:
        processed = [preprocess_text(d, keep_numbers=keep_numbers, use_lemmatization=use_lemma) for d in docs]
        n_docs = len(processed)

        config_min_df = float(getattr(CONFIG, "min_df", 0.01))
        config_max_df = float(getattr(CONFIG, "max_df", 1.0))
        adjusted_min_df = 1 if (n_docs <= 10) else (min_df if min_df is not None else max(1, int(n_docs * config_min_df)))
        adjusted_max_df = 1.0 if (n_docs <= 10) else (max_df if (max_df is not None and max_df < 1.0) else config_max_df)

        try:
            vectorizer = TfidfVectorizer(
                max_features=max_features or int(getattr(CONFIG, "tfidf_max_features", 5000)),
                ngram_range=CONFIG.ngram_range,
                min_df=adjusted_min_df,
                max_df=adjusted_max_df,
                stop_words="english",
                sublinear_tf=True,
                norm="l2",
                use_idf=True,
                smooth_idf=True,
            )
            X = vectorizer.fit_transform(processed)
        except Exception as e:
            logger.info("TF-IDF vectorization failed (%s). Using relaxed settings.", e)
            vectorizer = TfidfVectorizer(
                max_features=min(1000, max_features or int(getattr(CONFIG, "tfidf_max_features", 5000))),
                ngram_range=(1, 1),
                min_df=1,
                max_df=1.0,
                stop_words=None,
            )
            try:
                X = vectorizer.fit_transform(processed)
            except Exception as ex:
                logger.error("TF-IDF vectorization failed (relaxed settings): %s", ex)
                return _error_result("TF-IDF vectorization failed", str(ex))

    effective_n_clusters = min(int(n_clusters), max(2, n_docs - 1))

    kmeans = KMeans(n_clusters=effective_n_clusters, random_state=CONFIG.random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    if len(np.unique(labels)) < 2:
        return _error_result("Clustering produced a single cluster")

    try:
        sil = float(silhouette_score(X, labels))
    except Exception:
        sil = float("nan")

    try:
        dbi = float(davies_bouldin_score(X.toarray(), labels))
    except Exception:
        dbi = float("nan")

    try:
        pca = PCA(n_components=min(2, X.shape[1]))
        coords = pca.fit_transform(X.toarray())
    except Exception as e:
        return _error_result("PCA visualization failed", str(e))

    feature_names = vectorizer.get_feature_names_out()
    top_terms: Dict[int, List[tuple]] = {}

    for cluster_id in range(effective_n_clusters):
        if cluster_id >= len(kmeans.cluster_centers_):
            continue
        centroid = kmeans.cluster_centers_[cluster_id]
        top_idx = np.argsort(centroid)[-10:][::-1]
        top_terms[int(cluster_id)] = [(str(feature_names[i]), float(centroid[i])) for i in top_idx]

    return {
        "labels": labels,
        "coordinates": coords,
        "inertia": float(kmeans.inertia_),
        "silhouette_score": None if np.isnan(sil) else sil,
        "davies_bouldin_score": None if np.isnan(dbi) else dbi,
        "top_terms": top_terms,
        "vectorizer": vectorizer,
        "matrix": X,
        "names": list(name_list),
        "n_clusters": effective_n_clusters,
    }
