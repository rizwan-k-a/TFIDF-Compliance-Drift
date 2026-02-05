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

    docs = list(documents)
    if len(docs) < 2:
        return None

    if precomputed_vectorizer is not None and precomputed_matrix is not None:
        vectorizer = precomputed_vectorizer
        X = precomputed_matrix
        n_docs = X.shape[0]
        if n_docs < 2 or getattr(X, "nnz", 0) <= 0:
            return None
    else:
        processed = [preprocess_text(d, keep_numbers=keep_numbers, use_lemmatization=use_lemma) for d in docs]
        n_docs = len(processed)

        adjusted_min_df = 1 if (n_docs <= 10) else (min_df if min_df is not None else max(1, int(n_docs * CONFIG.min_df)))
        adjusted_max_df = 1.0 if (n_docs <= 10) else (max_df if (max_df is not None and max_df < 1.0) else CONFIG.max_df)

        try:
            vectorizer = TfidfVectorizer(
                max_features=max_features or CONFIG.tfidf_max_features,
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
                max_features=min(1000, max_features or CONFIG.tfidf_max_features),
                ngram_range=(1, 1),
                min_df=1,
                max_df=1.0,
                stop_words=None,
            )
            X = vectorizer.fit_transform(processed)

    effective_n_clusters = min(int(n_clusters), max(2, n_docs - 1))

    kmeans = KMeans(n_clusters=effective_n_clusters, random_state=CONFIG.random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    if len(np.unique(labels)) < 2:
        return None

    try:
        sil = float(silhouette_score(X, labels))
    except Exception:
        sil = float("nan")

    try:
        dbi = float(davies_bouldin_score(X.toarray(), labels))
    except Exception:
        dbi = float("nan")

    pca = PCA(n_components=min(2, X.shape[1]))
    coords = pca.fit_transform(X.toarray())

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
        "names": list(names),
        "n_clusters": effective_n_clusters,
    }
