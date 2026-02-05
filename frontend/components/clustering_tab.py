from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.clustering import perform_enhanced_clustering


def render_clustering_tab(
    cfg: dict,
    internal_docs: list[dict],
    *,
    shared_vectorizer=None,
    shared_internal_matrix=None,
) -> None:
    st.subheader("Clustering")

    if len(internal_docs) < 3:
        st.info("Upload at least 3 internal documents.")
        return

    n_clusters = st.slider("Clusters", min_value=2, max_value=8, value=3)

    docs = [d.get("text", "") for d in internal_docs]
    names = [d.get("name", "doc") for d in internal_docs]

    res = perform_enhanced_clustering(
        documents=docs,
        names=names,
        n_clusters=int(n_clusters),
        keep_numbers=bool(cfg.get("keep_numbers", True)),
        use_lemma=bool(cfg.get("use_lemma", False)),
        max_features=int(cfg.get("max_features", 5000)),
        precomputed_vectorizer=shared_vectorizer,
        precomputed_matrix=shared_internal_matrix,
    )

    if not res:
        st.warning("Clustering could not be computed.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Inertia", f"{res['inertia']:.2f}")
    c2.metric("Silhouette", "" if res["silhouette_score"] is None else f"{res['silhouette_score']:.3f}")
    c3.metric("Davies-Bouldin", "" if res["davies_bouldin_score"] is None else f"{res['davies_bouldin_score']:.3f}")

    df = pd.DataFrame(
        {
            "Document": res.get("names", names),
            "Cluster": res.get("labels"),
            "x": res.get("coordinates")[:, 0],
            "y": res.get("coordinates")[:, 1],
        }
    )

    st.scatter_chart(df, x="x", y="y", color="Cluster")

    st.write("Top terms per cluster")
    st.json(res.get("top_terms", {}))
