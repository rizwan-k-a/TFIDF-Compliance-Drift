from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.tfidf_engine import vectorize_documents


def render_tfidf_matrix_tab(cfg: dict, internal_docs: list[dict], guideline_docs: list[dict]) -> None:
    st.subheader("TF-IDF Matrix")

    docs = guideline_docs + internal_docs
    if len(docs) < 2:
        st.info("Upload at least 2 documents.")
        return

    names = [d.get("name", "doc") for d in docs]
    texts = [d.get("text", "") for d in docs]

    vectorizer, X = vectorize_documents(
        texts,
        keep_numbers=bool(cfg.get("keep_numbers", True)),
        use_lemma=bool(cfg.get("use_lemma", False)),
        max_features=int(cfg.get("max_features", 5000)),
    )

    if vectorizer is None or X is None:
        st.warning("Not enough meaningful text to build a TF-IDF matrix.")
        return

    terms = vectorizer.get_feature_names_out()

    max_terms = st.slider("Show top N terms", 20, 200, 50, step=10)
    matrix = X.toarray()
    df = pd.DataFrame(matrix[:, :max_terms], index=names, columns=[str(t) for t in terms[:max_terms]])

    st.dataframe(df, width="stretch")
