from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.tfidf_engine import compute_manual_tfidf_complete


def render_tfidf_math_tab(cfg: dict, internal_docs: list[dict], guideline_docs: list[dict]) -> None:
    st.subheader("TF-IDF Maths")

    corpus = [d.get("text", "") for d in (guideline_docs + internal_docs)]
    if len(corpus) < 2:
        st.info("Upload at least 2 documents to explore TF-IDF.")
        return

    st.markdown(
        """
- Term Frequency: $tf(t,d)$ variants (raw, normalized, log, etc.)
- Inverse Document Frequency: $idf(t)$ variants (standard, smoothed, sklearn-style)
- TF-IDF: $tf(t,d) \cdot idf(t)$
        """
    )

    words = st.text_input("Words to inspect (comma-separated)", value="aml, transaction, incident")
    sample_words = [w.strip() for w in words.split(",") if w.strip()]

    results = compute_manual_tfidf_complete(
        documents=corpus,
        sample_words=sample_words,
        keep_numbers=bool(cfg.get("keep_numbers", True)),
        use_lemma=bool(cfg.get("use_lemma", False)),
    )

    for word, data in results.items():
        st.markdown(f"### Word: `{word}`")
        tf_df = pd.DataFrame(data["tf_variants_per_doc"]) if data.get("tf_variants_per_doc") else pd.DataFrame()
        idf = data.get("idf_variants", {})
        tfidf_df = pd.DataFrame(data.get("tfidf_variants_per_doc", []))

        col1, col2 = st.columns(2)
        with col1:
            st.write("TF variants")
            st.dataframe(tf_df, use_container_width=True)
        with col2:
            st.write("IDF variants")
            st.json(idf)

        st.write("TF-IDF (normalized TF * sklearn-smooth IDF)")
        st.dataframe(tfidf_df, use_container_width=True)
