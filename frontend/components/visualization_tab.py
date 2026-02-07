from __future__ import annotations

import base64
from io import BytesIO

import streamlit as st


def render_visualization_tab(
    internal_docs: list[dict],
    guideline_docs: list[dict],
    *,
    shared_vectorizer=None,
    shared_matrix=None,
) -> None:
    st.subheader("Visualization")

    try:
        from wordcloud import STOPWORDS, WordCloud
    except ImportError:
        st.info("📦 Install `wordcloud` to enable word cloud visualization: `pip install wordcloud`")
        return

    docs = guideline_docs + internal_docs
    texts_list = [str(d.get("text", "")) for d in docs]
    joined = "\n".join(texts_list)
    if len(joined.strip()) < 50:
        st.info("Upload documents with more text.")
        return

    # Word cloud is TF-IDF weighted — used for intuitive explanation in viva.
    try:
        import numpy as np
    except ImportError as e:
        st.warning(f"📦 NumPy is required for word clouds. Install with: `pip install numpy`")
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("NumPy import failed: %s", e)
        return

    custom_stopwords = {
        "user",
        "system",
        "data",
        "api",
        "ctx",
        "json",
        "module",
        "service",
        "class",
        "function",
        "file",
        "document",
    }
    stopwords = set(STOPWORDS) | {s.lower() for s in custom_stopwords}

    if shared_vectorizer is not None and shared_matrix is not None:
        vectorizer, X = shared_vectorizer, shared_matrix
    else:
        try:
            from backend.tfidf_engine import vectorize_documents
        except (ImportError, ModuleNotFoundError) as e:
            st.warning(f"TF-IDF engine unavailable. Check backend module: {e}")
            import logging
            logger = logging.getLogger(__name__)
            logger.error("Failed to import vectorize_documents: %s", e)
            return

        result = vectorize_documents(
            texts_list,
            keep_numbers=True,
            use_lemma=False,
            max_features=5000,
        )

        if isinstance(result, dict) and result.get("error"):
            st.info(result.get("error"))
            return

        vectorizer, X = result.get("vectorizer"), result.get("matrix")

    if vectorizer is None or X is None:
        st.info("Not enough meaningful text to build a TF-IDF-based word cloud.")
        return

    terms = vectorizer.get_feature_names_out()
    try:
        weights = np.asarray(X.mean(axis=0)).ravel()
    except (AttributeError, ValueError):
        # Sparse matrix fallback
        try:
            weights = np.asarray(X.toarray()).mean(axis=0)
        except MemoryError:
            st.error("💾 Matrix too large to convert to dense format. Try fewer documents.")
            import logging
            logger = logging.getLogger(__name__)
            logger.error("Memory exhausted converting sparse to dense matrix")
            return

    weighted_terms = []
    for term, weight in zip(terms, weights):
        t = str(term)
        w = float(weight)
        if w <= 0:
            continue
        if len(t) <= 2:
            continue
        if t.lower() in stopwords:
            continue
        weighted_terms.append((t, w))

    if not weighted_terms:
        st.info("No informative TF-IDF terms found for the word cloud.")
        return

    weighted_terms.sort(key=lambda x: x[1], reverse=True)
    # Limit to a small set of the most informative terms.
    top_terms = weighted_terms[:80]
    frequencies = {t: w for t, w in top_terms}

    wc = WordCloud(width=1000, height=400, background_color="white").generate_from_frequencies(frequencies)

    img = wc.to_image()
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    st.markdown(
        f"""
        <div style="
            background: rgba(255, 242, 226, 0.75);
            border: 1px solid rgba(139,161,148,0.22);
            border-radius: 18px;
            padding: 14px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
            text-align: center;
        ">
            <img
                src="data:image/png;base64,{b64}"
                style="max-width: 100%; height: auto; border-radius: 14px;"
                alt="TF-IDF weighted word cloud"
            />
        </div>
        """,
        unsafe_allow_html=True,
    )
