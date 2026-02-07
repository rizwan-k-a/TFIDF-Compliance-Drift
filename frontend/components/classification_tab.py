from __future__ import annotations

import streamlit as st
from collections import Counter

from backend.classification import perform_classification
from backend.document_categorization import categorize_document


def render_classification_tab(
    cfg: dict,
    internal_docs: list[dict],
    *,
    shared_vectorizer=None,
    shared_internal_matrix=None,
) -> None:
    st.subheader("Document Classification")

    if len(internal_docs) < 1:
        st.info("Upload documents to begin")
        return

    docs = [d.get("text", "") for d in internal_docs]
    cats = [categorize_document(d.get("text", ""), d.get("name", "")) for d in internal_docs]

    if len(docs) != len(cats):
        st.error("Document/label mismatch. Please re-upload your files.")
        return

    class_counts = Counter(cats)
    st.write("Debug info")
    st.write({
        "docs": len(docs),
        "class_counts": dict(class_counts),
    })

    res = perform_classification(
        documents=docs,
        categories=cats,
        keep_numbers=bool(cfg.get("keep_numbers", True)),
        use_lemma=bool(cfg.get("use_lemma", False)),
        max_features=int(cfg.get("max_features", 5000)),
        use_cv=st.checkbox("Use 5-fold CV", value=False),
        precomputed_vectorizer=shared_vectorizer,
        precomputed_matrix=shared_internal_matrix,
    )

    if not res:
        st.warning("Classification could not be computed.")
        return

    if isinstance(res, dict) and res.get("error"):
        st.error(res.get("error"))
        return

    for msg in res.get("warnings", []):
        st.warning(msg)

    debug = res.get("debug", {})
    if debug:
        st.write({
            "train_size": debug.get("train_size"),
            "test_size": debug.get("test_size"),
        })

    c1, c2 = st.columns(2)
    c1.metric("Naive Bayes accuracy", f"{res['nb_accuracy']*100:.1f}%")
    c2.metric("LogReg accuracy", f"{res['lr_accuracy']*100:.1f}%")

    st.write("Category distribution")
    st.json(res.get("category_distribution", {}))

    st.write("LogReg top features per class")
    st.json(res.get("top_features", {}))

    st.write("Classification report (LogReg)")
    st.code(res.get("classification_report_lr", ""))
