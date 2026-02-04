from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from backend.config import CONFIG


def render_sidebar() -> dict:
    st.sidebar.header("Settings")

    divergence_threshold = st.sidebar.slider(
        "Divergence threshold (%)",
        min_value=10,
        max_value=90,
        value=int(CONFIG.default_divergence_threshold),
        step=5,
    )

    keep_numbers = st.sidebar.checkbox("Keep numbers", value=True)
    use_lemma = st.sidebar.checkbox("Use lemmatization (NLTK)", value=False)

    max_features = st.sidebar.number_input(
        "TF-IDF max features",
        min_value=500,
        max_value=20000,
        value=int(CONFIG.tfidf_max_features),
        step=500,
    )

    enable_ocr = st.sidebar.checkbox("Enable OCR fallback for PDFs", value=True)

    cfg = asdict(CONFIG)
    cfg.update(
        {
            "divergence_threshold": divergence_threshold,
            "keep_numbers": keep_numbers,
            "use_lemma": use_lemma,
            "max_features": max_features,
            "enable_ocr": enable_ocr,
        }
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: If PDFs are scanned images, enable OCR.")

    return cfg
