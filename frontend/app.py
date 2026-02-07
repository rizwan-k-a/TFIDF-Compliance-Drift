from __future__ import annotations

import logging

import streamlit as st

from backend.tfidf_engine import build_tfidf_vectors
from utils.logging_setup import setup_logging
from frontend.components.compliance_dashboard import render_compliance_dashboard
from frontend.components.classification_tab import render_classification_tab
from frontend.components.clustering_tab import render_clustering_tab
from frontend.components.file_upload import upload_documents
from frontend.components.header import render_header
from frontend.components.sidebar import render_sidebar
from frontend.components.tfidf_math_tab import render_tfidf_math_tab
from frontend.components.tfidf_matrix_tab import render_tfidf_matrix_tab
from frontend.components.visualization_tab import render_visualization_tab
from frontend.styles.custom_css import CUSTOM_CSS


@st.cache_resource(show_spinner=False)
def _cached_build_vectors(
    reference_docs: tuple[str, ...],
    internal_docs: tuple[str, ...],
    keep_numbers: bool,
    use_lemma: bool,
    max_features: int,
):
    return build_tfidf_vectors(
        reference_docs=reference_docs,
        internal_docs=internal_docs,
        keep_numbers=keep_numbers,
        use_lemma=use_lemma,
        max_features=max_features,
    )
def main() -> None:
    # Initialize logging before any application logic
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    st.set_page_config(
        page_title="Compliance Drift Monitoring",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    cfg = render_sidebar()
    render_header()

    docs = upload_documents(cfg)
    internal_docs = docs["internal"]
    guideline_docs = docs["guidelines"]
    
    logger.info(
        "Documents loaded: %d internal, %d guidelines",
        len(internal_docs),
        len(guideline_docs),
    )

    shared_vectorizer = None
    shared_ref_vectors = None
    shared_int_vectors = None
    shared_all_vectors = None
    shared_names = None

    # One shared TF-IDF pipeline for all tabs:
    # preprocessing → tfidf_engine → TF-IDF matrix → similarity/classification/clustering
    if internal_docs or guideline_docs:
        guideline_texts = [d.get("text", "") for d in guideline_docs]
        internal_texts = [d.get("text", "") for d in internal_docs]
        try:
            shared_vectorizer, shared_ref_vectors, shared_int_vectors = _cached_build_vectors(
                reference_docs=tuple(guideline_texts),
                internal_docs=tuple(internal_texts),
                keep_numbers=bool(cfg.get("keep_numbers", True)),
                use_lemma=bool(cfg.get("use_lemma", False)),
                max_features=int(cfg.get("max_features", 5000)),
            )
            try:
                from scipy.sparse import vstack

                shared_all_vectors = vstack([shared_ref_vectors, shared_int_vectors])
            except (ImportError, ValueError) as e:
                logger.warning("Matrix concatenation failed: %s", e)
                shared_all_vectors = None

            shared_names = [
                *[d.get("name", "doc") for d in guideline_docs],
                *[d.get("name", "doc") for d in internal_docs],
            ]
        except ValueError as e:
            st.error(f"🚨 Vectorization failed: {e}")
            logger.error("TF-IDF vectorization error: %s", e)
            shared_vectorizer = None
            shared_ref_vectors = None
            shared_int_vectors = None
            shared_all_vectors = None
            shared_names = None
        except MemoryError:
            st.error("🚨 Out of memory during vectorization. Try fewer documents or reduce max_features.")
            logger.error("Memory exhausted during vectorization")
            shared_vectorizer = None
            shared_ref_vectors = None
            shared_int_vectors = None
            shared_all_vectors = None
            shared_names = None
        except Exception as e:
            st.error(f"🚨 Unexpected error during vectorization: {type(e).__name__}: {e}")
            logger.exception("Unexpected vectorization error")
            shared_vectorizer = None
            shared_ref_vectors = None
            shared_int_vectors = None
            shared_all_vectors = None
            shared_names = None

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Compliance Dashboard",
            "📋 TF-IDF Matrix View",
            "🧮 TF-IDF Mathematics",
            "🏷️ Classification",
            "🧩 Clustering",
            "☁️ Word Cloud",
        ]
    )

    with tab1:
        render_compliance_dashboard(
            cfg,
            internal_docs,
            guideline_docs,
            shared_ref_vectors=shared_ref_vectors,
            shared_int_vectors=shared_int_vectors,
        )

    with tab2:
        render_tfidf_matrix_tab(
            cfg,
            internal_docs,
            guideline_docs,
            shared_vectorizer=shared_vectorizer,
            shared_matrix=shared_all_vectors,
            shared_names=shared_names,
        )

    with tab3:
        render_tfidf_math_tab(cfg, internal_docs, guideline_docs)

    with tab4:
        render_classification_tab(
            cfg,
            internal_docs,
            shared_vectorizer=shared_vectorizer,
            shared_internal_matrix=shared_int_vectors,
        )

    with tab5:
        render_clustering_tab(
            cfg,
            internal_docs,
            shared_vectorizer=shared_vectorizer,
            shared_internal_matrix=shared_int_vectors,
        )

    with tab6:
        render_visualization_tab(
            internal_docs,
            guideline_docs,
            shared_vectorizer=shared_vectorizer,
            shared_matrix=shared_all_vectors,
        )


if __name__ == "__main__":
    main()


