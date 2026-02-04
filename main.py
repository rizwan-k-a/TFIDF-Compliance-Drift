from __future__ import annotations

import streamlit as st

from frontend.components.clustering_tab import render_clustering_tab
from frontend.components.compliance_dashboard import render_compliance_dashboard
from frontend.components.classification_tab import render_classification_tab
from frontend.components.file_upload import upload_documents
from frontend.components.header import render_header
from frontend.components.sidebar import render_sidebar
from frontend.components.tfidf_math_tab import render_tfidf_math_tab
from frontend.components.tfidf_matrix_tab import render_tfidf_matrix_tab
from frontend.components.visualization_tab import render_visualization_tab
from frontend.styles.custom_css import CUSTOM_CSS


def main() -> None:
    st.set_page_config(
        page_title="Compliance Drift Monitoring | TF-IDF Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "Premium Compliance Drift Monitoring Dashboard v2.0",
        },
    )

    # Apply enhanced CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    cfg = render_sidebar()
    render_header()

    docs = upload_documents(cfg)
    internal_docs = docs["internal"]
    guideline_docs = docs["guidelines"]

    # Enhanced tabs with custom labels
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Compliance Dashboard",
            "🧮 TF-IDF Mathematics",
            "🔍 Document Classification",
            "🗂️ Clustering Analysis",
            "📋 TF-IDF Matrix View",
        ]
    )

    with tab1:
        # Note: Streamlit does not support st.container().style(...).
        # Styling is provided globally via CUSTOM_CSS.
        render_compliance_dashboard(cfg, internal_docs, guideline_docs)

    with tab2:
        render_tfidf_math_tab(cfg, internal_docs, guideline_docs)

    with tab3:
        render_classification_tab(cfg, internal_docs)

    with tab4:
        render_clustering_tab(cfg, internal_docs)

    with tab5:
        render_tfidf_matrix_tab(cfg, internal_docs, guideline_docs)

    st.markdown("---")
    with st.expander("☁️ Word Cloud Visualization", expanded=False):
        render_visualization_tab(internal_docs, guideline_docs)


if __name__ == "__main__":
    main()
