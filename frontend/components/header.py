from __future__ import annotations

import streamlit as st


def render_header() -> None:
    """Render compact header with title and subtitle only."""
    st.markdown(
        """
        <div class="header-compact">
            <h1 class="header-compact__title"><span class="emoji">📊</span> Compliance Drift Monitoring</h1>
            <p class="header-compact__subtitle">TF-IDF Analysis for Regulatory Document Compliance</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
