from __future__ import annotations

import streamlit as st


def render_header() -> None:
    """Render premium gradient header with subtitle."""

    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 3rem 0;">
            <h1 style="margin-bottom: 0.5rem;">📊 Compliance Drift Monitoring</h1>
            <p style="font-size: 1.1rem; color: #666; font-weight: 500; margin: 0;">
                Intelligent TF-IDF Analysis for Regulatory Document Compliance
            </p>
            <p style="font-size: 0.9rem; color: #999; margin-top: 0.6rem;">
                Upload internal documents and compare against regulatory guidelines by category
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
