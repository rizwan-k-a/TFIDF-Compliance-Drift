from __future__ import annotations

import streamlit as st


def render_header() -> None:
    """Render compact header with title and subtitle only."""
    st.markdown(
        """
        <a href="#main" class="skip-link">Skip to main content</a>
        <div class="header-compact" style="margin-bottom: 0.2rem;">
            <h1 class="header-compact__title" role="banner" aria-label="Compliance Drift Monitoring">
                <span class="header-compact__icon" aria-hidden="true">
                    <!-- Inline SVG logo: simple stylized bar chart matching theme -->
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" focusable="false" role="img">
                      <rect x="2" y="12" width="3" height="10" rx="0.5" fill="#8BA194"/>
                      <rect x="7" y="8" width="3" height="14" rx="0.5" fill="#6b8a7e"/>
                      <rect x="12" y="4" width="3" height="18" rx="0.5" fill="#4F633D"/>
                      <rect x="17" y="10" width="3" height="12" rx="0.5" fill="#A3B89A"/>
                    </svg>
                </span>
                Compliance Drift Monitoring
            </h1>
            <p class="header-compact__subtitle">TF-IDF Analysis for Regulatory Document Compliance</p>
            <a id="main"></a>
        </div>
        """,
        unsafe_allow_html=True,
    )
