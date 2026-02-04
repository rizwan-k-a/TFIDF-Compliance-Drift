from __future__ import annotations

import streamlit as st


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
          <h1>Compliance Drift Monitoring (TF-IDF)</h1>
          <p>Upload internal documents and compare them against regulatory guidelines by category.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Educational/demo tool. Results are heuristics and should be reviewed by qualified compliance/legal professionals."
    )
