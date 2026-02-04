from __future__ import annotations

import streamlit as st


def render_sidebar() -> dict:
    """Render simplified sidebar with hardcoded optimal settings.

    Returns a config dict used across the frontend/backend.
    Keys are kept stable to avoid breaking existing components.
    """

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        keep_numbers = st.checkbox(
            "Keep numbers in analysis",
            value=True,
            help="Preserve numerical values in TF-IDF processing",
        )

        st.markdown("---")

        with st.expander("🔧 Advanced Settings", expanded=False):
            st.markdown(
                """
                **Fixed System Defaults (v2):**
                - 🎯 Divergence Threshold: `70%`
                - 📝 Lemmatization (NLTK): `Always ON`
                - 🔍 OCR Fallback for PDFs: `Always ON`
                - 📊 TF-IDF Max Features: `10,000`

                These defaults are optimized for production-style compliance analysis.
                """
            )

        with st.expander("📈 Risk Classification Guide", expanded=False):
            st.markdown(
                """
                **Divergence Risk Levels:**
                - ✅ **< 50%**: Compliant
                - 🟢 **50–60%**: Low Risk
                - 🟡 **60–70%**: Medium Risk
                - 🟠 **70–80%**: High Risk
                - 🔴 **80%+**: Critical Risk
                """
            )

    # Return configuration dictionary with hardcoded optimal values.
    # Keep legacy keys for compatibility with existing components.
    cfg: dict = {
        "divergence_threshold": 70.0,
        "keep_numbers": bool(keep_numbers),
        "use_lemma": True,
        "enable_ocr": True,
        "max_features": 10000,
        # Aliases for future use (do not break consumers expecting these names)
        "use_lemmatization": True,
        "ocr_enabled": True,
    }
    return cfg
