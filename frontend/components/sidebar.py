from __future__ import annotations

import streamlit as st

from backend.config import CONFIG


def render_sidebar() -> dict:
    """Render simplified sidebar with hardcoded optimal settings.

    Returns a config dict used across the frontend/backend.
    Keys are kept stable to avoid breaking existing components.
    """

    with st.sidebar:
        st.markdown('<div class="sidebar-compact">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Settings")

        keep_numbers = st.checkbox(
            "Keep numbers in analysis",
            value=True,
            help="Preserve numerical values in TF-IDF processing",
        )

        use_lemma = st.checkbox(
            "Use lemmatization",
            value=True,
            help="Normalize words to their base form (e.g., 'monitoring' → 'monitor').",
        )

        max_features = st.slider(
            "TF-IDF max features",
            min_value=500,
            max_value=20000,
            value=10000,
            step=500,
            help="Caps vocabulary size for the TF-IDF matrix.",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Return configuration dictionary from backend CONFIG.
    # Keep legacy keys for compatibility with existing components.
    cfg: dict = {
        "divergence_threshold": float(getattr(CONFIG, "divergence_threshold", 70.0)),
        "keep_numbers": bool(keep_numbers),
        "use_lemma": bool(use_lemma),
        "enable_ocr": bool(getattr(CONFIG, "enable_ocr", True)),
        "max_features": int(max_features),
        "sample_autoload_enabled": bool(CONFIG.sample_autoload_enabled),
        "sample_autoload_internal_limit": int(CONFIG.sample_autoload_internal_limit),
        "sample_autoload_guideline_limit": int(CONFIG.sample_autoload_guideline_limit),
        # Aliases for future use (do not break consumers expecting these names)
        "use_lemmatization": bool(use_lemma),
        "ocr_enabled": bool(getattr(CONFIG, "enable_ocr", True)),
        "min_text_length": int(getattr(CONFIG, "min_text_length", 50)),
        "ocr_dpi": int(getattr(CONFIG, "ocr_dpi", 300)),
    }
    return cfg
