from __future__ import annotations

import streamlit as st


def render_visualization_tab(internal_docs: list[dict], guideline_docs: list[dict]) -> None:
    st.subheader("Visualization")

    try:
        from wordcloud import WordCloud
    except Exception:
        st.info("Install `wordcloud` to enable word cloud visualization.")
        return

    texts = "\n".join([d.get("text", "") for d in (guideline_docs + internal_docs)])
    if len(texts.strip()) < 50:
        st.info("Upload documents with more text.")
        return

    wc = WordCloud(width=1000, height=400, background_color="white").generate(texts)
    st.image(wc.to_array(), use_container_width=True)
