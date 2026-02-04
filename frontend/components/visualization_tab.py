from __future__ import annotations

import base64
from io import BytesIO

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

    img = wc.to_image()
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    st.markdown(
        f"""<img src="data:image/png;base64,{b64}" style="width: 100%; height: auto;"/>""",
        unsafe_allow_html=True,
    )
