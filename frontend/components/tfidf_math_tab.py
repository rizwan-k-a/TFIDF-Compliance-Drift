from __future__ import annotations

import math
import re

import pandas as pd
import streamlit as st


def render_tfidf_math_tab(cfg: dict, internal_docs: list[dict], guideline_docs: list[dict]) -> None:
    st.subheader("TF-IDF Maths")

    st.caption(
        "Two sentences only (editable). Pick 3–4 words and see TF, IDF, TF‑IDF computed step‑by‑step."
    )

    s1 = st.text_area(
        "Sentence 1",
        value="AML monitoring is essential for detecting suspicious transactions.",
        height=80,
    )
    s2 = st.text_area(
        "Sentence 2",
        value="Cyber incident response requires logging and quick containment.",
        height=80,
    )

    def tokenize(text: str) -> list[str]:
        # Simple, explainable tokenization for viva (manual math; no library shortcut)
        return re.findall(r"[a-z0-9']+", (text or "").lower())

    doc1 = tokenize(s1)
    doc2 = tokenize(s2)
    docs = [doc1, doc2]
    n_docs = 2

    vocab = sorted(set(doc1) | set(doc2))
    if not vocab:
        st.info("Type some text to compute TF‑IDF.")
        return

    default_terms = vocab[:4]
    selected_terms = st.multiselect(
        "Pick 3–4 sample words",
        options=vocab,
        default=default_terms[:4],
    )

    if len(selected_terms) < 3 or len(selected_terms) > 4:
        st.warning("Please select exactly 3 or 4 words.")
        return

    def term_count(term: str, tokens: list[str]) -> int:
        return sum(1 for t in tokens if t == term)

    def tf(term: str, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        return term_count(term, tokens) / float(len(tokens))

    def df(term: str) -> int:
        return sum(1 for d in docs if term in d)

    def idf(term: str) -> float:
        # Classic IDF for demo: idf(t) = ln(N / df(t))
        dft = df(term)
        return 0.0 if dft == 0 else math.log(n_docs / float(dft))

    st.markdown("### Formulas")
    st.latex(r"tf(t,d)=\frac{f_{t,d}}{|d|}")
    st.latex(r"idf(t)=\ln\left(\frac{N}{df(t)}\right)")
    st.latex(r"tfidf(t,d)=tf(t,d)\cdot idf(t)")

    rows = []
    for term in selected_terms:
        c1 = term_count(term, doc1)
        c2 = term_count(term, doc2)
        tf1 = tf(term, doc1)
        tf2 = tf(term, doc2)
        dft = df(term)
        idft = idf(term)
        rows.append(
            {
                "term": term,
                "count(d1)": c1,
                "tf(d1)": round(tf1, 4),
                "count(d2)": c2,
                "tf(d2)": round(tf2, 4),
                "df": dft,
                "idf": round(idft, 4),
                "tfidf(d1)": round(tf1 * idft, 4),
                "tfidf(d2)": round(tf2 * idft, 4),
            }
        )

    st.markdown("### Results")
    st.dataframe(pd.DataFrame(rows), width="stretch")

    st.markdown("### Substituted values (step‑by‑step)")
    for term in selected_terms:
        c1 = term_count(term, doc1)
        c2 = term_count(term, doc2)
        dft = df(term)
        st.markdown(f"**Term:** {term}")
        st.latex(rf"N = {n_docs},\; df({term}) = {dft}")
        st.latex(rf"idf({term}) = \ln\left(\frac{{{n_docs}}}{{{dft}}}\right)")
        st.latex(rf"tf({term}, d_1) = \frac{{{c1}}}{{{len(doc1)}}}")
        st.latex(rf"tf({term}, d_2) = \frac{{{c2}}}{{{len(doc2)}}}")

    st.info(
        "Why TF‑IDF matters: words that appear in fewer documents have higher IDF, so they receive higher weight and better separate documents."
    )
