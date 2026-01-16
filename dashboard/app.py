# ============================================================
# LEGAL COMPLIANCE SIMILARITY & DRIFT REVIEW DASHBOARD
# MCA Final Project – Decision Support + Academic Demonstration
# ============================================================

import sys
import os
from io import BytesIO

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns

from vectorize import build_tfidf_vectors
from similarity import compute_cosine_similarity
from drift import compute_drift
from alerts import generate_alerts
from manual_tfidf_math import get_manual_tfidf_output

from fpdf import FPDF
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Legal Compliance Similarity Review",
    page_icon="⚖️",
    layout="wide"
)

# ------------------------------------------------------------
# GLOBAL CONSTANTS
# ------------------------------------------------------------
DEFAULT_DIVERGENCE_THRESHOLD = 40  # professional baseline

# ------------------------------------------------------------
# SIDEBAR — ROLE SELECTION
# ------------------------------------------------------------
st.sidebar.title("User Role")

role = st.sidebar.radio(
    "Select role",
    ["Compliance Tester", "System Admin"]
)

st.sidebar.caption(
    "Compliance Tester: Legal review & prioritization\n\n"
    "System Admin: Data science & mathematical explanation"
)

# ------------------------------------------------------------
# ROLE-SPECIFIC HEADER
# ------------------------------------------------------------
if role == "Compliance Tester":
    st.markdown("""
    <h1>⚖️ Legal Compliance Review</h1>
    <p style="opacity:0.75; max-width:900px;">
    Decision-support dashboard to identify which legal or policy documents
    require review based on wording divergence from selected guidelines.
    </p>
    <p style="opacity:0.6;">
    <b>Disclaimer:</b> This system supports legal judgment.
    It does not certify compliance or replace legal advice.
    </p>
    <hr style="opacity:0.3;">
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <h1>📊 Data Science & Mathematical Analysis</h1>
    <p style="opacity:0.75; max-width:900px;">
    Academic view exposing the TF-IDF, similarity computation,
    divergence modeling, and explainability layers of the system.
    </p>
    <hr style="opacity:0.3;">
    """, unsafe_allow_html=True)

# ============================================================
# DOCUMENT LOADING UTILITIES
# ============================================================
def read_uploaded_files(files):
    texts, names = [], []
    for f in files:
        if f.name.lower().endswith(".txt"):
            text = f.read().decode("utf-8", errors="ignore")
        elif f.name.lower().endswith(".pdf"):
            text = ""
            with pdfplumber.open(f) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        else:
            continue
        texts.append(text)
        names.append(f.name)
    return texts, names


def load_default_docs(folder):
    texts, names = [], []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".txt"):
            with open(
                os.path.join(folder, fname),
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:
                texts.append(f.read())
                names.append(fname)
    return texts, names


# ============================================================
# PDF GENERATION
# ============================================================
def generate_audit_pdf(results_df, summary_text):
    buffer = BytesIO()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    pdf.multi_cell(0, 8, "Legal Compliance Similarity Audit Report\n")
    pdf.multi_cell(0, 8, summary_text)
    pdf.ln(6)

    for _, r in results_df.iterrows():
        pdf.multi_cell(
            0,
            8,
            f"""Document: {r['internal_document']}
Similarity: {r['similarity_percent']}%
Divergence: {r['divergence_percent']}%
Risk: {r['risk']}
"""
        )
        pdf.ln(2)

    pdf.output(buffer)
    buffer.seek(0)
    return buffer


# ============================================================
# COMPLIANCE TESTER VIEW
# ============================================================
if role == "Compliance Tester":

    st.header("📄 Upload Internal Documents")

    uploaded_internal = st.file_uploader(
        "Upload internal legal / policy documents (TXT or PDF)",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    st.header("⚖️ Select Reference Guidelines")
    st.caption("Select up to two guidelines (default or uploaded).")

    default_guidelines, default_names = load_default_docs(
        os.path.join(DATA_DIR, "reference")
    )

    selected_guidelines = {}
    cols = st.columns(3)

    for i, name in enumerate(default_names):
        with cols[i % 3]:
            if st.checkbox(name):
                selected_guidelines[name] = default_guidelines[i]

    st.subheader("Upload Custom Guidelines (Optional)")
    uploaded_guidelines = st.file_uploader(
        "Upload guideline files (TXT or PDF)",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_guidelines:
        g_texts, g_names = read_uploaded_files(uploaded_guidelines)
        for n, t in zip(g_names, g_texts):
            selected_guidelines[n] = t

    if len(selected_guidelines) > 2:
        st.error("Please select no more than two guidelines.")

    run = st.button("▶️ Run Legal Comparison")

    if run and uploaded_internal and selected_guidelines:

        with st.spinner("Analyzing documents..."):
            int_docs, int_names = read_uploaded_files(uploaded_internal)
            ref_docs = list(selected_guidelines.values())

            _, ref_vecs, int_vecs = build_tfidf_vectors(ref_docs, int_docs)
            sim_df = compute_cosine_similarity(ref_vecs, int_vecs, int_names)

            sim_df["similarity_percent"] = (sim_df["compliance_score"] * 100).round(1)
            sim_df["divergence_percent"] = (100 - sim_df["similarity_percent"]).round(1)

        def risk_label(div):
            if div <= 20:
                return "Safe – Closely Aligned"
            elif div <= DEFAULT_DIVERGENCE_THRESHOLD:
                return "Needs Attention"
            else:
                return "Review Required"

        sim_df["risk"] = sim_df["divergence_percent"].apply(risk_label)

        st.subheader("Detailed Results")
        st.dataframe(sim_df, use_container_width=True)


# ============================================================
# SYSTEM ADMIN VIEW (ACADEMIC)
# ============================================================
else:

    internal_docs, internal_names = load_default_docs(
        os.path.join(DATA_DIR, "internal")
    )
    guideline_docs, guideline_names = load_default_docs(
        os.path.join(DATA_DIR, "reference")
    )

    st.header("Select Documents for Analysis")

    sel_internal = st.multiselect(
        "Internal documents",
        internal_names,
        default=internal_names
    )

    sel_guidelines = st.multiselect(
        "Reference guidelines",
        guideline_names,
        default=guideline_names[:2]
    )

    st.header("⚖️ Legal Risk Threshold (Admin Only)")
    divergence_threshold = st.slider(
        "Maximum acceptable wording divergence (%)",
        20, 80, DEFAULT_DIVERGENCE_THRESHOLD, 5
    )

    run_admin = st.button("▶️ Run Admin Analysis")

    if run_admin:

        with st.spinner("Running TF-IDF and similarity analysis..."):

            int_docs = [internal_docs[internal_names.index(n)] for n in sel_internal]
            ref_docs = [guideline_docs[guideline_names.index(n)] for n in sel_guidelines]

            _, ref_vecs, int_vecs = build_tfidf_vectors(ref_docs, int_docs)
            sim_df = compute_cosine_similarity(ref_vecs, int_vecs, sel_internal)

            sim_df["similarity_percent"] = (sim_df["compliance_score"] * 100).round(1)
            sim_df["divergence_percent"] = (100 - sim_df["similarity_percent"]).round(1)

            # ----------------------------------------------------
            # ADD RISK LABELS (FIXED INDENTATION ONLY)
            # ----------------------------------------------------
            def risk_label(div):
                if div <= 20:
                    return "Safe – Closely Aligned"
                elif div <= DEFAULT_DIVERGENCE_THRESHOLD:
                    return "Needs Attention"
                else:
                    return "Review Required"

            sim_df["risk"] = sim_df["divergence_percent"].apply(risk_label)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Similarity Heatmap")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                cosine_similarity(int_vecs, ref_vecs) * 100,
                xticklabels=sel_guidelines,
                yticklabels=sel_internal,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                linewidths=0.5,
                cbar_kws={"shrink": 0.7},
                ax=ax
            )
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("Divergence per Document")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(sel_internal, sim_df["divergence_percent"])
            ax2.axhline(divergence_threshold, linestyle="--")
            ax2.set_ylabel("Divergence (%)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig2)

        st.subheader("Manual TF / IDF / TF-IDF Demonstration")

        math = get_manual_tfidf_output()
        st.markdown("**Term Frequency (TF)**")
        st.json(math["TF"])
        st.markdown("**Inverse Document Frequency (IDF)**")
        st.json(math["IDF"])
        st.markdown("**TF-IDF Weights**")
        st.json(math["TF-IDF"])

        st.info(
            "TF measures importance within a document. "
            "IDF reduces the impact of common legal terms. "
            "TF-IDF balances both to highlight discriminative wording."
        )

        st.subheader("Export Audit Summary (PDF)")

        summary_text = (
            f"Documents analyzed: {len(sel_internal)}\n"
            f"Guidelines used: {len(sel_guidelines)}\n"
            f"Divergence threshold: {divergence_threshold}%"
        )

        pdf_buffer = generate_audit_pdf(sim_df, summary_text)

        st.download_button(
            label="📄 Download Audit PDF",
            data=pdf_buffer,
            file_name="audit_summary.pdf",
            mime="application/pdf"
        )

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown(
    "<hr><p style='text-align:center; opacity:0.5;'>"
    "Legal Compliance Similarity Review — MCA Final Project"
    "</p>",
    unsafe_allow_html=True
)
