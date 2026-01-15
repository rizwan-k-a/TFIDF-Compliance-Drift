import sys
import os
import textwrap
from datetime import datetime

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# -------------------------------------------------
# IMPORTS
# -------------------------------------------------
import streamlit as st
import pandas as pd

from vectorize import load_documents, build_tfidf_vectors
from similarity import compute_cosine_similarity
from drift import compute_drift
from alerts import generate_alerts
from src.manual_tfidf_math import get_manual_tfidf_output

from sklearn.cluster import KMeans
from fpdf import FPDF

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Compliance Similarity Review",
    page_icon="📄",
    layout="wide"
)

# -------------------------------------------------
# BASIC STYLES
# -------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; max-width: 1400px; }
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px;
}
.kpi-title { font-size: 14px; color: #374151; }
.kpi-value { font-size: 30px; font-weight: 700; }
.help { font-size: 15px; color: #374151; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR — ROLE
# -------------------------------------------------
st.sidebar.title("User Role")

role = st.sidebar.radio(
    "Select role",
    ["Compliance Tester", "System Admin"]
)

st.sidebar.caption(
    "Compliance Tester: Reviews wording similarity\n"
    "System Admin: Adjusts thresholds and exports reports"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Compliance Similarity Review")

st.markdown(
    "<div class='help'>"
    "<b>What this screen shows:</b><br>"
    "Internal wording is compared with reference regulations to measure similarity.<br><br>"
    "<b>How to read the percentage:</b><br>"
    "Higher percentage = wording closer to regulation.<br>"
    "Lower percentage = wording more different from regulation."
    "</div>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
reference_path = os.path.join(BASE_DIR, "data", "reference")
internal_path = os.path.join(BASE_DIR, "data", "internal")
metadata_path = os.path.join(BASE_DIR, "data", "metadata.csv")

ref_docs, ref_names = load_documents(reference_path)
int_docs, int_names = load_documents(internal_path)

if not ref_docs or not int_docs:
    st.error("Reference or internal document folders are empty.")
    st.stop()

# -------------------------------------------------
# CORE PROCESSING
# -------------------------------------------------
_, ref_vectors, int_vectors = build_tfidf_vectors(ref_docs, int_docs)

similarity_df = compute_cosine_similarity(
    ref_vectors,
    int_vectors,
    int_names
)

drift_df = compute_drift(similarity_df, metadata_path)

# -------------------------------------------------
# THRESHOLD CONTROL
# -------------------------------------------------
if role == "System Admin":
    threshold = st.slider(
        "Similarity drop that triggers a review",
        -0.30, 0.0, -0.05, 0.01
    )
else:
    threshold = -0.05
    st.info("Standard review threshold is being used.")

alerts_df = generate_alerts(drift_df, threshold)

# -------------------------------------------------
# INTERPRETATION
# -------------------------------------------------
alerts_df["similarity_percent"] = (alerts_df["compliance_score"] * 100).round(1)

def priority_label(p):
    if p >= 25:
        return "Low Review Priority"
    elif p >= 10:
        return "Medium Review Priority"
    else:
        return "High Review Priority"

def meaning_text(p):
    if p >= 25:
        return "Very similar wording. No action needed."
    elif p >= 10:
        return "Some wording differences. Review later."
    else:
        return "Very different wording. Review urgently."

alerts_df["priority"] = alerts_df["similarity_percent"].apply(priority_label)
alerts_df["meaning"] = alerts_df["similarity_percent"].apply(meaning_text)

# -------------------------------------------------
# PRIORITY GROUPS
# -------------------------------------------------
high = alerts_df[alerts_df["priority"] == "High Review Priority"]
medium = alerts_df[alerts_df["priority"] == "Medium Review Priority"]
low = alerts_df[alerts_df["priority"] == "Low Review Priority"]

# -------------------------------------------------
# HUMAN-READABLE DECISION EXPLANATION (KEY ADDITION)
# -------------------------------------------------
if len(high) > 0:
    worst = high.sort_values("similarity_percent").iloc[0]
    others = alerts_df[alerts_df["filename"] != worst["filename"]]

    avg_other_sim = round(others["similarity_percent"].mean(), 1)
    worst_not_similar = round(100 - worst["similarity_percent"], 1)
    others_not_similar = round(100 - avg_other_sim, 1)

    st.error(
        f"🚨 **THIS is the main problem**\n\n"
        f"**{worst['filename']}** is **{worst_not_similar}% NOT similar** to the regulation.\n\n"
        f"Other documents are only about **{others_not_similar}% NOT similar**.\n\n"
        f"👉 This file is much more different than the others, "
        f"which is why it must be reviewed first."
    )

elif len(medium) > 0:
    st.warning(
        f"⚠️ **Some differences found**\n\n"
        f"{len(medium)} document(s) have moderate wording differences.\n"
        f"They are not the main problem, but should be reviewed later."
    )
else:
    st.success(
        "✅ **No immediate issue**\n\n"
        "All documents are very close to the regulation.\n"
        "No action is required right now."
    )

st.divider()

# -------------------------------------------------
# OVERALL SUMMARY
# -------------------------------------------------
st.header("Overall Review Summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"<div class='card'><div class='kpi-title'>High Priority</div>"
        f"<div class='kpi-value'>{len(high)}</div></div>",
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"<div class='card'><div class='kpi-title'>Medium Priority</div>"
        f"<div class='kpi-value'>{len(medium)}</div></div>",
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"<div class='card'><div class='kpi-title'>Low Priority</div>"
        f"<div class='kpi-value'>{len(low)}</div></div>",
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"<div class='card'><div class='kpi-title'>Average Similarity</div>"
        f"<div class='kpi-value'>{alerts_df['similarity_percent'].mean():.1f}%</div></div>",
        unsafe_allow_html=True
    )

# -------------------------------------------------
# ADMIN: TF-IDF CLUSTERING DEMO
# -------------------------------------------------
if role == "System Admin":
    st.divider()
    st.header("🔬 TF-IDF Matrix Based Clustering (Demonstration)")

    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(int_vectors)

    st.dataframe(
        pd.DataFrame({
            "Document": int_names,
            "Cluster": clusters
        }),
        width="stretch"
    )

    st.caption(
        "Clustering is shown only as a Data Science demonstration. "
        "It does not affect compliance decisions."
    )

# -------------------------------------------------
# ADMIN: MANUAL TF-IDF MATH
# -------------------------------------------------
if role == "System Admin":
    st.divider()
    st.header("📐 Mathematical Demonstration (Manual TF-IDF)")

    with st.expander("View Manual TF-IDF Computation"):
        math_output = get_manual_tfidf_output()

        st.subheader("Sample Documents")
        st.json(math_output["documents"])

        st.subheader("Term Frequency (TF)")
        st.json(math_output["TF"])

        st.subheader("Inverse Document Frequency (IDF)")
        st.json(math_output["IDF"])

        st.subheader("TF-IDF Weights")
        st.json(math_output["TF-IDF"])

        st.info(
            "This manual calculation is included only to demonstrate "
            "the mathematical foundation of TF-IDF."
        )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption("Compliance Similarity Review — Decision Support Tool")
