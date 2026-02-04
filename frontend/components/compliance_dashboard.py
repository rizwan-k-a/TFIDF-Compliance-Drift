from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from backend.document_categorization import categorize_document
from backend.report_generator import generate_pdf
from backend.similarity import compute_similarity_scores_by_category
from backend.utils import risk_color, risk_label


def render_compliance_dashboard(cfg: dict, internal_docs: list[dict], guideline_docs: list[dict]) -> None:
    st.subheader("Compliance Dashboard")

    if not internal_docs or not guideline_docs:
        st.info("Upload at least 1 internal document and 1 guideline document.")
        return

    categorized_docs: Dict = {}
    categorized_guidelines: Dict = {}

    for d in internal_docs:
        cat = categorize_document(d.get("text", ""), d.get("name", ""))
        categorized_docs.setdefault(cat, {"docs": [], "names": []})
        categorized_docs[cat]["docs"].append(d.get("text", ""))
        categorized_docs[cat]["names"].append(d.get("name", ""))

    for g in guideline_docs:
        cat = categorize_document(g.get("text", ""), g.get("name", ""))
        categorized_guidelines.setdefault(cat, {"docs": [], "names": []})
        categorized_guidelines[cat]["docs"].append(g.get("text", ""))
        categorized_guidelines[cat]["names"].append(g.get("name", ""))

    df = compute_similarity_scores_by_category(
        categorized_docs,
        categorized_guidelines,
        keep_numbers=bool(cfg.get("keep_numbers", True)),
        use_lemma=bool(cfg.get("use_lemma", False)),
        max_features=int(cfg.get("max_features", 5000)),
    )

    if df.empty:
        st.warning("No comparable category pairs found.")
        return

    divergence_threshold = float(cfg.get("divergence_threshold", 40))
    threshold_attention = divergence_threshold / 2.0

    def enrich_row(r):
        divergence = float(r.get("divergence_percent", 0.0))
        risk = risk_label(divergence, threshold_attention=threshold_attention, threshold_review=divergence_threshold)
        return pd.Series({"Risk Level": risk})

    df_out = df.copy()
    df_out["Risk Level"] = df_out.apply(enrich_row, axis=1)

    st.dataframe(
        df_out.sort_values(["category", "divergence_percent"], ascending=[True, False]),
        use_container_width=True,
    )

    high_risk = df_out[df_out["divergence_percent"] >= divergence_threshold]
    st.markdown(
        f"**Alerts:** {len(high_risk)} document(s) at or above {divergence_threshold:.0f}% divergence."
    )

    # PDF export
    export_df = df_out.rename(
        columns={
            "category": "Category",
            "internal_document": "Document",
            "matched_guideline": "Guideline",
            "similarity_percent": "Similarity (%)",
            "divergence_percent": "Divergence (%)",
        }
    )

    pdf_buf = generate_pdf(export_df)
    st.download_button(
        "Download PDF report",
        data=pdf_buf,
        file_name="compliance_audit_report.pdf",
        mime="application/pdf",
    )
