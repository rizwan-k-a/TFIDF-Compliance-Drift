from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from backend.document_categorization import categorize_document
from backend.report_generator import generate_pdf
from backend.similarity import (
    compute_similarity_scores_by_category,
    compute_similarity_scores_by_category_from_vectors,
)


def get_risk_level(divergence: float) -> tuple[str, str, str]:
    """Calculate risk level based on divergence percentage.

    Returns:
        (risk_label, css_class, icon)
    """

    d = float(divergence)
    if d < 50:
        return ("Compliant", "risk-compliant", "✅")
    if d < 60:
        return ("Low Risk", "risk-low", "🟢")
    if d < 70:
        return ("Medium Risk", "risk-medium", "🟡")
    if d < 80:
        return ("High Risk", "risk-high", "🟠")
    return ("Critical Risk", "risk-critical", "🔴")


def render_compliance_dashboard(
    cfg: dict,
    internal_docs: list[dict],
    guideline_docs: list[dict],
    *,
    shared_ref_vectors=None,
    shared_int_vectors=None,
) -> None:
    st.subheader("Compliance Dashboard")

    st.caption(
        "Divergence threshold is fixed at 70% for alerts. Risk levels are based on divergence bands (<50, 50–60, 60–70, 70–80, 80+)."
    )

    if not internal_docs or not guideline_docs:
        st.info("Upload at least 1 internal document and 1 guideline document.")
        return

    categorized_docs: Dict = {}
    categorized_guidelines: Dict = {}

    internal_names_by_cat: Dict[str, list[str]] = {}
    internal_idx_by_cat: Dict[str, list[int]] = {}
    guideline_names_by_cat: Dict[str, list[str]] = {}
    guideline_idx_by_cat: Dict[str, list[int]] = {}

    for i, d in enumerate(internal_docs):
        cat = categorize_document(d.get("text", ""), d.get("name", ""))
        categorized_docs.setdefault(cat, {"docs": [], "names": []})
        categorized_docs[cat]["docs"].append(d.get("text", ""))
        categorized_docs[cat]["names"].append(d.get("name", ""))

        internal_names_by_cat.setdefault(cat, []).append(d.get("name", ""))
        internal_idx_by_cat.setdefault(cat, []).append(i)

    for j, g in enumerate(guideline_docs):
        cat = categorize_document(g.get("text", ""), g.get("name", ""))
        categorized_guidelines.setdefault(cat, {"docs": [], "names": []})
        categorized_guidelines[cat]["docs"].append(g.get("text", ""))
        categorized_guidelines[cat]["names"].append(g.get("name", ""))

        guideline_names_by_cat.setdefault(cat, []).append(g.get("name", ""))
        guideline_idx_by_cat.setdefault(cat, []).append(j)

    if shared_ref_vectors is not None and shared_int_vectors is not None:
        df = compute_similarity_scores_by_category_from_vectors(
            internal_names_by_category=internal_names_by_cat,
            internal_indices_by_category=internal_idx_by_cat,
            guideline_names_by_category=guideline_names_by_cat,
            guideline_indices_by_category=guideline_idx_by_cat,
            ref_vectors=shared_ref_vectors,
            int_vectors=shared_int_vectors,
        )
    else:
        df = compute_similarity_scores_by_category(
            categorized_docs,
            categorized_guidelines,
            keep_numbers=bool(cfg.get("keep_numbers", True)),
            use_lemma=bool(cfg.get("use_lemma", False)),
            max_features=int(cfg.get("max_features", 5000)),
        )

    # Handle error dicts from similarity functions
    if isinstance(df, dict):
        if df.get("error"):
            st.error(df.get("error"))
            return
        # If no error key but is dict, something unexpected happened
        st.warning("Unexpected response from similarity computation.")
        return

    if df.empty:
        st.warning("No comparable category pairs found.")
        return

    divergence_threshold = float(cfg.get("divergence_threshold", 70.0))

    def enrich_row(r: pd.Series) -> pd.Series:
        divergence = float(r.get("divergence_percent", 0.0))
        risk_level, badge_class, icon = get_risk_level(divergence)
        return pd.Series(
            {
                "Risk Level": risk_level,
                "Risk Class": badge_class,
                "Risk Icon": icon,
            }
        )

    df_out = df.copy()
    df_out = pd.concat([df_out, df_out.apply(enrich_row, axis=1)], axis=1)

    # Summary pills
    max_div = float(df_out["divergence_percent"].max()) if not df_out.empty else 0.0
    top_risk, top_class, top_icon = get_risk_level(max_div)
    st.markdown(
        f"""
        <div style="text-align: center; margin: 1.0rem 0 1.5rem 0;">
            <span class="risk-badge {top_class}">{top_icon} Overall: {top_risk} · {max_div:.1f}% max divergence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk matrix as cards grid
    st.markdown("### Risk Matrix")
    cards = df_out.sort_values(["divergence_percent"], ascending=False).to_dict("records")
    cols_per_row = 3
    for i in range(0, len(cards), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx >= len(cards):
                break
            row = cards[idx]
            div = float(row.get("divergence_percent", 0.0))
            sim = float(row.get("similarity_percent", 0.0))
            risk_level, badge_class, icon = get_risk_level(div)
            category = str(row.get("category", ""))
            internal_name = str(row.get("internal_document", ""))
            guideline_name = str(row.get("matched_guideline", ""))

            with cols[j]:
                st.markdown(
                    f"""
                    <div style="background: rgba(255, 242, 226, 0.70); border: 1px solid rgba(139,161,148,0.20); border-radius: 16px; padding: 14px 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06);">
                        <div style="display:flex; justify-content: space-between; align-items: center; gap: 10px;">
                            <div style="font-weight: 700; color: #2d2d2d;">{category}</div>
                            <span class="risk-badge {badge_class}" style="padding: 6px 14px; font-size: 13px;">{icon} {risk_level}</span>
                        </div>
                        <div style="margin-top: 10px; color: #555; font-size: 13px; line-height: 1.3;">
                            <div title="Internal document">📄 <b>Internal</b>: {internal_name}</div>
                            <div title="Matched guideline">📘 <b>Guideline</b>: {guideline_name}</div>
                        </div>
                        <div style="display:flex; gap: 10px; margin-top: 12px;">
                            <div style="flex:1; background: #ffffff; border-radius: 12px; padding: 10px; border: 1px solid rgba(0,0,0,0.06);">
                                <div style="font-size: 12px; color: #777; font-weight: 600;">Similarity</div>
                                <div style="font-size: 20px; font-weight: 800; color: #4F633D;">{sim:.1f}%</div>
                            </div>
                            <div style="flex:1; background: #ffffff; border-radius: 12px; padding: 10px; border: 1px solid rgba(0,0,0,0.06);">
                                <div style="font-size: 12px; color: #777; font-weight: 600;">Divergence</div>
                                <div style="font-size: 20px; font-weight: 800; color: #2d2d2d;">{div:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("### Details")

    st.dataframe(
        df_out.drop(columns=["Risk Class", "Risk Icon"], errors="ignore").sort_values(
            ["category", "divergence_percent"], ascending=[True, False]
        ),
        width="stretch",
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
