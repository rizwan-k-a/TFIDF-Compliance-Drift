"""PDF report generation for compliance results."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from fpdf import FPDF


# Default footer text used on every page. Change as desired.
DEFAULT_FOOTER = "TF-IDF Compliance Drift — MCA DataScience Project — Rizwan K A"


class PDFReport(FPDF):
    """Custom FPDF with a professional footer on every page."""

    def __init__(self, footer_text: str = DEFAULT_FOOTER, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._footer_text = footer_text

    def footer(self):
        # Position at 15 mm from bottom
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        # Footer text on the left
        self.cell(0, 8, self._footer_text, align="L")
        # Page number on the right
        self.cell(0, 8, f"Page {self.page_no()}", align="R")


def generate_pdf(results_df: pd.DataFrame, footer_text: str = DEFAULT_FOOTER) -> BytesIO:
    """Generate a categorized compliance audit PDF.

    Args:
        results_df: DataFrame with columns similar to:
            Category, Document, Guideline, Similarity (%), Divergence (%), Risk Level

    Returns:
        BytesIO buffer containing the PDF.
    """

    buffer = BytesIO()
    pdf = PDFReport(footer_text=footer_text)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "COMPLIANCE AUDIT REPORT - CATEGORIZED", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 10)

    current_category = None
    for _, r in results_df.iterrows():
        category = str(r.get("Category", r.get("category", "Uncategorized")))
        if current_category != category:
            current_category = category
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, f"Category: {current_category}", ln=True)
            pdf.ln(2)

        doc = str(r.get("Document", r.get("internal_document", "")))
        guideline = str(r.get("Guideline", r.get("matched_guideline", "")))
        sim = str(r.get("Similarity (%)", r.get("similarity_percent", "")))
        div = str(r.get("Divergence (%)", r.get("divergence_percent", "")))
        risk_text = str(r.get("Risk Level", r.get("risk", "")))

        if "Safe" in risk_text:
            risk_plain = "[SAFE] Closely Aligned"
        elif "Attention" in risk_text:
            risk_plain = "[WARNING] Needs Attention"
        elif "Review" in risk_text or "Critical" in risk_text:
            risk_plain = "[CRITICAL] Review Required"
        else:
            risk_plain = risk_text.replace("✅", "").replace("⚠️", "").replace("🚨", "").strip()

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, f"Document: {doc}", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Guideline: {guideline}", ln=True)
        pdf.cell(0, 6, f"Similarity: {sim}%", ln=True)
        pdf.cell(0, 6, f"Divergence: {div}%", ln=True)
        pdf.cell(0, 6, f"Risk: {risk_plain}", ln=True)
        pdf.ln(4)

    pdf.output(buffer)
    buffer.seek(0)
    return buffer
