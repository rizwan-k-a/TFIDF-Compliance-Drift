"""
pdf_to_txt_once.py

Purpose:
- One-time extraction of statutory PDFs into clean .txt files
- Handles BOTH text PDFs and scanned Gazette PDFs (OCR fallback)
- Output is used for TF-IDF similarity analysis
- PDFs are NEVER processed during Streamlit runtime

IMPORTANT:
- Run this script ONCE
- Do NOT re-run unless PDFs change
"""

import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

# ============================================================
# CONFIGURATION
# ============================================================

PDF_ROOT = "data/guidelines_pdfs"
TXT_ROOT = "data/guidelines"

PDF_FILES = [
    "Criminal_Law/BNS_2023.pdf",
    "Cyber_Crime/IT_ACT_2021.pdf",
    "Financial_Law/PMLA_2002.pdf",
]

POPPLER_PATH = r"C:\Program Files\poppler-25.12.0\Library\bin"
MIN_LINE_LENGTH = 30

# ============================================================
# CLEANING
# ============================================================

def clean_legal_text(text: str) -> str:
    return "\n".join(
        line.strip()
        for line in text.splitlines()
        if len(line.strip()) >= MIN_LINE_LENGTH
    )

# ============================================================
# EXTRACTION (TEXT + OCR)
# ============================================================

def extract_pdf_text(pdf_path: str) -> str:
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text and len(text.strip()) > 100:
                pages.append(text)
            else:
                images = convert_from_path(
                    pdf_path,
                    first_page=i + 1,
                    last_page=i + 1,
                    poppler_path=POPPLER_PATH
                )

                ocr_text = pytesseract.image_to_string(
                    images[0],
                    lang="eng",
                    config="--psm 6"
                )
                pages.append(ocr_text)

    return "\n".join(pages)

# ============================================================
# PIPELINE
# ============================================================

def run_extraction():
    for rel_path in PDF_FILES:
        pdf_path = os.path.join(PDF_ROOT, rel_path)
        txt_path = os.path.join(TXT_ROOT, rel_path.replace(".pdf", ".txt"))

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Missing PDF: {pdf_path}")

        print(f"📄 Processing: {pdf_path}")

        raw_text = extract_pdf_text(pdf_path)
        cleaned_text = clean_legal_text(raw_text)

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"✅ Saved: {txt_path}")

    print("\n🎉 Extraction complete.")
    print("👉 All 3 statutes fully extracted (TEXT + OCR).")
    print("👉 Do NOT re-run unless PDFs change.")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_extraction()
