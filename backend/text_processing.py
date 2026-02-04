"""Text processing: preprocessing, PDF extraction, and validation.

Design goals:
- Backend-only: no Streamlit caching APIs.
- Safe defaults: robust to empty/invalid inputs.
- Performance: uses functools.lru_cache as requested.

Public API:
  - preprocess_text
  - extract_text_from_pdf
  - validate_text
"""

from __future__ import annotations

import functools
import logging
import os
import re
import tempfile
from io import BytesIO
from typing import Optional, Tuple

from .config import CONFIG, POPPLER_PATH


logger = logging.getLogger(__name__)


def _ensure_str(text: object) -> str:
    if not text or not isinstance(text, str):
        return ""
    return text


@functools.lru_cache(maxsize=512)
def preprocess_text(
    text: str,
    keep_numbers: bool = True,
    use_lemmatization: bool = False,
    use_lemma: Optional[bool] = None,
) -> str:
    """Preprocess text using regex cleaning and optional lemmatization.

    Args:
        text: Raw input text.
        keep_numbers: Whether to retain numeric characters.
        use_lemmatization: Whether to lemmatize tokens (requires NLTK).

    Returns:
        Preprocessed text.
    """

    text = _ensure_str(text)
    if not text:
        return ""

    cleaned = text.lower()

    if keep_numbers:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
    else:
        cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if use_lemma is not None:
        use_lemmatization = bool(use_lemma)

    if use_lemmatization:
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer

            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)

            lemmatizer = WordNetLemmatizer()
            tokens = cleaned.split()
            cleaned = " ".join(lemmatizer.lemmatize(tok) for tok in tokens)
        except Exception as e:
            # Lemmatization is optional; fall back silently.
            logger.info("Lemmatization unavailable: %s", e)

    return cleaned


@functools.lru_cache(maxsize=32)
def extract_text_from_pdf(file_bytes: bytes, filename: str = "document.pdf", use_ocr: bool = True) -> Tuple[str, bool, int]:
    """Extract text from a PDF with optional OCR fallback.

    Args:
        file_bytes: Raw PDF bytes.
        filename: For diagnostics only.
        use_ocr: Whether to attempt OCR if text extraction yields too little content.

    Returns:
        (text, ocr_used, page_count)

    Raises:
        RuntimeError: If pdfplumber is unavailable.
    """

    try:
        import pdfplumber
    except Exception as e:
        raise RuntimeError("pdfplumber is required for PDF extraction") from e

    text_chunks = []
    page_count = 0
    ocr_used = False

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with pdfplumber.open(tmp_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text and len(page_text.strip()) >= CONFIG.min_page_text_length:
                    text_chunks.append(page_text)

        full_text = "\n".join(text_chunks).strip()

        if len(full_text) < CONFIG.min_text_length and use_ocr:
            try:
                from pdf2image import convert_from_path
                import pytesseract

                images = convert_from_path(
                    tmp_path,
                    dpi=CONFIG.ocr_dpi,
                    poppler_path=POPPLER_PATH,
                )
                ocr_text = []
                for image in images:
                    t = pytesseract.image_to_string(image, config=CONFIG.ocr_config)
                    if t and t.strip():
                        ocr_text.append(t)
                if ocr_text:
                    full_text = "\n".join(ocr_text).strip()
                    ocr_used = True
            except Exception as e:
                logger.info("OCR fallback failed for %s: %s", filename, e)

        return full_text, ocr_used, page_count
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def validate_text(text: str, doc_name: str = "document") -> Tuple[bool, str]:
    """Validate that a document contains sufficient text.

    Args:
        text: Extracted/preprocessed text.
        doc_name: Name used in messages.

    Returns:
        (ok, message)
    """

    if not text or len(text.strip()) < CONFIG.min_text_length:
        return False, f"{doc_name}: insufficient text"

    words = preprocess_text(text, keep_numbers=True, use_lemmatization=False).split()
    if len(words) < CONFIG.min_words:
        return False, f"{doc_name}: too few words"

    return True, "Valid"
