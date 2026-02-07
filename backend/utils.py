"""Backend utility functions.

This module contains non-ML helpers that are shared across the UI and backend.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Optional, Sequence, Tuple

from .config import CONFIG


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileValidationResult:
    ok: bool
    reason: Optional[str] = None
    size_mb: Optional[float] = None


def validate_input_file(
    filename: str,
    file_bytes: bytes,
    max_size_mb: int = CONFIG.max_file_size_mb,
    allowed_extensions: Sequence[str] = (".pdf", ".txt"),
) -> FileValidationResult:
    """Validate an uploaded file for security and robustness.

    Security & validation checks:
      1. Filename sanity: no path traversal, null bytes, suspicious patterns
      2. Extension allowlist enforcement
      3. Per-type size limits (PDF stricter than TXT)
      4. Magic bytes validation (file type verification)
      5. PDF structure validation (not just header check)
      6. UTF-8 validation for text files

    Args:
        filename: Original filename.
        file_bytes: Full file bytes.
        max_size_mb: Maximum size in MB (default: CONFIG.max_file_size_mb)
        allowed_extensions: Allowed file extensions (default: .pdf, .txt)

    Returns:
        FileValidationResult: ok=True if valid, ok=False with reason if invalid.

    Security notes:
      - Path traversal attempts (../, .., //, etc.) are rejected
      - Null bytes in filename indicate possible encoding attacks
      - Malformed PDFs are caught before extraction
      - Empty or suspicious files are rejected
    """

    # ────────────────────────────────────────────────────────────────────────
    # 1. FILENAME VALIDATION
    # ────────────────────────────────────────────────────────────────────────

    name = (filename or "").strip()
    if not name:
        return FileValidationResult(False, reason="Missing filename")

    # Reject path traversal attempts
    if ".." in name or name.startswith("/") or name.startswith("\\"):
        return FileValidationResult(False, reason="Invalid filename: path traversal detected")

    # Reject null bytes (null byte injection attack)
    if "\x00" in name:
        return FileValidationResult(False, reason="Invalid filename: null byte detected")

    # ────────────────────────────────────────────────────────────────────────
    # 2. EXTENSION VALIDATION
    # ────────────────────────────────────────────────────────────────────────

    suffix = Path(name).suffix.lower()
    if suffix not in {e.lower() for e in allowed_extensions}:
        return FileValidationResult(False, reason=f"Unsupported file type: {suffix}")

    # Reject double extensions (e.g., document.pdf.txt used to hide content)
    lowered = name.lower()
    allowed_lower = [e.lower() for e in allowed_extensions]
    for ext in allowed_lower:
        if lowered.endswith(ext):
            continue
        if f"{ext}." in lowered:
            return FileValidationResult(False, reason="Suspicious filename (double extension)")

    # ────────────────────────────────────────────────────────────────────────
    # 3. SIZE VALIDATION (per-type limits)
    # ────────────────────────────────────────────────────────────────────────

    size_mb = len(file_bytes) / (1024 * 1024)

    # Enforce global maximum
    if size_mb > max_size_mb:
        logger.warning("File rejected: exceeds max size: %s (%.2f MB > %d MB)", name, size_mb, max_size_mb)
        return FileValidationResult(False, reason=f"File too large: {size_mb:.2f} MB > {max_size_mb} MB limit", size_mb=size_mb)

    # Enforce per-type stricter limits
    if suffix == ".pdf":
        pdf_max_mb = 50  # PDFs are bulky; stricter limit
        if size_mb > pdf_max_mb:
            logger.warning("PDF rejected: exceeds per-type max: %s (%.2f MB > %d MB)", name, size_mb, pdf_max_mb)
            return FileValidationResult(False, reason=f"PDF too large: {size_mb:.2f} MB (max {pdf_max_mb} MB)", size_mb=size_mb)

    if suffix == ".txt":
        txt_max_mb = 100  # Text files can be larger
        if size_mb > txt_max_mb:
            logger.warning("Text file rejected: exceeds per-type max: %s (%.2f MB > %d MB)", name, size_mb, txt_max_mb)
            return FileValidationResult(False, reason=f"Text file too large: {size_mb:.2f} MB (max {txt_max_mb} MB)", size_mb=size_mb)

    # ────────────────────────────────────────────────────────────────────────
    # 4. MAGIC BYTES VALIDATION (file type verification)
    # ────────────────────────────────────────────────────────────────────────

    if suffix == ".pdf":
        # PDF files should start with %PDF- (magic bytes)
        if not file_bytes.startswith(b"%PDF-"):
            logger.warning("PDF rejected: invalid magic bytes: %s", name)
            return FileValidationResult(False, reason="Invalid PDF header (magic bytes mismatch)", size_mb=size_mb)

    if suffix == ".txt":
        # Try to decode as UTF-8; reject binary files disguised as .txt
        try:
            file_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Text file rejected: not UTF-8: %s", name)
            return FileValidationResult(False, reason="Text file is not UTF-8 decodable", size_mb=size_mb)

    # ────────────────────────────────────────────────────────────────────────
    # 5. PDF STRUCTURE VALIDATION (open & check page count)
    # ────────────────────────────────────────────────────────────────────────

    if suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            # pdfplumber not available; skip structural validation
            logger.debug("pdfplumber not available; skipping PDF structure validation")
            return FileValidationResult(True, size_mb=size_mb)

        tmp_path = None
        try:
            # Write to temp file (pdfplumber needs a file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # Try to open and validate structure
            with pdfplumber.open(tmp_path) as pdf:
                page_count = len(pdf.pages)

                if page_count == 0:
                    logger.warning("PDF rejected: empty (0 pages): %s", name)
                    return FileValidationResult(False, reason="PDF is empty (0 pages)", size_mb=size_mb)

                if page_count > 500:
                    logger.warning("PDF rejected: too many pages: %s (%d pages > 500 max)", name, page_count)
                    return FileValidationResult(
                        False,
                        reason=f"PDF has too many pages: {page_count} (max 500)",
                        size_mb=size_mb,
                    )

        except Exception as e:
            # PDF is corrupted or malformed
            error_msg = str(e)[:100]  # Limit error message length
            logger.warning("PDF rejected: corrupted/malformed: %s (%s)", name, error_msg)
            return FileValidationResult(
                False,
                reason=f"Corrupted or malformed PDF: {error_msg}",
                size_mb=size_mb,
            )
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ────────────────────────────────────────────────────────────────────────
    # 6. ALL CHECKS PASSED
    # ────────────────────────────────────────────────────────────────────────

    logger.info("File validation passed: %s (%.2f MB, %s)", name, size_mb, suffix)
    return FileValidationResult(True, size_mb=size_mb)
