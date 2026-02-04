"""Backend utility functions.

This module contains non-ML helpers that are shared across the UI and backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Optional, Sequence, Tuple

from .config import CONFIG, get_poppler_path


logger = logging.getLogger(__name__)


def risk_label(
    divergence_percent: float,
    threshold_attention: float = 20,
    threshold_review: float = 40,
) -> str:
    """Convert divergence percent to a human-readable risk label.

    Args:
        divergence_percent: Divergence percent in [0, 100].

    Returns:
        Emoji-prefixed label for UI.
    """

    if divergence_percent < threshold_attention:
        return "✅ Safe"
    if divergence_percent < threshold_review:
        return "⚠️ Needs Attention"
    return "🚨 Review Required"


def risk_color(
    divergence_percent: float,
    threshold_attention: float = 20,
    threshold_review: float = 40,
) -> str:
    """Return a hex color representing the risk severity."""

    if divergence_percent < threshold_attention:
        return "#10b981"
    if divergence_percent < threshold_review:
        return "#f59e0b"
    return "#ef4444"


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

    Security checks:
      - Enforces extension allowlist.
      - Enforces size limits.
      - Validates PDF magic bytes for .pdf.
      - Ensures .txt is UTF-8 decodable.

    Args:
        filename: Original filename.
        file_bytes: Full file bytes.
        max_size_mb: Maximum size in MB.
        allowed_extensions: Allowed file extensions.

    Returns:
        FileValidationResult.
    """

    name = (filename or "").strip()
    if not name:
        return FileValidationResult(False, reason="Missing filename")

    suffix = Path(name).suffix.lower()
    if suffix not in {e.lower() for e in allowed_extensions}:
        return FileValidationResult(False, reason=f"Unsupported file type: {suffix}")

    lowered = name.lower()
    allowed_lower = [e.lower() for e in allowed_extensions]
    for ext in allowed_lower:
        if lowered.endswith(ext):
            continue
        if f"{ext}." in lowered:
            return FileValidationResult(False, reason="Suspicious filename (double extension)")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return FileValidationResult(False, reason=f"File too large: {size_mb:.2f} MB", size_mb=size_mb)

    if suffix == ".pdf":
        # PDF header: %PDF-
        if not file_bytes.startswith(b"%PDF-"):
            return FileValidationResult(False, reason="Invalid PDF header (magic bytes mismatch)", size_mb=size_mb)

    if suffix == ".txt":
        try:
            file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return FileValidationResult(False, reason="Text file is not UTF-8 decodable", size_mb=size_mb)

    return FileValidationResult(True, size_mb=size_mb)
