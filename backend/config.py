"""Backend configuration and constants.

This module centralizes configuration for the compliance review system.
It is intentionally UI-agnostic (no Streamlit imports).

Exports:
  - CONFIG: Config
  - CATEGORIES: dict
  - get_poppler_path: function

The Streamlit UI should import settings from here rather than duplicating them.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    """Global configuration constants.

    Attributes:
        min_text_length: Minimum number of characters to consider text usable.
        min_words: Minimum number of tokens to consider text usable.
        max_file_size_mb: Maximum accepted upload size.
        tfidf_max_features: Default TF-IDF feature cap.
        ngram_range: Default n-gram range.
        min_df: Default min_df.
        max_df: Default max_df.
        random_state: Default random seed.
        min_page_text_length: Minimum characters on a PDF page to accept as extracted.
        ocr_dpi: DPI used for OCR rasterization.
        ocr_config: Tesseract configuration string.
        sample_autoload_enabled: Whether sample auto-loading is enabled in the UI.
        sample_autoload_internal_limit: Max internal sample docs to auto-load.
        sample_autoload_guideline_limit: Max guideline sample docs to auto-load.
    """

    min_text_length: int = 100
    min_words: int = 20
    max_file_size_mb: int = 50
    tfidf_max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: float = 0.05
    max_df: float = 0.95

    random_state: int = 42
    min_page_text_length: int = 100

    ocr_dpi: int = 300
    ocr_config: str = "--psm 6"

    sample_autoload_enabled: bool = True
    sample_autoload_internal_limit: int = 10
    sample_autoload_guideline_limit: int = 5


def get_poppler_path() -> Optional[str]:
    """Detect Poppler binary directory across platforms.

    Returns:
        A filesystem path to Poppler's `bin` directory, or None if not found.

    Notes:
        - On Windows, users commonly install Poppler via prebuilt packages.
        - Users may also set POPPLER_PATH environment variable.
    """

    env = os.environ.get("POPPLER_PATH")
    if env and Path(env).exists():
        return env

    if sys.platform != "win32":
        return None

    candidates: List[str] = [
        r"C:\\Program Files\\poppler\\Library\\bin",
        r"C:\\Program Files (x86)\\poppler\\Library\\bin",
        r"C:\\Program Files\\poppler-24.08.0\\Library\\bin",
        r"C:\\Program Files\\poppler-25.12.0\\Library\\bin",
        r"C:\\poppler\\Library\\bin",
    ]

    for path in candidates:
        if Path(path).exists():
            return path

    return None


CONFIG = Config(
    sample_autoload_enabled=_env_bool("SAMPLE_AUTOLOAD_ENABLED", True),
    sample_autoload_internal_limit=_env_int("SAMPLE_AUTOLOAD_INTERNAL_LIMIT", 10),
    sample_autoload_guideline_limit=_env_int("SAMPLE_AUTOLOAD_GUIDELINE_LIMIT", 5),
)

POPPLER_PATH: Optional[str] = get_poppler_path()

# Document categories and keyword-based matching.
CATEGORIES: Dict[str, Dict[str, object]] = {
    "Criminal": {
        "keywords": [
            "criminal",
            "case",
            "intake",
            "incident",
            "investigation",
            "evidence",
            "prosecution",
            "bns",
            "nyaya",
            "sanhita",
        ],
        "guideline": "BNS_2023",
    },
    "Cyber": {
        "keywords": [
            "cyber",
            "data",
            "access",
            "control",
            "security",
            "information",
            "breach",
            "digital",
            "privacy",
            "encryption",
        ],
        "guideline": "IT_ACT_2021",
    },
    "Financial": {
        "keywords": [
            "financial",
            "transaction",
            "monitoring",
            "aml",
            "money",
            "laundering",
            "pmla",
            "kyc",
            "customer",
            "diligence",
        ],
        "guideline": "PMLA_2002",
    },
}
