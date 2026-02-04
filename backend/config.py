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


@dataclass(frozen=True)
class Config:
    """Global configuration constants.

    Attributes:
        min_text_length: Minimum number of characters to consider text usable.
        min_words: Minimum number of tokens to consider text usable.
        max_file_size_mb: Maximum accepted upload size.
        default_divergence_threshold: Divergence percent threshold for flagging.
        tfidf_max_features: Default TF-IDF feature cap.
        ngram_range: Default n-gram range.
        min_df: Default min_df.
        max_df: Default max_df.
        random_state: Default random seed.
        min_page_text_length: Minimum characters on a PDF page to accept as extracted.
        ocr_dpi: DPI used for OCR rasterization.
        ocr_config: Tesseract configuration string.
    """

    min_text_length: int = 100
    min_words: int = 20
    max_file_size_mb: int = 50
    default_divergence_threshold: int = 40

    tfidf_max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: float = 0.05
    max_df: float = 0.95

    random_state: int = 42
    min_page_text_length: int = 100

    ocr_dpi: int = 300
    ocr_config: str = "--psm 6"


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


CONFIG = Config()

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
        "guideline_name": "Bharatiya Nyaya Sanhita (BNS) 2023",
        "color": "#ef4444",
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
        "guideline_name": "IT Act 2021",
        "color": "#3b82f6",
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
        "guideline_name": "Prevention of Money Laundering Act (PMLA) 2002",
        "color": "#10b981",
    },
}
