import re
from typing import List


def clean_text(text: str) -> str:
    """Basic text cleaning for downstream TF-IDF: lowercasing, remove non-word chars."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def preprocess_documents(docs: List[str]) -> List[str]:
    return [clean_text(d) for d in docs]
