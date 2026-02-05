"""LEGACY — not used in main demo path.

Kept for reference/educational purposes.
"""

import os
from typing import List


def load_text_files(folder: str) -> List[str]:
    texts = []
    for fn in sorted(os.listdir(folder)):
        path = os.path.join(folder, fn)
        if os.path.isfile(path) and fn.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts


def save_csv(df, path: str):
    df.to_csv(path, index=False)
