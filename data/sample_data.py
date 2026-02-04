"""Small in-repo sample dataset helpers.

These are optional conveniences for demos/tests; production deployments should
load real regulatory/internal documents from your data sources.
"""

from __future__ import annotations

from pathlib import Path


def load_text(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="ignore")


def load_default_guidelines(data_root: str | Path = "data") -> dict[str, str]:
    root = Path(data_root) / "guidelines"
    mapping: dict[str, str] = {}
    for txt in root.rglob("*.txt"):
        mapping[txt.stem] = load_text(txt)
    return mapping


def load_default_internal(data_root: str | Path = "data") -> dict[str, str]:
    root = Path(data_root) / "internal"
    mapping: dict[str, str] = {}
    for txt in root.rglob("*.txt"):
        mapping[txt.stem] = load_text(txt)
    return mapping
