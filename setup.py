from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


def _read_requirements() -> list[str]:
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.exists():
        return []

    reqs: list[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


setup(
    name="tfidf-compliance-drift",
    version="0.1.0",
    description="TF-IDF based compliance drift monitoring system",
    packages=find_packages(exclude=("tests", "notebooks")),
    python_requires=">=3.10",
    install_requires=_read_requirements(),
)
