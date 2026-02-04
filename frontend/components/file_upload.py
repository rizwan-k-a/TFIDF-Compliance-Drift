from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st

from backend.text_processing import extract_text_from_pdf, validate_text
from backend.utils import validate_input_file


def _read_uploaded_file(uploaded_file) -> Tuple[bytes, str]:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    return file_bytes, filename


def upload_documents(cfg: dict) -> Dict[str, List[dict]]:
    """Upload internal and guideline documents.

    Returns:
        {"internal": [{name,text}], "guidelines": [{name,text}]}
    """

    st.subheader("Upload")

    col1, col2 = st.columns(2)

    with col1:
        internal_files = st.file_uploader(
            "Internal documents (TXT/PDF)",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="internal_upload",
        )

    with col2:
        guideline_files = st.file_uploader(
            "Guidelines (TXT/PDF)",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="guideline_upload",
        )

    def parse_files(files) -> List[dict]:
        docs: List[dict] = []
        if not files:
            return docs

        for f in files:
            file_bytes, filename = _read_uploaded_file(f)
            validation = validate_input_file(filename, file_bytes)
            if not validation.ok:
                st.warning(f"{filename}: {validation.reason or 'Invalid file'}")
                continue

            ext = filename.lower().split(".")[-1]
            if ext == "txt":
                try:
                    text = file_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""
                ok, msg = validate_text(text, filename)
                if not ok:
                    st.warning(msg)
                    continue
                docs.append({"name": filename, "text": text})
            elif ext == "pdf":
                text, ocr_used, _pages = extract_text_from_pdf(
                    file_bytes,
                    filename=filename,
                    use_ocr=bool(cfg.get("enable_ocr", True)),
                )
                ok, msg = validate_text(text, filename)
                if not ok:
                    st.warning(msg)
                    continue
                docs.append({"name": filename, "text": text, "ocr_used": ocr_used})

        return docs

    internal_docs = parse_files(internal_files)
    guideline_docs = parse_files(guideline_files)

    st.caption(
        f"Loaded {len(internal_docs)} internal docs and {len(guideline_docs)} guideline docs."
    )

    return {"internal": internal_docs, "guidelines": guideline_docs}
