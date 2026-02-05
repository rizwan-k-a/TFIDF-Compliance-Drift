from __future__ import annotations

from typing import Dict, List, Tuple
from pathlib import Path

import streamlit as st

from backend.utils import validate_input_file
from utils.file_loader import load_document_from_bytes
from utils.file_loader import discover_project_files, load_selected_files


def _read_uploaded_file(uploaded_file) -> Tuple[bytes, str]:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    return file_bytes, filename


def upload_documents(cfg: dict) -> Dict[str, List[dict]]:
    """Upload and/or select internal and guideline documents.

    Both upload boxes and existing file selectors are visible simultaneously.

    Returns:
        {"internal": [{name,text}], "guidelines": [{name,text}]}
    """

    # ─── Scan available project files ───
    def _scan_folder(candidates: List[str]) -> Dict[str, str]:
        """Returns relative path -> absolute path map (txt/pdf only)."""
        path_map: Dict[str, str] = {}
        for folder in candidates:
            if not Path(folder).exists():
                continue
            for info in discover_project_files(folder):
                rel_path = info["rel"]
                if not rel_path.lower().endswith((".txt", ".pdf")):
                    continue
                path_map[rel_path] = info["path"]
        return path_map

    internal_paths = _scan_folder([
        "data/internal", "data/internal_docs", "internal_docs", "samples/internal", "samples"
    ])
    guideline_paths = _scan_folder([
        "data/guidelines", "guidelines", "samples/guidelines", "samples"
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # DOCUMENT INPUT (TWO-COLUMN, TABBED)
    # ═══════════════════════════════════════════════════════════════════════════

    internal_folder_map = {
        "policy_documents": [
            "data/policy_documents",
            "policy_documents",
            "data/internal",
            "data/internal_docs",
            "internal_docs",
        ],
        "internal_samples": [
            "samples/internal",
            "samples",
            "data/internal",
            "data/internal_docs",
        ],
        "templates": [
            "templates",
            "data/templates",
            "samples/templates",
        ],
    }

    guideline_folder_map = {
        "guideline_samples": [
            "samples/guidelines",
            "samples",
            "data/guidelines",
            "guidelines",
        ],
        "regulatory_docs": [
            "data/regulatory_docs",
            "regulatory_docs",
            "data/guidelines",
            "guidelines",
        ],
        "standards": [
            "data/standards",
            "standards",
            "data/guidelines",
            "guidelines",
        ],
    }

    def get_files_from_folder(folder_name: str, doc_type: str) -> List[str]:
        if doc_type == "internal":
            candidates = internal_folder_map.get(folder_name, [])
        else:
            candidates = guideline_folder_map.get(folder_name, [])
        path_map = _scan_folder(candidates)
        return sorted(path_map.keys())

    def _get_path_map(folder_name: str, doc_type: str) -> Dict[str, str]:
        if doc_type == "internal":
            candidates = internal_folder_map.get(folder_name, [])
        else:
            candidates = guideline_folder_map.get(folder_name, [])
        return _scan_folder(candidates)

    def load_files_to_session(file_list: List[str], doc_type: str) -> None:
        st.session_state[f"{doc_type}_selected_files"] = list(file_list)

    with st.container():
        st.markdown("## 📁 Document Input")

        col_internal, col_guidelines = st.columns(2)

        # LEFT COLUMN — INTERNAL DOCUMENTS
        with col_internal:
            st.markdown("### 📄 Internal Documents")
            internal_tab_upload, internal_tab_existing = st.tabs(["⬆️ Upload", "📂 Choose Existing"])

            with internal_tab_upload:
                uploaded_internal = st.file_uploader(
                    "Internal documents",
                    type=["txt", "pdf"],
                    accept_multiple_files=True,
                    key="internal_upload",
                    help="Limit 200MB per file • TXT, PDF",
                )

            with internal_tab_existing:
                internal_folder = st.selectbox(
                    "Folder",
                    ["policy_documents", "internal_samples", "templates"],
                    key="internal_folder",
                )
                internal_files = get_files_from_folder(internal_folder, "internal")
                selected_internal = st.multiselect(
                    "Select files",
                    internal_files,
                    key="internal_selected",
                )
                if st.button("Load Internal Files", key="load_internal", use_container_width=True):
                    load_files_to_session(selected_internal, "internal")
                    st.success("Loaded internal files.")

        # RIGHT COLUMN — GUIDELINE DOCUMENTS
        with col_guidelines:
            st.markdown("### 📋 Guideline Documents")
            guideline_tab_upload, guideline_tab_existing = st.tabs(["⬆️ Upload", "📂 Choose Existing"])

            with guideline_tab_upload:
                uploaded_guidelines = st.file_uploader(
                    "Guideline documents",
                    type=["txt", "pdf"],
                    accept_multiple_files=True,
                    key="guideline_upload",
                    help="Limit 200MB per file • TXT, PDF",
                )

            with guideline_tab_existing:
                guideline_folder = st.selectbox(
                    "Folder",
                    ["guideline_samples", "regulatory_docs", "standards"],
                    key="guideline_folder",
                )
                guideline_files = get_files_from_folder(guideline_folder, "guideline")
                selected_guidelines = st.multiselect(
                    "Select files",
                    guideline_files,
                    key="guideline_selected",
                )
                if st.button("Load Guideline Files", key="load_guidelines", use_container_width=True):
                    load_files_to_session(selected_guidelines, "guideline")
                    st.success("Loaded guideline files.")

        def parse_uploaded(files) -> List[dict]:
            """Parse uploaded files into doc dicts."""
            docs: List[dict] = []
            if not files:
                return docs
            use_ocr = bool(cfg.get("enable_ocr", True))
            for f in files:
                file_bytes, filename = _read_uploaded_file(f)
                validation = validate_input_file(filename, file_bytes)
                if not validation.ok:
                    st.warning(f"{filename}: {validation.reason or 'Invalid'}")
                    continue
                doc, err = load_document_from_bytes(filename, file_bytes, source="upload", use_ocr=use_ocr)
                if err:
                    st.warning(err)
                    continue
                if doc:
                    docs.append(doc)
            return docs

        def load_existing(labels: List[str], folder_name: str, doc_type: str) -> List[dict]:
            """Load selected existing files into doc dicts."""
            if not labels:
                return []
            path_map = _get_path_map(folder_name, doc_type)
            paths = [path_map[l] for l in labels if l in path_map]
            docs, errors = load_selected_files(paths, use_ocr=bool(cfg.get("enable_ocr", True)))
            for err in errors:
                st.warning(err)
            return docs

        # Parse uploaded files
        internal_from_upload = parse_uploaded(uploaded_internal)
        guideline_from_upload = parse_uploaded(uploaded_guidelines)

        # Load selected existing files
        internal_from_existing: List[dict] = []
        guideline_from_existing: List[dict] = []

        internal_selected_effective = st.session_state.get("internal_selected_files", selected_internal)
        guideline_selected_effective = st.session_state.get("guideline_selected_files", selected_guidelines)

        if internal_selected_effective:
            internal_from_existing = load_existing(internal_selected_effective, internal_folder, "internal")

        if guideline_selected_effective:
            guideline_from_existing = load_existing(guideline_selected_effective, guideline_folder, "guideline")

        # Merge: uploaded + existing (dedupe by name)
        def merge_docs(list1: List[dict], list2: List[dict]) -> List[dict]:
            seen = set()
            merged = []
            for doc in list1 + list2:
                name = doc.get("name", "")
                if name not in seen:
                    seen.add(name)
                    merged.append(doc)
            return merged

        internal_docs = merge_docs(internal_from_upload, internal_from_existing)
        guideline_docs = merge_docs(guideline_from_upload, guideline_from_existing)

        # ─── Auto-load complementary files if only one side is populated ───
        autoload_enabled = bool(cfg.get("sample_autoload_enabled", True))
        int_limit = int(cfg.get("sample_autoload_internal_limit", 10) or 0)
        guide_limit = int(cfg.get("sample_autoload_guideline_limit", 5) or 0)

        if autoload_enabled and internal_docs and not guideline_docs and guideline_paths and guide_limit > 0:
            auto_paths = list(guideline_paths.values())[:guide_limit]
            auto_docs, errs = load_selected_files(auto_paths, use_ocr=bool(cfg.get("enable_ocr", True)))
            for e in errs:
                st.warning(e)
            guideline_docs = auto_docs

        if autoload_enabled and guideline_docs and not internal_docs and internal_paths and int_limit > 0:
            auto_paths = list(internal_paths.values())[:int_limit]
            auto_docs, errs = load_selected_files(auto_paths, use_ocr=bool(cfg.get("enable_ocr", True)))
            for e in errs:
                st.warning(e)
            internal_docs = auto_docs

    # ═══════════════════════════════════════════════════════════════════════
    # STATUS + METRICS ROW (IMMEDIATELY BELOW PANELS)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown(
        f"Loaded {len(internal_docs)} internal · {len(guideline_docs)} guideline document(s)"
    )

    m1, m2, m3 = st.columns(3, gap="small")
    with m1:
        st.metric("Internal", len(internal_docs))
    with m2:
        st.metric("Guidelines", len(guideline_docs))
    with m3:
        ready = bool(internal_docs) and bool(guideline_docs)
        st.metric("Status", "✓ Ready" if ready else "Waiting")

    return {"internal": internal_docs, "guidelines": guideline_docs}
