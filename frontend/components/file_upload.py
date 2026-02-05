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
        """Returns label -> absolute path map."""
        path_map: Dict[str, str] = {}
        for folder in candidates:
            if not Path(folder).exists():
                continue
            for info in discover_project_files(folder):
                label = f"{info['rel']} ({info['size']})"
                path_map[label] = info["path"]
        return path_map

    internal_paths = _scan_folder([
        "data/internal", "data/internal_docs", "internal_docs", "samples/internal", "samples"
    ])
    guideline_paths = _scan_folder([
        "data/guidelines", "guidelines", "samples/guidelines", "samples"
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # UNIFIED DOCUMENT INPUT CONTAINER
    # ═══════════════════════════════════════════════════════════════════════════

    with st.container():
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)

        # ─── Card title ───
        st.markdown(
            '<div class="upload-container__title">📁 Document Input</div>',
            unsafe_allow_html=True,
        )

        # ═══════════════════════════════════════════════════════════════════════
        # TWO SIDE-BY-SIDE PANELS
        # ═══════════════════════════════════════════════════════════════════════
        panel_left, panel_right = st.columns(2, gap="medium")

        # ───────────────────────────────────────────────────────────────────────
        # LEFT PANEL: Upload New Files
        # ───────────────────────────────────────────────────────────────────────
        with panel_left:
            st.markdown('<div class="input-panel">', unsafe_allow_html=True)
            st.markdown('<div class="input-panel__header">⬆️ Upload New Files</div>', unsafe_allow_html=True)

            # Two upload boxes side by side
            up_int, up_guide = st.columns(2, gap="small")

            with up_int:
                st.markdown('<div class="upload-box-label">Internal</div>', unsafe_allow_html=True)
                uploaded_internal = st.file_uploader(
                    "Internal docs",
                    type=["txt", "pdf"],
                    accept_multiple_files=True,
                    key="upload_internal",
                    label_visibility="collapsed",
                )

            with up_guide:
                st.markdown('<div class="upload-box-label">Guidelines</div>', unsafe_allow_html=True)
                uploaded_guidelines = st.file_uploader(
                    "Guideline docs",
                    type=["txt", "pdf"],
                    accept_multiple_files=True,
                    key="upload_guidelines",
                    label_visibility="collapsed",
                )

            st.markdown('</div>', unsafe_allow_html=True)

        # ───────────────────────────────────────────────────────────────────────
        # RIGHT PANEL: Choose Existing Files
        # ───────────────────────────────────────────────────────────────────────
        with panel_right:
            st.markdown('<div class="input-panel">', unsafe_allow_html=True)
            st.markdown('<div class="input-panel__header">📂 Choose Existing Files</div>', unsafe_allow_html=True)

            # Folder source dropdown
            folder_options = []
            if internal_paths:
                folder_options.append("Internal sample files")
            if guideline_paths:
                folder_options.append("Guideline sample files")

            if folder_options:
                selected_folder = st.selectbox(
                    "Source folder",
                    options=folder_options,
                    index=0,
                    key="existing_folder_source",
                    label_visibility="collapsed",
                )

                # Determine which file list to show
                if selected_folder == "Internal sample files":
                    current_paths = internal_paths
                    target_type = "internal"
                else:
                    current_paths = guideline_paths
                    target_type = "guideline"

                # Multiselect file picker
                selected_existing = st.multiselect(
                    "Select files",
                    options=sorted(current_paths.keys()),
                    default=[],
                    key="existing_file_select",
                    label_visibility="collapsed",
                    placeholder="Click to select files...",
                )

                if selected_existing:
                    st.markdown(
                        f'<div class="file-count-badge">📄 {len(selected_existing)} file(s) selected</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No sample files found in project folders.")
                selected_existing = []
                target_type = None
                current_paths = {}

            st.markdown('</div>', unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════════════
        # PARSE & MERGE DOCUMENTS
        # ═══════════════════════════════════════════════════════════════════════

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

        def load_existing(labels: List[str], path_map: Dict[str, str]) -> List[dict]:
            """Load selected existing files into doc dicts."""
            if not labels:
                return []
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

        if selected_existing and current_paths:
            existing_docs = load_existing(selected_existing, current_paths)
            if target_type == "internal":
                internal_from_existing = existing_docs
            else:
                guideline_from_existing = existing_docs

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
        # STATUS + METRICS ROW
        # ═══════════════════════════════════════════════════════════════════════
        st.markdown(
            f'<div class="upload-container__status">'
            f'Loaded {len(internal_docs)} internal · {len(guideline_docs)} guideline document(s)'
            f'</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3, gap="small")
        with m1:
            st.metric("Internal", len(internal_docs))
        with m2:
            st.metric("Guidelines", len(guideline_docs))
        with m3:
            ready = bool(internal_docs) and bool(guideline_docs)
            st.metric("Status", "✓ Ready" if ready else "Waiting")

        st.markdown('</div>', unsafe_allow_html=True)

    return {"internal": internal_docs, "guidelines": guideline_docs}
