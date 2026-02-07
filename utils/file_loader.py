from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


ALLOWED_EXTS = {".txt", ".pdf"}


def _safe_relpath(file_path: Path, base_dir: Path) -> str:
    try:
        return file_path.relative_to(base_dir).as_posix()
    except ValueError:
        # Path is not relative to base_dir; return just filename
        return file_path.name


def discover_project_files(base_folder: str) -> List[Dict[str, str]]:
    """Discover .txt/.pdf files under a folder (recursive).

    Args:
        base_folder: Folder path relative to repo root (e.g., 'data/internal').

    Returns:
        List of dicts: {name, path, rel, size, ext}.
    """

    base_dir = Path(base_folder)
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    out: List[Dict[str, str]] = []
    for file_path in sorted(base_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in ALLOWED_EXTS:
            continue
        # Ignore common metadata
        if file_path.name.lower() in {"metadata.csv"}:
            continue

        try:
            size_kb = file_path.stat().st_size / 1024.0
        except (OSError, FileNotFoundError):
            # File stat failed (permission, race condition, etc.)
            size_kb = 0.0

        rel = _safe_relpath(file_path, base_dir)
        out.append(
            {
                "name": file_path.name,
                "path": str(file_path),
                "rel": rel,
                "ext": file_path.suffix.lower(),
                "size": f"{size_kb:.1f}KB",
            }
        )

    return out


def read_text_preview(file_path: str, limit: int = 200) -> str:
    """Read a short preview (first N chars) for .txt files."""

    p = Path(file_path)
    if p.suffix.lower() != ".txt":
        return ""

    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except (OSError, FileNotFoundError, PermissionError):
        # File read failed; return empty
        return ""

    preview = (text or "").strip().replace("\r\n", "\n")
    if len(preview) > limit:
        return preview[:limit].rstrip() + "..."
    return preview


def load_document_from_bytes(
    filename: str,
    file_bytes: bytes,
    *,
    source: str,
    use_ocr: bool,
) -> Tuple[Dict[str, Any] | None, str | None]:
    """Convert raw bytes into the common doc dict used by the UI.

    Returns:
        (doc, error)
        doc is {name, text, source, ocr_used?}
    """

    from backend.text_processing import extract_text_from_pdf, validate_text

    name = (filename or "").strip() or "document"
    ext = Path(name).suffix.lower()

    try:
        if ext == ".txt":
            text = (file_bytes or b"").decode("utf-8", errors="ignore")
            ok, msg = validate_text(text, name)
            if not ok:
                return None, msg
            return {"name": name, "text": text, "source": source}, None

        if ext == ".pdf":
            text, ocr_used, _pages = extract_text_from_pdf(
                file_bytes or b"",
                filename=name,
                use_ocr=bool(use_ocr),
            )
            ok, msg = validate_text(text, name)
            if not ok:
                return None, msg
            return {"name": name, "text": text, "ocr_used": ocr_used, "source": source}, None

        return None, f"Unsupported file type: {ext or '(none)'}"
    except UnicodeDecodeError as e:
        return None, f"{name}: Text decoding failed (likely binary file)"
    except RuntimeError as e:
        # From extract_text_from_pdf missing pdfplumber
        return None, f"{name}: {e}"
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception("Unexpected error loading document %s", name)
        return None, f"{name}: Unexpected error: {type(e).__name__}"


def load_selected_files(
    file_paths: Sequence[str],
    *,
    use_ocr: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load selected project files as doc dicts.

    Returns:
        (docs, errors)
        docs are: {name, text, source, ocr_used?}
    """

    docs: List[Dict[str, Any]] = []
    errors: List[str] = []

    for path in file_paths:
        p = Path(path)
        if not p.exists() or not p.is_file():
            errors.append(f"Missing file: {path}")
            continue
        if p.suffix.lower() not in ALLOWED_EXTS:
            errors.append(f"Unsupported file type: {p.name}")
            continue

        try:
            file_bytes = p.read_bytes()
            doc, err = load_document_from_bytes(
                p.name,
                file_bytes,
                source="project",
                use_ocr=bool(use_ocr),
            )
            if err:
                errors.append(err)
                continue
            if doc:
                docs.append(doc)
        except (OSError, MemoryError) as e:
            errors.append(f"{p.name}: Failed to read file ({type(e).__name__})")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.exception("Unexpected error loading %s", p.name)
            errors.append(f"{p.name}: Unexpected error")

    return docs, errors
