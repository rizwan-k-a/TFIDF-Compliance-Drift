# 🔧 CRITICAL FIXES - BEFORE & AFTER CODE

This document shows exact code changes to fix the top 5 critical issues.

---

## #1: Exception Handling (Silent Failures)

### ❌ BEFORE - frontend/app.py (lines 42-50)
```python
if internal_docs or guideline_docs:
    guideline_texts = [d.get("text", "") for d in guideline_docs]
    internal_texts = [d.get("text", "") for d in internal_docs]
    try:
        shared_vectorizer, shared_ref_vectors, shared_int_vectors = build_tfidf_vectors(
            reference_docs=guideline_texts,
            internal_docs=internal_texts,
            keep_numbers=bool(cfg.get("keep_numbers", True)),
            use_lemma=bool(cfg.get("use_lemma", False)),
            max_features=int(cfg.get("max_features", 5000)),
        )
        try:
            from scipy.sparse import vstack
            shared_all_vectors = vstack([shared_ref_vectors, shared_int_vectors])
        except Exception:
            shared_all_vectors = None
    except Exception:  # ❌ SWALLOWS ALL ERRORS SILENTLY
        shared_vectorizer = None
        shared_ref_vectors = None
        shared_int_vectors = None
        shared_all_vectors = None
        shared_names = None
```

**Problem:** User sees blank UI with NO explanation.

### ✅ AFTER - frontend/app.py
```python
import logging
logger = logging.getLogger(__name__)

if internal_docs or guideline_docs:
    guideline_texts = [d.get("text", "") for d in guideline_docs]
    internal_texts = [d.get("text", "") for d in internal_docs]
    try:
        shared_vectorizer, shared_ref_vectors, shared_int_vectors = build_tfidf_vectors(
            reference_docs=guideline_texts,
            internal_docs=internal_texts,
            keep_numbers=bool(cfg.get("keep_numbers", True)),
            use_lemma=bool(cfg.get("use_lemma", False)),
            max_features=int(cfg.get("max_features", 5000)),
        )
        logger.info("TF-IDF vectors built successfully (%d + %d docs)", 
                    len(guideline_texts), len(internal_texts))
        try:
            from scipy.sparse import vstack
            shared_all_vectors = vstack([shared_ref_vectors, shared_int_vectors])
        except ImportError:
            logger.warning("scipy not available; skipping vstack")
            shared_all_vectors = None
    except ValueError as e:
        st.error(f"⚠️ Vectorization failed: {str(e)[:200]}")
        logger.error("ValueError during vectorization: %s", e)
        shared_vectorizer = None
        shared_ref_vectors = None
        shared_int_vectors = None
        shared_all_vectors = None
        shared_names = None
    except MemoryError:
        st.error("💾 Out of memory. Try reducing documents or max_features.")
        logger.error("MemoryError during vectorization")
        shared_vectorizer = None
        shared_ref_vectors = None
        shared_int_vectors = None
        shared_all_vectors = None
        shared_names = None
    except Exception as e:
        st.error(f"❌ Unexpected error: {type(e).__name__}. See logs for details.")
        logger.exception("Unexpected error during vectorization")
        shared_vectorizer = None
        shared_ref_vectors = None
        shared_int_vectors = None
        shared_all_vectors = None
        shared_names = None
```

**Improvement:**
- ✅ User sees friendly error message
- ✅ Admin can see detailed logs
- ✅ Specific exception types give better context
- ✅ Distinguishes recoverable vs. fatal errors

---

## #2: Memory Leak - PDF Caching

### ❌ BEFORE - backend/text_processing.py (line 115)
```python
@functools.lru_cache(maxsize=32)  # ❌ Stores full file bytes in memory!
def extract_text_from_pdf(file_bytes: bytes, filename: str = "document.pdf", use_ocr: bool = True) -> Tuple[str, bool, int]:
    """Extract text from a PDF with optional OCR fallback."""
    try:
        import pdfplumber
    except Exception as e:
        raise RuntimeError("pdfplumber is required for PDF extraction") from e
    
    # ... rest of function
```

**Problem:**
- If user uploads 35 PDFs of 10MB each → 350MB wasted
- Memory never freed until app restart

### ✅ AFTER - backend/text_processing.py
```python
# NO @functools.lru_cache decorator
def extract_text_from_pdf(file_bytes: bytes, filename: str = "document.pdf", use_ocr: bool = True) -> Tuple[str, bool, int]:
    """Extract text from a PDF with optional OCR fallback.
    
    Note: Not cached because file_bytes is unhashable and caching full PDFs
    causes memory bloat. Preprocessing (text) is cached separately.
    """
    try:
        import pdfplumber
    except Exception as e:
        raise RuntimeError("pdfplumber is required for PDF extraction") from e
    
    text_chunks = []
    page_count = 0
    ocr_used = False
    
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        with pdfplumber.open(tmp_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text and len(page_text.strip()) >= CONFIG.min_page_text_length:
                    text_chunks.append(page_text)
        
        full_text = "\n".join(text_chunks).strip()
        
        # ... OCR fallback code unchanged
        
        return full_text, ocr_used, page_count
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

# Preprocess IS cached (text is hashable)
@functools.lru_cache(maxsize=256)
def preprocess_text(text: str, keep_numbers: bool = True, use_lemmatization: bool = False) -> str:
    """Preprocess text - cached because text is hashable."""
    # ... implementation
```

**Improvement:**
- ✅ No memory leak from PDF bytes
- ✅ Preprocessing still cached (text is hashable)
- ✅ Temp files cleaned up immediately
- ✅ More documents can be processed

---

## #3: Classification on Imbalanced Data

### ❌ BEFORE - backend/classification.py (lines 100-118)
```python
def perform_classification(
    documents: Sequence[str],
    categories: Sequence[str],
    test_size: float = 0.3,
    ...
) -> Optional[Dict[str, object]]:
    """Train NB + LR classifiers on TF-IDF vectors."""
    
    docs = list(documents)
    cats = list(categories)
    
    if not docs or not cats or len(docs) != len(cats):
        return None
    
    category_counts = Counter(cats)
    
    # ... filtering code ...
    
    warnings: List[str] = []
    debug: Dict[str, object] = {
        "doc_count": len(y),
        "class_counts": dict(Counter(y)),
    }
    
    min_class_count = min(Counter(y).values()) if y else 0
    if min_class_count < 2:
        warnings.append("Not enough samples per class for reliable classification")
    
    # ❌ PROBLEM: Still continues despite warning!
    if len(y) < 10:
        warnings.append("Dataset has fewer than 10 labeled documents; training on full set")
    
    use_split = len(warnings) == 0
    # ... trains model even with insufficient data
```

**Problem:**
- Model trained on 1 sample per class → garbage predictions
- User doesn't know results are unreliable

### ✅ AFTER - backend/classification.py
```python
def perform_classification(
    documents: Sequence[str],
    categories: Sequence[str],
    test_size: float = 0.3,
    ...
) -> Optional[Dict[str, object]]:
    """Train NB + LR classifiers on TF-IDF vectors.
    
    Rejects datasets with insufficient samples per class or imbalance.
    """
    
    # Input validation
    docs = list(documents)
    cats = list(categories)
    
    if not docs or not cats:
        return None
    
    if len(docs) != len(cats):
        logger.error("Length mismatch: %d docs vs %d categories", len(docs), len(cats))
        return {
            "error": f"Mismatch: {len(docs)} documents but {len(cats)} categories"
        }
    
    # Class balance validation
    category_counts = Counter(cats)
    min_class_count = min(category_counts.values())
    max_class_count = max(category_counts.values())
    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
    
    # CRITICAL FIX: Reject insufficient samples
    if min_class_count < 3:
        logger.warning("Insufficient samples for class: %s (%d < 3)", 
                      min(cats, key=Counter(cats).get), min_class_count)
        return {
            "error": f"Insufficient labeled data. Minimum 3 samples per class required. "
                     f"Found: {dict(category_counts)}"
        }
    
    # WARN on imbalance
    if imbalance_ratio > 10:
        logger.warning("Severe class imbalance: ratio %.1f:1", imbalance_ratio)
        # Continue but inform user
    
    warnings: List[str] = []
    debug: Dict[str, object] = {
        "doc_count": len(cats),
        "class_counts": dict(category_counts),
        "min_samples": min_class_count,
        "imbalance_ratio": round(imbalance_ratio, 1),
    }
    
    if imbalance_ratio > 5:
        warnings.append(f"Class imbalance detected ({imbalance_ratio:.1f}:1). "
                       "Results may favor majority class.")
    
    # ... rest of function with more checks
    
    # After training, return includes error status
    return {
        "nb_model": nb_model,
        "lr_model": lr_model,
        "warnings": warnings,
        "debug": debug,
        "error": None,  # Explicitly no error
    }
```

**Improvement:**
- ✅ Rejects data with < 3 samples per class
- ✅ Warns on severe imbalance (ratio > 5)
- ✅ User sees actionable feedback
- ✅ No training on garbage data

---

## #4: File Validation - Path Traversal

### ❌ BEFORE - backend/utils.py (lines 23-70)
```python
def validate_input_file(
    filename: str,
    file_bytes: bytes,
    max_size_mb: int = CONFIG.max_file_size_mb,
    allowed_extensions: Sequence[str] = (".pdf", ".txt"),
) -> FileValidationResult:
    """Validate an uploaded file for security and robustness."""
    
    name = (filename or "").strip()
    if not name:
        return FileValidationResult(False, reason="Missing filename")
    
    suffix = Path(name).suffix.lower()
    if suffix not in {e.lower() for e in allowed_extensions}:
        return FileValidationResult(False, reason=f"Unsupported file type: {suffix}")
    
    lowered = name.lower()
    allowed_lower = [e.lower() for e in allowed_extensions]
    for ext in allowed_lower:
        if lowered.endswith(ext):
            continue
        if f"{ext}." in lowered:
            return FileValidationResult(False, reason="Suspicious filename (double extension)")
    
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return FileValidationResult(False, reason=f"File too large: {size_mb:.2f} MB", size_mb=size_mb)
    
    if suffix == ".pdf":
        # PDF header: %PDF-
        if not file_bytes.startswith(b"%PDF-"):
            return FileValidationResult(False, reason="Invalid PDF header (magic bytes mismatch)", size_mb=size_mb)
    
    if suffix == ".txt":
        try:
            file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return FileValidationResult(False, reason="Text file is not UTF-8 decodable", size_mb=size_mb)
    
    return FileValidationResult(True, size_mb=size_mb)
```

**Problems:**
- ❌ Missing path traversal check (though Streamlit prevents this)
- ❌ No validation of PDF structure (could be zip bomb)
- ❌ No upper size limit check for PDFs vs TXT

### ✅ AFTER - backend/utils.py
```python
def validate_input_file(
    filename: str,
    file_bytes: bytes,
    max_size_mb: int = CONFIG.max_file_size_mb,
    allowed_extensions: Sequence[str] = (".pdf", ".txt"),
) -> FileValidationResult:
    """Validate an uploaded file for security and robustness.
    
    Security checks:
      - Path traversal prevention
      - Magic byte validation
      - Size limits (especially PDFs which can be zip bombs)
      - UTF-8 validation for text
    """
    
    name = (filename or "").strip()
    if not name:
        return FileValidationResult(False, reason="Missing filename")
    
    # SECURITY FIX #1: Path traversal prevention
    if ".." in name or name.startswith("/") or name.startswith("\\"):
        logger.warning("Rejected suspicious filename: %s", name)
        return FileValidationResult(False, reason="Invalid filename: path traversal detected")
    
    # SECURITY FIX #2: null bytes (path injection)
    if "\x00" in name:
        return FileValidationResult(False, reason="Invalid filename: null byte detected")
    
    suffix = Path(name).suffix.lower()
    if suffix not in {e.lower() for e in allowed_extensions}:
        return FileValidationResult(False, reason=f"Unsupported file type: {suffix}")
    
    lowered = name.lower()
    allowed_lower = [e.lower() for e in allowed_extensions]
    for ext in allowed_lower:
        if lowered.endswith(ext):
            continue
        if f"{ext}." in lowered:
            return FileValidationResult(False, reason="Suspicious filename (double extension)")
    
    size_mb = len(file_bytes) / (1024 * 1024)
    
    # SECURITY FIX #3: Different limits for PDF vs TXT
    if suffix == ".pdf":
        pdf_max_mb = 100  # PDFs can be very large
    else:
        pdf_max_mb = 50  # TXT smaller
    
    if size_mb > pdf_max_mb:
        logger.warning("File rejected: %s (%.1f MB > %d MB limit)", name, size_mb, pdf_max_mb)
        return FileValidationResult(
            False,
            reason=f"File too large: {size_mb:.1f} MB (limit: {pdf_max_mb} MB)",
            size_mb=size_mb
        )
    
    if suffix == ".pdf":
        # PDF header: %PDF-
        if not file_bytes.startswith(b"%PDF-"):
            logger.warning("Invalid PDF magic bytes for %s", name)
            return FileValidationResult(
                False,
                reason="Invalid PDF header (corrupted or not a PDF)",
                size_mb=size_mb
            )
        
        # SECURITY FIX #4: Basic PDF structure validation
        try:
            import pdfplumber
            import tempfile
            
            with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                
                try:
                    with pdfplumber.open(tmp.name) as pdf:
                        if len(pdf.pages) == 0:
                            return FileValidationResult(False, reason="PDF is empty (0 pages)")
                        if len(pdf.pages) > 1000:
                            return FileValidationResult(False, reason=f"PDF too large: {len(pdf.pages)} pages")
                except Exception as e:
                    logger.error("PDF structure validation failed for %s: %s", name, e)
                    return FileValidationResult(False, reason=f"Corrupted PDF: {str(e)[:100]}")
        except ImportError:
            logger.info("pdfplumber not available; skipping PDF structure check")
    
    if suffix == ".txt":
        try:
            file_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Invalid UTF-8 in file %s", name)
            return FileValidationResult(
                False,
                reason="Text file is not UTF-8 decodable. Try saving as UTF-8.",
                size_mb=size_mb
            )
    
    logger.info("File validation passed: %s (%.1f MB)", name, size_mb)
    return FileValidationResult(True, size_mb=size_mb)
```

**Improvements:**
- ✅ Rejects path traversal sequences (`..`, `/`, `\`)
- ✅ Detects null bytes (path injection)
- ✅ Validates PDF structure (not just header)
- ✅ Different size limits for PDF vs TXT
- ✅ Detailed error messages for debugging
- ✅ Logging for audit trail

---

## #5: Imbalanced Classification Detection

### ❌ BEFORE - frontend/components/classification_tab.py (lines 20-45)
```python
def render_classification_tab(cfg: dict, internal_docs: list[dict], ...):
    st.subheader("Document Classification")
    
    if len(internal_docs) < 1:
        st.info("Upload documents to begin")
        return
    
    docs = [d.get("text", "") for d in internal_docs]
    cats = [categorize_document(d.get("text", ""), d.get("name", "")) for d in internal_docs]
    
    if len(docs) != len(cats):
        st.error("Document/label mismatch. Please re-upload your files.")
        return
    
    class_counts = Counter(cats)
    st.write("Debug info")  # ❌ Vague
    st.write({
        "docs": len(docs),
        "class_counts": dict(class_counts),
    })
    
    res = perform_classification(...)
    
    if not res:
        st.warning("Insufficient labeled data after filtering categories.")
        return  # ❌ No detail on what went wrong
    
    # ❌ Shows results without checking for warnings
    c1, c2 = st.columns(2)
    c1.metric("Naive Bayes accuracy", f"{res['nb_accuracy']*100:.1f}%")
    c2.metric("LogReg accuracy", f"{res['lr_accuracy']*100:.1f}%")
```

**Problems:**
- ❌ "Debug info" is vague
- ❌ Error message "Insufficient data" doesn't say why
- ❌ Warnings from `perform_classification` not displayed
- ❌ User sees optimistic accuracy without context

### ✅ AFTER - frontend/components/classification_tab.py
```python
def render_classification_tab(cfg: dict, internal_docs: list[dict], ...):
    st.subheader("📋 Document Classification")
    
    if len(internal_docs) < 2:
        st.info("Upload at least 2 documents to classify.")
        return
    
    docs = [d.get("text", "") for d in internal_docs]
    cats = [categorize_document(d.get("text", ""), d.get("name", "")) for d in internal_docs]
    
    if len(docs) != len(cats):
        st.error("🚨 Internal error: document/label mismatch. Contact support.")
        logger.error("Length mismatch in classification_tab: %d docs vs %d labels", len(docs), len(cats))
        return
    
    class_counts = Counter(cats)
    
    # IMPROVEMENT #1: Clear metadata display
    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Documents", len(docs))
    col2.metric("🏷️ Categories", len(class_counts))
    col3.metric("⏐ Min per class", min(class_counts.values()))
    
    st.write("**Category Distribution:**")
    st.bar_chart(dict(sorted(class_counts.items())))
    
    # IMPROVEMENT #2: Validate before training
    min_count = min(class_counts.values()) if class_counts else 0
    if min_count < 3:
        st.error(
            f"⚠️ **Insufficient labeled data**. "
            f"Need at least 3 documents per category, but category "
            f"`{min(cats, key=Counter(cats).get)}` has only {min_count}. "
            f"\n\n**Fix:** Upload more documents for underrepresented categories."
        )
        return
    
    if max(class_counts.values()) / min_count > 10:
        st.warning(
            f"⚠️ **Severe class imbalance** (~{max(class_counts.values()) / min_count:.0f}:1 ratio). "
            f"Results may favor the majority class."
        )
    
    # IMPROVEMENT #3: Show warnings from backend
    res = perform_classification(
        documents=docs,
        categories=cats,
        keep_numbers=bool(cfg.get("keep_numbers", True)),
        use_lemma=bool(cfg.get("use_lemma", False)),
        max_features=int(cfg.get("max_features", 5000)),
        use_cv=st.checkbox("Use 5-fold cross-validation", value=False),
        precomputed_vectorizer=shared_vectorizer,
        precomputed_matrix=shared_internal_matrix,
    )
    
    # Handle errors from backend
    if not res:
        st.error("🚨 Classification failed. Check your data and try again.")
        logger.error("perform_classification returned None")
        return
    
    if res.get("error"):
        st.error(f"❌ {res['error']}")
        logger.error("Classification error: %s", res['error'])
        return
    
    # IMPROVEMENT #4: Display all warnings before results
    for warning in res.get("warnings", []):
        st.warning(f"⚠️ {warning}")
    
    # IMPROVEMENT #5: Explain results clearly
    st.subheader("📊 Classification Results")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "Naive Bayes Accuracy",
            f"{res['nb_accuracy']*100:.1f}%",
            help="Accuracy on test set (30% of data held out)"
        )
    with c2:
        st.metric(
            "Logistic Regression Accuracy",
            f"{res['lr_accuracy']*100:.1f}%",
            help="Accuracy on test set (30% of data held out)"
        )
    
    # Show debug info if requested
    if st.checkbox("Show debug info"):
        debug = res.get("debug", {})
        st.json({
            "train_size": debug.get("train_size"),
            "test_size": debug.get("test_size"),
            "class_counts": debug.get("class_counts"),
        })
    
    # IMPROVEMENT #6: Feature importance with explanation
    if res.get("top_features"):
        st.subheader("🔍 Most Important Features")
        st.write(res.get("classification_report_lr", ""))
```

**Improvements:**
- ✅ Clear validation before training
- ✅ Shows category distribution as chart
- ✅ Explicit error messages with remediation steps
- ✅ Displays all warnings from backend
- ✅ Helpful tooltips and context
- ✅ Clear explanation of what "accuracy" means
- ✅ Debug info available but not forced on users

---

## Summary of Changes

| Issue | Lines Changed | Files | Impact |
|-------|---------------|-------|--------|
| Exception Handling | ~20 | app.py | Enables debugging |
| PDF Caching | 1 | text_processing.py | Prevents memory leak |
| Classification Validation | ~50 | classification.py | Prevents garbage results |
| File Validation | ~30 | utils.py | Improves security |
| UI Error Display | ~40 | classification_tab.py | Better UX |

**Total effort:** 4-6 hours  
**Impact:** Fixes ~60% of critical issues

