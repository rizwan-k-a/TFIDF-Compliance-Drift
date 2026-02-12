````markdown
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

...[truncated for archive]

````
