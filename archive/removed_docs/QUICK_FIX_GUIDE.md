# QUICK REFERENCE: CRITICAL FIXES

## 🚨 Top 5 Fixes (Do These First)

### 1. Fix Exception Handling (2-3 hours)
**File:** `frontend/app.py`, `frontend/components/*.py`

**Current:**
```python
try:
    result = some_operation()
except Exception:  # ❌ Swallows all errors
    result = None
```

**Fix:**
```python
try:
    result = some_operation()
except ValueError as e:
    st.error(f"Invalid input: {e}")
    logger.error("ValueError: %s", e)
except Exception as e:
    st.error("Unexpected error. Check logs.")
    logger.exception("Unexpected error in operation")
```

**Impact:** ⚡ Users see meaningful errors; debugging possible

---

### 2. Remove PDF Caching (30 minutes)
**File:** `backend/text_processing.py` line 115

**Current:**
```python
@functools.lru_cache(maxsize=32)  # ❌ Caches entire PDF in memory
def extract_text_from_pdf(file_bytes: bytes, ...):
```

**Fix:**
```python
def extract_text_from_pdf(file_bytes: bytes, ...):  # No decorator
    # Implementation unchanged
```

**Impact:** ⚡ Prevents memory leaks; handles 100+ PDFs

---

### 3. Add Input Validation (2 hours)
**File:** `backend/classification.py` line 40

**Add at function start:**
```python
def perform_classification(...):
    # New: Input validation
    if len(documents) != len(categories):
        raise ValueError(f"Length mismatch: {len(documents)} docs vs {len(categories)} labels")
    
    min_class_count = min(Counter(categories).values()) if categories else 0
    if min_class_count < 3:
        return {"error": f"Insufficient samples: {min_class_count} < 3 required"}
    
    # Rest of function...
```

**Impact:** ⚡ Prevents training on invalid data

---

### 4. Fix File Validation (1 hour)
**File:** `backend/utils.py` line 44

**Add before size check:**
```python
def validate_input_file(...):
    # Check for suspicious filenames
    if ".." in filename or "/" in filename or filename.startswith("-"):
        return FileValidationResult(False, "Suspicious filename")
    
    # Rest of validation...
```

**Impact:** ⚡ Prevents path traversal and injection attacks

---

### 5. Add Logging Framework (3 hours)
**File:** Create `utils/logging.py`

```python
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)

# In all backend modules:
logger = logging.getLogger(__name__)
logger.info("Processing %d documents", len(docs))
logger.error("Failed to vectorize: %s", error)
```

**Impact:** ⚡ Can now debug production issues

---

## 📋 Test Cases to Add

```python
# tests/test_critical_fixes.py
import pytest
from backend.classification import perform_classification

class TestInputValidation:
    def test_classification_length_mismatch(self):
        """Should reject if docs != categories."""
        with pytest.raises(ValueError, match="Length mismatch"):
            perform_classification(
                documents=["doc1", "doc2"],
                categories=["cat1"],  # ❌ Only 1 category
            )
    
    def test_classification_insufficient_samples(self):
        """Should reject if < 3 samples per class."""
        result = perform_classification(
            documents=["doc1", "doc2"],
            categories=["cat1", "cat1"],  # Only 1 class with 2 samples
        )
        assert "error" in result

class TestFileValidation:
    def test_path_traversal_rejection(self):
        """Should reject filenames with '..'."""
        from backend.utils import validate_input_file
        result = validate_input_file("../../../etc/passwd", b"data")
        assert not result.ok
    
    def test_pdf_too_large(self):
        """Should reject PDFs > 100MB."""
        from backend.utils import validate_input_file
        huge_pdf = b"%PDF-" + b"\x00" * (101 * 1024 * 1024)
        result = validate_input_file("huge.pdf", huge_pdf)
        assert not result.ok

class TestExceptionHandling:
    def test_vectorization_error_logged(self, caplog):
        """Should log vectorization errors."""
        from backend.tfidf_engine import build_tfidf_vectors
        import logging
        
        with caplog.at_level(logging.WARNING):
            # Trigger error (too few documents)
            try:
                build_tfidf_vectors([""], [""])
            except ValueError:
                pass
        
        # Should have logged something
        assert len(caplog.records) > 0
```

---

## 🔍 Code Cleanup (Delete These Files)

```bash
# These are dead code - remove to simplify maintenance
rm src/drift.py
rm src/alerts.py  
rm src/similarity.py
rm src/vectorize.py

# Keep only:
# - src/manual_tfidf_math.py (educational, referenced in tests)
# - src/preprocess.py (if used)

# These can be archived but not needed in main project:
# - scripts/pdf_to_txt_once.py (one-time utility)
# - scripts/audit_repo.py (development tool)
```

---

## 📊 Quick Test Coverage Check

```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage
pytest --cov=backend --cov=utils --cov-report=html tests/

# Open htmlcov/index.html to see what's NOT covered
```

**Target:** 70% coverage on critical paths (classification, similarity, vectorization)

---

## 🚀 Minimal Production Checklist

- [ ] All `except Exception:` replaced with specific types
- [ ] Error messages shown to user (not silently swallowed)
- [ ] `pytest -v` passes all tests
- [ ] No linter errors: `flake8 backend utils`
- [ ] No type errors: `mypy backend utils` 
- [ ] Logging configured and tested
- [ ] README documents known limitations
- [ ] Deployment instructions written

---

## 💡 Quick Wins (Do These in 30 minutes each)

1. **Add docstrings to public functions**
   ```python
   def compute_similarity_scores(docs: List[str]) -> pd.DataFrame:
       """Compute similarity between documents.
       
       Args:
           docs: List of document texts
       
       Returns:
           DataFrame with similarity scores
       """
   ```

2. **Fix hardcoded values**
   ```python
   # ❌ Don't: st.slider(..., value=10000)
   # ✅ Do:    st.slider(..., value=CONFIG.tfidf_max_features)
   ```

3. **Improve error messages**
   ```python
   # ❌ st.error("Error occurred")
   # ✅ st.error("Could not vectorize documents. Please check PDF quality or reduce max_features.")
   ```

4. **Add type hints to function signatures**
   ```python
   # ❌ def process(data):
   # ✅ def process(data: List[str]) -> Dict[str, float]:
   ```

5. **Use config values everywhere**
   ```python
   # ❌ max_size = 50
   # ✅ max_size = CONFIG.max_file_size_mb
   ```

---

## 📝 Files to Review First

**Priority order for fixing:**
1. `backend/classification.py` - ML quality (add validation)
2. `frontend/app.py` - Error handling (add logging)
3. `backend/text_processing.py` - Memory safety (remove PDF cache)
4. `backend/utils.py` - File security (add path checks)
5. `frontend/components/*.py` - UI error messages (improve UX)

---

## ✅ Validation Checklist

After implementing fixes, verify:

```bash
# 1. Code style
black --check backend/ utils/ frontend/tests/

# 2. Linting
flake8 backend/ utils/ --max-line-length=120

# 3. Type safety
mypy backend/ utils/ --ignore-missing-imports

# 4. Tests pass
pytest tests/ -v

# 5. Coverage minimum
pytest --cov=backend tests/ | grep -i "covered"

# 6. No security issues
bandit -r backend/ utils/
```

---

## 🎯 Success Metrics

After fixes, you should see:

- ✅ **Error messages** appear in UI when things fail
- ✅ **Test suite** takes < 10 seconds
- ✅ **Memory usage** stable with 100+ documents
- ✅ **No crashes** on invalid input
- ✅ **Type checker satisfied**: `mypy` has 0 errors
- ✅ **Production-ready** for small-scale deployments (<100 concurrent users)

