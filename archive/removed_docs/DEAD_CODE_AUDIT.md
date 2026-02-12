````markdown
# 🗑️ Dead Code & Legacy Module Audit

**Date:** 2026-02-07  
**Status:** Analysis Complete | 2,000+ lines identified for cleanup

---

## Executive Summary

The `src/` folder contains **legacy/educational modules** totaling **2,000+ SLOC** that are partially unused or duplicate the `backend/` implementations. Only **1 module** (`manual_tfidf_math.py`) has active usage in the test suite.

**Recommendation: Archive or delete 6/7 modules in `src/`**

**Impact:**
- ✅ Reduce codebase by ~30% (cleaning dead code)
- ✅ Eliminate duplication and confusion
- ✅ Simplify maintenance and refactoring
- ✅ Faster code search and navigation

---

## Detailed Analysis

### 1. ✅ `src/manual_tfidf_math.py` — KEEP (Educational Value)

**Status:** KEEP  
**Size:** 642 lines

**Usage:**
- ✅ Used in `tests/test_tfidf_math.py` (7+ imports)
- ✅ Used in `tests/test_preprocessing.py` (1 import)
- ✅ Referenced in `README.md` for validation script

**Purpose:** Educational implementation of TF-IDF algorithm from scratch (manual matrix operations, no sklearn).

**Code Quality:** Excellent pedagogical value; thorough comments explaining each step.

**Recommendation:**
```bash
# KEEP in src/, but optionally move to docs/
# mkdir -p docs/educational
# cp src/manual_tfidf_math.py docs/educational/manual_tfidf_math.py
```

...[truncated for archive]

````
