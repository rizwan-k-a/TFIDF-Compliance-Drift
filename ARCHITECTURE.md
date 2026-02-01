# TF-IDF COMPLIANCE DRIFT - ARCHITECTURE DEEP DIVE

This document provides a concise map of the primary modules and code regions inside the project, using `dashboard/app.py` as the canonical line-mapped reference for developers and reviewers.

## Code Organization (app.py Line Map)

### 1️⃣ Configuration & Imports (Lines 1-150)
- **Lines 1-50**: Library imports (Streamlit, scikit-learn, NLTK, PDF/OCR libraries, plotting)
- **Lines 51-100**: `Config` dataclass and global configuration values (thresholds, defaults)
- **Lines 101-150**: Document category definitions and metadata (Criminal, Cyber, Financial)

### 2️⃣ Preprocessing Pipeline (Lines 151-350)
- **Lines 151-200**: `preprocess_text()` — cleaning, normalization, optional lemmatization
- **Lines 201-280**: `extract_text_from_pdf()` — pdfplumber primary extraction + pdf2image/pytesseract fallback
- **Lines 281-350**: `get_tfidf_vectorizer()` / cached vectorizer factory and small-corpus fallbacks

### 3️⃣ Manual TF-IDF Mathematics (Lines 351-550)
- **Lines 351-400**: `compute_tf_variants()` — Binary, Raw, Normalized, Log-normalized, Double-normalization
- **Lines 401-450**: `compute_idf_variants()` — Standard, Smooth, Sklearn Smooth, Probabilistic
- **Lines 451-550**: `compute_manual_tfidf_complete()` — orchestrator that produces per-doc, per-variant scores

### 4️⃣ Document Categorization (Lines 551-650)
- **Lines 551-650**: `categorize_document()` — lightweight keyword-based category mapping + guideline linking

### 5️⃣ Core Analysis Engine (Lines 651-1200)
- **Lines 651-750**: `build_tfidf_vectors()` — adaptive vectorization, dynamic `min_df`/`max_df`, `max_features`
- **Lines 751-900**: `compute_similarity_scores_by_category()` — cosine similarity, best-match selection, aggregation
- **Lines 901-1100**: `perform_classification()` — supervised pipeline (MultinomialNB, LogisticRegression) with filtering and metrics
- **Lines 1101-1200**: `perform_enhanced_clustering()` — K-Means exploration, silhouette/Davies-Bouldin/Calinski-Harabasz

### 6️⃣ Visualization & UI Utilities (Lines 1201-1500)
- **Lines 1201-1300**: `display_tfidf_matrix()` — shape, sparsity, sample matrix rendering
- **Lines 1301-1400**: `generate_wordcloud()` / plotting helpers — word clouds, bar charts, heatmaps
- **Lines 1401-1450**: `render_header()`, `render_disclaimer()` — reusable UI fragments
- **Lines 1451-1500**: `explain_tfidf_weighting()` — educational content shown in Tab 2

### 7️⃣ Main Application Flow (Lines 1501-2000)
- **Lines 1501-1550**: Sidebar configuration — UI controls for `max_features`, `min_df`, `max_df`, `ngram_range`, etc.
- **Lines 1551-1600**: File upload and processing loop — PDF/TXT ingestion, OCR messaging, session-state updates
- **Lines 1601-1700**: Tab 1 — Compliance Dashboard (comparison, risk scoring, PDF report generation)
- **Lines 1701-1800**: Tab 2 — TF-IDF Mathematics (manual computations, decision tables, validation)
- **Lines 1801-1900**: Tab 3 — Classification (training, performance, predictive features)
- **Lines 1901-2000**: Tab 4-6 — Clustering, Matrix Inspection, Visualizations

## Data Flow Diagram (textual)

1. Input: `data/internal/` and `data/guidelines/` (TXT/PDF)
2. Ingestion: `dashboard/app.py` file upload → `extract_text_from_pdf()` or direct read
3. Preprocessing: `preprocess_text()` (clean, tokenize, optional lemmatize)
4. Vectorization:
   - Manual TF-IDF: `compute_manual_tfidf_complete()` → per-variant matrices for demonstration
   - Sklearn TF-IDF: `build_tfidf_vectors()` / `TfidfVectorizer` for production pipelines
5. Analysis:
   - Similarity: `compute_similarity_scores_by_category()` (cosine similarity)
   - Classification: `perform_classification()` (train/test, metrics, feature importance)
   - Clustering: `perform_enhanced_clustering()` (K-Means + diagnostics)
6. Detection & Alerts: `compute_divergence()` → `generate_alerts()` → `results/drift_alerts.csv`
7. Presentation: Streamlit tabs and downloadable PDF audit reports

## Module Responsibilities (brief)
- `src/manual_tfidf_math.py`: Ground-up TF and IDF formulas, tokenization helpers, validation utilities
- `src/vectorize.py`: Adaptive TF-IDF wrapper, caching, vocabulary control, n-gram handling
- `src/preprocess.py`: Text normalization, stopword handling, lemmatization wrappers
- `src/similarity.py`: Cosine similarity helpers, matrix utilities
- `src/drift.py`: Drift scoring, aggregation across documents/categories, helper thresholds
- `src/alerts.py`: Alert logic, risk-label mapping, export helpers
- `src/utils.py`: I/O helpers, PDF text extraction wrappers, safe file reads

## Operational Notes
- The app is intentionally defensive: default `max_features` and dynamic `min_df/max_df` protect against memory blowups on large corpora and over-filtering on small corpora.
- Manual TF-IDF is validated against `sklearn.TfidfVectorizer` in the UI to provide reproducible proof of correctness.
- OCR is optional but recommended for scanned PDFs; the code includes a fallback chain and user-visible cues when OCR is used.

---

For detailed walkthroughs, refer to `dashboard/app.py` line ranges mentioned above and to the notebooks in `notebooks/` for experiments and derivations.
