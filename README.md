# TF-IDF COMPLIANCE DRIFT DETECTION SYSTEM

**Professional compliance review system using manual TF-IDF implementation with supervised and unsupervised learning techniques.**

---

## 📋 Project Overview

This system monitors compliance drift by comparing internal organizational policies against regulatory guidelines using a custom TF-IDF vectorization engine. It identifies discrepancies, quantifies risk levels, and provides actionable audit trails through an interactive Streamlit dashboard.

**Key Features:**
- 🔧 Manual TF-IDF implementation with 20 algorithm variants (5 TF × 4 IDF combinations)
- 🤖 Supervised classification (Naive Bayes, Logistic Regression) with feature importance
- 📊 Unsupervised clustering (K-Means) with visualization and profiling
- 🛡️ Drift detection with 3-tier risk scoring (Safe/Warning/Critical)
- 📈 Interactive dashboard with real-time PDF processing
- 🔐 Robust error handling for edge cases and missing data

---

## 🏗️ Project Architecture

### Directory Structure

```
TFIDF-COMPLIANCE-DRIFT/
├── frontend/
│   └── app.py                          # Main Streamlit dashboard (modular UI)
├── backend/                             # UI-agnostic backend (TF-IDF, similarity, ML, PDF, validation)
├── dashboard/
│   └── app.py                          # Compatibility entrypoint (defaults to frontend; legacy via env var)
├── data/
│   ├── guidelines/                     # Reference legal documents
│   │   ├── Criminal_Law/
│   │   │   └── BNS_2023.txt
│   │   ├── Cyber_Crime/
│   │   │   └── IT_ACT_2021.txt
│   │   └── Financial_Law/
│   │       └── PMLA_2002.txt
│   ├── guidelines_pdfs/                # PDF versions of guidelines
│   │   ├── Criminal_Law/
│   │   ├── Cyber_Crime/
│   │   └── Financial_Law/
│   ├── internal/                       # Sample internal policies
│   │   ├── aml_customer_due_diligence_procedure_v1.txt
│   │   ├── criminal_case_intake_procedure_v1.txt
│   │   ├── criminal_incident_response_protocol_v1.txt
│   │   ├── cyber_data_access_control_policy_v1.txt
│   │   ├── financial_transaction_monitoring_policy_v1.txt
│   │   └── information_security_incident_response_plan_v1.txt
│   └── metadata.csv                    # Document metadata and category labels
├── notebooks/
│   ├── experiments.ipynb               # Mathematical derivations and tests
│   └── maths_derivations.ipynb         # TF-IDF variant analysis
├── results/
│   ├── drift_alerts.csv                # Compliance drift results
│   └── similarity_scores.csv           # Document similarity matrix
├── scripts/
│   └── pdf_to_txt_once.py              # Batch PDF text extraction utility
├── src/                                # Legacy/educational modules
│   ├── alerts.py                       # Drift detection & risk scoring logic
│   ├── drift.py                        # Core comparison engine
│   ├── manual_tfidf_math.py            # From-scratch TF-IDF implementation
│   ├── preprocess.py                   # Text preprocessing pipeline
│   ├── similarity.py                   # Cosine similarity calculations
│   ├── utils.py                        # Helper functions and utilities
│   └── vectorize.py                    # Adaptive vectorization engine
├── requirements.txt
├── README.md                           # This file
└── .gitignore
```

---

## 🔑 Key Components

### 1. Manual TF-IDF Engine (`src/manual_tfidf_math.py`)

**From-scratch implementation** with no scikit-learn shortcuts for maximum transparency and control.

**TF (Term Frequency) Variants:**
| Variant | Formula | Use Case |
|---------|---------|----------|
| Binary | 1 if count > 0 else 0 | Boolean retrieval systems |
| Raw Count | count | Short documents, frequency matters |
| **Normalized** ✅ | count / doc_length | **Default - prevents length bias** |
| Log Normalized | 1 + log(count) | Diminishing returns for repetition |
| Double Normalization | 0.5 + 0.5×(count/max_count) | Balances rare vs. common terms |

**IDF (Inverse Document Frequency) Variants:**
| Variant | Formula | Use Case |
|---------|---------|----------|
| Standard | log(N/DF) | Basic information theory |
| Smooth | log(N/DF) + 1 | Prevents zero division |
| **Sklearn Smooth** ✅ | log((1+N)/(1+DF)) + 1 | **Default - industry standard** |
| Probabilistic | log((N-DF)/DF) | Emphasizes discriminative terms |

**Implementation validates against sklearn's TfidfVectorizer** to ensure correctness across all 20 combinations.

---

### 2. Adaptive Vectorization (`src/vectorize.py`)

Intelligent parameter selection that scales with corpus characteristics:

- **Dynamic min_df/max_df:** Adjusts for corpus size (2-10,000 docs)
  - Small corpus (< 50 docs): min_df=1, max_df=1.0 (no filtering)
  - Medium corpus (50-500): min_df=2, max_df=0.8
  - Large corpus (> 500): min_df=5, max_df=0.95

- **Small dataset handling:** Falls back to unigrams + no stopwords when needed
- **Edge case resilience:**
  - Empty documents → replaced with placeholder tokens
  - Single-word documents → preserved without filtering
  - All-stopword documents → handled gracefully

---

### 3. Supervised Classification (`src/drift.py`)

Two-algorithm classification pipeline:

- **Algorithms:** Multinomial Naive Bayes (speed), Logistic Regression (accuracy)
- **Category filtering:** Auto-removes classes with < 2 samples (prevents training errors)
- **Feature importance:** Extracts top 10 predictive terms per category
- **Stratified splitting:** Uses stratified K-fold when possible; falls back to random 80/20
- **Performance metrics:** Accuracy, Precision, Recall, F1-score per category

---

### 4. Unsupervised Clustering (`src/drift.py`)

K-Means clustering with quality assessment:

- **K-range exploration:** Tests K=2 to min(10, sqrt(n_docs))
- **Quality metrics:**
  - Silhouette Score: [-1, 1], higher is better (separateness vs. cohesion)
  - Davies-Bouldin Index: [0, ∞], lower is better (ratio of within/between distances)
  - Calinski-Harabasz Score: Higher is better (density/separation tradeoff)

- **PCA visualization:** 2D projection of document space using 2 principal components
- **Cluster profiling:** Top 10 terms per cluster extracted from K-Means centroids

---

### 5. Compliance Drift Detection (`src/alerts.py`)

Automated risk quantification for internal policy vs. regulatory guidelines:

**Matching Logic:**
- Categories (Criminal_Law, Cyber_Crime, Financial_Law) auto-mapped to internal policies
- Cosine similarity scores computed between each internal doc and relevant guideline
- Scores aggregated to category-level and organization-level metrics

**Risk Scoring (3-Tier System):**
| Tier | Similarity Range | Status | Action |
|------|-----------------|--------|--------|
| Safe | > 80% | ✅ Compliant | No action needed |
| Warning | 60-80% | ⚠️ Partial alignment | Review and align |
| Critical | < 60% | ❌ Drift detected | Immediate remediation |

**PDF Reporting:**
- Automated audit trail generation with timestamps
- Side-by-side comparison tables
- Recommendations for policy updates

---

### 6. UI/UX Dashboard (`frontend/app.py`)

Six-tab Streamlit interface (modular frontend calling backend modules):

1. **Compliance Tab:** Upload internal docs → Compare vs. guidelines → View drift alerts
2. **Math Tab:** Explore TF-IDF variants, manual calculations, and theoretical foundations
3. **Classification Tab:** Train supervised models, view feature importance, inspect predictions
4. **Clustering Tab:** K-Means exploration, silhouette analysis, cluster membership
5. **Similarity Matrix:** Heatmap of document-to-document cosine similarities
6. **Visualizations:** PCA scatter plots, topic distributions, drift trend charts

**Sidebar Hyperparameter Controls:**
- `max_features`: TF-IDF feature space size (100-10000, default 5000)
- `min_df`: Minimum document frequency filter (1-10)
- `max_df`: Maximum document frequency filter (0.5-1.0)
- `ngram_range`: Unigram or bigram tokenization
- `k_clusters`: Number of K-Means clusters (2-10)

**Real-Time Processing:**
- Upload PDFs/TXT files → OCR with fallback chain → Vectorize → Analyze → Display results

**Legacy note:** `dashboard/app.py` is kept for compatibility. By default it launches the modular UI.
To force the legacy monolithic dashboard, set `TFIDF_USE_LEGACY_DASHBOARD=1`.

---

## 🧮 Mathematical Foundation

### Why TF-IDF Weighting?

**Problem with Raw Term Frequency:**
- Favors long documents (more words = higher counts)
- Common words dominate ("the", "and", "is")
- No discrimination between important vs. trivial terms

**Solution: TF-IDF Weighting**

$$\text{TF-IDF}(\text{term}, \text{doc}) = \text{TF}(\text{term}, \text{doc}) \times \text{IDF}(\text{term})$$

Where:
- **TF (Normalized):** $\frac{\text{count of term in document}}{\text{total terms in document}}$
- **IDF (Sklearn Smooth):** $\log\left(\frac{1+N}{1+\text{DF}}\right) + 1$

### Why Our Choice: Normalized TF × Sklearn IDF?

| Criterion | Benefit |
|-----------|---------|
| **Normalized TF** | Eliminates document length bias; proportional term importance |
| **Sklearn IDF** | Industry-standard smoothing; robust to corpus size changes; prevents extreme weights |
| **Combination** | Balanced approach: neither over-weights rare terms nor under-weights discriminative ones |

### Cosine Similarity

For two TF-IDF vectors $\vec{a}$ and $\vec{b}$:

$$\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| \times |\vec{b}|}$$

**Range:** [0, 1] where 1 = identical documents, 0 = no overlap

---

## 🛡️ Robust Error Handling

The system gracefully handles 10+ edge cases:

| Scenario | Handling Strategy |
|----------|-------------------|
| **Small datasets** (< 5 docs) | Adaptive min_df=1, max_df=1.0; no stopwords |
| **Imbalanced categories** | Auto-filters insufficient samples (< 2); shows warnings |
| **Empty clusters** | Silhouette score → NaN; displays "N/A" in UI |
| **OCR failures** | Pdfplumber → Tesseract → Plain text fallback |
| **Missing NLTK** | Auto-download in quiet mode; suppresses verbose output |
| **Single-word documents** | Preserved in vectorization; not filtered |
| **All-stopword documents** | Treated as valid (not discarded); placeholder tokens added |
| **Division by zero** | Sklearn IDF smoothing prevents inf/NaN |
| **Empty vocabulary** | Falls back to character n-grams |
| **Memory overflow** | max_features=5000 default prevents sparse matrix explosion |

---

## 📊 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Corpus Size** | 2-10,000 docs | Adaptive parameters scale automatically |
| **Processing Speed** | ~2-5s for 50 docs | Includes TF-IDF + clustering + classification |
| **Memory Usage** | ~150-500 MB | Depends on max_features (default 5000) |
| **Classification Accuracy** | 75-95% | Varies by category distribution and text clarity |
| **Clustering Quality** | Silhouette ≥ 0.5 | For well-separated document groups |
| **TF-IDF Sparsity** | ~95-99% | Typical for natural language data |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/rizwan-kh/tfidf-compliance-drift.git
cd tfidf-compliance-drift

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install OCR support
# Windows - Tesseract:
# Download: https://github.com/UB-Mannheim/tesseract/wiki
# Set environment variable: TESSERACT_PATH = C:\Program Files\Tesseract-OCR\tesseract.exe

# Windows - Poppler:
# Download: https://github.com/oschwartz10612/poppler-windows/releases
# Add to PATH or set: POPPLER_PATH = C:\path\to\poppler\bin
```

### Running the Dashboard

```bash
# From project root directory

# Recommended (modular frontend + backend)
streamlit run frontend/app.py

# Legacy monolithic app (kept for reference)
# streamlit run dashboard/app.py

# Dashboard opens at: http://localhost:8501
```

#### Windows helper scripts (avoids `python` vs `py` interpreter mismatch)

```powershell
# Recommended modular UI
./scripts/run_streamlit.ps1 -Target frontend

# Compatibility entrypoint (defaults to frontend)
./scripts/run_streamlit.ps1 -Target dashboard

# Force legacy monolithic dashboard
./scripts/run_streamlit.ps1 -Target legacy-dashboard

# Run on a different port/headless
./scripts/run_streamlit.ps1 -Target frontend -Port 8502 -Headless

# Run tests
./scripts/run_tests.ps1
```

### Basic Workflow

1. **Prepare Data:**
   - Place regulatory guidelines in `data/guidelines/{category}/`
   - Place internal policies in `data/internal/`
   - Create `data/metadata.csv` with document metadata (optional)

2. **Run Analysis:**
   - Open dashboard → Upload documents → Select analysis type
   - Choose TF-IDF variants and hyperparameters
   - Click "Analyze" to generate results

3. **Review Results:**
   - **Compliance Tab:** View drift alerts and risk scores
   - **Classification Tab:** Inspect category predictions
   - **Clustering Tab:** Explore document groupings
   - **Visualizations:** Analyze PCA projections and distributions

4. **Export Results:**
   - Download CSV reports: `results/drift_alerts.csv`, `results/similarity_scores.csv`
   - Generate PDF compliance audit trail (from UI)

---

## 🧪 Testing & Validation

### Run Unit Tests
```bash
pytest -q tests/
```

### Validate TF-IDF Implementation
```bash
python -c "from src.manual_tfidf_math import validate_against_sklearn; validate_against_sklearn()"
```

### Run Setup Validator (environment checks)
A quick environment and project-health check script is included at `setup_validate.py`. It verifies project structure, required packages, imports, and data layout.

```bash
# From project root
python setup_validate.py
```


### Experiment with Hyperparameters
- Open notebooks/experiments.ipynb in Jupyter
- Run hyperparameter sweeps and diagnostic analyses
- Review mathematical derivations in notebooks/maths_derivations.ipynb

---

## 📚 Dependencies

See requirements.txt for complete list. Key packages:

- **streamlit:** Interactive dashboard framework
- **scikit-learn:** ML algorithms (Naive Bayes, K-Means, PCA)
- **pandas:** Data manipulation
- **numpy:** Numerical computing
- **pdfplumber:** PDF text extraction (primary)
- **pytesseract:** OCR fallback for scanned PDFs
- **nltk:** Text preprocessing (tokenization, stopwords)
- **matplotlib, plotly:** Data visualization

---

## 🔍 Project Conventions

- **Code Location:** Production code in `src/`, notebooks in `notebooks/`, app in `dashboard/`
- **Data:** Store documents in `data/`, results in `results/`
- **Naming:** Snake_case for files/functions, SCREAMING_SNAKE_CASE for constants
- **Testing:** Use pytest with tests under `tests/`
- **Reproducibility:** Random seed set globally for consistent results across runs

---

## 📖 Documentation

- **Mathematical Details:** See notebooks/maths_derivations.ipynb for TF-IDF derivations and theoretical background
- **Experiments:** notebooks/experiments.ipynb contains hyperparameter sweeps and diagnostic analysis
- **Code Examples:** Each module in `src/` includes docstrings with usage examples

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- Additional TF-IDF variants (BM25, TF-IDF-cosine normalization)
- Deep learning baselines (BERT, LLM-based embeddings)
- Temporal drift tracking across policy versions
- PDF form extraction and structured data handling
- Multi-language support

---

## 📄 License

<--------------------------->

---

## 👤 Author

**Rizwan K A**

---

## ⭐ Acknowledgments

- Regulatory texts: Indian legal frameworks (BNS 2023, IT Act 2021, PMLA 2002)
- Framework inspiration: Industry best practices in compliance monitoring
- Dashboard design: Streamlit community examples

---

**Last Updated:** February 2026  
**Version:** 1.0.0
