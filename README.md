tfidf-compliance-drift

Purpose
- Detect compliance drift in internal documents vs regulatory/reference texts using TF-IDF and cosine similarity.

# Universal Compliance Review System

## Features
- PDF + OCR text extraction
- TF-IDF vectorization
- Cosine similarity analysis
- Risk assessment dashboard
- Mathematical transparency

## Installation
```bash
pip install -r requirements.txt
```

Quickstart

1. Create and activate a Python virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Streamlit dashboard (after running the analysis to generate `results/drift_alerts.csv`)

```powershell
streamlit run dashboard\app.py
```

Project layout
- `data/`: Stores public regulatory texts (guidelines, rules), sample internal documents (consent forms, SOPs), and version history used for drift analysis over time
- `src/`: Core Mathematics + Data Science logic; each module should be simple, explainable, and independent
- `dashboard/`: Streamlit-based visual monitoring system showing similarity scores, drift trends, and alerts
- `notebooks/`: `maths.ipynb` for mathematical explanations, formulas, derivations; `experiments.ipynb` for data science experiments and analysis
- `results/`: Output files such as similarity scores and drift alerts, used for demo and report submission

Next steps
- Replace placeholder text files in `data/` with real documents.
- Run analysis script (not yet implemented) or use the `src/` modules interactively.
