# 🚀 TF-IDF Compliance Drift Monitoring System

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rizwan-k-a/tfidf-compliance-drift/main/compliance_monitor.py)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automated monitoring for **legal/medical document compliance** using **TF-IDF**, **cosine similarity**, and **drift detection**. Flags revisions drifting from regulatory references.

**RVCE MCA Portfolio Project** | Math + Data Science Dual Evaluation

## ✨ Features
- **TF-IDF Vectorization** (unigrams/bigrams)
- **Cosine Similarity** scoring vs reference docs
- **Drift Detection** (\( \Delta = s_t - s_{t-1} \))
- **Interactive Streamlit Dashboard**
- **Version Tracking** & visualizations

## 📊 Live Demo
[Launch App](https://tfidf-compliance-drift.streamlit.app) *(Deploy after push)*

## 🚀 Quick Start
```bash
git clone https://github.com/rizwan-k-a/TFIDF-Compliance-Drift.git
cd TFIDF-Compliance-Drift
pip install -r requirements.txt
streamlit run compliance_monitor.py
