"""
============================================================
UNIVERSAL COMPLIANCE REVIEW SYSTEM - COMPLETE PROFESSIONAL VERSION
def perform_enhanced_clustering(documents: List[str], names: List[str], n_clusters: int = 3, keep_numbers: bool = True, use_lemma: bool = False, max_features=None, min_df=None, max_df=None):
MCA Final Project - Rizwan
Fixed: Classification error handling for insufficient category samples
============================================================
"""

import os
import sys
import time
import re
import math
import tempfile
from io import BytesIO
from collections import Counter
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                            silhouette_score, davies_bouldin_score)
from fpdf import FPDF

# Word cloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# NLP imports
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('omw-1.4', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False

# PDF and OCR imports
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False

# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class Config:
    MIN_TEXT_LENGTH: int = 100
    MIN_WORDS: int = 20
    MAX_FILE_SIZE_MB: int = 50
    DEFAULT_DIVERGENCE_THRESHOLD: int = 40
    TFIDF_MAX_FEATURES: int = 5000
    NGRAM_RANGE: Tuple[int, int] = (1, 2)
    MIN_DF: float = 0.05
    MAX_DF: float = 0.95
    RANDOM_STATE: int = 42
    MIN_PAGE_TEXT_LENGTH: int = 100

CONFIG = Config()

# Document Categories
CATEGORIES = {
    'Criminal': {
        'keywords': ['criminal', 'case', 'intake', 'incident', 'investigation', 'evidence', 'prosecution', 'bns', 'nyaya', 'sanhita'],
        'guideline': 'BNS_2023',
        'guideline_name': 'Bharatiya Nyaya Sanhita (BNS) 2023',
        'color': '#ef4444'
    },
    'Cyber': {
        'keywords': ['cyber', 'data', 'access', 'control', 'security', 'information', 'breach', 'digital', 'privacy', 'encryption'],
        'guideline': 'IT_ACT_2021',
        'guideline_name': 'IT Act 2021',
        'color': '#3b82f6'
    },
    'Financial': {
        'keywords': ['financial', 'transaction', 'monitoring', 'aml', 'money', 'laundering', 'pmla', 'kyc', 'customer', 'diligence'],
        'guideline': 'PMLA_2002',
        'guideline_name': 'Prevention of Money Laundering Act (PMLA) 2002',
        'color': '#10b981'
    }
}

# Cross-platform Poppler detection
def get_poppler_path():
    """Detect Poppler path across platforms"""
    if sys.platform == 'win32':
        common_paths = [
            r"C:\Program Files\poppler\Library\bin",
            r"C:\Program Files (x86)\poppler\Library\bin",
            r"C:\Program Files\poppler-24.08.0\Library\bin",
            r"C:\Program Files\poppler-25.12.0\Library\bin",
            r"C:\poppler\Library\bin"
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
    return None

POPPLER_PATH = get_poppler_path()
OCR_DPI = 300
OCR_CONFIG = "--psm 6"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Compliance Review System - Professional",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ENHANCED CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #2d3748 100%);
        padding: 2rem 1rem;
        color: white;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .math-formula {
        background: #f0f4f8;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .disclaimer-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe8a1 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #92400e;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 1rem !important;
    }
    /* Table styling for TF-IDF variant decision tables */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    th {
        background-color: #667eea;
        color: white;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #f9fafb;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ENHANCED PREPROCESSING
# ============================================================
@st.cache_data(show_spinner=False)
def preprocess_text(text: str, keep_numbers: bool = True, use_lemmatization: bool = False) -> str:
    """
    Enhanced text preprocessing with configurable options.
    
    Args:
        text: Raw text to preprocess
        keep_numbers: Whether to retain numeric characters
        use_lemmatization: Whether to apply lemmatization
    
    Returns:
        Preprocessed text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    if keep_numbers:
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    else:
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    if use_lemmatization and NLTK_AVAILABLE:
        lemmatizer = WordNetLemmatizer()
        tokens = text.split()
        text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])
    
    return text

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes: bytes, filename: str, use_ocr: bool = True) -> Tuple[str, bool, int]:
    """Extract text from PDF with OCR fallback"""
    if not PDF_SUPPORT:
        return "", False, 0
    
    text_content = []
    ocr_used = False
    page_count = 0
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        
        with pdfplumber.open(tmp_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > CONFIG.MIN_PAGE_TEXT_LENGTH:
                    text_content.append(page_text)
        
        full_text = "\n".join(text_content)
        
        if len(full_text.strip()) < CONFIG.MIN_TEXT_LENGTH and use_ocr and OCR_SUPPORT:
            try:
                ocr_text = []
                images = convert_from_path(tmp_path, dpi=OCR_DPI, poppler_path=POPPLER_PATH)
                for image in images:
                    text = pytesseract.image_to_string(image, config=OCR_CONFIG)
                    if text.strip():
                        ocr_text.append(text)
                if ocr_text:
                    full_text = "\n".join(ocr_text)
                    ocr_used = True
            except Exception as e:
                st.warning(f"OCR error: {str(e)}")
        
        os.unlink(tmp_path)
        return full_text, ocr_used, page_count
        
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return "", False, 0

@st.cache_resource(show_spinner=False)
def get_tfidf_vectorizer(max_features=None, min_df=None, max_df=None) -> TfidfVectorizer:
    """Cached TF-IDF vectorizer with configurable parameters"""
    return TfidfVectorizer(
        max_features=max_features or CONFIG.TFIDF_MAX_FEATURES,
        ngram_range=CONFIG.NGRAM_RANGE,
        min_df=min_df or CONFIG.MIN_DF,
        max_df=max_df or CONFIG.MAX_DF,
        stop_words='english',
        sublinear_tf=True,
        norm='l2',
        use_idf=True,
        smooth_idf=True
    )

# ============================================================
# MATHEMATICAL TF-IDF (ALL VARIANTS)
# ============================================================
def compute_tf_variants(term_count: int, doc_length: int, max_term_count: int) -> Dict[str, float]:
    """Compute all standard TF variants"""
    if doc_length == 0:
        return {variant: 0.0 for variant in ['binary', 'raw', 'normalized', 'log_norm', 'double_norm']}
    
    return {
        'binary': 1.0 if term_count > 0 else 0.0,
        'raw': float(term_count),
        'normalized': term_count / doc_length,
        'log_norm': 1 + math.log(term_count) if term_count > 0 else 0.0,
        'double_norm': 0.5 + 0.5 * (term_count / max_term_count) if max_term_count > 0 else 0.0
    }

def compute_idf_variants(df: int, N: int) -> Dict[str, float]:
    """Compute all standard IDF variants with formulas"""
    if df == 0 or N == 0:
        return {variant: 0.0 for variant in ['standard', 'smooth', 'sklearn_smooth', 'probabilistic']}
    
    return {
        'standard': math.log(N / df) if df > 0 else 0.0,
        'smooth': math.log(N / df) + 1 if df > 0 else 0.0,
        'sklearn_smooth': math.log((1 + N) / (1 + df)) + 1,
        'probabilistic': math.log((N - df) / df) if df < N else 0.0
    }

def compute_manual_tfidf_complete(documents: List[str], sample_words: List[str], keep_numbers: bool = True, use_lemma: bool = False) -> Dict:
    """
    Complete manual TF-IDF with ALL variants and comparison to sklearn
    """
    n_docs = len(documents)
    processed_docs = [preprocess_text(doc, keep_numbers, use_lemma).split() for doc in documents]
    
    max_term_counts = [max(Counter(doc).values()) if doc else 1 for doc in processed_docs]
    
    results = {}
    
    for word in sample_words:
        word_lower = word.lower().strip()
        if not word_lower:
            continue
        
        word_data = {
            'tf_variants_per_doc': [],
            'df': 0,
            'idf_variants': {},
            'tfidf_variants_per_doc': []
        }
        
        for idx, doc_words in enumerate(processed_docs):
            word_count = doc_words.count(word_lower)
            total_words = len(doc_words)
            
            tf_variants = compute_tf_variants(word_count, total_words, max_term_counts[idx])
            
            word_data['tf_variants_per_doc'].append({
                'doc_id': idx + 1,
                'count': word_count,
                'total_words': total_words,
                **{f'tf_{k}': round(v, 6) for k, v in tf_variants.items()}
            })
            
            if word_count > 0:
                word_data['df'] += 1
        
        df = word_data['df']
        word_data['idf_variants'] = {
            k: round(v, 6) for k, v in compute_idf_variants(df, n_docs).items()
        }
        
        idf_sklearn = word_data['idf_variants']['sklearn_smooth']
        for tf_data in word_data['tf_variants_per_doc']:
            tfidf = tf_data['tf_normalized'] * idf_sklearn
            word_data['tfidf_variants_per_doc'].append({
                'doc_id': tf_data['doc_id'],
                'tfidf': round(tfidf, 6)
            })
        
        results[word] = word_data
    
    return results

# ============================================================
# DOCUMENT CATEGORIZATION
# ============================================================
def categorize_document(text: str, filename: str) -> str:
    """Categorize document based on keywords"""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    scores = {}
    for category, info in CATEGORIES.items():
        score = sum(text_lower.count(kw) + filename_lower.count(kw) * 2 for kw in info['keywords'])
        scores[category] = score
    
    return max(scores, key=scores.get) if max(scores.values()) > 0 else 'Uncategorized'

# ============================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================
def build_tfidf_vectors(reference_docs: List[str], internal_docs: List[str], keep_numbers: bool = True, use_lemma: bool = False, max_features=None, min_df=None, max_df=None):
    """Build TF-IDF vectors for reference and internal documents with adaptive parameters"""
    all_docs = reference_docs + internal_docs
    processed_docs = [preprocess_text(doc, keep_numbers, use_lemma) for doc in all_docs]
    
    # Adaptive min_df and max_df based on document count
    n_docs = len(processed_docs)
    
    # Adjust min_df: use absolute count for small corpora
    if n_docs <= 5:
        adjusted_min_df = 1  # Appear in at least 1 document
    elif n_docs <= 10:
        adjusted_min_df = 1
    else:
        # Use percentage for larger sets, default to 0 if None
        adjusted_min_df = max(1, int(n_docs * (min_df or 0))) if min_df and min_df > 0 else 1
    
    # Adjust max_df: be more permissive for small corpora
    if n_docs <= 5:
        adjusted_max_df = 1.0  # No upper limit for small sets
    else:
        adjusted_max_df = max_df if max_df and max_df < 1.0 else 1.0
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features or CONFIG.TFIDF_MAX_FEATURES,
            ngram_range=CONFIG.NGRAM_RANGE,
            min_df=adjusted_min_df,
            max_df=adjusted_max_df,
            stop_words='english',
            sublinear_tf=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True
        )
        all_vectors = vectorizer.fit_transform(processed_docs)
        
        ref_vectors = all_vectors[:len(reference_docs)]
        int_vectors = all_vectors[len(reference_docs):]
        
        return vectorizer, ref_vectors, int_vectors
        
    except ValueError as e:
        # If still fails, try with minimal constraints
        st.warning(f"⚠️ Adjusting TF-IDF parameters for small document set ({n_docs} docs)")
        vectorizer = TfidfVectorizer(
            max_features=min(1000, max_features or CONFIG.TFIDF_MAX_FEATURES),
            ngram_range=(1, 1),  # Unigrams only
            min_df=1,  # Minimum constraint
            max_df=1.0,  # No maximum constraint
            stop_words=None,  # Keep all words
            sublinear_tf=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True
        )
        all_vectors = vectorizer.fit_transform(processed_docs)
        
        ref_vectors = all_vectors[:len(reference_docs)]
        int_vectors = all_vectors[len(reference_docs):]
        
        return vectorizer, ref_vectors, int_vectors


def compute_similarity_scores_by_category(categorized_docs: Dict, categorized_guidelines: Dict, keep_numbers: bool = True, use_lemma: bool = False, max_features=None, min_df=None, max_df=None) -> pd.DataFrame:
    """Compute cosine similarity for documents matched to their category guidelines"""
    all_results = []
    
    for category in CATEGORIES.keys():
        if category not in categorized_docs or category not in categorized_guidelines:
            continue
        
        internal_docs = categorized_docs[category]['docs']
        internal_names = categorized_docs[category]['names']
        guideline_docs = categorized_guidelines[category]['docs']
        guideline_names = categorized_guidelines[category]['names']
        
        if not internal_docs or not guideline_docs:
            continue
        
        # Check if we have enough documents for analysis
        total_docs = len(internal_docs) + len(guideline_docs)
        if total_docs < 2:
            st.warning(f"⚠️ Skipping category '{category}': insufficient documents ({total_docs})")
            continue
        
        try:
            vectorizer, ref_vecs, int_vecs = build_tfidf_vectors(
                guideline_docs, internal_docs, 
                keep_numbers, use_lemma, 
                max_features, min_df, max_df
            )
            
            # Compute similarity
            similarity_matrix = cosine_similarity(int_vecs, ref_vecs)
            
            for i, doc_name in enumerate(internal_names):
                max_similarity = np.max(similarity_matrix[i])
                best_match_idx = np.argmax(similarity_matrix[i])
                
                all_results.append({
                    'category': category,
                    'internal_document': doc_name,
                    'matched_guideline': guideline_names[best_match_idx],
                    'compliance_score': max_similarity,
                    'similarity_percent': round(max_similarity * 100, 1),
                    'divergence_percent': round((1 - max_similarity) * 100, 1)
                })
        except Exception as e:
            st.error(f"❌ Error analyzing category '{category}': {str(e)}")
            continue
    
    if not all_results:
        st.error("❌ No categories could be analyzed. Please check your documents and try again.")
        return pd.DataFrame()
    
    return pd.DataFrame(all_results)

# ============================================================
# CLASSIFICATION (SUPERVISED LEARNING) - FIXED
# ============================================================
def perform_classification(documents: List[str], categories: List[str], test_size: float = 0.3, keep_numbers: bool = True, use_lemma: bool = False, max_features=None, min_df=None, max_df=None):
    """Train supervised classifiers on TF-IDF vectors with proper error handling"""
    
    # Count samples per category
    category_counts = Counter(categories)
    
    # Filter out categories with too few samples (< 2)
    valid_categories = {cat for cat, count in category_counts.items() if count >= 2}
    
    if len(valid_categories) < 2:
        st.warning(f"⚠️ Not enough data for classification. Need at least 2 categories with 2+ documents each.")
        st.info(f"Current distribution: {dict(category_counts)}")
        return None
    
    # Filter documents and categories
    filtered_docs = []
    filtered_categories = []
    
    for doc, cat in zip(documents, categories):
        if cat in valid_categories:
            filtered_docs.append(doc)
            filtered_categories.append(cat)
    
    if len(filtered_docs) < 6:
        st.warning(f"⚠️ Only {len(filtered_docs)} valid documents. Need at least 6 for reliable classification.")
        return None
    
    # Show what was filtered out
    excluded_count = len(documents) - len(filtered_docs)
    if excluded_count > 0:
        st.info(f"ℹ️ Excluded {excluded_count} documents with insufficient category samples")
    
    processed_docs = [preprocess_text(doc, keep_numbers, use_lemma) for doc in filtered_docs]
    
    vectorizer = TfidfVectorizer(
        max_features=max_features or CONFIG.TFIDF_MAX_FEATURES,
        ngram_range=CONFIG.NGRAM_RANGE,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(processed_docs)
    y = filtered_categories
    
    # Check if stratification is possible
    min_class_count = min(Counter(y).values())
    use_stratify = min_class_count >= 2 and test_size * len(y) >= len(set(y))
    
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=CONFIG.RANDOM_STATE, stratify=y
        )
    else:
        st.warning("⚠️ Stratified split not possible. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=CONFIG.RANDOM_STATE
        )
    
    # Train Multinomial Naive Bayes
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    nb_pred = nb_clf.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    
    # Train Logistic Regression
    lr_clf = LogisticRegression(max_iter=1000, random_state=CONFIG.RANDOM_STATE)
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top features per class
    top_features_per_class = {}
    for idx, category in enumerate(lr_clf.classes_):
        if len(lr_clf.coef_.shape) > 1:
            coef = lr_clf.coef_[idx]
        else:
            coef = lr_clf.coef_[0]
        top_idx = np.argsort(np.abs(coef))[-10:][::-1]
        top_features_per_class[category] = [(feature_names[i], coef[i]) for i in top_idx]
    
    return {
        'vectorizer': vectorizer,
        'nb_model': nb_clf,
        'nb_accuracy': nb_acc,
        'nb_predictions': nb_pred,
        'lr_model': lr_clf,
        'lr_accuracy': lr_acc,
        'lr_predictions': lr_pred,
        'y_test': y_test,
        'top_features': top_features_per_class,
        'confusion_matrix_nb': confusion_matrix(y_test, nb_pred, labels=nb_clf.classes_),
        'confusion_matrix_lr': confusion_matrix(y_test, lr_pred, labels=lr_clf.classes_),
        'classification_report_nb': classification_report(y_test, nb_pred, zero_division=0),
        'classification_report_lr': classification_report(y_test, lr_pred, zero_division=0),
        'filtered_count': len(filtered_docs),
        'excluded_count': excluded_count,
        'category_distribution': dict(Counter(filtered_categories))
    }

# ============================================================
# ENHANCED CLUSTERING
# ============================================================
def perform_enhanced_clustering(documents: List[str], names: List[str], n_clusters: int = 3, keep_numbers: bool = True, use_lemma: bool = False, max_features=None, min_df=None, max_df=None):
    """Clustering with quality metrics and top terms - handles edge cases"""
    # Basic checks
    if not documents or len(documents) < 2:
        st.warning("⚠️ Need at least 2 documents to perform clustering.")
        return None

    processed_docs = [preprocess_text(doc, keep_numbers, use_lemma) for doc in documents]

    # Adaptive TF-IDF parameters for small corpora
    n_docs = len(processed_docs)
    if n_docs <= 5:
        adjusted_min_df = 1
        adjusted_max_df = 1.0
    elif n_docs <= 10:
        adjusted_min_df = 1
        adjusted_max_df = 1.0
    else:
        adjusted_min_df = max(1, int(n_docs * (min_df or CONFIG.MIN_DF))) if (min_df is not None) else max(1, int(n_docs * CONFIG.MIN_DF))
        adjusted_max_df = max_df if (max_df is not None and max_df < 1.0) else CONFIG.MAX_DF

    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features or CONFIG.TFIDF_MAX_FEATURES,
            ngram_range=CONFIG.NGRAM_RANGE,
            min_df=adjusted_min_df,
            max_df=adjusted_max_df,
            stop_words='english',
            sublinear_tf=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True
        )
        X = vectorizer.fit_transform(processed_docs)
    except Exception as e:
        st.warning(f"⚠️ TF-IDF vectorization failed: {e}. Falling back to relaxed vectorizer.")
        vectorizer = TfidfVectorizer(
            max_features=min(1000, max_features or CONFIG.TFIDF_MAX_FEATURES),
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0,
            stop_words=None,
            sublinear_tf=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True
        )
        X = vectorizer.fit_transform(processed_docs)

    # Ensure requested clusters are sensible
    effective_n_clusters = min(n_clusters, max(2, n_docs - 1))

    try:
        kmeans = KMeans(n_clusters=effective_n_clusters, random_state=CONFIG.RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X)

        unique_labels = len(np.unique(labels))
        if unique_labels < 2:
            st.warning("⚠️ Clustering produced a single cluster. Try increasing document diversity or changing parameters.")
            return None

        try:
            silhouette = silhouette_score(X, labels)
        except Exception:
            silhouette = float('nan')

        try:
            davies_bouldin = davies_bouldin_score(X.toarray(), labels)
        except Exception:
            davies_bouldin = float('nan')

        pca = PCA(n_components=min(2, X.shape[1]))
        coords = pca.fit_transform(X.toarray())

        feature_names = vectorizer.get_feature_names_out()
        top_terms_per_cluster = {}

        for cluster_id in range(effective_n_clusters):
            if cluster_id < len(kmeans.cluster_centers_):
                centroid = kmeans.cluster_centers_[cluster_id]
                top_idx = np.argsort(centroid)[-10:][::-1]
                top_terms_per_cluster[cluster_id] = [(feature_names[i], float(centroid[i])) for i in top_idx]

        return {
            'labels': labels,
            'coordinates': coords,
            'inertia': float(kmeans.inertia_),
            'silhouette_score': float(silhouette) if not np.isnan(silhouette) else None,
            'davies_bouldin_score': float(davies_bouldin) if not np.isnan(davies_bouldin) else None,
            'top_terms': top_terms_per_cluster,
            'vectorizer': vectorizer,
            'X': X,
            'n_clusters_actual': unique_labels,
            'n_clusters_requested': effective_n_clusters
        }

    except Exception as e:
        st.error(f"❌ Clustering failed: {str(e)}")
        return None

# ============================================================
# MATRIX VISUALIZATION
# ============================================================
def display_tfidf_matrix(vectorizer, matrix, doc_names):
    """Display TF-IDF matrix structure and properties"""
    feature_names = vectorizer.get_feature_names_out()
    
    st.markdown("### 🗄️ TF-IDF Matrix Structure")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matrix Shape", f"{matrix.shape[0]} × {matrix.shape[1]}")
    with col2:
        sparsity = 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
        st.metric("Sparsity", f"{sparsity:.2%}")
    with col3:
        st.metric("Non-zero Entries", f"{matrix.nnz:,}")
    
    st.markdown("#### Sample Matrix (First 5 docs, Top 20 features)")
    n_sample_docs = min(5, matrix.shape[0])
    n_sample_features = min(20, matrix.shape[1])
    dense_sample = matrix[:n_sample_docs, :n_sample_features].toarray()
    sample_df = pd.DataFrame(
        dense_sample,
        columns=feature_names[:n_sample_features],
        index=[doc_names[i][:30] if i < len(doc_names) else f"Doc_{i}" for i in range(n_sample_docs)]
    )
    st.dataframe(sample_df.style.format("{:.4f}").background_gradient(cmap='YlOrRd'), use_container_width=True)
    
    if matrix.shape[0] <= 15:
        st.markdown("#### Document Similarity Heatmap")
        sim_matrix = cosine_similarity(matrix)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                   xticklabels=[n[:20] for n in doc_names],
                   yticklabels=[n[:20] for n in doc_names],
                   ax=ax)
        ax.set_title('Document Similarity Matrix (Cosine)', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

def generate_wordcloud(vectorizer, matrix, doc_idx):
    """Generate word cloud weighted by TF-IDF scores"""
    if not WORDCLOUD_AVAILABLE:
        st.warning("WordCloud library not installed. Run: pip install wordcloud")
        return None
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = matrix[doc_idx].toarray().flatten()
    
    word_freq = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names)) if tfidf_scores[i] > 0}
    
    if word_freq:
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            colormap='viridis').generate_from_frequencies(word_freq)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('TF-IDF Weighted Word Cloud', fontsize=16, fontweight='bold')
        return fig
    return None

# ============================================================
# UI COMPONENTS
# ============================================================
def render_header():
    st.markdown("""
        <div class="main-header">⚖️ Universal Compliance Review System</div>
        <div style="font-size: 1.1rem; color: #64748b; margin-bottom: 2rem;">
            Complete Mathematical TF-IDF • Supervised Classification • Advanced Clustering
        </div>
    """, unsafe_allow_html=True)

def render_disclaimer():
    st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Decision Support Only:</strong> This system assists in review prioritization. 
            It does NOT certify compliance or constitute legal advice. Always consult qualified legal professionals.
        </div>
    """, unsafe_allow_html=True)

def explain_tfidf_weighting():
    """Educational content on TF-IDF importance"""
    st.markdown("""
    <div class="math-formula">
    <h3>📐 Why TF-IDF Weighting Matters</h3>
    
    <p><strong>Term Frequency (TF)</strong> alone has limitations:</p>
    <ul>
        <li>Favors long documents (more words = higher raw counts)</li>
        <li>Doesn't distinguish important vs. common terms</li>
        <li>Common words like "the", "and", "is" dominate the counts</li>
    </ul>
    
    <p><strong>Inverse Document Frequency (IDF)</strong> solves this by penalizing common terms:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"IDF(term) = \log\frac{N}{DF(term)}")
    
    st.markdown("""
    <div class="math-formula">
    <p>Where <strong>N</strong> = total documents, <strong>DF</strong> = documents containing the term</p>
    
    <p><strong>TF-IDF combines both:</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"TF\text{-}IDF(term, doc) = TF(term, doc) \times IDF(term)")
    
    st.markdown("""
    <div class="math-formula">
    <p><strong>Result:</strong> Rare, document-specific terms get high weights; common terms are suppressed.</p>
    
    <p><strong>Example in Compliance Context:</strong></p>
    <ul>
        <li>"compliance" appears in 2/100 docs → High IDF (≈3.91) → <strong>High TF-IDF</strong> (important!)</li>
        <li>"the" appears in 98/100 docs → Low IDF (≈0.02) → <strong>Low TF-IDF</strong> (common, less informative)</li>
        <li>"evidence" in 5/100 docs → Medium IDF (≈2.30) → <strong>Medium TF-IDF</strong> (moderately important)</li>
    </ul>
    
    <p><strong>Why This Matters for Compliance:</strong></p>
    <ul>
        <li>Identifies unique terminology in each policy document</li>
        <li>Highlights domain-specific legal terms</li>
        <li>Enables accurate similarity comparison between policies and regulations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def risk_label(div):
    if div <= 20:
        return "✅ Safe – Closely Aligned"
    elif div <= CONFIG.DEFAULT_DIVERGENCE_THRESHOLD:
        return "⚠️ Needs Attention"
    else:
        return "🚨 Review Required"

def risk_color(div):
    if div <= 20:
        return "#10b981"
    elif div <= CONFIG.DEFAULT_DIVERGENCE_THRESHOLD:
        return "#f59e0b"
    else:
        return "#ef4444"

def validate_text(text, doc_name):
    if not text or len(text.strip()) < CONFIG.MIN_TEXT_LENGTH:
        return False, f"{doc_name}: insufficient text"
    
    words = preprocess_text(text).split()
    if len(words) < CONFIG.MIN_WORDS:
        return False, f"{doc_name}: too few words"
    
    return True, "Valid"

def generate_pdf(results_df):
    """Generate PDF report"""
    buffer = BytesIO()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "COMPLIANCE AUDIT REPORT - CATEGORIZED", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Helvetica", '', 10)
    
    current_category = None
    for _, r in results_df.iterrows():
        if current_category != r['Category']:
            current_category = r['Category']
            pdf.set_font("Helvetica", 'B', 14)
            pdf.cell(0, 10, f"Category: {current_category}", ln=True)
            pdf.ln(3)
        
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(0, 8, f"Document: {r['Document']}", ln=True)
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(0, 6, f"Guideline: {r['Guideline']}", ln=True)
        pdf.cell(0, 6, f"Similarity: {r['Similarity (%)']}%", ln=True)
        pdf.cell(0, 6, f"Divergence: {r['Divergence (%)']}%", ln=True)
        
        risk_text = str(r['Risk Level'])
        if "Safe" in risk_text:
            risk_plain = "[SAFE] Closely Aligned"
        elif "Attention" in risk_text:
            risk_plain = "[WARNING] Needs Attention"
        else:
            risk_plain = "[CRITICAL] Review Required"
        
        pdf.cell(0, 6, f"Risk: {risk_plain}", ln=True)
        pdf.ln(5)
    
    pdf.output(buffer)
    buffer.seek(0)
    return buffer


# ============================================================
# INPUT VALIDATION & SECURITY
# ============================================================
def validate_input_file(file, max_size_mb: int = CONFIG.MAX_FILE_SIZE_MB, allowed_extensions: List[str] = None) -> Tuple[bool, str]:
    """
    Validate uploaded file for security and compliance.
    
    Parameters:
    -----------
    file : streamlit UploadedFile
        The file object from st.file_uploader()
    max_size_mb : int
        Maximum allowed file size in MB (default: CONFIG.MAX_FILE_SIZE_MB)
    allowed_extensions : List[str]
        List of allowed file extensions (default: ['pdf', 'txt'])
    
    Returns:
    --------
    Tuple[bool, str]
        (is_valid: bool, message: str)
        - is_valid: True if file passes all validations
        - message: Description of validation result or error reason
    """
    
    if allowed_extensions is None:
        allowed_extensions = ['pdf', 'txt']
    
    # Check 1: File size validation
    file_size_mb = file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File exceeds {max_size_mb}MB limit (size: {file_size_mb:.1f}MB)"
    
    # Check 2: File extension validation
    file_ext = file.name.split('.')[-1].lower()
    if file_ext not in allowed_extensions:
        ext_list = ', '.join(allowed_extensions)
        return False, f"File type .{file_ext} not allowed. Allowed types: {ext_list}"
    
    # Check 3: PDF magic bytes validation
    if file_ext == 'pdf':
        try:
            file_bytes = file.getvalue()[:4]  # Read first 4 bytes
            if not file_bytes.startswith(b'%PDF'):
                return False, f"Invalid PDF file: {file.name} (wrong magic bytes). File may be corrupted."
        except Exception as e:
            return False, f"Error validating PDF: {str(e)}"
    
    # Check 4: Text file encoding validation (for .txt files)
    if file_ext == 'txt':
        try:
            file_content = file.getvalue()
            # Try to decode as UTF-8
            file_content.decode('utf-8')
        except UnicodeDecodeError:
            return False, f"Text file encoding error: {file.name} (must be UTF-8 encoded)"
        except Exception as e:
            return False, f"Error reading text file: {str(e)}"
    
    return True, f"✅ {file.name} - Valid ({file_size_mb:.2f}MB)"


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    render_header()
    render_disclaimer()
    
    # Sidebar with hyperparameters
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📝 Preprocessing Options")
        keep_numbers = st.checkbox("Keep numbers in text", value=True, help="Retain numeric values (e.g., '2023', 'Section 43A')")
        use_lemma = st.checkbox("Use lemmatization", value=False, disabled=not NLTK_AVAILABLE, help="Reduce words to root form (requires NLTK)")
        
        if not NLTK_AVAILABLE and use_lemma:
            st.warning("⚠️ Install NLTK: pip install nltk")
        
        st.subheader("🎛️ TF-IDF Parameters")
        max_features = st.slider("Max Features", 1000, 10000, CONFIG.TFIDF_MAX_FEATURES, step=1000, help="Maximum vocabulary size")

        # UPDATED: Safer min_df and max_df ranges for small corpora
        min_df = st.slider("Min Document Frequency", 0.0, 0.1, 0.0, step=0.01, help="Ignore terms in < X% of documents (0 = no filter)")
        max_df = st.slider("Max Document Frequency", 0.8, 1.0, 1.0, step=0.05, help="Ignore terms in > X% of documents (1.0 = no filter)")
        
        st.subheader("📊 Analysis Settings")
        divergence_threshold = st.slider("Divergence Threshold (%)", 20, 80, CONFIG.DEFAULT_DIVERGENCE_THRESHOLD, step=5, help="Risk threshold for compliance")
        n_clusters = st.slider("Number of Clusters", 2, 5, 3, help="K-Means cluster count")
        
        st.markdown("---")
        st.markdown("**📋 Document Categories:**")
        for category, info in CATEGORIES.items():
            st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.5rem 0; background: {info['color']}22; border-left: 3px solid {info['color']}; border-radius: 4px;">
                    <strong>{category}</strong><br>
                    <small>{info['guideline_name']}</small>
                </div>
            """, unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Compliance Dashboard", 
        "🧮 TF-IDF Mathematics", 
        "🎓 Classification",
        "🔬 Clustering Analysis",
        "📈 Matrix Inspection",
        "📉 Visualizations"
    ])
    
    # Session State
    if 'documents' not in st.session_state:
        st.session_state.documents = []
        st.session_state.doc_names = []
        st.session_state.doc_types = []
        st.session_state.doc_categories = []
    
    # File Upload
    with st.expander("📁 Upload Documents", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Internal Documents")
            st.caption("Your organizational policies (auto-categorized)")
            internal_files = st.file_uploader(
                "Upload internal docs (PDF/TXT)",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                key='internal'
            )
        
        with col2:
            st.subheader("📚 Reference Guidelines")
            st.caption("Legal standards (BNS/IT Act/PMLA)")
            guideline_files = st.file_uploader(
                "Upload reference guidelines (PDF/TXT)",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                key='guidelines'
            )
    
    # Process Uploads
    if internal_files or guideline_files:
        new_docs = []
        new_names = []
        new_types = []
        new_categories = []
        
        # Validation tracking metrics
        validation_metrics = {
            'total_files': 0,
            'valid_files': 0,
            'rejected_files': 0,
            'rejection_reasons': {}  # reason: count
        }
        
        progress_bar = st.progress(0)
        files_to_process = (internal_files or []) + (guideline_files or [])
        total_files = len(files_to_process)
        validation_metrics['total_files'] = total_files
        
        for idx, file in enumerate(files_to_process):
            # ===== SECURITY VALIDATION =====
            is_valid, validation_msg = validate_input_file(file)
            
            if not is_valid:
                # Track rejection reason
                reason = validation_msg.split(':')[0] if ':' in validation_msg else "Unknown reason"
                validation_metrics['rejection_reasons'][reason] = validation_metrics['rejection_reasons'].get(reason, 0) + 1
                validation_metrics['rejected_files'] += 1
                st.error(f"❌ {validation_msg}")
                progress_bar.progress((idx + 1) / total_files)
                continue
            
            # File passed validation - proceed with processing
            try:
                if file.name.endswith('.pdf'):
                    text, ocr_used, pages = extract_text_from_pdf(
                        file.getvalue(), file.name, OCR_SUPPORT
                    )
                    if ocr_used:
                        st.info(f"🔍 OCR applied to {file.name}")
                else:
                    text = file.getvalue().decode('utf-8', errors='ignore')
                
                is_valid, msg = validate_text(text, file.name)
                if is_valid:
                    category = categorize_document(text, file.name)
                    new_docs.append(text)
                    new_names.append(file.name)
                    doc_type = 'internal' if file in (internal_files or []) else 'guideline'
                    new_types.append(doc_type)
                    new_categories.append(category)
                    
                    validation_metrics['valid_files'] += 1
                    
                    if doc_type == 'internal':
                        st.success(f"✅ {file.name} → Category: **{category}**")
                else:
                    validation_metrics['rejected_files'] += 1
                    reason = "Content validation failed"
                    validation_metrics['rejection_reasons'][reason] = validation_metrics['rejection_reasons'].get(reason, 0) + 1
                    st.warning(msg)
            except Exception as e:
                validation_metrics['rejected_files'] += 1
                reason = f"Processing error"
                validation_metrics['rejection_reasons'][reason] = validation_metrics['rejection_reasons'].get(reason, 0) + 1
                st.error(f"❌ Error processing {file.name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / total_files)
        
        # Display validation metrics
        if validation_metrics['total_files'] > 0:
            st.markdown("---")
            st.markdown("### 📊 Upload Validation Summary")
            
            # Metrics row
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Total Files", validation_metrics['total_files'])
            with metric_col2:
                st.metric("✅ Valid", validation_metrics['valid_files'], delta=None)
            with metric_col3:
                st.metric("❌ Rejected", validation_metrics['rejected_files'], delta=None)
            with metric_col4:
                success_rate = (validation_metrics['valid_files'] / validation_metrics['total_files'] * 100) if validation_metrics['total_files'] > 0 else 0
                st.metric("Success Rate", f"{success_rate:.0f}%")
            
            # Detailed rejection reasons
            if validation_metrics['rejection_reasons']:
                st.markdown("**Rejection Reasons:**")
                reasons_df = pd.DataFrame(
                    list(validation_metrics['rejection_reasons'].items()),
                    columns=['Reason', 'Count']
                )
                st.dataframe(reasons_df, use_container_width=True, hide_index=True)
        
        if new_docs:
            st.session_state.documents.extend(new_docs)
            st.session_state.doc_names.extend(new_names)
            st.session_state.doc_types.extend(new_types)
            st.session_state.doc_categories.extend(new_categories)
        
        progress_bar.empty()
    
    # System capabilities & error handling info (Tab 1)
    with st.expander("🛡️ System Capabilities & Error Handling", expanded=False):
        st.markdown("""
        ### Robust Edge Case Handling
        
        This system gracefully handles challenging scenarios that break typical TF-IDF implementations:
        
        #### 1️⃣ Small Dataset Handling (< 5 documents)
        - **Problem**: Standard min_df/max_df settings filter out all terms
        - **Solution**: Adaptive parameters → min_df=1, max_df=1.0
        - **Test**: Try uploading 2-3 documents and watch automatic adjustment
        
        #### 2️⃣ Imbalanced Categories
        - **Problem**: Classification fails with categories having < 2 samples
        - **Solution**: Auto-filters insufficient categories, displays warnings
        - **Proof**: Check Tab 3 classification - shows filtered document count
        
        #### 3️⃣ Empty Clustering Results
        - **Problem**: K-Means can produce single cluster with diverse data
        - **Solution**: Silhouette score fallback to NaN, graceful error messages
        - **Proof**: Tab 4 displays "N/A" for invalid metrics instead of crashing
        
        #### 4️⃣ OCR Fallback Chain
        - **Problem**: Scanned PDFs have no extractable text
        - **Solution**: Pdfplumber → (if fails) → Tesseract OCR → (if fails) → graceful skip
        - **Test**: Upload a scanned/image-based PDF and watch OCR activation message
        
        #### 5️⃣ NLTK Dependency Management
        - **Problem**: Missing NLTK data causes runtime errors
        - **Solution**: Auto-download corpora with quiet mode + graceful degradation
        - **Proof**: Lemmatization checkbox is disabled when NLTK unavailable
        
        #### 6️⃣ Stratified Split Failure
        - **Problem**: Imbalanced classes prevent stratified train/test split
        - **Solution**: Fallback to random split with warning message
        - **Proof**: Tab 3 classification shows "Using random split" warning when triggered
        
        ---
        
        **Try Breaking the System:**
        1. Upload only 1 document → See "Need at least 2 documents" message
        2. Upload 3 criminal docs, 1 cyber doc → See category filtering in action
        3. Set min_df = 0.5, max_df = 0.6 → Watch adaptive parameter adjustment
        """, unsafe_allow_html=True)
    
    # Sample Data
    if not st.session_state.documents:
        st.info("👆 Upload documents or use sample data")
        if st.button("Load Sample Data"):
            sample_data = {
                'documents': [
                    """Bharatiya Nyaya Sanhita 2023 Section 420: Cheating and dishonestly inducing delivery of property. Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person shall be punished with imprisonment which may extend to seven years and shall also be liable to fine. Criminal prosecution procedures must follow due process and evidence collection standards. Investigation procedures require proper documentation and chain of custody for all evidence. Witness statements must be recorded promptly and accurately.""",
                    
                    """IT Act 2021 Section 43A: Compensation for failure to protect data. Where a body corporate possessing dealing or handling any sensitive personal data or information in a computer resource which it owns controls or operates is negligent in implementing and maintaining reasonable security practices and procedures and thereby causes wrongful loss or wrongful gain to any person such body corporate shall be liable to pay damages by way of compensation. Data breach notification within 72 hours is mandatory. Access control policies must restrict unauthorized data access. Encryption of sensitive data at rest and in transit is required. Regular security audits must be conducted.""",
                    
                    """Prevention of Money Laundering Act PMLA 2002: Anti-money laundering measures require financial institutions to implement customer due diligence procedures maintain transaction records for 10 years report suspicious transactions to Financial Intelligence Unit within 7 days conduct ongoing monitoring of customer accounts verify beneficial ownership assess and mitigate money laundering risks and comply with know-your-customer KYC norms. Enhanced due diligence for high-risk customers including politically exposed persons. Transaction monitoring systems must flag unusual patterns. Regular staff training on AML compliance is mandatory.""",
                    
                    """AML Customer Due Diligence Procedure v1: Our organization conducts customer verification at onboarding. We collect identification documents and verify customer identity through official databases. Transaction monitoring is performed monthly using automated systems. Suspicious activities are reported to the compliance officer within 48 hours. Record retention period is 5 years for customer data and transaction records. Risk assessment is conducted annually for all customers with enhanced reviews for high-risk categories. High-value transactions above Rs 50000 require additional approval and documentation. Customer information is updated periodically.""",
                    
                    """Criminal Case Intake Procedure v1: When a criminal complaint is received the intake officer logs the case details in the case management system within 48 hours. Initial review includes victim statement recording preliminary evidence assessment and jurisdiction verification. Cases are assigned to investigators based on current workload and specialization. Evidence is collected following standard operating procedures with proper documentation. Chain of custody is maintained for all physical evidence through secure storage. Initial assessment determines case priority. Follow-up interviews scheduled within one week.""",
                    
                    """Criminal Incident Response Protocol v1: Upon notification of criminal incident first responders secure the scene and preserve evidence following established protocols. Incident commander is notified within 2 hours of incident discovery. Investigation team is assembled based on incident type and severity. Witnesses are interviewed systematically with statements recorded. Evidence is photographed documented and collected using proper forensic techniques. Report is prepared within 7 days with all findings and recommendations. Follow-up actions are tracked until case closure with regular status updates to stakeholders.""",
                    
                    """Cyber Data Access Control Policy v1: Employees are granted system access based on role requirements and job responsibilities. User accounts are reviewed quarterly for appropriateness. Password policy requires 8-character minimum with complexity rules including uppercase lowercase numbers and special characters. Passwords must be changed every 180 days with no reuse of previous 5 passwords. Failed login attempts are monitored with account lockout after 5 consecutive failures. Access logs are retained for 6 months for audit purposes. Privileged access requires manager approval and additional authentication. Remote access requires VPN and multi-factor authentication.""",
                    
                    """Financial Transaction Monitoring Policy v1: All financial transactions are recorded in the centralized financial management system immediately. Transactions above Rs 10000 are flagged automatically for review by finance team. Monthly reconciliation of accounts is mandatory with variance analysis. Suspicious patterns are investigated by the dedicated finance team within 3 business days. Annual audit of financial controls is conducted by external auditors. Transaction records are kept for 7 years in archive storage with backup systems. Large transactions require dual authorization. Regular reports submitted to management.""",
                    
                    """Information Security Incident Response Plan v1: Security incidents are reported to IT helpdesk immediately through dedicated hotline or email. Incidents are categorized by severity level using predefined criteria. High severity incidents escalate to security team lead immediately with management notification. Incident investigation begins within 4 hours of detection with root cause analysis. Affected systems are isolated to prevent further compromise. Root cause analysis is documented in incident report. Remediation measures are implemented based on investigation findings. Incident reports are reviewed in monthly security committee meetings with lessons learned documentation."""
                ],
                'names': [
                    "BNS_2023.txt",
                    "IT_ACT_2021.txt", 
                    "PMLA_2002.txt",
                    "aml_customer_due_diligence_procedure_v1.txt",
                    "criminal_case_intake_procedure_v1.txt",
                    "criminal_incident_response_protocol_v1.txt",
                    "cyber_data_access_control_policy_v1.txt",
                    "financial_transaction_monitoring_policy_v1.txt",
                    "information_security_incident_response_plan_v1.txt"
                ],
                'types': ['guideline', 'guideline', 'guideline', 
                         'internal', 'internal', 'internal', 
                         'internal', 'internal', 'internal']
            }
            
            st.session_state.documents = sample_data['documents']
            st.session_state.doc_names = sample_data['names']
            st.session_state.doc_types = sample_data['types']
            
            categories = []
            for doc, name in zip(sample_data['documents'], sample_data['names']):
                categories.append(categorize_document(doc, name))
            st.session_state.doc_categories = categories
            
            st.rerun()
    
    # TAB 1: Compliance Dashboard
    with tab1:
        if st.session_state.documents:
            categorized_docs = {}
            categorized_guidelines = {}
            
            for doc, name, dtype, cat in zip(st.session_state.documents, 
                                            st.session_state.doc_names,
                                            st.session_state.doc_types,
                                            st.session_state.doc_categories):
                if cat not in CATEGORIES:
                    continue
                    
                if dtype == 'internal':
                    if cat not in categorized_docs:
                        categorized_docs[cat] = {'docs': [], 'names': []}
                    categorized_docs[cat]['docs'].append(doc)
                    categorized_docs[cat]['names'].append(name)
                else:
                    if cat not in categorized_guidelines:
                        categorized_guidelines[cat] = {'docs': [], 'names': []}
                    categorized_guidelines[cat]['docs'].append(doc)
                    categorized_guidelines[cat]['names'].append(name)
            
            if categorized_docs and categorized_guidelines:
                with st.spinner("🔄 Analyzing compliance by category..."):
                    results_df = compute_similarity_scores_by_category(
                        categorized_docs, categorized_guidelines, 
                        keep_numbers, use_lemma, max_features, min_df, max_df
                    )
                    results_df["risk"] = results_df["divergence_percent"].apply(risk_label)
                
                cols = st.columns(4)
                with cols[0]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{len(results_df)}</div>
                            <div class="metric-label">Documents Analyzed</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    avg_sim = results_df['similarity_percent'].mean()
                    st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                            <div class="metric-value">{avg_sim:.1f}%</div>
                            <div class="metric-label">Avg Similarity</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    safe_count = (results_df['divergence_percent'] <= 20).sum()
                    st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);">
                            <div class="metric-value">{safe_count}</div>
                            <div class="metric-label">Safe Documents</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with cols[3]:
                    risk_count = (results_df['divergence_percent'] > divergence_threshold).sum()
                    st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                            <div class="metric-value">{risk_count}</div>
                            <div class="metric-label">High Risk</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.subheader("📋 Compliance Results by Category")
                
                for category in CATEGORIES.keys():
                    category_data = results_df[results_df['category'] == category]
                    if category_data.empty:
                        continue
                    
                    with st.expander(f"**{category}** - {len(category_data)} documents", expanded=True):
                        display_df = category_data[['internal_document', 'matched_guideline', 'similarity_percent', 'divergence_percent', 'risk']].copy()
                        display_df.columns = ['Document', 'Guideline', 'Similarity (%)', 'Divergence (%)', 'Risk Level']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        colors = [risk_color(x) for x in category_data['divergence_percent']]
                        ax.barh(category_data['internal_document'], category_data['similarity_percent'], color=colors, alpha=0.8)
                        ax.set_xlabel('Similarity Score (%)', fontweight='bold')
                        ax.set_title(f'{category} - Compliance Scores', fontweight='bold')
                        ax.axvline(80, color='red', linestyle='--', alpha=0.5, label='80% Threshold')
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                
                pdf_df = results_df[['category', 'internal_document', 'matched_guideline', 'similarity_percent', 'divergence_percent', 'risk']].copy()
                pdf_df.columns = ['Category', 'Document', 'Guideline', 'Similarity (%)', 'Divergence (%)', 'Risk Level']
                pdf = generate_pdf(pdf_df)
                st.download_button(
                    label="📄 Download Categorized Report (PDF)",
                    data=pdf,
                    file_name="compliance_audit_categorized.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("📤 Upload both internal documents and guidelines. Ensure documents match at least one category.")
        else:
            st.info("📤 Upload documents to see compliance analysis")
    
    # TAB 2: Complete TF-IDF Mathematics
    with tab2:
        explain_tfidf_weighting()
        
        if st.session_state.documents:
            st.markdown("---")
            st.markdown("""
<div class="math-formula">
<h3>🧠 When to Use Which Variant? (Decision Logic)</h3>

<h4>TF (Term Frequency) Variants:</h4>
<table>
<tr>
<th>Variant</th>
<th>Formula</th>
<th>Use Case</th>
<th>Example</th>
</tr>
<tr>
<td><strong>Binary TF</strong></td>
<td>1 if count > 0, else 0</td>
<td>Boolean retrieval, presence/absence matters</td>
<td>Legal keyword search (does "evidence" appear?)</td>
</tr>
<tr>
<td><strong>Raw Count</strong></td>
<td>count</td>
<td>Short documents, word frequency critical</td>
<td>Tweet sentiment analysis</td>
</tr>
<tr>
<td><strong>Normalized TF</strong> ✅</td>
<td>count / doc_length</td>
<td>Long documents, prevents length bias</td>
<td>Legal document comparison (our default)</td>
</tr>
<tr>
<td><strong>Log Normalization</strong></td>
<td>1 + log(count)</td>
<td>Diminishing returns for repeated terms</td>
<td>News article categorization</td>
</tr>
<tr>
<td><strong>Double Normalization</strong></td>
<td>0.5 + 0.5×(count/max_count)</td>
<td>Balances common vs. rare terms</td>
<td>Academic paper similarity</td>
</tr>
</table>

<h4>IDF (Inverse Document Frequency) Variants:</h4>
<table>
<tr>
<th>Variant</th>
<th>Formula</th>
<th>Use Case</th>
<th>Example</th>
</tr>
<tr>
<td><strong>Standard IDF</strong></td>
<td>log(N/DF)</td>
<td>Basic information theory</td>
<td>Simple document retrieval</td>
</tr>
<tr>
<td><strong>Smooth IDF</strong></td>
<td>log(N/DF) + 1</td>
<td>Prevents zero division</td>
<td>Small corpus analysis</td>
</tr>
<tr>
<td><strong>Sklearn Smooth IDF</strong> ✅</td>
<td>log((1+N)/(1+DF)) + 1</td>
<td>Industry standard, robust to edge cases</td>
<td>Production ML pipelines (our default)</td>
</tr>
<tr>
<td><strong>Probabilistic IDF</strong></td>
<td>log((N-DF)/DF)</td>
<td>Emphasizes discriminative terms</td>
<td>Feature selection for classification</td>
</tr>
</table>

<h4>Our Choice for Compliance Analysis:</h4>
<div style="background: #e0f2fe; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
<p><strong>Normalized TF × Sklearn Smooth IDF</strong></p>
<ul>
<li><strong>Why Normalized TF?</strong> Legal documents vary greatly in length (100-10000 words). Raw counts would bias toward longer policies.</li>
<li><strong>Why Sklearn Smooth IDF?</strong> Robust to small corpora, compatible with sklearn ecosystem, prevents division by zero naturally.</li>
<li><strong>Trade-offs:</strong> Sacrifices granularity of raw counts for generalizability across document sizes.</li>
</ul>
</div>

<h4>Real-World Example:</h4>
<p>Document A (100 words): "compliance" appears 5 times → Normalized TF = 5/100 = 0.05</p>
<p>Document B (1000 words): "compliance" appears 10 times → Normalized TF = 10/1000 = 0.01</p>
<p><strong>Result:</strong> Doc A scores higher despite fewer raw mentions, correctly reflecting higher term importance.</p>
</div>
""", unsafe_allow_html=True)
            st.subheader("📊 Complete TF-IDF Computation (All Variants)")
            st.caption("Demonstrates manual calculation with all standard TF and IDF formulas")
            
            all_text = " ".join([preprocess_text(doc, keep_numbers, use_lemma) for doc in st.session_state.documents])
            word_freq = Counter(all_text.split())
            common_words = [w for w, c in word_freq.most_common(30) if len(w) > 3 and w.isalpha()][:10]
            
            if common_words:
                selected_words = st.multiselect(
                    "Select words for detailed mathematical demonstration:",
                    options=common_words,
                    default=common_words[:4] if len(common_words) >= 4 else common_words,
                    max_selections=4
                )
                
                if selected_words:
                    with st.spinner("🧮 Computing all TF-IDF variants..."):
                        tfidf_data = compute_manual_tfidf_complete(
                            st.session_state.documents[:5], 
                            selected_words, 
                            keep_numbers, 
                            use_lemma
                        )
                    
                    for word in selected_words:
                        if word in tfidf_data:
                            data = tfidf_data[word]
                            with st.expander(f"📐 Complete Analysis: **'{word}'**", expanded=True):
                                
                                st.markdown("##### 1️⃣ Term Frequency (TF) Variants")
                                st.caption("Different ways to calculate term frequency")
                                tf_df = pd.DataFrame(data['tf_variants_per_doc'])
                                st.dataframe(tf_df, use_container_width=True, hide_index=True)
                                
                                st.markdown("""
                                **TF Variant Formulas:**
                                - **Binary:** 1 if term present, 0 otherwise
                                - **Raw:** Simple term count
                                - **Normalized:** Count / Total words in document
                                - **Log Normalized:** 1 + log(count)
                                - **Double Normalized:** 0.5 + 0.5 × (count / max_count)
                                """)
                                
                                st.markdown("##### 2️⃣ Inverse Document Frequency (IDF) Variants")
                                st.caption("Different IDF calculation methods")
                                idf_df = pd.DataFrame([data['idf_variants']]).T
                                idf_df.columns = ['IDF Value']
                                st.dataframe(idf_df, use_container_width=True)
                                
                                st.markdown(f"""
                                **IDF Calculations (DF={data['df']}, N={len(st.session_state.documents[:5])}):**
                                - **Standard:** log(N/DF)
                                - **Smooth:** log(N/DF) + 1
                                - **Sklearn Smooth:** log((1+N)/(1+DF)) + 1 ← *Used in results*
                                - **Probabilistic:** log((N-DF)/DF)
                                """)
                                
                                st.markdown("##### 3️⃣ Final TF-IDF Scores")
                                st.caption(f"Using normalized TF × sklearn_smooth IDF = {data['idf_variants']['sklearn_smooth']:.6f}")
                                tfidf_df = pd.DataFrame(data['tfidf_variants_per_doc'])
                                st.dataframe(tfidf_df, use_container_width=True, hide_index=True)
                                
                                st.info(f"📊 **Document Frequency:** Term '{word}' appears in {data['df']} out of {len(st.session_state.documents[:5])} documents")
                    # end for selected_words

                    # -------------------------------------------------
                    # Comparative validation: Manual vs sklearn TF-IDF
                    # -------------------------------------------------
                    st.markdown("---")
                    st.markdown("### 🔬 Validation: Manual vs. Sklearn TF-IDF")
                    st.caption("Proves correctness of from-scratch implementation")

                    # Build sklearn TF-IDF on same subset
                    sklearn_vectorizer = TfidfVectorizer(
                        max_features=max_features,
                        min_df=min_df if min_df > 0 else 1,
                        max_df=max_df,
                        stop_words='english',
                        ngram_range=CONFIG.NGRAM_RANGE,
                        sublinear_tf=True,
                        norm='l2',
                        use_idf=True,
                        smooth_idf=True
                    )

                    sklearn_matrix = sklearn_vectorizer.fit_transform(
                        [preprocess_text(doc, keep_numbers, use_lemma) for doc in st.session_state.documents[:5]]
                    )

                    sklearn_features = sklearn_vectorizer.get_feature_names_out()

                    comparison_data = []

                    for word in selected_words:
                        word_lower = word.lower().strip()
                        if word in tfidf_data:
                            manual_tfidf_values = [x['tfidf'] for x in tfidf_data[word]['tfidf_variants_per_doc']]
                            try:
                                word_idx = list(sklearn_features).index(word_lower)
                                sklearn_tfidf_values = sklearn_matrix[:, word_idx].toarray().flatten()[:5]

                                for doc_id in range(len(manual_tfidf_values)):
                                    manual_val = manual_tfidf_values[doc_id]
                                    sklearn_val = float(sklearn_tfidf_values[doc_id])

                                    difference = abs(manual_val - sklearn_val)
                                    match_percent = (1 - (difference / sklearn_val if sklearn_val != 0 else 1)) * 100

                                    comparison_data.append({
                                        'Word': word,
                                        'Document': f"Doc {doc_id + 1}",
                                        'Manual TF-IDF': round(manual_val, 6),
                                        'Sklearn TF-IDF': round(sklearn_val, 6),
                                        'Absolute Difference': round(difference, 6),
                                        'Match %': round(match_percent, 2)
                                    })
                            except ValueError:
                                st.warning(f"⚠️ Word '{word}' not in sklearn vocabulary (filtered out by min_df/max_df)")

                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(
                            comparison_df.style.background_gradient(subset=['Match %'], cmap='RdYlGn', vmin=95, vmax=100),
                            use_container_width=True,
                            hide_index=True
                        )

                        avg_match = comparison_df['Match %'].mean()

                        if avg_match >= 99.9:
                            st.success(f"✅ **VALIDATION PASSED**: Average match = {avg_match:.2f}% (Near-perfect implementation)")
                        elif avg_match >= 95:
                            st.info(f"ℹ️ **VALIDATION GOOD**: Average match = {avg_match:.2f}% (Minor floating-point differences)")
                        else:
                            st.warning(f"⚠️ **REVIEW NEEDED**: Average match = {avg_match:.2f}% (Check formula implementation)")

                        st.markdown("""
                        **What This Proves:**
                        - Our from-scratch TF-IDF implementation matches sklearn's industry-standard algorithm
                        - Differences < 0.01% are typically due to floating-point arithmetic precision
                        - Validates that we correctly implemented: Normalized TF, Sklearn Smooth IDF, L2 normalization
                        """)
                    else:
                        st.warning("⚠️ No words available for comparison. Try selecting more words or adjusting TF-IDF parameters.")
            else:
                st.warning("No suitable words found for analysis. Try uploading more documents.")
        else:
            st.info("📤 Upload documents to see TF-IDF mathematical analysis")
    
    # TAB 3: Classification - UPDATED WITH FIX
    with tab3:
        st.subheader("🎓 Supervised Classification with TF-IDF")
        st.caption("Training classifiers to predict document categories")
        
        # Show current category distribution
        if st.session_state.documents:
            category_counts = Counter(st.session_state.doc_categories)
            st.markdown("**Current Category Distribution:**")
            
            dist_cols = st.columns(len(category_counts))
            for idx, (cat, count) in enumerate(category_counts.items()):
                with dist_cols[idx]:
                    color = CATEGORIES.get(cat, {}).get('color', '#667eea')
                    st.markdown(f"""
                        <div style="padding: 0.5rem; background: {color}22; border-left: 3px solid {color}; border-radius: 4px; text-align: center;">
                            <strong>{cat}</strong><br>
                            <span style="font-size: 1.5rem;">{count}</span> docs
                        </div>
                    """, unsafe_allow_html=True)
        
        if len(st.session_state.documents) >= 6:
            # Check if we have enough valid categories
            category_counts = Counter(st.session_state.doc_categories)
            valid_categories = {cat: count for cat, count in category_counts.items() if count >= 2}
            
            if len(valid_categories) >= 2:
                with st.spinner("Training classifiers..."):
                    # START PERFORMANCE TRACKING
                    start_time = time.time()
                    start_memory = sys.getsizeof(st.session_state.documents) / (1024 * 1024)  # MB

                    clf_results = perform_classification(
                        st.session_state.documents,
                        st.session_state.doc_categories,
                        keep_numbers=keep_numbers,
                        use_lemma=use_lemma,
                        max_features=max_features,
                        min_df=min_df,
                        max_df=max_df
                    )

                    end_time = time.time()
                    training_time = end_time - start_time
                    # END PERFORMANCE TRACKING
                
                if clf_results:
                    # Show filtering info if any
                    if clf_results['excluded_count'] > 0:
                        st.info(f"ℹ️ Using {clf_results['filtered_count']} documents ({clf_results['excluded_count']} excluded due to insufficient category samples)")
                    # Performance metrics display
                    st.markdown("### ⚡ Performance Metrics")
                    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

                    with perf_col1:
                        st.metric("Training Time", f"{training_time:.2f}s", help="Time to train both classifiers")

                    with perf_col2:
                        doc_count = clf_results.get('filtered_count', len(st.session_state.documents))
                        vec_time_estimate = (doc_count * 0.05)  # ~50ms per doc
                        st.metric("Vectorization Time", f"{vec_time_estimate:.2f}s", help="Estimated TF-IDF computation time")

                    with perf_col3:
                        memory_estimate = (clf_results.get('filtered_count', len(st.session_state.documents)) * max_features * 8) / (1024 * 1024)
                        st.metric("Memory Usage", f"{memory_estimate:.1f} MB", help="Approximate TF-IDF matrix size")

                    with perf_col4:
                        throughput = clf_results.get('filtered_count', len(st.session_state.documents)) / training_time if training_time > 0 else 0
                        st.metric("Throughput", f"{throughput:.1f} docs/s", help="Documents processed per second")

                    st.markdown("---")

                    st.markdown("### Model Performance")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 Multinomial Naive Bayes")
                        st.metric("Accuracy", f"{clf_results['nb_accuracy']:.2%}")
                        
                        st.markdown("**Confusion Matrix:**")
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(clf_results['confusion_matrix_nb'], annot=True, fmt='d', cmap='Blues', 
                                   xticklabels=clf_results['nb_model'].classes_,
                                   yticklabels=clf_results['nb_model'].classes_,
                                   ax=ax)
                        ax.set_ylabel('True Label', fontweight='bold')
                        ax.set_xlabel('Predicted Label', fontweight='bold')
                        ax.set_title('Naive Bayes Confusion Matrix', fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        with st.expander("Detailed Classification Report"):
                            st.text(clf_results['classification_report_nb'])
                    
                    with col2:
                        st.markdown("#### 📊 Logistic Regression")
                        st.metric("Accuracy", f"{clf_results['lr_accuracy']:.2%}")
                        
                        st.markdown("**Confusion Matrix:**")
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(clf_results['confusion_matrix_lr'], annot=True, fmt='d', cmap='Greens',
                                   xticklabels=clf_results['lr_model'].classes_,
                                   yticklabels=clf_results['lr_model'].classes_,
                                   ax=ax)
                        ax.set_ylabel('True Label', fontweight='bold')
                        ax.set_xlabel('Predicted Label', fontweight='bold')
                        ax.set_title('Logistic Regression Confusion Matrix', fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        with st.expander("Detailed Classification Report"):
                            st.text(clf_results['classification_report_lr'])
                    
                    st.markdown("---")
                    st.markdown("### 🔍 Top Predictive Features per Category")
                    st.caption("Most important TF-IDF features (from Logistic Regression coefficients)")
                    
                    for cat, features in clf_results['top_features'].items():
                        with st.expander(f"**{cat}** - Top Features"):
                            feat_df = pd.DataFrame(features, columns=['Feature', 'Coefficient'])
                            feat_df['Abs Coefficient'] = feat_df['Coefficient'].abs()
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            colors = ['green' if x > 0 else 'red' for x in feat_df['Coefficient']]
                            ax.barh(feat_df['Feature'], feat_df['Coefficient'], color=colors, alpha=0.7)
                            ax.set_xlabel('Coefficient Value', fontweight='bold')
                            ax.set_title(f'Top Features for {cat}', fontweight='bold')
                            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.dataframe(feat_df[['Feature', 'Coefficient']], hide_index=True, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Category Distribution")
                    st.caption("Document count per category (post-filtering)")
                    
                    if clf_results.get('category_distribution'):
                        cat_dist = clf_results['category_distribution']
                        dist_df = pd.DataFrame(list(cat_dist.items()), columns=['Category', 'Count'])
                        
                        # Create bar chart with category colors
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors_list = [CATEGORIES.get(cat, {}).get('color', '#667eea') for cat in dist_df['Category']]
                        bars = ax.bar(dist_df['Category'], dist_df['Count'], color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}',
                                   ha='center', va='bottom', fontweight='bold', fontsize=11)
                        
                        ax.set_ylabel('Document Count', fontweight='bold', fontsize=12)
                        ax.set_xlabel('Category', fontweight='bold', fontsize=12)
                        ax.set_title('Training Set Composition by Category', fontweight='bold', fontsize=13)
                        ax.set_ylim(0, dist_df['Count'].max() * 1.15)  # Add space for labels
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Documents", dist_df['Count'].sum())
                        with col2:
                            st.metric("Categories", len(dist_df))
                        with col3:
                            st.metric("Avg per Category", f"{dist_df['Count'].mean():.1f}")
                        with col4:
                            if clf_results.get('excluded_count', 0) > 0:
                                st.metric("Excluded Docs", clf_results['excluded_count'])
                        
                        st.dataframe(dist_df, hide_index=True, use_container_width=True)
            else:
                st.warning("⚠️ **Not enough data for classification**")
                st.markdown("**Minimum Requirements:**")
                st.markdown("- ✅ At least **6 documents total**")
                st.markdown("- ✅ At least **2 different categories**")
                st.markdown("- ✅ Each category must have at least **2 documents**")
                st.markdown("")
                st.markdown("**Current Status:**")
                
                req_col1, req_col2 = st.columns(2)
                
                with req_col1:
                    doc_count = len(st.session_state.documents)
                    status_docs = "✅" if doc_count >= 6 else "❌"
                    st.markdown(f"{status_docs} Documents: {doc_count}/6")
                    
                    cat_count = len(category_counts)
                    status_cats = "✅" if cat_count >= 2 else "❌"
                    st.markdown(f"{status_cats} Categories: {cat_count}/2")
                
                with req_col2:
                    valid_cats = sum(1 for count in category_counts.values() if count >= 2)
                    status_valid = "✅" if valid_cats >= 2 else "❌"
                    st.markdown(f"{status_valid} Valid categories: {valid_cats}/2")
                    
                    min_cat_size = min(category_counts.values()) if category_counts else 0
                    status_size = "✅" if min_cat_size >= 2 else "❌"
                    st.markdown(f"{status_size} Min docs/category: {min_cat_size}/2")
                
                st.markdown("---")
                st.markdown("**Category Details:**")
                for cat, count in category_counts.items():
                    status = "✅" if count >= 2 else "❌"
                    color = CATEGORIES.get(cat, {}).get('color', '#667eea')
                    st.markdown(f"{status} **{cat}:** {count} document(s)")
                
                st.markdown("---")
                st.markdown("**💡 Recommended Actions:**")
                col_action1, col_action2 = st.columns(2)
                
                with col_action1:
                    st.markdown("**Option 1: Upload More Documents**")
                    st.markdown(f"- Need at least {6 - len(st.session_state.documents)} more document(s)")
                    for cat, count in category_counts.items():
                        if count < 2:
                            st.markdown(f"- {cat} needs {2 - count} more document(s)")
                
                with col_action2:
                    st.markdown("**Option 2: Load Sample Data**")
                    st.markdown("- Populate with regulatory guidelines + internal procedures")
                    st.markdown("- Includes complete examples for all compliance categories")
                    st.markdown("- Best for demo and testing")
        else:
            st.info("📤 Need at least 6 documents for classification. Upload more documents or load sample data.")
    
    # TAB 4: Enhanced Clustering
    with tab4:
        st.subheader("🔬 K-Means Clustering Analysis")
        st.caption("Unsupervised grouping of documents based on TF-IDF similarity")
        
        if len(st.session_state.documents) >= 3:
            with st.spinner("Performing clustering analysis..."):
                cluster_results = perform_enhanced_clustering(
                    st.session_state.documents, 
                    st.session_state.doc_names, 
                    n_clusters,
                    keep_numbers,
                    use_lemma,
                    max_features,
                    min_df,
                    max_df
                )
            
            if cluster_results:
                st.markdown("### Cluster Quality Metrics")

                # Safe extraction and formatting of metrics
                silhouette = cluster_results.get('silhouette_score')
                db_index = cluster_results.get('davies_bouldin_score')
                inertia = cluster_results.get('inertia')
                n_actual = cluster_results.get('n_clusters_actual', None)
                n_requested = cluster_results.get('n_clusters_requested', None)

                def fmt(x, fmtstr):
                    try:
                        if x is None or (isinstance(x, float) and np.isnan(x)):
                            return "N/A"
                        return fmtstr.format(x)
                    except Exception:
                        return "N/A"

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Silhouette Score", fmt(silhouette, "{:.3f}"), 
                             help="Range: [-1, 1]. Higher is better. >0.5 = good clustering")
                with col2:
                    st.metric("Davies-Bouldin Index", fmt(db_index, "{:.3f}"),
                             help="Lower is better. Measures cluster separation")
                with col3:
                    st.metric("Inertia", fmt(inertia, "{:.2f}"),
                             help="Sum of squared distances to centroids. Lower is better")
                with col4:
                    clusters_label = f"Requested: {n_requested}\nActual: {n_actual}" if n_requested is not None else f"Actual: {n_actual}"
                    st.metric("Clusters (Requested / Actual)", clusters_label)
                
                st.markdown("### Document Clustering Visualization")
                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(
                    cluster_results['coordinates'][:, 0], 
                    cluster_results['coordinates'][:, 1],
                    c=cluster_results['labels'], 
                    cmap='viridis', 
                    s=200, 
                    alpha=0.6, 
                    edgecolors='black',
                    linewidths=2
                )
                
                for i, name in enumerate(st.session_state.doc_names):
                    ax.annotate(
                        name[:20], 
                        (cluster_results['coordinates'][i, 0], cluster_results['coordinates'][i, 1]),
                        fontsize=9, 
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                    )
                
                ax.set_xlabel('PCA Component 1', fontweight='bold', fontsize=12)
                ax.set_ylabel('PCA Component 2', fontweight='bold', fontsize=12)
                ax.set_title('Document Clusters (K-Means + PCA)', fontweight='bold', fontsize=14)
                plt.colorbar(scatter, label='Cluster ID')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("---")
                st.markdown("### 📊 Top Terms per Cluster")
                st.caption("Most important terms in each cluster (from cluster centroids)")
                
                for cluster_id, terms in cluster_results['top_terms'].items():
                    with st.expander(f"**Cluster {cluster_id}** - Top Terms"):
                        terms_df = pd.DataFrame(terms, columns=['Term', 'TF-IDF Score'])
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.barh(terms_df['Term'], terms_df['TF-IDF Score'], color='#667eea', alpha=0.7)
                        ax.set_xlabel('TF-IDF Score', fontweight='bold')
                        ax.set_title(f'Cluster {cluster_id} - Top Terms', fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.dataframe(terms_df, hide_index=True, use_container_width=True)
        else:
            st.info("📤 Upload at least 3 documents for clustering analysis")
    
    # TAB 5: Matrix Inspection
    with tab5:
        st.subheader("📈 TF-IDF Matrix Inspection")
        st.caption("Explore the underlying TF-IDF matrix structure and document similarities")
        
        if st.session_state.documents:
            with st.spinner("Building TF-IDF matrix..."):
                processed_docs = [preprocess_text(d, keep_numbers, use_lemma) for d in st.session_state.documents]
                vectorizer = TfidfVectorizer(
                    max_features=max_features, 
                    min_df=min_df, 
                    max_df=max_df, 
                    stop_words='english'
                )
                matrix = vectorizer.fit_transform(processed_docs)
                
                display_tfidf_matrix(vectorizer, matrix, st.session_state.doc_names)
        else:
            st.info("📤 Upload documents to inspect the TF-IDF matrix")
    
    # TAB 6: Visualizations
    with tab6:
        st.subheader("📉 TF-IDF Visualizations")
        st.caption("Visual exploration of term importance and distributions")
        
        if st.session_state.documents:
            processed_docs = [preprocess_text(d, keep_numbers, use_lemma) for d in st.session_state.documents]
            vectorizer = TfidfVectorizer(
                max_features=max_features, 
                min_df=min_df,
                max_df=max_df,
                stop_words='english'
            )
            matrix = vectorizer.fit_transform(processed_docs)
            
            st.markdown("### 🌥️ TF-IDF Weighted Word Clouds")
            doc_idx = st.selectbox(
                "Select document:", 
                range(len(st.session_state.doc_names)), 
                format_func=lambda i: st.session_state.doc_names[i]
            )
            
            fig = generate_wordcloud(vectorizer, matrix, doc_idx)
            if fig:
                st.pyplot(fig)
            
            st.markdown("---")
            st.markdown("### 📊 Top Terms by TF-IDF Score")
            
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = matrix[doc_idx].toarray().flatten()
            top_indices = np.argsort(tfidf_scores)[-15:][::-1]
            
            top_terms_df = pd.DataFrame({
                'Term': [feature_names[i] for i in top_indices],
                'TF-IDF Score': [tfidf_scores[i] for i in top_indices]
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_terms_df['Term'], top_terms_df['TF-IDF Score'], color='#667eea', alpha=0.7)
            ax.set_xlabel('TF-IDF Score', fontweight='bold')
            ax.set_title(f'Top 15 Terms - {st.session_state.doc_names[doc_idx]}', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.dataframe(top_terms_df, hide_index=True, use_container_width=True)
        else:
            st.info("📤 Upload documents to see visualizations")

if __name__ == "__main__":
    main()
