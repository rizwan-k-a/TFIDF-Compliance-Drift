"""
============================================================
UNIVERSAL COMPLIANCE REVIEW SYSTEM - PREMIUM UI WITH OCR
MCA Final Project – Professional Grade Dashboard
============================================================
"""

import os
import sys
from io import BytesIO
import re
import math
from collections import Counter

# Path setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Third-party imports
import streamlit as st
import pandas as pd
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from sklearn.metrics.pairwise import cosine_similarity

# Project module imports
from vectorize import build_tfidf_vectors
from similarity import compute_cosine_similarity
# Import only what exists in manual_tfidf_math
try:
    from manual_tfidf_math import tokenize, build_tfidf_matrix, compute_pairwise_similarity
except ImportError:
    # Fallback if imports fail
    def tokenize(text):
        return text.lower().split()
    
    def build_tfidf_matrix(documents):
        return [], {}, set()
    
    def compute_pairwise_similarity(v1, v2):
        return [[]]

# Define compute_manual_tfidf directly in dashboard
def compute_manual_tfidf(documents, sample_words):
    """
    Compute TF, IDF, and TF-IDF manually for sample words.
    Dashboard-specific implementation.
    """
    import re
    import math
    from collections import Counter
    
    def preprocess_simple(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Preprocess and tokenize documents
    processed_docs = [preprocess_simple(doc).split() for doc in documents]
    n_docs = len(processed_docs)
    
    results = {}
    
    for word in sample_words:
        word_lower = word.lower()
        results[word] = {
            'tf_per_doc': [],
            'df': 0,
            'idf': 0,
            'tfidf_per_doc': []
        }
        
        # Calculate TF for each document
        for doc_idx, doc_words in enumerate(processed_docs):
            word_count = doc_words.count(word_lower)
            total_words = len(doc_words)
            tf = word_count / total_words if total_words > 0 else 0
            
            results[word]['tf_per_doc'].append({
                'doc': f'Doc{doc_idx+1}',
                'count': word_count,
                'total': total_words,
                'tf': round(tf, 6)
            })
            
            # Count document frequency
            if word_count > 0:
                results[word]['df'] += 1
        
        # Calculate IDF (with smoothing)
        df = results[word]['df']
        idf = math.log((1 + n_docs) / (1 + df)) + 1
        results[word]['idf'] = round(idf, 6)
        
        # Calculate TF-IDF for each document
        for doc_data in results[word]['tf_per_doc']:
            tfidf = doc_data['tf'] * idf
            results[word]['tfidf_per_doc'].append({
                'doc': doc_data['doc'],
                'tfidf': round(tfidf, 6)
            })
    
    return results, n_docs


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Compliance Review System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM CUSTOM CSS - SENIOR UI DESIGNER LEVEL
# ============================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #2d3748 100%);
        padding: 2rem 1rem;
    }
    
    /* Logo and Branding */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 1.5rem 1rem;
        margin-bottom: 2rem;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    
    .sidebar-logo-icon {
        font-size: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sidebar-logo-text h1 {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }
    
    .sidebar-logo-text p {
        font-size: 0.75rem;
        color: #a0aec0;
        margin: 0;
    }
    
    /* Page Header */
    .page-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .page-header-icon {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .page-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    
    .page-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin: 0.5rem 0 2rem 0;
        line-height: 1.6;
    }
    
    /* Disclaimer Badge */
    .disclaimer-badge {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(251, 191, 36, 0.3);
    }
    
    /* Disclaimer Box */
    .disclaimer-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe8a1 100%);
        border-left: 4px solid #fbbf24;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(251, 191, 36, 0.1);
    }
    
    .disclaimer-box strong {
        color: #f59e0b;
        font-weight: 700;
    }
    
    /* Document Section */
    .doc-section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    
    .doc-section-icon {
        font-size: 1.8rem;
    }
    
    .doc-section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }
    
    /* Primary Button */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border: none;
        padding: 1rem 3rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        width: 100%;
        margin: 2rem 0;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.5);
    }
    
    /* Metrics Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.8rem;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card.success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .metric-card.warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    
    .metric-card.danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        line-height: 1;
    }
    
    /* Processing Badge */
    .processing-badge {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .ocr-badge {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 3px solid #e2e8f0;
    }
    
    .section-header h2 {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b;
        margin: 0;
    }
    
    .section-icon {
        font-size: 1.8rem;
        color: #667eea;
    }
    
    /* Mathematical Box */
    .math-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        font-family: 'Courier New', monospace;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .math-box strong {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Guidelines Used Sidebar */
    .guidelines-used {
        background: rgba(16, 185, 129, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    
    .guidelines-used h4 {
        color: #10b981;
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
    }
    
    .guideline-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
        color: #cbd5e1;
        font-size: 0.85rem;
    }
    
    .guideline-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 4rem;
        border-top: 2px solid #e2e8f0;
        color: #64748b;
        font-size: 0.9rem;
    }
    
    /* Dataframe Styling */
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 1rem !important;
        border: none !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .dataframe tbody tr:hover {
        background: #f8fafc !important;
        transform: scale(1.01);
    }
    
    .dataframe tbody tr td {
        padding: 1rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #10b981;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #ef4444;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #f59e0b;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
DEFAULT_DIVERGENCE_THRESHOLD = 40
MIN_TEXT_LENGTH = 100
MIN_PAGE_TEXT_LENGTH = 100

# OCR Configuration
POPPLER_PATH = r"C:\Program Files\poppler-25.12.0\Library\bin"
OCR_DPI = 300
OCR_CONFIG = "--psm 6"

# ============================================================
# PDF PROCESSING WITH OCR
# ============================================================

def extract_text_from_pdf(uploaded_file, use_ocr=True):
    """
    Extract text from PDF with OCR fallback for scanned pages.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        use_ocr: Whether to use OCR for scanned pages
    
    Returns:
        Tuple of (extracted_text, ocr_pages_count, total_pages)
    """
    pages = []
    ocr_pages = 0
    total_pages = 0
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Extract text
        with pdfplumber.open(temp_path) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages, 1):
                # Try text extraction first
                text = page.extract_text()
                
                if text and len(text.strip()) > MIN_PAGE_TEXT_LENGTH:
                    pages.append(text)
                elif use_ocr:
                    # Use OCR for scanned pages
                    try:
                        images = convert_from_path(
                            temp_path,
                            first_page=i,
                            last_page=i,
                            poppler_path=POPPLER_PATH,
                            dpi=OCR_DPI
                        )
                        
                        ocr_text = pytesseract.image_to_string(
                            images[0],
                            lang="eng",
                            config=OCR_CONFIG
                        )
                        
                        if ocr_text.strip():
                            pages.append(ocr_text)
                            ocr_pages += 1
                    except Exception as ocr_error:
                        st.warning(f"⚠️ OCR failed for page {i}: {str(ocr_error)}")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return "\n".join(pages), ocr_pages, total_pages
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"PDF extraction failed: {str(e)}")

def clean_extracted_text(text: str, min_line_length: int = 30) -> str:
    """Clean extracted text by removing short lines and extra whitespace."""
    cleaned_lines = []
    
    for line in text.splitlines():
        line = line.strip()
        
        # Skip very short lines (likely headers/footers)
        if len(line) < min_line_length:
            continue
        
        # Skip lines that are mostly numbers (page numbers)
        if sum(c.isdigit() for c in line) > len(line) * 0.5:
            continue
        
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def preprocess_text_simple(text):
    """Simple preprocessing for TF-IDF"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def validate_text(text, doc_name):
    """Validate if text has sufficient content"""
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False, f"Document '{doc_name}' has insufficient text ({len(text)} chars)"
    
    words = preprocess_text_simple(text).split()
    if len(words) < 20:
        return False, f"Document '{doc_name}' has too few words ({len(words)} words)"
    
    return True, "Valid"

def read_uploaded_files(files, use_ocr=True):
    """
    Read uploaded files with OCR support for PDFs.
    
    Returns:
        Tuple of (texts, names, errors, processing_info)
    """
    texts, names = [], []
    errors = []
    processing_info = []
    
    for f in files:
        try:
            if f.name.lower().endswith(".txt"):
                text = f.read().decode("utf-8", errors="ignore")
                processing_info.append({
                    "filename": f.name,
                    "type": "Text",
                    "ocr_used": False,
                    "pages": 1
                })
            elif f.name.lower().endswith(".pdf"):
                with st.spinner(f"📄 Processing PDF: {f.name}..."):
                    text, ocr_pages, total_pages = extract_text_from_pdf(f, use_ocr)
                    text = clean_extracted_text(text)
                    
                    processing_info.append({
                        "filename": f.name,
                        "type": "PDF",
                        "ocr_used": ocr_pages > 0,
                        "ocr_pages": ocr_pages,
                        "total_pages": total_pages
                    })
            else:
                continue
            
            is_valid, message = validate_text(text, f.name)
            if is_valid:
                texts.append(text)
                names.append(f.name)
            else:
                errors.append(message)
        except Exception as e:
            errors.append(f"Error reading {f.name}: {str(e)}")
    
    return texts, names, errors, processing_info

def load_text_files(folder):
    """Load text files from folder with validation"""
    texts, names = [], []
    errors = []
    
    if not os.path.exists(folder):
        return texts, names, errors
    
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.endswith(".txt"):
                try:
                    filepath = os.path.join(root, f)
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                        text = file.read()
                    
                    is_valid, message = validate_text(text, f)
                    if is_valid:
                        texts.append(text)
                        names.append(f)
                    else:
                        errors.append(message)
                except Exception as e:
                    errors.append(f"Error loading {f}: {str(e)}")
    
    return texts, names, errors

def risk_label(div):
    """Determine risk level based on divergence"""
    if div <= 20:
        return "✅ Safe – Closely Aligned"
    elif div <= DEFAULT_DIVERGENCE_THRESHOLD:
        return "⚠️ Needs Attention"
    else:
        return "🚨 Review Required"

def risk_color(div):
    """Get color based on divergence"""
    if div <= 20:
        return "#10b981"
    elif div <= DEFAULT_DIVERGENCE_THRESHOLD:
        return "#f59e0b"
    else:
        return "#ef4444"

def generate_pdf(results_df):
    """Generate PDF report"""
    buffer = BytesIO()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "UNIVERSAL COMPLIANCE AUDIT REPORT", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Helvetica", '', 10)
    for _, r in results_df.iterrows():
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(0, 8, f"Document: {r['Document']}", ln=True)
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(0, 6, f"Similarity: {r['Similarity (%)']}%", ln=True)
        pdf.cell(0, 6, f"Divergence: {r['Divergence (%)']}%", ln=True)
        pdf.cell(0, 6, f"Risk: {r['Risk Level']}", ln=True)
        pdf.ln(5)
    
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

def display_manual_tfidf(documents, doc_names):
    """Display manual TF-IDF calculations"""
    st.markdown('<div class="section-header"><span class="section-icon">🧮</span><h2>Manual TF-IDF Computation</h2></div>', unsafe_allow_html=True)
    
    st.info("**📚 Academic Requirement 20:** Mathematical demonstration of TF-IDF calculations")
    
    first_doc = preprocess_text_simple(documents[0]).split()
    word_freq = Counter(first_doc)
    sample_words = [word for word, _ in word_freq.most_common(10)[2:6]][:4]
    
    if len(sample_words) < 3:
        sample_words = list(set(first_doc))[:4]
    
    st.markdown(f"**📝 Sample words selected:** `{', '.join(sample_words)}`")
    
    results, n_docs = compute_manual_tfidf(documents, sample_words)
    
    st.markdown("#### 📐 Mathematical Formulas")
    st.markdown("""
    <div class="math-box">
    <strong>Term Frequency (TF):</strong><br>
    TF(t, d) = (Number of times term t appears in document d) / (Total terms in document d)<br><br>
    
    <strong>Inverse Document Frequency (IDF) with smoothing:</strong><br>
    IDF(t) = log((1 + Total documents) / (1 + Documents containing term t)) + 1<br><br>
    
    <strong>TF-IDF Score:</strong><br>
    TF-IDF(t, d) = TF(t, d) × IDF(t)
    </div>
    """, unsafe_allow_html=True)
    
    for word in sample_words:
        with st.expander(f"📊 Detailed Calculation: **'{word}'**", expanded=False):
            word_data = results[word]
            
            st.markdown(f"**Step 1: Calculate TF (Term Frequency) for each document**")
            tf_df = pd.DataFrame(word_data['tf_per_doc'])
            st.dataframe(tf_df, use_container_width=True, hide_index=True)
            
            st.markdown(f"**Step 2: Calculate IDF (Inverse Document Frequency)**")
            st.code(f"""
Document Frequency (DF) = {word_data['df']} documents contain '{word}'
Total Documents (N) = {n_docs}

IDF = log((1 + {n_docs}) / (1 + {word_data['df']})) + 1
    = {word_data['idf']:.6f}
            """)
            
            st.markdown(f"**Step 3: Calculate TF-IDF for each document**")
            tfidf_df = pd.DataFrame(word_data['tfidf_per_doc'])
            st.dataframe(tfidf_df, use_container_width=True, hide_index=True)
            
            st.success(f"**💡 Interpretation:** Higher TF-IDF values indicate that '{word}' is more distinctive in that document")
    
    st.markdown("#### 💡 Importance of TF-IDF Weighting")
    st.markdown("""
    <div class="math-box">
    <strong>Why TF-IDF is important:</strong><br>
    1. <strong>Balances frequency with uniqueness:</strong> Common words get low scores<br>
    2. <strong>Highlights distinctive terms:</strong> Unique words get high scores<br>
    3. <strong>Better than raw counts:</strong> Accounts for document length<br>
    4. <strong>Enables similarity comparison:</strong> Similar vectors = related documents<br>
    5. <strong>Foundation for ML:</strong> Used in clustering and classification
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # Logo and Branding
    st.markdown("""
    <div class="sidebar-logo">
        <span class="sidebar-logo-icon">⚖️</span>
        <div class="sidebar-logo-text">
            <h1>Compliance<br>Review System</h1>
            <p>Decision Support Tool</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔐 USER ROLE")
    
    role = st.radio(
        "Select your access level:",
        ["Compliance Tester", "System Admin"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # OCR Settings
    st.markdown("### ⚙️ OCR SETTINGS")
    use_ocr = st.checkbox("Enable OCR for scanned PDFs", value=True, help="Uses Tesseract OCR for scanned documents")
    
    if use_ocr:
        st.success("✅ OCR Enabled")
        st.caption("PDFs will be processed with OCR fallback for scanned pages")
    else:
        st.info("ℹ️ OCR Disabled")
        st.caption("Only text-based PDFs will be processed")
    
    st.markdown("---")
    
    # Guidelines Used
    st.markdown("""
    <div class="guidelines-used">
        <h4>📚 LEGAL GUIDELINES</h4>
        <div class="guideline-item">
            <span class="guideline-dot"></span>
            Bharatiya Nyaya Sanhita 2023
        </div>
        <div class="guideline-item">
            <span class="guideline-dot"></span>
            Information Technology Act 2021
        </div>
        <div class="guideline-item">
            <span class="guideline-dot"></span>
            Prevention of Money Laundering Act 2002
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #94a3b8; font-size: 0.75rem; margin-top: 2rem;">
        MCA Final Project<br>
        PDF + OCR Processing<br>
        Decision Support Tool
    </p>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT
# ============================================================

# ============================================================
# COMPLIANCE TESTER VIEW
# ============================================================
if role == "Compliance Tester":
    # Header
    st.markdown("""
    <div class="page-header">
        <span class="page-header-icon">⚖️</span>
        <h1 class="page-title">Compliance Review</h1>
    </div>
    <p class="page-subtitle">
        Compare internal documents against statutory legal guidelines to identify compliance risks
    </p>
    """, unsafe_allow_html=True)
    
    # Disclaimer Badge
    st.markdown("""
    <span class="disclaimer-badge">⚠️ Decision Support Only</span>
    """, unsafe_allow_html=True)
    
    # Disclaimer Box
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Important Disclaimer:</strong> This system assists in review prioritization only. 
        It does NOT certify compliance or constitute legal advice. Always consult qualified legal 
        professionals for compliance decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Load built-in documents
    builtin_internal_docs, builtin_internal_names, internal_errors = load_text_files(
        os.path.join(DATA_DIR, "internal")
    )
    builtin_guideline_docs, builtin_guideline_names, guideline_errors = load_text_files(
        os.path.join(DATA_DIR, "guidelines")
    )
    
    # Display errors if any
    all_errors = internal_errors + guideline_errors
    if all_errors:
        with st.expander("⚠️ Document Loading Warnings", expanded=False):
            for error in all_errors:
                st.warning(error)
    
    # Document Selection
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="doc-section-header">
            <span class="doc-section-icon">📄</span>
            <h3 class="doc-section-title">Legal Guideline</h3>
        </div>
        <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">
            Select one reference guideline or upload your own PDF
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("**Built-in Guidelines**")
        st.caption("Choose from pre-loaded legal frameworks")
        
        selected_builtin_guidelines = st.multiselect(
            "Select guidelines",
            builtin_guideline_names,
            default=builtin_guideline_names[:2] if len(builtin_guideline_names) >= 2 else builtin_guideline_names,
            max_selections=2,
            label_visibility="collapsed",
            key="guideline_select"
        )
        
        st.markdown("**OR Upload Custom Guideline (PDF/TXT with OCR)**")
        
        uploaded_guidelines = st.file_uploader(
            "Upload custom guideline",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="PDFs will be processed with OCR if enabled",
            key="guideline_upload"
        )
    
    with col2:
        st.markdown("""
        <div class="doc-section-header">
            <span class="doc-section-icon">📋</span>
            <h3 class="doc-section-title">Internal Document</h3>
        </div>
        <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">
            Select document to analyze or upload PDF with OCR
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("**Sample Internal Documents**")
        st.caption("Choose from example organizational policies")
        
        selected_builtin_internal = st.multiselect(
            "Select internal documents",
            builtin_internal_names,
            default=builtin_internal_names if len(builtin_internal_names) > 0 else [],
            label_visibility="collapsed",
            key="internal_select"
        )
        
        st.markdown("**OR Upload Your Document (PDF/TXT with OCR)**")
        
        uploaded_internal = st.file_uploader(
            "Upload your document",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="PDFs will be processed with OCR if enabled",
            key="internal_upload"
        )
    
    # Build corpus
    internal_texts, internal_names = [], []
    upload_errors = []
    processing_info = []
    
    for name in selected_builtin_internal:
        i = builtin_internal_names.index(name)
        internal_texts.append(builtin_internal_docs[i])
        internal_names.append(name)
    
    if uploaded_internal:
        t, n, e, p_info = read_uploaded_files(uploaded_internal, use_ocr)
        internal_texts.extend(t)
        internal_names.extend(n)
        upload_errors.extend(e)
        processing_info.extend(p_info)
    
    guideline_texts, guideline_names = [], []
    for name in selected_builtin_guidelines:
        i = builtin_guideline_names.index(name)
        guideline_texts.append(builtin_guideline_docs[i])
        guideline_names.append(name)
    
    if uploaded_guidelines:
        t, n, e, p_info = read_uploaded_files(uploaded_guidelines, use_ocr)
        guideline_texts.extend(t)
        guideline_names.extend(n)
        upload_errors.extend(e)
        processing_info.extend(p_info)
    
    # Display processing info
    if processing_info:
        with st.expander("📊 Document Processing Summary", expanded=True):
            for info in processing_info:
                if info["type"] == "PDF":
                    if info["ocr_used"]:
                        st.markdown(f"""
                        <span class="ocr-badge">🔍 OCR: {info['filename']}</span>
                        <br><small>Pages processed: {info['total_pages']} | OCR used on: {info['ocr_pages']} pages</small>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <span class="processing-badge">📄 PDF: {info['filename']}</span>
                        <br><small>Pages processed: {info['total_pages']} (text extraction)</small>
                        """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ Text file: {info['filename']}")
    
    if upload_errors:
        for error in upload_errors:
            st.error(error)
    
    if len(guideline_names) > 2:
        st.error("⚠️ Maximum two guidelines allowed for optimal performance.")
        st.stop()
    
    # Run Analysis Button
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("▶️ Run Compliance Analysis", use_container_width=True, type="primary")
    
    if run:
        if not internal_texts:
            st.error("❌ Please select or upload at least one internal document.")
        elif not guideline_texts:
            st.error("❌ Please select or upload at least one guideline document.")
        else:
            try:
                with st.spinner("🔄 Analyzing compliance similarity with TF-IDF vectorization..."):
                    _, ref_vecs, int_vecs = build_tfidf_vectors(
                        guideline_texts,
                        internal_texts
                    )
                    
                    sim_df = compute_cosine_similarity(
                        ref_vecs,
                        int_vecs,
                        internal_names
                    )
                    
                    sim_df["similarity_percent"] = (sim_df["compliance_score"] * 100).round(1)
                    sim_df["divergence_percent"] = (100 - sim_df["similarity_percent"]).round(1)
                    sim_df["risk"] = sim_df["divergence_percent"].apply(risk_label)
                
                st.success("✅ Analysis completed successfully!")
                
                # Metrics
                st.markdown('<div class="section-header"><span class="section-icon">📊</span><h2>Compliance Overview</h2></div>', unsafe_allow_html=True)
                
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Documents Analyzed</div>
                        <div class="metric-value">{}</div>
                    </div>
                    """.format(len(internal_names)), unsafe_allow_html=True)
                
                with metric_cols[1]:
                    avg_sim = sim_df["similarity_percent"].mean()
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Avg Similarity</div>
                        <div class="metric-value">{:.1f}%</div>
                    </div>
                    """.format(avg_sim), unsafe_allow_html=True)
                
                with metric_cols[2]:
                    safe_count = (sim_df["divergence_percent"] <= 20).sum()
                    st.markdown("""
                    <div class="metric-card success">
                        <div class="metric-label">✅ Safe Documents</div>
                        <div class="metric-value">{}</div>
                    </div>
                    """.format(safe_count), unsafe_allow_html=True)
                
                with metric_cols[3]:
                    risk_count = (sim_df["divergence_percent"] > DEFAULT_DIVERGENCE_THRESHOLD).sum()
                    st.markdown("""
                    <div class="metric-card danger">
                        <div class="metric-label">🚨 High Risk</div>
                        <div class="metric-value">{}</div>
                    </div>
                    """.format(risk_count), unsafe_allow_html=True)
                
                # Results Table
                st.markdown('<div class="section-header"><span class="section-icon">📋</span><h2>Detailed Results</h2></div>', unsafe_allow_html=True)
                
                display_df = sim_df[['internal_document', 'similarity_percent', 'divergence_percent', 'risk']].copy()
                display_df.columns = ['Document', 'Similarity (%)', 'Divergence (%)', 'Risk Level']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualizations
                st.markdown('<div class="section-header"><span class="section-icon">📈</span><h2>Visual Analysis</h2></div>', unsafe_allow_html=True)
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    colors = [risk_color(d) for d in sim_df["divergence_percent"]]
                    bars = ax1.barh(internal_names, sim_df["similarity_percent"], color=colors, alpha=0.8, height=0.6)
                    
                    ax1.set_xlabel("Similarity Score (%)", fontsize=12, fontweight='600')
                    ax1.set_title("Compliance Similarity Scores", fontsize=14, fontweight='bold', pad=20)
                    ax1.set_xlim(0, 100)
                    ax1.grid(axis='x', alpha=0.3, linestyle='--')
                    ax1.spines['top'].set_visible(False)
                    ax1.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig1)
                
                with viz_col2:
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    risk_counts = sim_df["risk"].value_counts()
                    colors_pie = ['#10b981', '#f59e0b', '#ef4444']
                    
                    wedges, texts, autotexts = ax2.pie(
                        risk_counts,
                        labels=risk_counts.index,
                        autopct='%1.1f%%',
                        colors=colors_pie,
                        startangle=90,
                        textprops={'fontweight': 'bold', 'fontsize': 11}
                    )
                    
                    ax2.set_title("Risk Distribution", fontsize=14, fontweight='bold', pad=20)
                    st.pyplot(fig2)
                
                # Download Report
                st.markdown("<br>", unsafe_allow_html=True)
                pdf = generate_pdf(display_df)
                st.download_button(
                    label="📄 Download Compliance Audit Report (PDF)",
                    data=pdf,
                    file_name="compliance_audit_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("💡 **Possible causes:**\n- Documents may be too short or empty\n- Text preprocessing removed all content\n- Invalid file format\n- OCR processing failed")
                st.exception(e)

# ============================================================
# SYSTEM ADMIN VIEW
# ============================================================
else:
    st.markdown("""
    <div class="page-header">
        <span class="page-header-icon">📊</span>
        <h1 class="page-title">System Admin</h1>
    </div>
    <p class="page-subtitle">
        Deep dive into TF-IDF vectorization, cosine similarity matrices, and mathematical foundations
    </p>
    """, unsafe_allow_html=True)
    
    # Load documents
    internal_docs, internal_names, int_errors = load_text_files(
        os.path.join(DATA_DIR, "internal")
    )
    guideline_docs, guideline_names, guide_errors = load_text_files(
        os.path.join(DATA_DIR, "guidelines")
    )
    
    all_errors = int_errors + guide_errors
    if all_errors:
        with st.expander("⚠️ Document Loading Issues", expanded=True):
            for error in all_errors:
                st.warning(error)
    
    if not internal_docs or not guideline_docs:
        st.error("❌ Unable to load sufficient documents for analysis.")
        st.info("💡 **Tip:** Run `python scripts/pdf_to_txt_once.py` to extract guidelines from PDFs")
        st.stop()
    
    # Configuration
    st.markdown('<div class="section-header"><span class="section-icon">⚙️</span><h2>Analysis Configuration</h2></div>', unsafe_allow_html=True)
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        sel_internal = st.multiselect(
            "**Internal Documents:**",
            internal_names,
            default=internal_names
        )
    
    with config_col2:
        sel_guidelines = st.multiselect(
            "**Legal Guidelines:**",
            guideline_names,
            default=guideline_names[:2]
        )
    
    divergence_threshold = st.slider(
        "**Divergence Threshold (%):**",
        min_value=20,
        max_value=80,
        value=DEFAULT_DIVERGENCE_THRESHOLD,
        step=5,
        help="Documents above this threshold are flagged as high risk"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_admin = st.button("▶️ Run Admin Analysis", use_container_width=True, type="primary")
    
    if run_admin:
        if not sel_internal or not sel_guidelines:
            st.error("❌ Please select at least one internal document and one guideline.")
        else:
            try:
                with st.spinner("🔄 Computing similarity matrices and TF-IDF vectors..."):
                    int_docs = [internal_docs[internal_names.index(n)] for n in sel_internal]
                    ref_docs = [guideline_docs[guideline_names.index(n)] for n in sel_guidelines]
                    
                    vectorizer, ref_vecs, int_vecs = build_tfidf_vectors(ref_docs, int_docs)
                    
                    sim_df = compute_cosine_similarity(ref_vecs, int_vecs, sel_internal)
                    
                    sim_df["similarity_percent"] = (sim_df["compliance_score"] * 100).round(1)
                    sim_df["divergence_percent"] = (100 - sim_df["similarity_percent"]).round(1)
                    sim_df["risk"] = sim_df["divergence_percent"].apply(risk_label)
                
                st.success("✅ Analysis completed!")
                
                # System Metrics
                st.markdown('<div class="section-header"><span class="section-icon">📈</span><h2>System Metrics</h2></div>', unsafe_allow_html=True)
                
                metrics = st.columns(5)
                
                with metrics[0]:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">📄 Documents</div>
                        <div class="metric-value">{}</div>
                    </div>
                    """.format(len(sel_internal)), unsafe_allow_html=True)
                
                with metrics[1]:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">📚 Guidelines</div>
                        <div class="metric-value">{}</div>
                    </div>
                    """.format(len(sel_guidelines)), unsafe_allow_html=True)
                
                with metrics[2]:
                    vocab_size = len(vectorizer.vocabulary_)
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">🔤 Vocabulary</div>
                        <div class="metric-value">{}</div>
                    </div>
                    """.format(vocab_size), unsafe_allow_html=True)
                
                with metrics[3]:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">📊 Matrix Size</div>
                        <div class="metric-value">{}×{}</div>
                    </div>
                    """.format(int_vecs.shape[0], int_vecs.shape[1]), unsafe_allow_html=True)
                
                with metrics[4]:
                    avg_sim = sim_df["similarity_percent"].mean()
                    st.markdown("""
                    <div class="metric-card success">
                        <div class="metric-label">📈 Avg Similarity</div>
                        <div class="metric-value">{:.1f}%</div>
                    </div>
                    """.format(avg_sim), unsafe_allow_html=True)
                
                # Visualizations
                st.markdown('<div class="section-header"><span class="section-icon">🔥</span><h2>Similarity Matrix Heatmap</h2></div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    sim_matrix = cosine_similarity(int_vecs, ref_vecs) * 100
                    im = ax.imshow(sim_matrix, cmap="YlGnBu", aspect='auto')
                    
                    ax.set_xticks(range(len(sel_guidelines)))
                    ax.set_yticks(range(len(sel_internal)))
                    ax.set_xticklabels([g[:20] for g in sel_guidelines], rotation=45, ha="right", fontsize=10)
                    ax.set_yticklabels([d[:20] for d in sel_internal], fontsize=10)
                    
                    for i in range(len(sel_internal)):
                        for j in range(len(sel_guidelines)):
                            text = ax.text(j, i, f'{sim_matrix[i, j]:.1f}',
                                         ha="center", va="center", color="black", fontsize=10, fontweight='600')
                    
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label("Similarity (%)", fontsize=11, fontweight='600')
                    ax.set_title("Cosine Similarity Matrix", fontweight='bold', fontsize=14, pad=15)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(10, 7))
                    bars = ax2.bar(range(len(sel_internal)), sim_df["divergence_percent"], width=0.6)
                    
                    for i, (bar, div) in enumerate(zip(bars, sim_df["divergence_percent"])):
                        bar.set_color(risk_color(div))
                    
                    ax2.axhline(divergence_threshold, color='#ef4444', linestyle='--', 
                               linewidth=2.5, label=f'Threshold ({divergence_threshold}%)', alpha=0.7)
                    ax2.set_xticks(range(len(sel_internal)))
                    ax2.set_xticklabels([d[:15] for d in sel_internal], rotation=45, ha="right", fontsize=10)
                    ax2.set_ylabel("Divergence (%)", fontsize=12, fontweight='600')
                    ax2.set_title("Compliance Divergence Analysis", fontweight='bold', fontsize=14, pad=15)
                    ax2.legend(fontsize=10)
                    ax2.grid(axis='y', alpha=0.3, linestyle='--')
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig2)
                
                # Results Table
                st.markdown('<div class="section-header"><span class="section-icon">📋</span><h2>Detailed Compliance Results</h2></div>', unsafe_allow_html=True)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)
                
                # Manual TF-IDF Section
                st.markdown("<br><br>", unsafe_allow_html=True)
                display_manual_tfidf(int_docs, sel_internal)
                
                # TF-IDF Matrix Exploration
                st.markdown('<div class="section-header"><span class="section-icon">🔍</span><h2>TF-IDF Matrix Exploration</h2></div>', unsafe_allow_html=True)
                
                feature_names = vectorizer.get_feature_names_out()
                
                selected_doc = st.selectbox(
                    "**Select document to view top terms:**",
                    sel_internal
                )
                
                doc_idx = sel_internal.index(selected_doc)
                doc_vector = int_vecs[doc_idx].toarray().flatten()
                
                top_indices = doc_vector.argsort()[-20:][::-1]
                top_terms = [(feature_names[i], doc_vector[i]) for i in top_indices if doc_vector[i] > 0]
                
                if top_terms:
                    terms_df = pd.DataFrame(top_terms, columns=['Term', 'TF-IDF Score'])
                    terms_df['Rank'] = range(1, len(terms_df) + 1)
                    terms_df = terms_df[['Rank', 'Term', 'TF-IDF Score']]
                    
                    col_table, col_chart = st.columns([1, 1])
                    
                    with col_table:
                        st.dataframe(terms_df, use_container_width=True, hide_index=True)
                    
                    with col_chart:
                        fig3, ax3 = plt.subplots(figsize=(10, 7))
                        bars = ax3.barh(terms_df['Term'], terms_df['TF-IDF Score'], color='#667eea', alpha=0.8)
                        ax3.set_xlabel('TF-IDF Score', fontsize=12, fontweight='600')
                        ax3.set_title(f'Top 20 Terms in {selected_doc[:30]}', fontweight='bold', fontsize=14, pad=15)
                        ax3.invert_yaxis()
                        ax3.grid(axis='x', alpha=0.3, linestyle='--')
                        ax3.spines['top'].set_visible(False)
                        ax3.spines['right'].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig3)
                
            except Exception as e:
                st.error(f"❌ Admin analysis failed: {str(e)}")
                st.exception(e)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <p style="font-weight: 600; color: #475569; margin-bottom: 0.5rem;">
        Universal Compliance Review System with OCR
    </p>
    <p style="font-size: 0.85rem;">
        MCA Final Project — PDF Processing + Tesseract OCR + TF-IDF Analysis<br>
        Decision Support Tool | Not Legal Advice
    </p>
</div>
""", unsafe_allow_html=True)
