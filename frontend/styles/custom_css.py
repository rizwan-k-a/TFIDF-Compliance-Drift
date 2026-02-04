CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ============ GLOBAL STYLES ============ */
html, body, [class*="css"], .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.block-container {
  padding: 2rem 3rem;
  background: linear-gradient(135deg, #FFF2E2 0%, #faf6ef 100%);
  max-width: 1400px;
}

/* Remove default outlines/borders */
*:focus { outline: none !important; box-shadow: none !important; }

/* ============ SIDEBAR PREMIUM ============ */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #4F633D 0%, #3d4d2f 100%);
  border-right: none;
}
section[data-testid="stSidebar"] > div:first-child {
  background: transparent;
}

.stSidebar [data-testid="stMarkdownContainer"] p,
.stSidebar [data-testid="stMarkdownContainer"] label,
.stSidebar .stCheckbox label {
  color: #FFF2E2 !important;
  font-weight: 500;
}

.stSidebar .stCheckbox > label {
  background: rgba(255, 242, 226, 0.10);
  padding: 12px 16px;
  border-radius: 12px;
  transition: all 0.3s ease;
}
.stSidebar .stCheckbox > label:hover {
  background: rgba(255, 242, 226, 0.20);
  transform: translateX(4px);
}

.stSidebar hr {
  border-color: rgba(255, 242, 226, 0.30);
  margin: 1.5rem 0;
}

/* ============ HEADER SECTION ============ */
h1 {
  background: linear-gradient(135deg, #4F633D 0%, #8BA194 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-weight: 800;
  font-size: 3rem;
  margin-bottom: 0.5rem;
}

/* ============ TABS PREMIUM ============ */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background: transparent;
  border-bottom: 2px solid #e5e5e5;
  padding-bottom: 0;
}

.stTabs [data-baseweb="tab"] {
  height: 56px;
  background: #f5f5f5;
  border-radius: 16px 16px 0 0;
  border: none;
  padding: 0 28px;
  font-weight: 600;
  font-size: 15px;
  color: #666;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stTabs [data-baseweb="tab"]:hover {
  background: linear-gradient(135deg, #4F633D15, #8BA19415);
  color: #4F633D;
  transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #8BA194, #6b8a7e) !important;
  color: #FFF2E2 !important;
  box-shadow: 0 6px 20px rgba(139, 161, 148, 0.40);
}

.stTabs [data-baseweb="tab-panel"] {
  padding-top: 2rem;
}

/* ============ FILE UPLOAD CARDS ============ */
[data-testid="stFileUploader"] {
  background: linear-gradient(135deg, #8BA19420, #8BA19410);
  border: 2px dashed #8BA194;
  border-radius: 20px;
  padding: 2rem;
  transition: all 0.4s ease;
}

[data-testid="stFileUploader"]:hover {
  border-color: #4F633D;
  background: linear-gradient(135deg, #4F633D15, #8BA19420);
  box-shadow: 0 8px 24px rgba(79, 99, 61, 0.15);
  transform: translateY(-4px);
}

[data-testid="stFileUploader"] button {
  background: linear-gradient(135deg, #4F633D, #5a7348) !important;
  color: #FFF2E2 !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 12px 28px !important;
  font-weight: 600 !important;
  transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"] button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(79, 99, 61, 0.40);
}

/* ============ BUTTONS PREMIUM ============ */
.stButton > button {
  background: linear-gradient(135deg, #4F633D 0%, #5a7348 100%);
  color: #FFF2E2;
  border-radius: 14px;
  border: none;
  padding: 14px 32px;
  font-weight: 600;
  font-size: 15px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 4px 12px rgba(79, 99, 61, 0.25);
}

.stButton > button:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(79, 99, 61, 0.40);
}

.stButton > button:active {
  transform: translateY(-1px);
}

/* ============ METRICS & CARDS ============ */
[data-testid="stMetric"] {
  background: linear-gradient(135deg, #FFF2E2, #f8f4ed);
  padding: 1.5rem;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(139, 161, 148, 0.20);
  transition: transform 0.3s ease;
}

[data-testid="stMetric"]:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.12);
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #4F633D;
  font-size: 2.5rem;
  font-weight: 700;
}

[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: #666;
  font-weight: 600;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* ============ EXPANDER PREMIUM ============ */
.streamlit-expanderHeader {
  background: linear-gradient(135deg, #8BA194, #6b8a7e);
  color: #FFF2E2 !important;
  border-radius: 14px;
  padding: 1rem 1.5rem;
  font-weight: 600;
  border: none;
  transition: all 0.3s ease;
}

.streamlit-expanderHeader:hover {
  transform: translateX(4px);
  box-shadow: 0 4px 16px rgba(139, 161, 148, 0.30);
}

.streamlit-expanderContent {
  border: 2px solid #8BA194;
  border-radius: 0 0 14px 14px;
  border-top: none;
  padding: 1.5rem;
  background: #FFF2E2;
}

/* ============ DATAFRAMES & TABLES ============ */
[data-testid="stDataFrame"] {
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.10);
}

/* best-effort striped rows for dataframe */
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
  background-color: rgba(139, 161, 148, 0.06);
}

/* ============ RISK BADGES (Custom Classes) ============ */
.risk-badge {
  display: inline-block;
  padding: 8px 20px;
  border-radius: 24px;
  font-weight: 700;
  font-size: 14px;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.risk-compliant {
  background: linear-gradient(135deg, #4F633D, #5a7348);
  color: #FFF2E2;
}

.risk-low {
  background: linear-gradient(135deg, #7cb342, #8bc34a);
  color: #ffffff;
}

.risk-medium {
  background: linear-gradient(135deg, #ffa726, #ffb74d);
  color: #ffffff;
}

.risk-high {
  background: linear-gradient(135deg, #ff7043, #ff8a65);
  color: #ffffff;
}

.risk-critical {
  background: linear-gradient(135deg, #e53935, #ef5350);
  color: #ffffff;
}

/* ============ SPINNER BRANDING ============ */
[data-testid="stSpinner"] div {
  border-top-color: #4F633D !important;
}

/* ============ SCROLLBARS ============ */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #FFF2E2; border-radius: 10px; }
::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #8BA194, #4F633D); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: linear-gradient(135deg, #4F633D, #8BA194); }

/* ============ REMOVE DEFAULTS ============ */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
header { visibility: hidden; }

/* ============ RESPONSIVE ============ */
@media (max-width: 768px) {
  .block-container { padding: 1rem; }
  h1 { font-size: 2rem; }
  .stTabs [data-baseweb="tab"] { padding: 0 16px; font-size: 13px; }
}
</style>
"""
