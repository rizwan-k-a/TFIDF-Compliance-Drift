CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ============ GLOBAL STYLES ============ */
html, body, [class*="css"], .stApp {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', BlinkMacSystemFont, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', sans-serif;
  -webkit-font-smoothing: antialiased;
}

.block-container {
  padding: 0.75rem 1.75rem 1rem 1.75rem;
  background: linear-gradient(135deg, #FFF2E2 0%, #faf6ef 100%);
  max-width: 1600px;
  margin-left: auto;
  margin-right: auto;
  width: 100%;
}

/* Reduce Streamlit default vertical gaps */
.stApp [data-testid="stVerticalBlock"] > div {
  gap: 0.5rem;
}

/* Remove default outlines/borders */
*:focus { outline: none !important; box-shadow: none !important; }

/* ============ SIDEBAR PREMIUM ============ */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #4F633D 0%, #3d4d2f 100%);
  border-right: none;
  width: 280px !important;
}
section[data-testid="stSidebar"] > div:first-child {
  background: transparent;
  width: 280px !important;
}

.stSidebar [data-testid="stMarkdownContainer"] p,
.stSidebar [data-testid="stMarkdownContainer"] label,
.stSidebar .stCheckbox label {
  color: #FFF2E2 !important;
  font-weight: 500;
}

.stSidebar [data-testid="stMarkdownContainer"] h3 {
  color: #FFF2E2 !important;
  font-weight: 700 !important;
  font-size: 1.25rem !important;
  letter-spacing: 0.02em;
  margin: 0.5rem 0 0.75rem 0 !important;
  padding: 0 !important;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}

.stSidebar .stCheckbox > label {
  background: rgba(255, 242, 226, 0.10);
  padding: 8px 12px;
  border-radius: 10px;
  border: 1px solid rgba(255, 242, 226, 0.25);
  transition: all 0.3s ease;
}
.stSidebar .stCheckbox > label:hover {
  background: rgba(255, 242, 226, 0.20);
  border: 1px solid rgba(255, 242, 226, 0.35);
  transform: translateX(4px);
}

.stSidebar hr {
  border-color: rgba(255, 242, 226, 0.30);
  margin: 0.75rem 0;
}

/* Compact sidebar content spacing */
.sidebar-compact {
  padding-top: 0.25rem;
}
.sidebar-compact h3 {
  margin: 0.5rem 0 0.75rem 0;
}

/* ============ HEADER SECTION ============ */
h1 {
  background: linear-gradient(135deg, #4F633D 0%, #8BA194 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-weight: 800;
  font-size: 2rem;
  margin: 0;
  line-height: 1.1;
}

/* Compact header block */
.header-compact {
  text-align: center;
  padding: 0.4rem 0 0.5rem 0;
  margin-bottom: 0.6rem;
}
.header-compact__title {
  margin: 0;
  line-height: 1.1;
}
.header-compact__subtitle {
  font-size: 0.92rem;
  color: #666;
  font-weight: 500;
  margin: 0.15rem 0 0 0;
  line-height: 1.2;
}

/* Ensure emojis/icons remain visible inside gradient headings */
h1 .emoji {
  background: none !important;
  -webkit-background-clip: initial !important;
  -webkit-text-fill-color: #4F633D !important;
  color: #4F633D !important;
}

/* Keep the sidebar toggle (collapsed control) visible and clickable */
header[data-testid="stHeader"] {
  visibility: visible;
  background: transparent;
  border-bottom: none;
  z-index: 1000;
}

header[data-testid="stHeader"] [data-testid="collapsedControl"] {
  visibility: visible;
  opacity: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: 10px;
  color: #4F633D;
}

header[data-testid="stHeader"] [data-testid="collapsedControl"] svg {
  width: 22px;
  height: 22px;
}

header[data-testid="stHeader"] [data-testid="collapsedControl"] svg,
header[data-testid="stHeader"] [data-testid="collapsedControl"] path {
  fill: #4F633D;
  stroke: #4F633D;
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
  padding-top: 0.9rem;
}

/* ============ UPLOAD CONTAINER (single bordered card) ============ */
.upload-container {
  border: 1px solid rgba(139, 161, 148, 0.40);
  background: rgba(255, 242, 226, 0.85);
  border-radius: 14px;
  padding: 0.75rem 1rem 0.85rem 1rem;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
  margin-bottom: 0.6rem;
}

.upload-container__title {
  font-size: 1.1rem;
  font-weight: 700;
  color: #4F633D;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.upload-container__status {
  font-size: 0.82rem;
  color: #777;
  margin: 0.5rem 0 0.4rem 0;
  text-align: center;
}

.upload-pane {
  background: rgba(139, 161, 148, 0.08);
  border: 1px dashed rgba(139, 161, 148, 0.45);
  border-radius: 10px;
  padding: 0.6rem 0.7rem;
  min-height: 90px;
}

.upload-pane__label {
  font-size: 0.85rem;
  font-weight: 600;
  color: #4F633D;
  margin-bottom: 0.35rem;
}

/* ============ INPUT PANELS (side-by-side upload + existing) ============ */
.input-panel {
  background: rgba(139, 161, 148, 0.06);
  border: 1px solid rgba(139, 161, 148, 0.30);
  border-radius: 10px;
  padding: 0.6rem 0.75rem;
  min-height: 140px;
}

.input-panel__header {
  font-size: 0.9rem;
  font-weight: 700;
  color: #4F633D;
  margin-bottom: 0.5rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid rgba(139, 161, 148, 0.20);
}

.upload-box-label {
  font-size: 0.78rem;
  font-weight: 600;
  color: #555;
  margin-bottom: 0.25rem;
}

.file-count-badge {
  font-size: 0.78rem;
  color: #4F633D;
  font-weight: 500;
  margin-top: 0.35rem;
  padding: 0.2rem 0.5rem;
  background: rgba(79, 99, 61, 0.12);
  border-radius: 5px;
  display: inline-block;
}

/* ============ FILE PICKER SECTION ============ */
.file-picker-section {
  background: rgba(139, 161, 148, 0.06);
  border: 1px solid rgba(139, 161, 148, 0.25);
  border-radius: 10px;
  padding: 0.7rem 0.9rem;
  margin-bottom: 0.5rem;
}

.file-picker__label {
  font-size: 0.85rem;
  font-weight: 600;
  color: #4F633D;
  display: flex;
  align-items: center;
  height: 100%;
}

.file-picker__list {
  margin-top: 0.4rem;
}

.file-picker__count {
  font-size: 0.8rem;
  color: #4F633D;
  font-weight: 500;
  margin-top: 0.3rem;
  padding: 0.25rem 0.5rem;
  background: rgba(79, 99, 61, 0.1);
  border-radius: 6px;
  display: inline-block;
}

/* Multiselect styling inside file picker */
.file-picker-section [data-testid="stMultiSelect"] {
  background: #fff;
  border-radius: 8px;
}

/* ============ FILE UPLOAD CARDS ============ */
[data-testid="stFileUploader"] {
  background: linear-gradient(135deg, #8BA19420, #8BA19410);
  border: 2px dashed #8BA194;
  border-radius: 16px;
  padding: 1.1rem;
  transition: all 0.4s ease;
}

/* Upload container column alignment */
.upload-container [data-testid="stHorizontalBlock"] {
  gap: 0.75rem;
}

.upload-container [data-testid="stFileUploader"] {
  margin-top: 0;
}

[data-testid="stFileUploader"]:hover {
  border-color: #4F633D;
  background: linear-gradient(135deg, #4F633D15, #8BA19420);
  box-shadow: 0 4px 12px rgba(79, 99, 61, 0.12);
}

[data-testid="stFileUploader"] button {
  background: linear-gradient(135deg, #4F633D, #5a7348) !important;
  color: #FFF2E2 !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 8px 20px !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"] button:hover {
  box-shadow: 0 4px 12px rgba(79, 99, 61, 0.35);
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

/* ============ METRICS & CARDS (compact) ============ */
[data-testid="stMetric"] {
  background: linear-gradient(135deg, #FFF2E2, #f8f4ed);
  padding: 0.6rem 0.8rem;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
  border: 1px solid rgba(139, 161, 148, 0.20);
  transition: transform 0.2s ease;
}

[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #4F633D;
  font-size: 1.5rem;
  font-weight: 700;
}

[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: #666;
  font-weight: 600;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.4px;
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
/* Keep header visible; it contains the sidebar toggle button */
header { visibility: visible; }

/* ============ RESPONSIVE ============ */
@media (max-width: 768px) {
  .block-container { padding: 1rem; }
  h1 { font-size: 2rem; }
  .stTabs [data-baseweb="tab"] { padding: 0 16px; font-size: 13px; }
  .upload-section [data-testid="stHorizontalBlock"] { grid-template-columns: 1fr; }
}
</style>
"""
