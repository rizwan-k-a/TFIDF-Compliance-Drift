CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ============ GLOBAL STYLES ============ */
html, body, [class*="css"], .stApp {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', BlinkMacSystemFont, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', sans-serif;
  -webkit-font-smoothing: antialiased;
}

.block-container {
  /* authoritative content container for Streamlit main content */
  padding: 0.75rem 2rem 1rem 2rem;
  background: linear-gradient(135deg, #FFF2E2 0%, #faf6ef 100%);
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
  box-sizing: border-box;
}

/* Centered container wrapper used in app.py to scope main content */
.centered-container {
  max-width: 1200px;
  margin: 0 auto;
  padding-left: 24px;
  padding-right: 24px;
  box-sizing: border-box;
  width: 100%;
}

/* Ensure block-container remains centered and does not shift right */
@media (min-width: 769px) {
  .block-container {
    box-sizing: border-box;
    margin: 0 auto !important;
    width: 100%;
    max-width: 1200px !important;
    padding-top: 1rem;
  }
  /* avoid accidental horizontal scroll from small layout shifts */
  html, body { overflow-x: hidden; }
}
/* Reduce Streamlit default vertical gaps */
.stApp [data-testid="stVerticalBlock"] > div {
  gap: 0.5rem;
}

/* Accessibility: keep a subtle focus ring for keyboard users */
*:focus { outline: 3px solid rgba(139,161,148,0.12) !important; box-shadow: none !important; }

/* ============ SIDEBAR PREMIUM ============ */
section[data-testid="stSidebar"] {
  /* soften sidebar tone and slightly reduce width to improve main content space */
  background: #4F633D !important;
  border-right: 1px solid #3f5131;
  box-shadow: 3px 0 12px rgba(0, 0, 0, 0.12);
  width: 312px !important;
  min-width: 312px !important;
  max-width: 312px !important;
}
section[data-testid="stSidebar"] > div:first-child {
  background: #4F633D !important;
  width: 312px !important;
  min-width: 312px !important;
  max-width: 312px !important;
  padding: 1.1rem 0.85rem 1rem 0.85rem;
}

/* Make sidebar controls use full available width and match reference spacing */
.stSidebar .stCheckbox > label,
.stSidebar [data-testid="stSlider"] {
  display: block !important;
  width: 100% !important;
  box-sizing: border-box !important;
}
.stSidebar .stCheckbox > label {
  padding: 10px 12px !important;
  border-radius: 10px !important;
}
.sidebar-compact {
  padding-top: 0.4rem;
  padding-bottom: 0.6rem;
}

/* Fine tuning to better match reference visuals */
section[data-testid="stSidebar"] {
  min-height: 100vh;
  padding-top: 1.6rem;
  padding-left: 14px;
  padding-right: 14px;
}

.stSidebar .stCheckbox > label {
  background: #5B7150 !important;
  border: 1px solid #6f8463 !important;
  color: #FFF2E2 !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
}

/* Make checkboxes appear as compact option cards */
.stSidebar .stCheckbox > label {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 0.5rem !important;
}
.stSidebar .stCheckbox > label .stMarkdown { flex: 1 1 auto; }

/* Style the slider to match the aesthetic */
.stSidebar .stSlider > div {
  padding: 6px 8px !important;
  background: #5B7150;
  border-radius: 8px;
  border: 1px solid #6f8463;
}
.stSidebar .stSlider input[type="range"] { accent-color: #8BA194; }

/* Sidebar heading spacing */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.sidebar-settings-header {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 0.55rem;
  margin: 0.25rem 0 0.9rem 0;
}

.sidebar-settings-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 1.05rem;
  line-height: 1;
}

.sidebar-settings-title {
  color: #FFF2E2;
  font-weight: 700;
  font-size: 1.25rem;
  letter-spacing: 0.02em;
  line-height: 1.1;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}

.stSidebar .stCheckbox {
  margin: 0 0 0.75rem 0 !important;
}

.stSidebar .stCheckbox > label {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 0.5rem !important;
  width: 100% !important;
  box-sizing: border-box !important;
  overflow: visible !important;
}

.stSidebar .stCheckbox > label .stMarkdown {
  flex: 1 1 auto;
  min-width: 0;
}

.stSidebar .stCheckbox > label .stMarkdown p {
  margin: 0 !important;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.stSidebar .stCheckbox [data-testid="stWidgetLabel"] {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 0.5rem !important;
  width: 100% !important;
  box-sizing: border-box !important;
}

.stSidebar .stCheckbox [data-testid="stWidgetLabel"] > div:first-child {
  flex: 1 1 auto !important;
  min-width: 0 !important;
}

.stSidebar .stCheckbox [data-testid="stWidgetLabel"] p {
  margin: 0 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

.stSidebar .stCheckbox > label [data-testid="stTooltipIcon"],
.stSidebar .stCheckbox > label button {
  margin-left: auto !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  align-self: center !important;
  flex: 0 0 auto !important;
  width: 20px !important;
  height: 20px !important;
}

.stSidebar .stCheckbox [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"],
.stSidebar .stCheckbox [data-testid="stWidgetLabel"] button,
.stSidebar [data-testid="stSlider"] [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"],
.stSidebar [data-testid="stSlider"] [data-testid="stWidgetLabel"] button {
  position: static !important;
  inset: auto !important;
  margin-left: auto !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  align-self: center !important;
  flex: 0 0 auto !important;
  width: 20px !important;
  height: 20px !important;
}

.stSidebar .stSlider,
.stSidebar [data-testid="stSlider"] {
  width: 100% !important;
  margin: 0.35rem 0 0.8rem 0 !important;
}

.stSidebar .stSlider .css-1q8dd3e { /* slider track container fallback */
  width: 100% !important;
}

/* Reduce top gap so header aligns visually closer to reference */
.block-container { padding-top: 0.4rem; }

/* Slightly bump header toward top center */
h1 {
  margin-top: 0.2rem;
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
  background: #5B7150;
  padding: 8px 12px;
  border-radius: 10px;
  border: 1px solid #748968;
  transition: all 0.3s ease;
}
.stSidebar .stCheckbox > label:hover {
  background: #647a57;
  border: 1px solid #829877;
  transform: none;
}

.stSidebar hr {
  border-color: rgba(255, 242, 226, 0.30);
  margin: 0.75rem 0;
}

/* Compact sidebar content spacing */
.sidebar-compact {
  padding-top: 0.4rem;
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
  font-size: 1.85rem; /* slightly smaller for balance */
  margin: 0;
  line-height: 1.05;
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
.header-compact__icon { display:inline-block; vertical-align:middle; margin-right:0.9rem; }
.header-compact__icon svg { vertical-align: middle; border-radius: 8px; box-shadow: 0 4px 12px rgba(75,90,65,0.06); }
.header-compact__subtitle {
  font-size: 0.95rem;
  color: #6b6b63; /* slightly darker for better legibility */
  font-weight: 500;
  margin: 0.15rem 0 0 0;
  line-height: 1.2;
}

/* Make header icon larger and center-aligned like the screenshot */
.header-compact .emoji { font-size: 2.2rem; margin-right: 0.4rem; vertical-align: middle; }

/* Ensure header scales nicely on very large screens */
@media (min-width: 1400px) {
  .header-compact__title { font-size: 2.2rem; }
  .header-compact__icon svg { width: 44px; height: 44px; }
}

/* Center header in viewport on larger screens for perfect alignment */
@media (min-width: 769px) {

    /* Keep header centered within the content container (no absolute positioning).
      This ensures header aligns with the block-container grid without shifting the container. */
    .block-container .header-compact {
    position: relative !important;
    left: 0 !important;
    transform: none !important;
    top: auto !important;
    z-index: 950 !important;
    width: 100% !important;
    pointer-events: auto !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    margin: 0 auto 0.6rem auto !important;
  }

  /* small top spacing so header doesn't collide with top chrome */
  .block-container { padding-top: 1rem !important; }
}

/* Keep header in normal flow on small screens */
@media (max-width: 768px) {
  .header-compact { position: relative; transform: none; left: auto; top: auto; }
}

/* Emphasize Document Input section */
.block-container h2, .block-container h1 { color: #3e4a3f; font-weight: 800; }
.block-container h2::before { content: "📁"; margin-right: 0.5rem; }

/* Ensure main section headings align to left edge of centered container */
.centered-container h1, .centered-container h2, .centered-container h3, .centered-container .stSubheader {
  text-align: left !important;
  margin-left: 0 !important;
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
  height: 44px; /* reduced to reduce visual weight */
  background: #f7f7f7;
  border-radius: 12px 12px 0 0;
  border: none;
  padding: 0 20px;
  font-weight: 600;
  font-size: 14px;
  color: #555;
  transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Make tab icons stand out and add small separators */
.stTabs [data-baseweb="tab"] svg { margin-right: 8px; vertical-align: middle; }

.stTabs [data-baseweb="tab"]:hover {
  background: linear-gradient(135deg, #4F633D15, #8BA19415);
  color: #4F633D;
  transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #8BA194, #6b8a7e) !important;
  color: #FFF2E2 !important;
  box-shadow: 0 6px 16px rgba(139, 161, 148, 0.28);
}

.stTabs [data-baseweb="tab-panel"] {
  padding-top: 0.9rem;
}

/* ============ UPLOAD CONTAINER (single bordered card) ============ */
.upload-container {
  border: 1px solid rgba(139, 161, 148, 0.36);
  background: rgba(255, 242, 226, 0.92);
  border-radius: 12px;
  padding: 0.6rem 0.9rem 0.6rem 0.9rem;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.04);
  margin-bottom: 0.8rem;
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
  background: rgba(139, 161, 148, 0.06);
  border: 1px dashed rgba(139, 161, 148, 0.35);
  border-radius: 10px;
  padding: 0.5rem 0.6rem;
  min-height: 88px;
}

.upload-pane__label {
  font-size: 0.85rem;
  font-weight: 600;
  color: #4F633D;
  margin-bottom: 0.35rem;
}

/* Align browse button to the right in uploader for a clean affordance */
[data-testid="stFileUploader"] .css-1kyxreq { display:flex; align-items:center; justify-content:space-between; gap:12px; }
[data-testid="stFileUploader"] .css-1kyxreq button { flex: 0 0 auto; }

/* Metrics row spacing similar to screenshot */
[data-testid="stMetric"] { min-width: 220px; margin-right: 12px; }

/* Slightly reduce overall block padding for visual density similar to screenshot */
.block-container { padding: 0.6rem 2rem 1rem 2rem; }

/* ============ INPUT PANELS (side-by-side upload + existing) ============ */
.input-panel {
  background: rgba(139, 161, 148, 0.06);
  border: 1px solid rgba(139, 161, 148, 0.30);
  border-radius: 10px;
  padding: 0.6rem 0.75rem;
  min-height: 140px;
}

/* ============ LAYOUT FIXES FOR ALIGNMENT ============ */
/* ensure consistent centered max-width and symmetric padding */
.block-container {
  max-width: 1200px !important;
  padding-left: 28px !important;
  padding-right: 28px !important;
}

/* Keep section headings left-aligned with the content column edge */
.block-container h2, .block-container h3, .block-container h1, .block-container .stSubheader {
  text-align: left !important;
  margin-left: 0 !important;
}

/* Tabs: keep in one row with even spacing and prevent wrapping */
.stTabs [data-baseweb="tab-list"] {
  display: flex !important;
  flex-wrap: nowrap !important;
  white-space: nowrap !important;
  gap: 14px !important;
  justify-content: space-between !important;
  overflow-x: auto !important;
}
.stTabs [data-baseweb="tab"] { flex: 1 1 0 !important; text-align: center; }

/* Equal-height upload columns: wrapper class used in component */

.upload-eq { display: flex; flex-direction: column; height: 100%; width: 100%; box-sizing: border-box; }
.upload-eq .upload-container, .upload-eq [data-testid="stFileUploader"], .upload-eq .input-panel {
  height: 170px !important;
  min-height: 170px !important;
  max-height: 180px !important;
  box-sizing: border-box !important;
  padding: 0.45rem 0.6rem !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  align-items: stretch !important;
}

/* Ensure both upload boxes have identical visual dimensions within the .upload-eq scope */
.upload-eq [data-testid="stFileUploader"] { width: 100% !important; height: 170px !important; min-height: 170px !important; }

/* Align buttons to the same baseline and keep consistent sizing */
.stButton > button { display: inline-flex !important; align-items: center !important; vertical-align: baseline !important; }

/* Encourage column children to stretch so the upload wrappers match heights */
.stApp [data-testid="stHorizontalBlock"] > div {
  display: flex !important;
  flex-direction: column !important;
  align-items: stretch !important;
}

/* Prevent accidental overlapping text by ensuring header occupies flow */
.header-compact { z-index: 10; }

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

.stButton > button {
  background: linear-gradient(135deg, #4F633D 0%, #5a7348 100%);
  color: #FFF2E2;
  border-radius: 12px;
  border: none;
  padding: 10px 22px;
  font-weight: 600;
  font-size: 14px;
  transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 6px 14px rgba(79, 99, 61, 0.18);
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(79, 99, 61, 0.28);
}
.stButton > button:active {
  transform: translateY(-1px);
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

/* Mobile-first: collapse sidebar into a top panel and make content full-width */
@media (max-width: 900px) {
  section[data-testid="stSidebar"] {
    position: relative !important;
    top: auto;
    left: auto;
    right: auto;
    height: 100vh !important;
    z-index: auto;
    padding: 1.1rem 0.85rem 1rem 0.85rem !important;
    border-right: 1px solid #3f5131 !important;
    border-bottom: none !important;
  }
  header[data-testid="stHeader"] [data-testid="collapsedControl"] { display: block !important; }
  .stTabs [data-baseweb="tab-list"] { overflow-x: auto; }
  .upload-container, .input-panel { min-width: auto; }
}

/* Skip link for keyboard users (visible on focus) */
.skip-link {
  position: absolute;
  left: -999px;
  top: 8px;
  background: #f7f7f7;
  color: #2f3b2f;
  padding: 8px 12px;
  border-radius: 6px;
  z-index: 2000;
}
.skip-link:focus { left: 12px; }

/* ============ SIDEBAR ALIGNMENT FINAL OVERRIDE ============ */
html, body {
  margin: 0 !important;
  padding: 0 !important;
  height: 100% !important;
}

.stApp,
[data-testid="stAppViewContainer"] {
  margin: 0 !important;
  padding: 0 !important;
  min-height: 100vh !important;
}

[data-testid="stAppViewContainer"] > .main {
  margin: 0 !important;
  padding-left: 0 !important;
}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child {
  width: 312px !important;
  min-width: 312px !important;
  max-width: 312px !important;
  box-sizing: border-box !important;
  overflow-x: hidden !important;
  background: #4F633D !important;
  height: 100vh !important;
}

section[data-testid="stSidebar"] {
  margin: 0 !important;
  padding: 0 !important;
  min-height: 100vh !important;
  border-right: 1px solid #3f5131 !important;
  box-shadow: none !important;
}

section[data-testid="stSidebar"] > div:first-child {
  margin: 0 !important;
  padding: 1rem 0.7rem 1rem 0.7rem !important;
  min-height: 100vh !important;
}

.stSidebar .stCheckbox {
  margin: 0 0 0.65rem 0 !important;
}

.stSidebar .stCheckbox > label {
  width: 100% !important;
  box-sizing: border-box !important;
  padding: 8px 10px !important;
  overflow: hidden !important;
}

.stSidebar .stCheckbox [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stWidgetLabel"] {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 0.45rem !important;
  width: 100% !important;
  box-sizing: border-box !important;
  padding-right: 8px !important;
}

.stSidebar .stCheckbox [data-testid="stWidgetLabel"] > div:first-child,
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stWidgetLabel"] > div:first-child {
  flex: 1 1 auto !important;
  min-width: 0 !important;
}

.stSidebar .stCheckbox [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {
  margin: 0 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] button {
  position: static !important;
  margin-left: auto !important;
  margin-right: 0 !important;
  flex: 0 0 20px !important;
  width: 20px !important;
  height: 20px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}

.stSidebar .stSlider,
.stSidebar [data-testid="stSlider"] {
  width: 100% !important;
  box-sizing: border-box !important;
  margin: 0.35rem 0 0.8rem 0 !important;
}

/* Stronger focus outline for interactive elements */
button:focus, a:focus, input:focus, select:focus, textarea:focus { outline: 3px solid rgba(139,161,148,0.18) !important; }

/* ============ SIDEBAR WIDGET LABEL NORMALIZATION ============ */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
  display: flex !important;
  justify-content: space-between !important;
  align-items: center !important;
  width: 100% !important;
  box-sizing: border-box !important;
  padding-right: 8px !important;
  gap: 0.5rem !important;
}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] > div:first-child {
  flex: 1 1 auto !important;
  min-width: 0 !important;
}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  margin: 0 !important;
}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] button {
  margin-left: auto !important;
  margin-right: 0 !important;
  align-self: center !important;
  flex: 0 0 auto !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}
</style>
"""
