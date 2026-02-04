CUSTOM_CSS = """
<style>
:root {
  --brand: #7c3aed;
  --text: #111827;
  --muted: #6b7280;
  --panel: #0b1220;
}

.block-container { padding-top: 1.2rem; }

.app-header {
  padding: 1rem 1.25rem;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(59,130,246,0.10));
  border: 1px solid rgba(124,58,237,0.25);
  margin-bottom: 1rem;
}

.app-header h1 { margin: 0; font-size: 1.6rem; color: var(--text); }
.app-header p { margin: .25rem 0 0 0; color: var(--muted); }

.metric-card {
  padding: .75rem 1rem;
  border-radius: 12px;
  border: 1px solid rgba(17,24,39,0.12);
  background: rgba(255,255,255,0.65);
}

.small-muted { color: var(--muted); font-size: .9rem; }

</style>
"""
