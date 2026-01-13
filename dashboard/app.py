import streamlit as st
import pandas as pd

st.title("TF-IDF Compliance Drift Dashboard")

st.markdown("Upload or preview `results/drift_alerts.csv` to inspect flagged documents.")

uploaded = st.file_uploader("Upload a drift_alerts.csv", type=["csv"]) 

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.dataframe(df)
else:
    try:
        df = pd.read_csv("results/drift_alerts.csv")
        st.dataframe(df)
    except Exception:
        st.info("No results/drift_alerts.csv found. Run the analysis to generate alerts.")
