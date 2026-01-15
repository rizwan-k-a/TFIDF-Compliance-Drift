print("ALERTS.PY IS RUNNING")

"""
alerts.py

Purpose:
- Flag documents where compliance drift exceeds a threshold
"""

import pandas as pd


def generate_alerts(drift_df: pd.DataFrame, threshold: float = -0.05) -> pd.DataFrame:
    """
    Generate alert flags based on drift threshold
    """
    print("\nGenerating alerts...")

    drift_df = drift_df.copy()

    drift_df["alert"] = drift_df["drift"].apply(
        lambda x: "ALERT" if pd.notnull(x) and x < threshold else "OK"
    )

    return drift_df


# ===========================
# MAIN / DEBUG EXECUTION
# ===========================
if __name__ == "__main__":
    print("ENTERED MAIN BLOCK")

    # IMPORT PIPELINE MODULES
    from vectorize import load_documents, build_tfidf_vectors
    from similarity import compute_cosine_similarity
    from drift import compute_drift

    # LOAD DOCUMENTS
    print("\nLoading documents...")
    ref_docs, _ = load_documents("data/reference")
    int_docs, int_names = load_documents("data/internal")

    print("Reference docs:", len(ref_docs))
    print("Internal docs :", len(int_docs))

    if len(ref_docs) == 0 or len(int_docs) == 0:
        print("\n❌ ERROR: Document folders are empty.")
        exit()

    # TF-IDF
    print("\nBuilding TF-IDF vectors...")
    _, ref_vecs, int_vecs = build_tfidf_vectors(ref_docs, int_docs)

    print("TF-IDF shapes:", ref_vecs.shape, int_vecs.shape)

    # SIMILARITY
    print("\nComputing similarity...")
    sim_df = compute_cosine_similarity(ref_vecs, int_vecs, int_names)
    print(sim_df)

    # DRIFT
    print("\nComputing drift...")
    drift_df = compute_drift(sim_df, "data/metadata.csv")
    print(drift_df)

    # ALERTS
    alerts_df = generate_alerts(drift_df, threshold=-0.05)

    print("\n=== FINAL ALERT OUTPUT ===")
    print(alerts_df[[
        "doc_id",
        "filename",
        "version",
        "compliance_score",
        "drift",
        "alert"
    ]])
    
