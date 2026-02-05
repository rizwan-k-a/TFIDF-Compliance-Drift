"""LEGACY — not used in main demo path.

Kept for reference/educational purposes.
"""

"""
drift.py

Purpose:
- Track compliance score changes across document versions
- Quantify compliance drift over time
"""

import pandas as pd


def compute_drift(similarity_df: pd.DataFrame, metadata_path: str) -> pd.DataFrame:
    """
    Computes drift (change in compliance score) across document versions.

    Parameters:
    - similarity_df: DataFrame with internal_document, compliance_score
    - metadata_path: path to metadata.csv

    Returns:
    - DataFrame with version-wise drift values
    """

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Normalize filenames for safe merge
    metadata["filename"] = metadata["filename"].astype(str)
    similarity_df["internal_document"] = similarity_df["internal_document"].astype(str)

    # Merge similarity scores with metadata
    merged = metadata.merge(
        similarity_df,
        left_on="filename",
        right_on="internal_document",
        how="inner"
    )

    # Sort by document id and version
    merged = merged.sort_values(by=["doc_id", "version"])

    # Compute drift: Δ score between versions
    merged["drift"] = merged.groupby("doc_id")["compliance_score"].diff()

    return merged


# ===========================
# TEST / DEBUG EXECUTION
# ===========================
if __name__ == "__main__":
    print("ENTERED MAIN BLOCK")

    from similarity import compute_cosine_similarity
    from vectorize import load_documents, build_tfidf_vectors

    # Load documents
    ref_docs, _ = load_documents("data/reference")
    int_docs, int_names = load_documents("data/internal")

    # Vectorize
    _, ref_vecs, int_vecs = build_tfidf_vectors(ref_docs, int_docs)

    # Similarity
    sim_df = compute_cosine_similarity(ref_vecs, int_vecs, int_names)
    print("\nSimilarity Scores:")
    print(sim_df)

    # Drift
    drift_df = compute_drift(sim_df, "data/metadata.csv")

    print("\n=== DRIFT ANALYSIS ===")
    print(drift_df[[
        "doc_id",
        "filename",
        "version",
        "date",
        "compliance_score",
        "drift"
    ]])
