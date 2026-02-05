"""LEGACY — not used in main demo path.

Kept for reference/educational purposes.
The Streamlit demo uses backend/similarity.py.
"""

"""
similarity.py

Purpose:
- Compute cosine similarity between internal documents and reference documents
- Output compliance scores for each internal document
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(ref_vectors, int_vectors, int_names):
    """
    Computes maximum cosine similarity score for each internal document
    against all reference documents.
    """
    similarity_matrix = cosine_similarity(int_vectors, ref_vectors)

    compliance_scores = similarity_matrix.max(axis=1)

    results_df = pd.DataFrame({
        "internal_document": int_names,
        "compliance_score": compliance_scores
    })

    return results_df


# ===========================
# TEST / DEBUG EXECUTION
# ===========================
if __name__ == "__main__":
    print("ENTERED MAIN BLOCK")

    # Import vectorization utilities
    from vectorize import load_documents, build_tfidf_vectors

    # Load reference documents
    print("\nLoading reference documents...")
    ref_docs, ref_names = load_documents("data/reference")
    print("Reference files found:", ref_names)

    # Load internal documents
    print("\nLoading internal documents...")
    int_docs, int_names = load_documents("data/internal")
    print("Internal files found:", int_names)

    # Check counts
    print("\nDocument counts:")
    print("Reference:", len(ref_docs))
    print("Internal :", len(int_docs))

    # Guard against empty folders
    if len(ref_docs) == 0 or len(int_docs) == 0:
        print("\n❌ ERROR: One or more document folders are empty.")
        print("Make sure .txt files exist in:")
        print(" - data/reference/")
        print(" - data/internal/")
        exit()

    # Build TF-IDF vectors
    print("\nBuilding TF-IDF vectors...")
    _, ref_vecs, int_vecs = build_tfidf_vectors(ref_docs, int_docs)

    print("TF-IDF shapes:")
    print("Reference vectors:", ref_vecs.shape)
    print("Internal vectors :", int_vecs.shape)

    # Compute similarity
    print("\nComputing cosine similarity...")
    df = compute_cosine_similarity(ref_vecs, int_vecs, int_names)

    print("\n=== COMPLIANCE SCORES ===")
    print(df)
