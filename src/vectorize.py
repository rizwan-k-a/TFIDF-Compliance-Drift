"""
vectorize.py

Purpose:
- Convert text documents into TF-IDF vectors
- Fit on combined corpus to ensure fair comparison
"""

import os
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


def load_documents(folder_path: str) -> Tuple[List[str], List[str]]:
    documents = []
    filenames = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".txt"):
            with open(
                os.path.join(folder_path, file),
                "r",
                encoding="utf-8"
            ) as f:
                documents.append(f.read())
                filenames.append(file)

    return documents, filenames


def build_tfidf_vectors(
    reference_docs: List[str],
    internal_docs: List[str]
):
    """
    Build TF-IDF vectors using a shared vocabulary
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 1),     # 🔑 important for short documents
        min_df=1,
        max_df=0.95,
        norm="l2"
    )


    # 🔑 Fit on combined corpus
    all_docs = reference_docs + internal_docs
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    # Split back
    ref_vectors = tfidf_matrix[:len(reference_docs)]
    int_vectors = tfidf_matrix[len(reference_docs):]

    return vectorizer, ref_vectors, int_vectors


# -------------------------------------------------
# DEBUG / SANITY CHECK
# -------------------------------------------------
if __name__ == "__main__":
    ref_docs, _ = load_documents("data/reference")
    int_docs, _ = load_documents("data/internal")

    vectorizer, ref_vecs, int_vecs = build_tfidf_vectors(ref_docs, int_docs)

    print("Reference vectors shape:", ref_vecs.shape)
    print("Internal vectors shape:", int_vecs.shape)
    print("TF-IDF vocabulary size:", len(vectorizer.vocabulary_))
