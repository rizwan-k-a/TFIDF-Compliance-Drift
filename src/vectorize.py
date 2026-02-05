# LEGACY — not used in main demo path
# ============================================================
# ============================================================
# TF-IDF VECTORIZATION MODULE
# Robust, Academic-Safe Implementation
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectors(reference_docs, internal_docs):
    """
    Builds TF-IDF vectors for reference and internal documents.

    This implementation is intentionally defensive to handle:
    - Identical documents
    - Very short texts
    - Single document uploads
    - Stopword-heavy content

    This prevents:
    ValueError: After pruning, no terms remain
    """

    # --------------------------------------------------------
    # Combine documents
    # --------------------------------------------------------
    all_docs = reference_docs + internal_docs

    # --------------------------------------------------------
    # Defensive cleaning
    # --------------------------------------------------------
    cleaned_docs = []
    for doc in all_docs:
        if doc and doc.strip():
            cleaned_docs.append(doc.strip())

    if len(cleaned_docs) < 2:
        raise ValueError(
            "Not enough meaningful text for TF-IDF computation."
        )

    # --------------------------------------------------------
    # Primary TF-IDF attempt (normal configuration)
    # --------------------------------------------------------
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,            # IMPORTANT: allow rare terms
            max_df=1.0,          # IMPORTANT: no aggressive pruning
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(all_docs)

    # --------------------------------------------------------
    # Fallback TF-IDF (no stopwords, maximum tolerance)
    # --------------------------------------------------------
    except ValueError:
        vectorizer = TfidfVectorizer(
            stop_words=None,
            min_df=1,
            max_df=1.0
        )
        tfidf_matrix = vectorizer.fit_transform(all_docs)

    # --------------------------------------------------------
    # Split reference and internal vectors
    # --------------------------------------------------------
    ref_count = len(reference_docs)
    ref_vectors = tfidf_matrix[:ref_count]
    int_vectors = tfidf_matrix[ref_count:]

    return vectorizer, ref_vectors, int_vectors