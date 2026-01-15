"""
Manual TF-IDF and Cosine Similarity Implementation

This module provides step-by-step implementations of TF-IDF weighting and 
cosine similarity calculations for educational purposes and verification.

References:
- TF-IDF formula: tfidf_{t,d} = tf_{t,d} * log(N / df_t)
- Cosine similarity: cos(a,b) = (a · b) / (||a|| * ||b||)
"""

import math
from collections import Counter
from typing import List, Dict, Tuple


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization: lowercase and split by whitespace.
    
    Args:
        text: Input document text.
        
    Returns:
        List of tokens.
    """
    return text.lower().split()


def compute_term_frequency(tokens: List[str]) -> Dict[str, float]:
    """
    Compute raw term frequency (term count / total tokens).
    
    Args:
        tokens: List of tokens from a document.
        
    Returns:
        Dictionary mapping term -> frequency.
    """
    if not tokens:
        return {}
    
    counter = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counter.items()}


def compute_document_frequency(documents: List[List[str]]) -> Dict[str, int]:
    """
    Compute document frequency (number of documents containing each term).
    
    Args:
        documents: List of tokenized documents.
        
    Returns:
        Dictionary mapping term -> count of documents containing it.
    """
    df = Counter()
    for doc in documents:
        unique_terms = set(doc)
        df.update(unique_terms)
    return dict(df)


def compute_idf(doc_freq: Dict[str, int], num_documents: int) -> Dict[str, float]:
    """
    Compute inverse document frequency: log(N / df_t).
    
    Args:
        doc_freq: Document frequency dictionary.
        num_documents: Total number of documents.
        
    Returns:
        Dictionary mapping term -> IDF value.
    """
    idf = {}
    for term, df in doc_freq.items():
        # Add 1 to avoid division by zero and log(1) = 0
        idf[term] = math.log(num_documents / (1 + df))
    return idf


def compute_tfidf_vector(
    tf: Dict[str, float],
    idf: Dict[str, float],
    vocabulary: set = None
) -> Dict[str, float]:
    """
    Compute TF-IDF vector for a single document.
    
    Args:
        tf: Term frequency dictionary for the document.
        idf: IDF dictionary (computed once for corpus).
        vocabulary: Optional set of terms to include. If None, use all terms in tf.
        
    Returns:
        Dictionary mapping term -> TF-IDF weight.
    """
    if vocabulary is None:
        vocabulary = set(tf.keys())
    
    tfidf = {}
    for term in vocabulary:
        tf_val = tf.get(term, 0.0)
        idf_val = idf.get(term, 0.0)
        tfidf[term] = tf_val * idf_val
    
    return tfidf


def build_tfidf_matrix(
    documents: List[List[str]]
) -> Tuple[List[Dict[str, float]], Dict[str, float], set]:
    """
    Build TF-IDF vectors for all documents in a corpus.
    
    Args:
        documents: List of tokenized documents.
        
    Returns:
        Tuple of:
            - tfidf_vectors: List of TF-IDF dictionaries (one per document).
            - idf: IDF dictionary for the corpus.
            - vocabulary: Set of all unique terms.
    """
    # Compute IDF
    doc_freq = compute_document_frequency(documents)
    idf = compute_idf(doc_freq, len(documents))
    
    # Build vocabulary
    vocabulary = set(doc_freq.keys())
    
    # Compute TF-IDF for each document
    tfidf_vectors = []
    for doc in documents:
        tf = compute_term_frequency(doc)
        tfidf = compute_tfidf_vector(tf, idf, vocabulary)
        tfidf_vectors.append(tfidf)
    
    return tfidf_vectors, idf, vocabulary


def vector_norm(vec: Dict[str, float]) -> float:
    """
    Compute Euclidean norm of a vector: sqrt(sum of squares).
    
    Args:
        vec: Dictionary representation of a vector.
        
    Returns:
        Euclidean norm.
    """
    return math.sqrt(sum(val ** 2 for val in vec.values()))


def dot_product(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute dot product of two vectors.
    
    Args:
        vec1: First vector (dictionary).
        vec2: Second vector (dictionary).
        
    Returns:
        Dot product value.
    """
    result = 0.0
    for term in set(vec1.keys()) & set(vec2.keys()):
        result += vec1[term] * vec2[term]
    return result


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Formula: cos(a, b) = (a · b) / (||a|| * ||b||)
    
    Args:
        vec1: First vector (dictionary).
        vec2: Second vector (dictionary).
        
    Returns:
        Cosine similarity in range [0, 1].
    """
    dot = dot_product(vec1, vec2)
    norm1 = vector_norm(vec1)
    norm2 = vector_norm(vec2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)


def compute_pairwise_similarity(
    query_vectors: List[Dict[str, float]],
    reference_vectors: List[Dict[str, float]]
) -> List[List[float]]:
    """
    Compute all pairwise cosine similarities between query and reference vectors.
    
    Args:
        query_vectors: List of query TF-IDF vectors.
        reference_vectors: List of reference TF-IDF vectors.
        
    Returns:
        2D list of shape (len(query_vectors), len(reference_vectors))
        where element [i, j] is the similarity between query_vectors[i] 
        and reference_vectors[j].
    """
    similarities = []
    for q_vec in query_vectors:
        row = [cosine_similarity(q_vec, r_vec) for r_vec in reference_vectors]
        similarities.append(row)
    return similarities


def max_similarity_per_query(
    similarities: List[List[float]]
) -> List[float]:
    """
    Extract maximum similarity for each query document.
    
    Args:
        similarities: Pairwise similarity matrix.
        
    Returns:
        List of maximum similarities, one per query.
    """
    return [max(row) if row else 0.0 for row in similarities]


def compute_drift(max_similarities: List[float]) -> List[float]:
    """
    Compute drift score as 1 - max_similarity for each document.
    
    Drift ranges from 0 (perfect match) to 1 (no match).
    
    Args:
        max_similarities: List of maximum similarities per document.
        
    Returns:
        List of drift scores.
    """
    return [1.0 - sim for sim in max_similarities]


# =============================================================================
# MANUAL EXAMPLE: Step-by-step TF, IDF, TF-IDF computation on small corpus
# =============================================================================
# Purpose: Demonstrate manual computation for mathematics evaluation
# Uses a very small sample corpus for clarity and verification

def get_manual_tfidf_output():
    """
    Compute TF, IDF, and TF-IDF manually on a small, fixed corpus.
    Returns results as a dictionary for display and verification.
    """
    # Small sample documents
    documents_dict = {
        "D1": "patient consent explains risks",
        "D2": "patient consent includes risks",
        "D3": "hospital policy explains patient rights"
    }
    
    terms = ["patient", "consent", "risks", "policy"]
    N = len(documents_dict)
    
    # Compute Term Frequency (TF)
    tf_values = {}
    for doc, text in documents_dict.items():
        words = text.split()
        total_words = len(words)
        counts = Counter(words)
        tf_values[doc] = {
            term: counts.get(term, 0) / total_words 
            for term in terms
        }
    
    # Compute Document Frequency (DF)
    df = {}
    for term in terms:
        df[term] = sum(
            1 for text in documents_dict.values() 
            if term in text.split()
        )
    
    # Compute Inverse Document Frequency (IDF)
    idf = {term: math.log(N / df[term]) for term in terms}
    
    # Compute TF-IDF
    tfidf_values = {}
    for doc in documents_dict:
        tfidf_values[doc] = {
            term: tf_values[doc][term] * idf[term]
            for term in terms
        }
    
    return {
        "documents": documents_dict,
        "terms": terms,
        "N": N,
        "TF": tf_values,
        "DF": df,
        "IDF": idf,
        "TF-IDF": tfidf_values
    }


# Example usage and testing
if __name__ == "__main__":
    from pprint import pprint
    
    print("=" * 70)
    print("MANUAL TF-IDF COMPUTATION ON SMALL CORPUS")
    print("=" * 70)
    
    # Get manual example results
    manual_results = get_manual_tfidf_output()
    
    print("\nDocuments:")
    for doc_id, text in manual_results["documents"].items():
        print(f"  {doc_id}: \"{text}\"")
    
    print("\nTerms:", manual_results["terms"])
    print("N (total documents):", manual_results["N"])
    
    print("\n--- TERM FREQUENCY (TF) ---")
    pprint(manual_results["TF"])
    
    print("\n--- DOCUMENT FREQUENCY (DF) ---")
    pprint(manual_results["DF"])
    
    print("\n--- INVERSE DOCUMENT FREQUENCY (IDF) ---")
    print("Formula: IDF(t) = log(N / DF(t))")
    pprint(manual_results["IDF"])
    
    print("\n--- TF-IDF VALUES ---")
    print("Formula: TF-IDF(t,d) = TF(t,d) × IDF(t)")
    pprint(manual_results["TF-IDF"])
    
    print("\n" + "=" * 70)
    print("SKLEARN-STYLE EXAMPLE (for comparison)")
    print("=" * 70)
    
    # Sample documents for sklearn-style example
    docs_text = [
        "privacy policy data protection consent",
        "privacy terms service user agreement",
        "cookie policy tracking user consent",
    ]
    
    # Tokenize
    documents = [tokenize(doc) for doc in docs_text]
    print("\nDocuments (tokenized):")
    for i, doc in enumerate(documents):
        print(f"  Doc {i}: {doc}")
    
    # Build TF-IDF
    tfidf_vectors, idf, vocab = build_tfidf_matrix(documents)
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"IDF scores (sample): {dict(list(idf.items())[:5])}")
    
    # Compute pairwise similarity
    similarities = compute_pairwise_similarity(tfidf_vectors, tfidf_vectors)
    print(f"\nSimilarity matrix shape: {len(similarities)} x {len(similarities[0])}")
    print("Similarity matrix:")
    for i, row in enumerate(similarities):
        print(f"  Doc {i}: {[f'{s:.3f}' for s in row]}")
    
    # Compute drift for first two documents
    max_sims = max_similarity_per_query(similarities[:2])
    drift_scores = compute_drift(max_sims)
    print(f"\nMax similarities (docs 0-1): {max_sims}")
    print(f"Drift scores (docs 0-1): {drift_scores}")
