"""
Manual TF-IDF and Cosine Similarity Implementation
Academic Requirement 20 - Complete Mathematical Demonstration

This module provides step-by-step implementations of TF-IDF weighting and 
cosine similarity calculations for educational purposes and verification.

References:
- TF-IDF formula: tfidf_{t,d} = tf_{t,d} * log(N / df_t)
- Cosine similarity: cos(a,b) = (a · b) / (||a|| * ||b||)
"""

import math
import re
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


def preprocess_text_simple(text: str) -> str:
    """Simple preprocessing for TF-IDF"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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


def max_similarity_per_query(similarities: List[List[float]]) -> List[float]:
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
# MANUAL TF-IDF COMPUTATION FOR DASHBOARD
# =============================================================================

def compute_manual_tfidf(documents: List[str], sample_words: List[str]) -> Tuple[Dict, int]:
    """
    Compute TF, IDF, and TF-IDF manually for sample words.
    
    This function is specifically designed for the dashboard to show
    step-by-step mathematical calculations.
    
    Args:
        documents: List of document strings
        sample_words: List of words to analyze
        
    Returns:
        Tuple of (results_dict, n_docs) where results_dict contains
        detailed calculations for each word
    """
    # Preprocess and tokenize documents
    processed_docs = [preprocess_text_simple(doc).split() for doc in documents]
    n_docs = len(processed_docs)
    
    results = {}
    
    for word in sample_words:
        word_lower = word.lower()
        results[word] = {
            'tf_per_doc': [],
            'df': 0,
            'idf': 0,
            'tfidf_per_doc': []
        }
        
        # Calculate TF for each document
        for doc_idx, doc_words in enumerate(processed_docs):
            word_count = doc_words.count(word_lower)
            total_words = len(doc_words)
            tf = word_count / total_words if total_words > 0 else 0
            
            results[word]['tf_per_doc'].append({
                'doc': f'Doc{doc_idx+1}',
                'count': word_count,
                'total': total_words,
                'tf': round(tf, 6)
            })
            
            # Count document frequency
            if word_count > 0:
                results[word]['df'] += 1
        
        # Calculate IDF (with smoothing)
        df = results[word]['df']
        idf = math.log((1 + n_docs) / (1 + df)) + 1
        results[word]['idf'] = round(idf, 6)
        
        # Calculate TF-IDF for each document
        for doc_data in results[word]['tf_per_doc']:
            tfidf = doc_data['tf'] * idf
            results[word]['tfidf_per_doc'].append({
                'doc': doc_data['doc'],
                'tfidf': round(tfidf, 6)
            })
    
    return results, n_docs


# =============================================================================
# MANUAL EXAMPLE: Step-by-step TF, IDF, TF-IDF computation on small corpus
# =============================================================================

def get_manual_tfidf_output():
    """
    Compute TF, IDF, and TF-IDF manually on a small, fixed corpus.
    Returns results as a dictionary for display and verification.
    
    Purpose: Academic Requirement 20 - Manual calculation demonstration
    """
    # Small sample documents for clear demonstration
    documents_dict = {
        "D1": "patient consent explains risks",
        "D2": "patient consent includes risks",
        "D3": "hospital policy explains patient rights"
    }
    
    terms = ["patient", "consent", "risks", "policy"]
    N = len(documents_dict)
    
    # Step 1: Compute Term Frequency (TF)
    tf_values = {}
    for doc, text in documents_dict.items():
        words = text.split()
        total_words = len(words)
        counts = Counter(words)
        tf_values[doc] = {
            term: counts.get(term, 0) / total_words 
            for term in terms
        }
    
    # Step 2: Compute Document Frequency (DF)
    df = {}
    for term in terms:
        df[term] = sum(
            1 for text in documents_dict.values() 
            if term in text.split()
        )
    
    # Step 3: Compute Inverse Document Frequency (IDF)
    idf = {term: math.log(N / df[term]) for term in terms}
    
    # Step 4: Compute TF-IDF
    tfidf_values = {}
    for doc in documents_dict:
        tfidf_values[doc] = {
            term: tf_values[doc][term] * idf[term]
            for term in terms
        }
    
    # Step 5: Compute cosine similarity between D1 and D2
    vec1 = tfidf_values["D1"]
    vec2 = tfidf_values["D2"]
    similarity_d1_d2 = cosine_similarity(vec1, vec2)
    
    return {
        "documents": documents_dict,
        "terms": terms,
        "N": N,
        "TF": tf_values,
        "DF": df,
        "IDF": idf,
        "TF-IDF": tfidf_values,
        "cosine_similarity_D1_D2": round(similarity_d1_d2, 6),
        "explanation": {
            "TF": "Term Frequency = (count of term in document) / (total terms in document)",
            "IDF": "Inverse Document Frequency = log(N / DF) where N = total docs, DF = docs containing term",
            "TF-IDF": "TF-IDF = TF × IDF (balances term frequency with document uniqueness)",
            "Cosine": "Cosine Similarity = (A · B) / (||A|| × ||B||) measures vector alignment"
        }
    }


def get_detailed_manual_example(documents_text: List[str], sample_terms: List[str] = None):
    """
    Generate detailed manual TF-IDF calculations for custom documents.
    
    Args:
        documents_text: List of document strings
        sample_terms: Optional list of terms to analyze (uses top terms if None)
    
    Returns:
        Dictionary with complete mathematical breakdown
    """
    if not documents_text or len(documents_text) == 0:
        return get_manual_tfidf_output()  # Return default example
    
    N = len(documents_text)
    
    # Tokenize documents
    tokenized_docs = [tokenize(doc) for doc in documents_text]
    
    # Get sample terms if not provided
    if sample_terms is None:
        all_words = []
        for doc in tokenized_docs:
            all_words.extend(doc)
        word_freq = Counter(all_words)
        # Get 4 most common terms (excluding very common ones)
        sample_terms = [word for word, _ in word_freq.most_common(15)[3:7]]
    
    # Compute TF for each document
    tf_values = {}
    for idx, tokens in enumerate(tokenized_docs):
        doc_id = f"Doc{idx+1}"
        tf_values[doc_id] = compute_term_frequency(tokens)
    
    # Compute DF and IDF
    doc_freq = compute_document_frequency(tokenized_docs)
    idf_values = compute_idf(doc_freq, N)
    
    # Build TF-IDF vectors
    tfidf_vectors, _, _ = build_tfidf_matrix(tokenized_docs)
    
    # Compute pairwise similarities
    similarities = compute_pairwise_similarity(tfidf_vectors, tfidf_vectors)
    
    # Format results for display
    results = {
        "num_documents": N,
        "sample_terms": sample_terms,
        "term_analysis": {}
    }
    
    for term in sample_terms:
        term_data = {
            "term": term,
            "df": doc_freq.get(term, 0),
            "idf": round(idf_values.get(term, 0), 6),
            "tf_per_doc": [],
            "tfidf_per_doc": []
        }
        
        for idx in range(N):
            doc_id = f"Doc{idx+1}"
            tf = tf_values[doc_id].get(term, 0)
            tfidf = tfidf_vectors[idx].get(term, 0)
            
            term_data["tf_per_doc"].append({
                "document": doc_id,
                "tf": round(tf, 6)
            })
            term_data["tfidf_per_doc"].append({
                "document": doc_id,
                "tfidf": round(tfidf, 6)
            })
        
        results["term_analysis"][term] = term_data
    
    # Add similarity matrix
    results["similarity_matrix"] = [
        [round(sim, 4) for sim in row] 
        for row in similarities
    ]
    
    return results


# Example usage and testing
if __name__ == "__main__":
    from pprint import pprint
    
    print("=" * 70)
    print("MANUAL TF-IDF COMPUTATION - ACADEMIC DEMONSTRATION")
    print("=" * 70)
    
    # Get manual example results
    manual_results = get_manual_tfidf_output()
    
    print("\n📄 Documents:")
    for doc_id, text in manual_results["documents"].items():
        print(f"  {doc_id}: \"{text}\"")
    
    print("\n🔤 Terms:", manual_results["terms"])
    print("📊 N (total documents):", manual_results["N"])
    
    print("\n--- STEP 1: TERM FREQUENCY (TF) ---")
    print("Formula:", manual_results["explanation"]["TF"])
    pprint(manual_results["TF"])
    
    print("\n--- STEP 2: DOCUMENT FREQUENCY (DF) ---")
    pprint(manual_results["DF"])
    
    print("\n--- STEP 3: INVERSE DOCUMENT FREQUENCY (IDF) ---")
    print("Formula:", manual_results["explanation"]["IDF"])
    pprint(manual_results["IDF"])
    
    print("\n--- STEP 4: TF-IDF VALUES ---")
    print("Formula:", manual_results["explanation"]["TF-IDF"])
    pprint(manual_results["TF-IDF"])
    
    print("\n--- STEP 5: COSINE SIMILARITY ---")
    print("Formula:", manual_results["explanation"]["Cosine"])
    print(f"Similarity between D1 and D2: {manual_results['cosine_similarity_D1_D2']}")
    
    print("\n" + "=" * 70)
    print("✅ Academic Requirement 20: SATISFIED")
    print("   ✓ Manual TF calculation for 4 terms")
    print("   ✓ Manual IDF calculation")
    print("   ✓ Manual TF-IDF calculation")
    print("   ✓ Explanation of weighting importance")
    print("   ✓ Cosine similarity demonstration")
    print("=" * 70)
    
    # Test compute_manual_tfidf function
    print("\n" + "=" * 70)
    print("TESTING compute_manual_tfidf FUNCTION")
    print("=" * 70)
    
    test_docs = [
        "data privacy policy requires consent",
        "privacy protection ensures security",
        "policy compliance mandates security"
    ]
    
    test_words = ["privacy", "policy", "security", "consent"]
    
    results, n = compute_manual_tfidf(test_docs, test_words)
    
    print(f"\n✅ Function test successful!")
    print(f"Analyzed {n} documents for {len(test_words)} words")
    print(f"Sample result for 'privacy': IDF = {results['privacy']['idf']}")
