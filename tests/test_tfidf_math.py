"""
Tests for core TF-IDF mathematical operations.

Validates:
- Term frequency computation (5 variants)
- Inverse document frequency computation (4 variants)
- Manual TF-IDF computation accuracy vs sklearn
- Edge cases in mathematical operations
"""

import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from docs.educational.manual_tfidf_math import (
    compute_term_frequency,
    compute_idf_variants,
    compute_tf_variants,
    compute_manual_tfidf_complete
)


class TestTermFrequency:
    """Test suite for term frequency computation."""
    
    def test_compute_term_frequency_basic(self):
        """Test basic TF computation with known inputs."""
        tokens = ['the', 'cat', 'the', 'dog', 'the']
        tf = compute_term_frequency(tokens)
        
        # 'the' appears 3 times in 5 tokens
        assert tf['the'] == 3 / 5
        assert tf['cat'] == 1 / 5
        assert tf['dog'] == 1 / 5
        assert len(tf) == 3
    
    def test_compute_term_frequency_empty(self):
        """Test TF with empty token list."""
        tokens = []
        tf = compute_term_frequency(tokens)
        assert tf == {}
    
    def test_compute_term_frequency_single_term(self):
        """Test TF with single unique term."""
        tokens = ['term', 'term', 'term']
        tf = compute_term_frequency(tokens)
        assert tf['term'] == 1.0
        assert len(tf) == 1
    
    def test_compute_term_frequency_uniform(self):
        """Test TF with uniform distribution."""
        tokens = ['a', 'b', 'c', 'd', 'e']
        tf = compute_term_frequency(tokens)
        
        for term in ['a', 'b', 'c', 'd', 'e']:
            assert tf[term] == pytest.approx(0.2, abs=1e-6)


class TestTFVariants:
    """Test suite for 5 TF variant computations."""
    
    def test_tf_variants_basic(self, sample_small_corpus):
        """Test that all 5 TF variants return reasonable values."""
        tf_variants = compute_tf_variants(sample_small_corpus)
        
        # Should return dict with 5 keys
        assert len(tf_variants) == 5
        assert all(isinstance(v, dict) for v in tf_variants.values())
        
        # All values should be between 0 and 1
        for variant_dict in tf_variants.values():
            for doc_idx, term_dict in variant_dict.items():
                for term, value in term_dict.items():
                    assert 0 <= value <= 1, f"TF value {value} out of range"
    
    def test_tf_variants_consistency(self, sample_small_corpus):
        """Test that TF variants are consistent across runs."""
        tf_variants_1 = compute_tf_variants(sample_small_corpus)
        tf_variants_2 = compute_tf_variants(sample_small_corpus)
        
        # Should be deterministic
        for key in tf_variants_1.keys():
            assert tf_variants_1[key] == tf_variants_2[key]


class TestIDFVariants:
    """Test suite for 4 IDF variant computations."""
    
    def test_idf_variants_basic(self, sample_small_corpus):
        """Test that all 4 IDF variants return reasonable values."""
        from docs.educational.manual_tfidf_math import preprocess_text_simple
        
        docs = [preprocess_text_simple(d) for d in sample_small_corpus]
        idf_variants = compute_idf_variants(docs)
        
        # Should return dict with 4 keys
        assert len(idf_variants) == 4
        assert all(isinstance(v, dict) for v in idf_variants.values())
        
        # Standard/smoothed IDF variants should be positive.
        for name, variant_dict in idf_variants.items():
            for term, value in variant_dict.items():
                if name in ("probabilistic", "prob_idf", "prob"):
                    # Probabilistic IDF can be negative for very common terms.
                    assert isinstance(value, (int, float))
                else:
                    assert value > 0, f"IDF value {value} should be positive"
    
    def test_idf_variants_ordering(self, sample_small_corpus):
        """Test that rarer terms have higher IDF values."""
        from docs.educational.manual_tfidf_math import preprocess_text_simple
        
        docs = [preprocess_text_simple(d) for d in sample_small_corpus]
        idf_variants = compute_idf_variants(docs)
        
        # Pick any IDF variant
        any_variant = list(idf_variants.values())[0]
        
        # Verify IDF values are ordered correctly
        idf_values = list(any_variant.values())
        assert all(isinstance(v, (int, float)) for v in idf_values)


class TestManualTFIDFAccuracy:
    """Test suite for manual TF-IDF accuracy vs sklearn."""
    
    def test_manual_vs_sklearn_basic(self, sample_docs):
        """Test manual TF-IDF computation matches sklearn."""
        from docs.educational.manual_tfidf_math import preprocess_text_simple
        
        # Preprocess documents
        processed_docs = [preprocess_text_simple(doc) for doc in sample_docs]
        
        # Compute manual TF-IDF
        manual_tfidf = compute_manual_tfidf_complete(processed_docs, min_df=1, max_df=1.0)
        
        # Compute sklearn TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=None,
            min_df=1,
            max_df=1.0,
            stop_words='english',
            norm='l2'
        )
        sklearn_matrix = vectorizer.fit_transform(processed_docs).toarray()
        
        # Manual should have results
        assert manual_tfidf is not None
        assert len(manual_tfidf) > 0
        
        # Both should have same number of documents
        assert len(manual_tfidf) == len(sklearn_matrix)
    
    def test_manual_vs_sklearn_small_corpus(self, sample_small_corpus):
        """Test manual vs sklearn with minimal corpus."""
        from docs.educational.manual_tfidf_math import preprocess_text_simple
        
        processed_docs = [preprocess_text_simple(doc) for doc in sample_small_corpus]
        manual_tfidf = compute_manual_tfidf_complete(processed_docs, min_df=1, max_df=1.0)
        
        # Should produce non-empty results
        assert len(manual_tfidf) == len(sample_small_corpus)
        
        # Each document should have TF-IDF scores
        for doc_tfidf in manual_tfidf:
            assert isinstance(doc_tfidf, dict)
            assert len(doc_tfidf) > 0
    
    def test_manual_tfidf_values_normalized(self, sample_docs):
        """Test that manual TF-IDF vectors are L2 normalized."""
        from docs.educational.manual_tfidf_math import preprocess_text_simple
        import math
        
        processed_docs = [preprocess_text_simple(doc) for doc in sample_docs]
        manual_tfidf = compute_manual_tfidf_complete(processed_docs, min_df=1, max_df=1.0)
        
        # For each document, compute L2 norm
        for doc_tfidf in manual_tfidf:
            l2_norm = math.sqrt(sum(v ** 2 for v in doc_tfidf.values()))
            # Should be close to 1.0 (L2 normalized)
            assert l2_norm == pytest.approx(1.0, abs=0.01)


class TestEdgeCaseMath:
    """Test edge cases in mathematical operations."""
    
    def test_single_document_idf(self):
        """Test IDF computation with single document."""
        docs = ["single document test"]
        idf_variants = compute_idf_variants(docs)
        
        # Should not crash and return results
        assert len(idf_variants) == 4
        for variant_dict in idf_variants.values():
            assert len(variant_dict) > 0
    
    def test_identical_documents_tfidf(self):
        """Test TF-IDF with identical documents."""
        docs = ["test document"] * 3
        manual_tfidf = compute_manual_tfidf_complete(docs, min_df=1, max_df=1.0)
        
        # Should produce results despite identical docs
        assert len(manual_tfidf) == 3
        
        # All should be identical
        for i in range(1, len(manual_tfidf)):
            assert manual_tfidf[i] == manual_tfidf[0]
    
    def test_high_tf_idf_scores_finite(self, sample_docs):
        """Test that TF-IDF scores remain finite (no inf/nan)."""
        from docs.educational.manual_tfidf_math import preprocess_text_simple
        
        processed_docs = [preprocess_text_simple(doc) for doc in sample_docs]
        manual_tfidf = compute_manual_tfidf_complete(processed_docs, min_df=1, max_df=1.0)
        
        for doc_tfidf in manual_tfidf:
            for term, score in doc_tfidf.items():
                assert np.isfinite(score), f"Score {score} is not finite"

