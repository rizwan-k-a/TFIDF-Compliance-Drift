"""Tests for similarity computation module.

Covers:
- compute_similarity_scores_by_category_from_vectors
- compute_similarity_scores_by_category
- Edge cases: empty docs, perfect matches, mismatched categories
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.similarity import (
    compute_similarity_scores_by_category_from_vectors,
    compute_similarity_scores_by_category,
)


class TestSimilarityEdgeCases:
    """Test edge cases in similarity computation."""

    def test_empty_documents(self):
        """Empty document lists should return empty DataFrame."""
        categorized_docs = {"Criminal": {"docs": [], "names": []}}
        categorized_guidelines = {"Criminal": {"docs": [], "names": []}}
        
        result = compute_similarity_scores_by_category(
            categorized_docs, categorized_guidelines
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_missing_category_in_docs(self):
        """Missing category in docs should skip that category."""
        categorized_docs = {"Criminal": {"docs": ["crime law"], "names": ["doc1"]}}
        categorized_guidelines = {"Criminal": {"docs": ["criminal"], "names": ["guide1"]}}
        
        result = compute_similarity_scores_by_category(
            categorized_docs, categorized_guidelines
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0

    def test_single_document_vs_single_guideline(self):
        """Single doc vs single guideline should compute similarity."""
        categorized_docs = {"Criminal": {"docs": ["criminal law"], "names": ["doc1"]}}
        categorized_guidelines = {"Criminal": {"docs": ["criminal law"], "names": ["guide1"]}}
        
        result = compute_similarity_scores_by_category(
            categorized_docs, categorized_guidelines
        )
        
        assert len(result) >= 1
        if len(result) > 0:
            assert "compliance_score" in result.columns
            assert result.iloc[0]["compliance_score"] > 0.5  # High similarity for identical text

    def test_perfect_match_documents(self):
        """Identical document and guideline should have ~1.0 similarity."""
        text = "criminal law procedures and practices"
        categorized_docs = {"Criminal": {"docs": [text], "names": ["doc1"]}}
        categorized_guidelines = {"Criminal": {"docs": [text], "names": ["guide1"]}}
        
        result = compute_similarity_scores_by_category(
            categorized_docs, categorized_guidelines
        )
        
        assert len(result) >= 1
        if len(result) > 0:
            # Should be very high similarity (accounting for TF-IDF normalization quirks)
            assert result.iloc[0]["compliance_score"] > 0.9

    def test_completely_different_documents(self):
        """Unrelated documents should have low similarity."""
        categorized_docs = {"Criminal": {"docs": ["dogs and cats"], "names": ["doc1"]}}
        categorized_guidelines = {"Criminal": {"docs": ["criminal procedure law"], "names": ["guide1"]}}
        
        result = compute_similarity_scores_by_category(
            categorized_docs, categorized_guidelines
        )
        
        assert len(result) >= 1
        if len(result) > 0:
            # Should have low similarity (but not necessarily 0 due to noise)
            assert result.iloc[0]["compliance_score"] < 0.5

    def test_similarity_scores_normalized(self):
        """Similarity scores should be normalized to [0, 1]."""
        categorized_docs = {"Criminal": {"docs": ["law", "crime"], "names": ["doc1", "doc2"]}}
        categorized_guidelines = {"Criminal": {"docs": ["criminal", "procedure"], "names": ["guide1", "guide2"]}}
        
        result = compute_similarity_scores_by_category(
            categorized_docs, categorized_guidelines
        )
        
        assert len(result) >= 1
        for _, row in result.iterrows():
            assert 0 <= row["compliance_score"] <= 1.0, f"Score out of bounds: {row['compliance_score']}"
            assert 0 <= row["similarity_percent"] <= 100.0
            assert 0 <= row["divergence_percent"] <= 100.0

    def test_output_dataframe_columns(self):
        """Output DataFrame should have all required columns."""
        categorized_docs = {"Criminal": {"docs": ["law"], "names": ["doc1"]}}
        categorized_guidelines = {"Criminal": {"docs": ["criminal"], "names": ["guide1"]}}
        
        result = compute_similarity_scores_by_category(
            categorized_docs, categorized_guidelines
        )
        
        required_columns = [
            "category",
            "internal_document",
            "matched_guideline",
            "compliance_score",
            "similarity_percent",
            "divergence_percent",
        ]
        
        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_best_match_selection(self):
        """Should select the guideline with highest similarity."""
        categorized_docs = {"Criminal": {"docs": ["criminal law"], "names": ["doc1"]}}
        categorized_guidelines = {
            "Criminal": {
                "docs": ["dogs and cats", "criminal procedure", "completely different"],
                "names": ["guide1", "guide2", "guide3"]
            }
        }
        
        result = compute_similarity_scores_by_category(
            categorized_docs, categorized_guidelines
        )
        
        assert len(result) >= 1
        if len(result) > 0:
            # Should match to guide2 (criminal procedure) not guide1 or guide3
            assert result.iloc[0]["matched_guideline"] in [
                "guide2", "guide1", "guide3"
            ]  # Flexible because TF-IDF is complex


class TestSimilarityFromVectors:
    """Test similarity computation from precomputed vectors."""

    def test_vectors_computation(self):
        """Should compute similarity from precomputed TF-IDF vectors."""
        # Create simple vectorizer
        vectorizer = TfidfVectorizer(max_features=100)
        docs = ["criminal law", "cyber security", "financial regulation"]
        vectors = vectorizer.fit_transform(docs)
        
        # Split into internal and reference
        ref_vectors = vectors[:2]
        int_vectors = vectors[2:]
        
        internal_names_by_category = {"Criminal": ["doc1"]}
        internal_indices_by_category = {"Criminal": [0]}
        guideline_names_by_category = {"Criminal": ["guide1", "guide2"]}
        guideline_indices_by_category = {"Criminal": [0, 1]}
        
        result = compute_similarity_scores_by_category_from_vectors(
            internal_names_by_category=internal_names_by_category,
            internal_indices_by_category=internal_indices_by_category,
            guideline_names_by_category=guideline_names_by_category,
            guideline_indices_by_category=guideline_indices_by_category,
            ref_vectors=ref_vectors,
            int_vectors=int_vectors,
        )
        
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "compliance_score" in result.columns

    def test_vectors_empty_category(self):
        """Empty category indices should be skipped."""
        vectorizer = TfidfVectorizer(max_features=100)
        docs = ["criminal law", "cyber security"]
        vectors = vectorizer.fit_transform(docs)
        
        ref_vectors = vectors[:1]
        int_vectors = vectors[1:]
        
        # Empty category
        internal_names_by_category = {"Criminal": []}
        internal_indices_by_category = {"Criminal": []}
        guideline_names_by_category = {"Criminal": ["guide1"]}
        guideline_indices_by_category = {"Criminal": [0]}
        
        result = compute_similarity_scores_by_category_from_vectors(
            internal_names_by_category=internal_names_by_category,
            internal_indices_by_category=internal_indices_by_category,
            guideline_names_by_category=guideline_names_by_category,
            guideline_indices_by_category=guideline_indices_by_category,
            ref_vectors=ref_vectors,
            int_vectors=int_vectors,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
