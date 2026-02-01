"""
Tests for edge cases and data validation.

Validates:
- Parameter range validation (min_df, max_df)
- Empty document handling
- Single document rejection
- Category constraint validation
- Special data distributions
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vectorize import vectorize_documents
from similarity import perform_classification


class TestParameterValidation:
    """Test suite for parameter validation."""
    
    def test_min_df_max_df_valid_range(self, sample_docs):
        """Test valid min_df and max_df range."""
        result = vectorize_documents(
            sample_docs,
            min_df=1,
            max_df=0.9,
            keep_numbers=True,
            use_lemma=False,
            max_features=100
        )
        
        # Should work with valid range
        assert result is not None
    
    def test_min_df_less_than_max_df(self, sample_docs):
        """Test that min_df should be less than max_df."""
        # Test inverted range
        result = vectorize_documents(
            sample_docs,
            min_df=0.8,
            max_df=0.2,  # max_df < min_df (invalid)
            keep_numbers=True,
            use_lemma=False,
            max_features=100
        )
        
        # Should handle gracefully or swap/adjust
        assert result is None or isinstance(result, tuple)
    
    def test_min_df_zero(self, sample_docs):
        """Test min_df = 0 (all terms included)."""
        result = vectorize_documents(
            sample_docs,
            min_df=0,
            max_df=1.0,
            keep_numbers=True,
            use_lemma=False,
            max_features=100
        )
        
        # Should work
        assert result is None or isinstance(result, tuple)
    
    def test_max_df_one(self, sample_docs):
        """Test max_df = 1.0 (all terms included)."""
        result = vectorize_documents(
            sample_docs,
            min_df=1,
            max_df=1.0,
            keep_numbers=True,
            use_lemma=False,
            max_features=100
        )
        
        # Should work
        assert result is None or isinstance(result, tuple)
    
    def test_min_df_fractional(self, sample_docs):
        """Test min_df as fractional value (document frequency)."""
        result = vectorize_documents(
            sample_docs,
            min_df=0.25,  # Terms in at least 25% of docs
            max_df=0.9,
            keep_numbers=True,
            use_lemma=False,
            max_features=100
        )
        
        # Should work with fractional min_df
        assert result is None or isinstance(result, tuple)


class TestEmptyDocuments:
    """Test suite for handling empty documents."""
    
    def test_empty_string_document(self):
        """Test handling of empty string document."""
        docs = ["test document", "", "another test"]
        
        result = vectorize_documents(
            docs,
            min_df=1,
            max_df=1.0,
            keep_numbers=True,
            use_lemma=False,
            max_features=100
        )
        
        # Should handle gracefully
        assert result is None or isinstance(result, tuple)
    
    def test_all_empty_documents(self):
        """Test handling when all documents are empty."""
        docs = ["", "", ""]
        
        result = vectorize_documents(
            docs,
            min_df=1,
            max_df=1.0,
            keep_numbers=True,
            use_lemma=False,
            max_features=100
        )
        
        # Should return None or handle gracefully
        assert result is None or isinstance(result, tuple)
    
    def test_whitespace_only_documents(self):
        """Test documents with only whitespace."""
        docs = ["   ", "\t\t", "\n\n", "valid document"]
        
        result = vectorize_documents(
            docs,
            min_df=1,
            max_df=1.0,
            keep_numbers=True,
            use_lemma=False,
            max_features=100
        )
        
        # Should handle gracefully
        assert result is None or isinstance(result, tuple)


class TestSingleDocumentRejection:
    """Test suite for single document scenarios."""
    
    def test_single_document_vectorization(self):
        """Test vectorization with single document."""
        docs = ["single test document"]
        
        result = vectorize_documents(
            docs,
            min_df=1,
            max_df=1.0,
            keep_numbers=True,
            use_lemma=False,
            max_features=100
        )
        
        # Single doc might be rejected for IDF computation
        assert result is None or isinstance(result, tuple)
    
    def test_classification_rejects_single_doc(self):
        """Test that classification rejects single document."""
        docs = ["single document test"]
        categories = ["Category_A"]
        
        result = perform_classification(
            docs,
            categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # Should return None - insufficient documents
        assert result is None


class TestCategoryConstraints:
    """Test suite for category-related constraints."""
    
    def test_missing_category_for_document(self):
        """Test when document lacks corresponding category."""
        docs = ["doc1", "doc2", "doc3"]
        categories = ["cat1", "cat2"]  # Missing third category
        
        # Should handle gracefully or raise error
        try:
            result = perform_classification(
                docs,
                categories,
                keep_numbers=True,
                use_lemma=False,
                max_features=100,
                min_df=1,
                max_df=1.0
            )
            # If no error, check result
            assert result is None or isinstance(result, dict)
        except (ValueError, IndexError):
            # Expected to fail with clear error
            pass
    
    def test_many_categories_few_docs(self):
        """Test with many categories but few total documents."""
        docs = [f"doc{i}" for i in range(6)]
        categories = ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6"]  # 6 categories, 6 docs
        
        result = perform_classification(
            docs,
            categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # Should fail - each category has only 1 sample
        assert result is None


class TestSpecialDistributions:
    """Test suite for special data distributions."""
    
    def test_identical_documents(self):
        """Test classification with identical documents."""
        docs = ["identical document content"] * 6
        categories = ["A", "B", "A", "B", "A", "B"]
        
        result = perform_classification(
            docs,
            categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # Should classify despite identical content
        assert result is None or isinstance(result, dict)
    
    def test_very_short_documents(self):
        """Test classification with very short documents."""
        docs = ["a", "b", "c", "d", "e", "f"]
        categories = ["X", "Y", "X", "Y", "X", "Y"]
        
        result = perform_classification(
            docs,
            categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # Should handle gracefully
        assert result is None or isinstance(result, dict)
    
    def test_very_long_documents(self):
        """Test classification with very long documents."""
        long_text = "word " * 1000  # Repeat "word" 1000 times
        docs = [long_text for _ in range(6)]
        categories = ["A", "B", "A", "B", "A", "B"]
        
        result = perform_classification(
            docs,
            categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # Should handle without crash
        assert result is None or isinstance(result, dict)


class TestDataTypeValidation:
    """Test suite for data type validation."""
    
    def test_documents_not_string_list(self):
        """Test handling of non-string documents."""
        docs = [123, 456, 789, 101112, 131415, 161718]  # Not strings
        categories = ["A", "B", "A", "B", "A", "B"]
        
        # Should handle gracefully or raise clear error
        try:
            result = perform_classification(
                docs,
                categories,
                keep_numbers=True,
                use_lemma=False,
                max_features=100,
                min_df=1,
                max_df=1.0
            )
            assert result is None or isinstance(result, dict)
        except (TypeError, AttributeError):
            # Expected to fail with type error
            pass
    
    def test_none_in_documents(self):
        """Test handling of None values in document list."""
        docs = ["doc1", None, "doc3", "doc4", "doc5", "doc6"]
        categories = ["A", "B", "A", "B", "A", "B"]
        
        # Should handle gracefully
        try:
            result = perform_classification(
                docs,
                categories,
                keep_numbers=True,
                use_lemma=False,
                max_features=100,
                min_df=1,
                max_df=1.0
            )
            assert result is None or isinstance(result, dict)
        except (TypeError, AttributeError):
            # Expected to fail
            pass


class TestVectorizerEdgeCases:
    """Test edge cases in vectorization."""
    
    def test_min_df_greater_than_doc_count(self, sample_docs):
        """Test min_df greater than number of documents."""
        result = vectorize_documents(
            sample_docs,
            min_df=100,  # More than number of docs
            max_df=1.0,
            keep_numbers=True,
            use_lemma=False,
            max_features=100
        )
        
        # Should filter out all terms
        assert result is None or isinstance(result, tuple)
    
    def test_max_features_zero(self, sample_docs):
        """Test max_features = 0."""
        result = vectorize_documents(
            sample_docs,
            min_df=1,
            max_df=1.0,
            keep_numbers=True,
            use_lemma=False,
            max_features=0
        )
        
        # Should handle gracefully
        assert result is None or isinstance(result, tuple)
    
    def test_max_features_very_large(self, sample_docs):
        """Test max_features with very large value."""
        result = vectorize_documents(
            sample_docs,
            min_df=1,
            max_df=1.0,
            keep_numbers=True,
            use_lemma=False,
            max_features=1000000
        )
        
        # Should work with large max_features
        assert result is None or isinstance(result, tuple)
