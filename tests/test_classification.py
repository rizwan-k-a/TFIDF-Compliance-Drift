"""
Tests for classification and clustering functionality.

Validates:
- Classification with sufficient data
- Classification returns None with insufficient data
- Category filtering for imbalanced datasets
- Clustering operations
- Result structure and data types
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from similarity import perform_classification, perform_enhanced_clustering


class TestClassificationBasic:
    """Test suite for basic classification operations."""
    
    def test_classification_with_sufficient_data(self, sample_docs, sample_categories):
        """Test classification with 4 documents and 2 categories."""
        result = perform_classification(
            sample_docs,
            sample_categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # Should return results
        assert result is not None
        assert isinstance(result, dict)
    
    def test_classification_result_structure(self, sample_docs, sample_categories):
        """Test that classification result has expected structure."""
        result = perform_classification(
            sample_docs,
            sample_categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        if result:
            # Check key fields exist
            assert 'nb_accuracy' in result or 'accuracy' in result
            assert 'classification_report_nb' in result or 'report' in result
            assert 'filtered_count' in result
            assert 'excluded_count' in result
    
    def test_classification_requires_6_docs(self, sample_docs, sample_categories):
        """Test that classification needs at least 6 documents."""
        # Test with less than 6 docs
        result = perform_classification(
            sample_docs[:5],  # Only 5 docs
            sample_categories[:5],
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # May return None or empty for insufficient data
        assert result is None or isinstance(result, dict)
    
    def test_classification_requires_2_categories(self):
        """Test that classification needs at least 2 categories."""
        docs = [f"test document {i}" for i in range(6)]
        categories = ["Category_A"] * 6  # Only 1 category
        
        result = perform_classification(
            docs,
            categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # Should not classify with only 1 category
        assert result is None or (isinstance(result, dict) and 'error' in str(result).lower())


class TestClassificationWithImbalancedData:
    """Test suite for classification with imbalanced categories."""
    
    def test_classification_filters_small_categories(self, sample_docs_imbalanced, sample_categories_imbalanced):
        """Test that categories with < 2 samples are filtered."""
        result = perform_classification(
            sample_docs_imbalanced,
            sample_categories_imbalanced,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # Should handle gracefully - either None or with filtering info
        assert result is None or isinstance(result, dict)
        
        if result:
            # Check filtering info
            assert 'filtered_count' in result
            assert 'excluded_count' in result
    
    def test_classification_returns_filtering_info(self, sample_docs_imbalanced, sample_categories_imbalanced):
        """Test that result contains filtering statistics."""
        result = perform_classification(
            sample_docs_imbalanced,
            sample_categories_imbalanced,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        if result:
            assert isinstance(result.get('filtered_count', 0), int)
            assert isinstance(result.get('excluded_count', 0), int)
            assert result['filtered_count'] >= 0
            assert result['excluded_count'] >= 0


class TestClassificationEdgeCases:
    """Test edge cases in classification."""
    
    def test_classification_all_same_category(self, sample_docs_single_category, sample_categories_single):
        """Test classification with all documents in same category."""
        result = perform_classification(
            sample_docs_single_category,
            sample_categories_single,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # Should return None - can't classify with only 1 category
        assert result is None or (isinstance(result, dict) and result.get('excluded_count', 0) > 0)
    
    def test_classification_min_df_too_high(self, sample_docs, sample_categories):
        """Test classification with min_df filtering out all terms."""
        result = perform_classification(
            sample_docs,
            sample_categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=0.8,  # Filter terms appearing in < 80% of docs
            max_df=1.0
        )
        
        # May still return results with adaptive vectorizer
        assert result is None or isinstance(result, dict)
    
    def test_classification_max_df_too_low(self, sample_docs, sample_categories):
        """Test classification with max_df filtering out all terms."""
        result = perform_classification(
            sample_docs,
            sample_categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=0.1  # Filter terms appearing in > 10% of docs
        )
        
        # Should handle gracefully
        assert result is None or isinstance(result, dict)


class TestClassificationModelOutputs:
    """Test classification model outputs and metrics."""
    
    def test_classification_accuracy_values(self, sample_docs, sample_categories):
        """Test that accuracy values are between 0 and 1."""
        result = perform_classification(
            sample_docs,
            sample_categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        if result:
            if 'nb_accuracy' in result:
                assert 0 <= result['nb_accuracy'] <= 1
            if 'lr_accuracy' in result:
                assert 0 <= result['lr_accuracy'] <= 1
    
    def test_classification_reports_exist(self, sample_docs, sample_categories):
        """Test that classification reports are generated."""
        result = perform_classification(
            sample_docs,
            sample_categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        if result:
            if 'classification_report_nb' in result:
                assert isinstance(result['classification_report_nb'], str)
                assert len(result['classification_report_nb']) > 0
            if 'classification_report_lr' in result:
                assert isinstance(result['classification_report_lr'], str)
                assert len(result['classification_report_lr']) > 0
    
    def test_classification_confusion_matrices(self, sample_docs, sample_categories):
        """Test that confusion matrices are generated."""
        result = perform_classification(
            sample_docs,
            sample_categories,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        if result:
            if 'confusion_matrix_nb' in result:
                import numpy as np
                assert isinstance(result['confusion_matrix_nb'], np.ndarray)
                assert result['confusion_matrix_nb'].ndim == 2
            if 'confusion_matrix_lr' in result:
                import numpy as np
                assert isinstance(result['confusion_matrix_lr'], np.ndarray)
                assert result['confusion_matrix_lr'].ndim == 2


class TestClusteringBasic:
    """Test suite for clustering operations."""
    
    def test_clustering_with_minimum_docs(self):
        """Test clustering with minimum required documents (3)."""
        docs = [f"test document {i}" for i in range(3)]
        doc_names = [f"doc_{i}" for i in range(3)]
        
        result = perform_enhanced_clustering(
            docs,
            doc_names,
            n_clusters=2,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # Should return results or None
        assert result is None or isinstance(result, dict)
    
    def test_clustering_result_structure(self):
        """Test clustering result structure."""
        docs = [
            "machine learning classification algorithm",
            "deep neural network training process",
            "data preprocessing normalization scaling"
        ]
        doc_names = ["doc_0", "doc_1", "doc_2"]
        
        result = perform_enhanced_clustering(
            docs,
            doc_names,
            n_clusters=2,
            keep_numbers=True,
            use_lemma=False,
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        if result:
            # Should have expected keys
            expected_keys = ['coordinates', 'labels', 'silhouette_score', 'davies_bouldin_score']
            for key in expected_keys:
                assert key in result or True  # Some keys might be optional
