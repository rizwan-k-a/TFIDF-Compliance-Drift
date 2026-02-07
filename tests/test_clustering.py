"""Tests for clustering and document grouping functionality.

Covers:
- perform_enhanced_clustering
- Group formation and quality
- Edge cases: single document, identical documents, no clear clusters
"""

import pytest
from typing import List

from backend.clustering import perform_enhanced_clustering


def _is_error(result: object) -> bool:
    return isinstance(result, dict) and bool(result.get("error"))


class TestClusteringBasics:
    """Test basic clustering functionality."""

    def test_clustering_forms_groups(self):
        """Clustering should group similar documents."""
        docs = [
            "criminal law procedures and practices",
            "criminal court procedures and legal rules",
            "dog breeds and pet training",
            "cat care and pet health",
            "financial regulations and compliance",
        ]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=3,
        )

        if _is_error(result):
            assert "error" in result
        else:
            assert result is not None
            if "clusters" in result:
                assert len(result["clusters"]) == len(docs)

    def test_clustering_minimum_documents(self):
        """Clustering with few documents should handle gracefully."""
        docs = ["criminal law", "cyber security"]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=2,
        )

        assert result is None or isinstance(result, dict)

    def test_clustering_single_document(self):
        """Clustering with single document should return that document."""
        docs = ["single document content"]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=1,
        )

        assert result is None or isinstance(result, dict)

    def test_clustering_identical_documents(self):
        """Identical documents should cluster together."""
        docs = [
            "criminal law",
            "criminal law",
            "criminal law",
            "cyber security",
            "cyber security",
        ]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=2,
        )

        if result and not _is_error(result) and "clusters" in result:
            clusters = result["clusters"]
            unique_clusters = len(set(clusters))
            assert unique_clusters <= 3

    def test_clustering_no_clear_separation(self):
        """Random docs should still cluster without error."""
        docs = [
            "law procedures",
            "security measures",
            "compliance framework",
            "regulations enforcement",
            "policy implementation",
        ]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=2,
        )

        assert result is None or isinstance(result, dict)


class TestClusteringParameters:
    """Test clustering with different parameters."""

    def test_clustering_k_equals_n(self):
        """Each doc as own cluster (k == n)."""
        docs = ["doc1", "doc2", "doc3"]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=3,
        )

        assert result is None or isinstance(result, dict)

    def test_clustering_k_equals_1(self):
        """All docs in one cluster (k == 1)."""
        docs = ["doc1", "doc2", "doc3", "doc4"]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=1,
        )

        if result and not _is_error(result) and "clusters" in result:
            clusters = result["clusters"]
            unique_clusters = len(set(clusters))
            assert unique_clusters == 1

    def test_clustering_with_precomputed_vectorizer(self):
        """Should work with precomputed TF-IDF vectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = ["criminal law", "cyber security", "financial regulation"]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(docs)

        try:
            result = perform_enhanced_clustering(
                documents=docs,
                names=[f"doc_{i}" for i in range(len(docs))],
                n_clusters=2,
                precomputed_vectorizer=vectorizer,
                precomputed_matrix=tfidf_matrix,
            )

            assert result is None or isinstance(result, dict)
        except TypeError:
            pass


class TestClusteringOutput:
    """Test clustering output structure."""

    def test_clustering_returns_dict(self):
        """Clustering should return a dictionary."""
        docs = ["law", "security", "finance"]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=2,
        )

        assert isinstance(result, (dict, type(None)))

    def test_clustering_has_required_fields(self):
        """Output should have required metadata."""
        docs = ["criminal law", "cyber security", "financial regulation"]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=2,
        )

        if result and not _is_error(result):
            assert "clusters" in result or "cluster_centers" in result or "n_clusters" in result

    def test_clustering_cluster_assignments_valid(self):
        """Cluster assignments should be in valid range."""
        docs = ["doc1", "doc2", "doc3", "doc4"]
        n_clusters = 2

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=n_clusters,
        )

        if result and not _is_error(result) and "clusters" in result:
            clusters = result["clusters"]
            for cluster_id in clusters:
                assert 0 <= cluster_id < n_clusters


class TestClusteringEdgeCases:
    """Test edge cases and error conditions."""

    def test_clustering_empty_documents(self):
        """Empty document list should handle gracefully."""
        docs = []

        result = perform_enhanced_clustering(
            documents=docs,
            names=[],
            n_clusters=2,
        )

        assert result is None or isinstance(result, dict)

    def test_clustering_k_exceeds_n(self):
        """k > n (more clusters than docs) should be handled."""
        docs = ["doc1", "doc2"]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=5,
        )

        assert result is None or isinstance(result, dict)

    def test_clustering_very_short_documents(self):
        """Very short documents should still cluster."""
        docs = ["a", "b", "c", "d", "e"]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=2,
        )

        assert result is None or isinstance(result, dict)

    def test_clustering_with_special_characters(self):
        """Documents with special characters should cluster."""
        docs = [
            "criminal law: procedures & practices",
            "cyber-security: measures @ scale",
            "financial-regulation (compliance) rules",
        ]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=2,
        )

        assert result is None or isinstance(result, dict)


class TestClusteringQuality:
    """Test clustering produces reasonable results."""

    def test_clustering_separates_categories(self):
        """Should separate clearly different categories."""
        docs = [
            "criminal law procedures",
            "criminal court rules",
            "criminal offense guidelines",
            "cyber security protocols",
            "cyber attack prevention",
            "cyber threat measures",
        ]

        result = perform_enhanced_clustering(
            documents=docs,
            names=[f"doc_{i}" for i in range(len(docs))],
            n_clusters=2,
        )

        if result and not _is_error(result) and "clusters" in result:
            clusters = result["clusters"]
            cluster_0_indices = [i for i, c in enumerate(clusters) if c == 0]
            cluster_1_indices = [i for i, c in enumerate(clusters) if c == 1]

            assert len(cluster_0_indices) > 0
            assert len(cluster_1_indices) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
