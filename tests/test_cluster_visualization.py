"""Tests for cluster_visualization module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.cluster_visualization import (
    create_cluster_scatter_pca,
    create_distance_distribution_plot,
    create_cluster_size_barplot,
    create_bic_aic_comparison_plot,
    create_silhouette_plot,
    create_dendrogram,
)


# ============================================================================
# Tests for create_cluster_scatter_pca
# ============================================================================


class TestCreateClusterScatterPCA:
    """Tests for PCA scatter plot with cluster coloring."""

    def test_basic_scatter_with_pca_result(
        self, kmeans_cluster_result, pca_result_for_clustering
    ):
        """Test basic scatter plot using provided PCA results."""
        fig = create_cluster_scatter_pca(
            kmeans_cluster_result, pca_result=pca_result_for_clustering
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "PC1" in ax.get_xlabel()
        assert "PC2" in ax.get_ylabel()
        assert "Clustering Results" in ax.get_title()
        plt.close(fig)

    def test_scatter_compute_pca_when_none(self, kmeans_cluster_result):
        """Test that PCA is computed when pca_result=None."""
        fig = create_cluster_scatter_pca(kmeans_cluster_result, pca_result=None)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "PC1" in ax.get_xlabel()
        assert "PC2" in ax.get_ylabel()
        plt.close(fig)

    def test_scatter_with_highlight_indices(
        self, kmeans_cluster_result, pca_result_for_clustering, highlight_indices_data
    ):
        """Test scatter plot with highlighted samples."""
        fig = create_cluster_scatter_pca(
            kmeans_cluster_result,
            pca_result=pca_result_for_clustering,
            highlight_indices=highlight_indices_data["indices"],
        )

        assert isinstance(fig, plt.Figure)
        # Check that highlighted points were added
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().texts]
        assert "Highlighted" in legend_labels
        plt.close(fig)

    def test_scatter_with_empty_highlights(
        self, kmeans_cluster_result, pca_result_for_clustering, highlight_indices_data
    ):
        """Test scatter plot with empty highlight list."""
        fig = create_cluster_scatter_pca(
            kmeans_cluster_result,
            pca_result=pca_result_for_clustering,
            highlight_indices=highlight_indices_data["empty"],
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().texts]
        assert "Highlighted" not in legend_labels
        plt.close(fig)

    def test_scatter_with_no_matching_highlights(
        self, kmeans_cluster_result, pca_result_for_clustering, highlight_indices_data
    ):
        """Test scatter plot when highlight indices don't match data."""
        fig = create_cluster_scatter_pca(
            kmeans_cluster_result,
            pca_result=pca_result_for_clustering,
            highlight_indices=highlight_indices_data["no_match"],
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().texts]
        assert "Highlighted" not in legend_labels
        plt.close(fig)

    def test_scatter_edge_case_insufficient_components(self, minimal_pca_result):
        """Test handling of PCA with < 2 components."""
        # Create a minimal cluster result
        cluster_result = {
            "cluster_labels": np.array([0, 0, 1, 1]),
            "data_indices": [0, 1, 2, 3],
            "data_processed": np.random.randn(4, 2),
        }

        fig = create_cluster_scatter_pca(cluster_result, pca_result=minimal_pca_result)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # Should show warning text
        text_objects = [
            child for child in ax.get_children() if hasattr(child, "get_text")
        ]
        # Check if warning message exists somewhere in the plot
        assert any(
            "Need at least 2 PCA components" in str(child)
            for child in ax.get_children()
        )
        plt.close(fig)

    def test_scatter_custom_figsize_and_title(
        self, kmeans_cluster_result, pca_result_for_clustering
    ):
        """Test custom figure size and title."""
        custom_title = "My Custom Cluster Plot"
        fig = create_cluster_scatter_pca(
            kmeans_cluster_result,
            pca_result=pca_result_for_clustering,
            figsize=(12, 10),
            title=custom_title,
        )

        assert isinstance(fig, plt.Figure)
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 10
        ax = fig.axes[0]
        assert ax.get_title() == custom_title
        plt.close(fig)

    def test_scatter_multiple_clusters(self, simple_cluster_data):
        """Test scatter plot with different numbers of clusters."""
        from sleap_roots_analyze.clustering import perform_kmeans_clustering

        for n_clusters in [2, 3, 5]:
            result = perform_kmeans_clustering(
                simple_cluster_data, n_clusters=n_clusters, random_state=42
            )
            fig = create_cluster_scatter_pca(result)

            assert isinstance(fig, plt.Figure)
            ax = fig.axes[0]
            legend_labels = [t.get_text() for t in ax.get_legend().texts]
            assert len(legend_labels) == n_clusters
            plt.close(fig)


# ============================================================================
# Tests for create_distance_distribution_plot
# ============================================================================


class TestCreateDistanceDistributionPlot:
    """Tests for distance distribution histogram."""

    def test_basic_distance_plot(self, distance_array):
        """Test basic distance distribution plot."""
        fig = create_distance_distribution_plot(
            distance_array, threshold=10.0, method_name="K-Means"
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Two subplots
        plt.close(fig)

    def test_distance_plot_different_thresholds(self, distance_array):
        """Test distance plot with different threshold values."""
        for threshold in [5.0, 10.0, 15.0]:
            fig = create_distance_distribution_plot(
                distance_array, threshold=threshold, method_name="Test"
            )

            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_distance_plot_custom_figsize(self, distance_array):
        """Test custom figure size."""
        fig = create_distance_distribution_plot(
            distance_array, threshold=10.0, method_name="GMM", figsize=(8, 6)
        )

        assert isinstance(fig, plt.Figure)
        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 6
        plt.close(fig)

    def test_distance_plot_edge_cases(self):
        """Test with edge case distance arrays."""
        # Empty array
        distances_empty = np.array([])
        fig1 = create_distance_distribution_plot(
            distances_empty, threshold=0, method_name="Empty"
        )
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # All identical values
        distances_identical = np.array([5.0] * 50)
        fig2 = create_distance_distribution_plot(
            distances_identical, threshold=5.0, method_name="Identical"
        )
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)


# ============================================================================
# Tests for create_cluster_size_barplot
# ============================================================================


class TestCreateClusterSizeBarplot:
    """Tests for cluster size bar plot."""

    def test_basic_cluster_size_plot(self, kmeans_cluster_result):
        """Test basic cluster size bar plot."""
        fig = create_cluster_size_barplot(
            kmeans_cluster_result["cluster_labels"],
            kmeans_cluster_result["n_clusters"],
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Cluster" in ax.get_xlabel()
        assert "Number of Samples" in ax.get_ylabel()
        plt.close(fig)

    def test_cluster_size_different_n_clusters(self, simple_cluster_data):
        """Test cluster size plot with different numbers of clusters."""
        from sleap_roots_analyze.clustering import perform_kmeans_clustering

        for n_clusters in [2, 3, 5]:
            result = perform_kmeans_clustering(
                simple_cluster_data, n_clusters=n_clusters, random_state=42
            )
            fig = create_cluster_size_barplot(
                result["cluster_labels"], result["n_clusters"]
            )

            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_cluster_size_custom_figsize(self, kmeans_cluster_result):
        """Test custom figure size."""
        fig = create_cluster_size_barplot(
            kmeans_cluster_result["cluster_labels"],
            kmeans_cluster_result["n_clusters"],
            figsize=(12, 8),
        )

        assert isinstance(fig, plt.Figure)
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8
        plt.close(fig)


# ============================================================================
# Tests for create_bic_aic_comparison_plot
# ============================================================================


class TestCreateBICAICComparisonPlot:
    """Tests for BIC/AIC comparison plot."""

    def test_basic_bic_aic_plot(self, bic_aic_data):
        """Test basic BIC/AIC comparison plot."""
        fig = create_bic_aic_comparison_plot(bic_aic_data["bic"], bic_aic_data["aic"])

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Number of Components" in ax.get_xlabel()
        assert "Information Criterion" in ax.get_ylabel()
        plt.close(fig)

    def test_bic_aic_custom_figsize(self, bic_aic_data):
        """Test custom figure size."""
        fig = create_bic_aic_comparison_plot(
            bic_aic_data["bic"], bic_aic_data["aic"], figsize=(10, 6)
        )

        assert isinstance(fig, plt.Figure)
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        plt.close(fig)

    def test_bic_aic_different_lengths(self):
        """Test with different length arrays."""
        bic = [100, 90, 85, 80, 85]
        aic = [95, 85, 80, 78, 82]

        fig = create_bic_aic_comparison_plot(bic, aic)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bic_aic_single_value(self):
        """Test with single value arrays."""
        bic = [100]
        aic = [95]

        fig = create_bic_aic_comparison_plot(bic, aic)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# Tests for create_silhouette_plot
# ============================================================================


class TestCreateSilhouettePlot:
    """Tests for silhouette coefficient plot."""

    def test_basic_silhouette_plot(self, kmeans_cluster_result):
        """Test basic silhouette plot."""
        fig = create_silhouette_plot(kmeans_cluster_result)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Silhouette Coefficient" in ax.get_xlabel()
        assert "Cluster" in ax.get_ylabel()
        assert "Silhouette Plot" in ax.get_title()
        plt.close(fig)

    def test_silhouette_plot_different_n_clusters(self, simple_cluster_data):
        """Test silhouette plot with different numbers of clusters."""
        from sleap_roots_analyze.clustering import perform_kmeans_clustering

        for n_clusters in [2, 3, 4]:
            result = perform_kmeans_clustering(
                simple_cluster_data, n_clusters=n_clusters, random_state=42
            )
            fig = create_silhouette_plot(result)

            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_silhouette_plot_custom_figsize(self, kmeans_cluster_result):
        """Test custom figure size."""
        fig = create_silhouette_plot(kmeans_cluster_result, figsize=(12, 8))

        assert isinstance(fig, plt.Figure)
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8
        plt.close(fig)

    def test_silhouette_plot_gmm_result(self, gmm_cluster_result):
        """Test silhouette plot with GMM clustering result."""
        fig = create_silhouette_plot(gmm_cluster_result)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# Tests for create_dendrogram
# ============================================================================


class TestCreateDendrogram:
    """Tests for hierarchical clustering dendrogram."""

    def test_basic_dendrogram(self, hierarchical_cluster_result):
        """Test basic dendrogram creation."""
        fig = create_dendrogram(hierarchical_cluster_result)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Hierarchical Clustering" in ax.get_title()
        plt.close(fig)

    def test_dendrogram_with_cut_height(self, hierarchical_cluster_result):
        """Test dendrogram with horizontal cut line."""
        cut_height = 5.0
        fig = create_dendrogram(hierarchical_cluster_result, cut_height=cut_height)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # Should have horizontal line at cut height
        lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
        assert len(lines) > 0
        plt.close(fig)

    def test_dendrogram_with_n_clusters(self, hierarchical_cluster_result):
        """Test dendrogram with n_clusters specified."""
        fig = create_dendrogram(hierarchical_cluster_result, n_clusters=3)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "3 clusters" in ax.get_title()
        plt.close(fig)

    def test_dendrogram_custom_color_threshold(self, hierarchical_cluster_result):
        """Test dendrogram with custom color threshold."""
        fig = create_dendrogram(
            hierarchical_cluster_result, color_threshold=10.0, cut_height=None
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dendrogram_minimal_linkage(self, linkage_matrix_small):
        """Test dendrogram with minimal linkage matrix."""
        hierarchical_result = {
            "linkage_matrix": linkage_matrix_small,
            "linkage_method": "ward",
            "distance_metric": "euclidean",
        }

        fig = create_dendrogram(hierarchical_result)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dendrogram_custom_title(self, hierarchical_cluster_result):
        """Test dendrogram with custom title."""
        custom_title = "My Custom Dendrogram"
        fig = create_dendrogram(hierarchical_cluster_result, title=custom_title)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert custom_title in ax.get_title()
        plt.close(fig)

    def test_dendrogram_with_labels(self, hierarchical_cluster_result):
        """Test dendrogram with custom sample labels."""
        n_samples = len(hierarchical_cluster_result["data_indices"])
        labels = [f"S{i}" for i in range(n_samples)]

        fig = create_dendrogram(hierarchical_cluster_result, labels=labels)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# Integration Tests
# ============================================================================


class TestVisualizationIntegration:
    """Integration tests combining clustering and visualization."""

    def test_full_kmeans_visualization_pipeline(self, simple_cluster_data):
        """Test complete K-Means clustering visualization workflow."""
        from sleap_roots_analyze.clustering import perform_kmeans_clustering
        from sleap_roots_analyze.pca import perform_pca_analysis

        # Perform clustering
        cluster_result = perform_kmeans_clustering(
            simple_cluster_data, n_clusters=3, random_state=42
        )

        # Perform PCA
        pca_result = perform_pca_analysis(
            simple_cluster_data, n_components=3, standardize=True, random_state=42
        )

        # Create visualizations
        fig1 = create_cluster_scatter_pca(cluster_result, pca_result=pca_result)
        fig2 = create_cluster_size_barplot(
            cluster_result["cluster_labels"], cluster_result["n_clusters"]
        )
        fig3 = create_silhouette_plot(cluster_result)

        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
        assert isinstance(fig3, plt.Figure)

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

    def test_full_hierarchical_visualization_pipeline(self, simple_cluster_data):
        """Test complete hierarchical clustering visualization workflow."""
        from sleap_roots_analyze.clustering import (
            perform_hierarchical_clustering,
            cut_dendrogram,
        )

        # Perform hierarchical clustering
        hier_result = perform_hierarchical_clustering(simple_cluster_data)

        # Cut dendrogram
        cut_result = cut_dendrogram(hier_result, n_clusters=3)

        # Create visualizations
        fig1 = create_dendrogram(hier_result, cut_height=cut_result["cut_height"])
        fig2 = create_cluster_size_barplot(
            cut_result["cluster_labels"], cut_result["n_clusters"]
        )

        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)

        plt.close(fig1)
        plt.close(fig2)

    def test_gmm_with_bic_aic_visualization(self, multimodal_data):
        """Test GMM clustering with BIC/AIC visualization."""
        from sleap_roots_analyze.clustering import perform_gmm_clustering

        # Test multiple n_components
        k_range = range(2, 6)
        bic_scores = []
        aic_scores = []

        for k in k_range:
            result = perform_gmm_clustering(
                multimodal_data, n_components=k, random_state=42
            )
            bic_scores.append(result["bic_scores"][0])
            aic_scores.append(result["aic_scores"][0])

        # Create BIC/AIC comparison plot
        fig = create_bic_aic_comparison_plot(bic_scores, aic_scores)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
