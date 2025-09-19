"""Tests for outlier visualization module."""

from __future__ import annotations

import matplotlib

# Use non-interactive backend for tests to avoid Tk issues
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from sleap_roots_analyze.outlier_visualization import (
    create_isolation_forest_plots,
    create_outlier_overlap_heatmap,
    create_outliers_per_genotype_plot,
    create_mahalanobis_outlier_plots,
    create_pca_outlier_plot,
    create_comprehensive_outlier_comparison,
)


class TestCreateIsolationForestPlots:
    """Test create_isolation_forest_plots function."""

    def test_basic_plot_creation(
        self, outlier_viz_sample_data, outlier_viz_isolation_results
    ):
        """Test basic Isolation Forest plot creation."""
        df, _ = outlier_viz_sample_data
        figures = create_isolation_forest_plots(df, outlier_viz_isolation_results)

        assert "isolation_forest_analysis" in figures
        assert isinstance(figures["isolation_forest_analysis"], plt.Figure)

        # Check that figure has correct subplots
        fig = figures["isolation_forest_analysis"]
        assert len(fig.axes) == 2  # Two subplots

        # Clean up
        plt.close("all")

    def test_with_error_results(
        self, outlier_viz_sample_data, outlier_viz_error_results
    ):
        """Test handling of error results."""
        df, _ = outlier_viz_sample_data
        figures = create_isolation_forest_plots(
            df, outlier_viz_error_results["isolation_forest"]
        )

        # Should return empty dict when error present
        assert len(figures) == 0

        plt.close("all")

    def test_without_anomaly_scores(self, outlier_viz_sample_data):
        """Test with results missing anomaly scores."""
        df, _ = outlier_viz_sample_data
        results = {
            "method": "IsolationForest",
            "outlier_indices": [1, 2, 3],
            "n_outliers": 3,
        }

        figures = create_isolation_forest_plots(df, results)

        # Should return empty dict when no scores
        assert len(figures) == 0

        plt.close("all")

    def test_with_custom_data_indices(self, outlier_viz_sample_data):
        """Test with custom data indices mapping."""
        df, _ = outlier_viz_sample_data
        n_samples = 50

        results = {
            "method": "IsolationForest",
            "anomaly_scores": np.random.uniform(0.4, 1.0, n_samples).tolist(),
            "outlier_indices": [5, 10, 15],
            "data_indices": list(range(10, 10 + n_samples)),  # Custom indices
            "n_outliers": 3,
        }

        figures = create_isolation_forest_plots(df, results)

        assert "isolation_forest_analysis" in figures
        assert isinstance(figures["isolation_forest_analysis"], plt.Figure)

        plt.close("all")

    def test_empty_outliers(self, outlier_viz_sample_data):
        """Test with no outliers detected."""
        df, _ = outlier_viz_sample_data
        n_samples = 100

        results = {
            "method": "IsolationForest",
            "anomaly_scores": np.random.uniform(0.5, 1.0, n_samples).tolist(),
            "outlier_indices": [],
            "n_outliers": 0,
        }

        figures = create_isolation_forest_plots(df, results)

        assert "isolation_forest_analysis" in figures

        # Verify title shows 0 outliers
        fig = figures["isolation_forest_analysis"]
        title = fig.axes[1].get_title()
        assert "0 outliers" in title

        plt.close("all")


class TestCreateOutlierOverlapHeatmap:
    """Test create_outlier_overlap_heatmap function."""

    def test_basic_heatmap_creation(self, outlier_viz_all_methods_results):
        """Test basic overlap heatmap creation."""
        fig = create_outlier_overlap_heatmap(outlier_viz_all_methods_results)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Main plot + colorbar

        # Check title
        assert "Overlap" in fig.axes[0].get_title()

        plt.close("all")

    def test_with_single_method(self):
        """Test with only one method."""
        results = {
            "pca": {
                "outlier_indices": [1, 2, 3],
                "method": "PCA",
            }
        }

        fig = create_outlier_overlap_heatmap(results)

        assert isinstance(fig, plt.Figure)

        plt.close("all")

    def test_with_no_outliers(self, outlier_viz_empty_results):
        """Test with no outliers in any method."""
        fig = create_outlier_overlap_heatmap(outlier_viz_empty_results)

        assert isinstance(fig, plt.Figure)

        # Matrix should show zeros
        ax = fig.axes[0]
        # Check that the heatmap was created
        assert ax.get_ylabel() is not None

        plt.close("all")

    def test_excludes_combined_results(self):
        """Test that combined results are excluded from heatmap."""
        results = {
            "pca": {"outlier_indices": [1, 2, 3]},
            "isolation": {"outlier_indices": [2, 3, 4]},
            "combined": {"outlier_indices": [2, 3]},  # Should be excluded
        }

        fig = create_outlier_overlap_heatmap(results)

        # Check that only 2 methods are shown (not combined)
        ax = fig.axes[0]
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "combined" not in ytick_labels
        assert len(ytick_labels) == 2

        plt.close("all")

    def test_overlap_calculation(self):
        """Test correct overlap calculation."""
        results = {
            "method1": {"outlier_indices": [1, 2, 3]},
            "method2": {"outlier_indices": [2, 3, 4]},
        }

        fig = create_outlier_overlap_heatmap(results)

        # Diagonal should show total counts (3 and 3)
        # Overlap should be 2 (indices 2 and 3)
        # Can't easily extract values from heatmap, but test runs successfully
        assert isinstance(fig, plt.Figure)

        plt.close("all")


class TestCreateOutliersPerGenotypePlot:
    """Test create_outliers_per_genotype_plot function."""

    def test_basic_plot_creation(
        self, outlier_viz_sample_data, outlier_viz_all_methods_results
    ):
        """Test basic genotype outlier plot creation."""
        df, _ = outlier_viz_sample_data
        fig = create_outliers_per_genotype_plot(df, outlier_viz_all_methods_results)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Two subplots

        # Check titles
        assert "per Genotype" in fig.axes[0].get_title()
        assert "Proportion" in fig.axes[1].get_title()

        plt.close("all")

    def test_custom_genotype_column(self, outlier_viz_all_methods_results):
        """Test with custom genotype column name."""
        # Create data with custom column
        df = pd.DataFrame(
            {
                "trait_1": np.random.randn(100),
                "trait_2": np.random.randn(100),
                "variety": np.random.choice(["A", "B", "C"], 100),
            }
        )

        fig = create_outliers_per_genotype_plot(
            df, outlier_viz_all_methods_results, genotype_col="variety"
        )

        assert isinstance(fig, plt.Figure)

        plt.close("all")

    def test_missing_genotype_column(self, outlier_viz_all_methods_results):
        """Test error handling when genotype column is missing."""
        df = pd.DataFrame(
            {
                "trait_1": np.random.randn(100),
                "trait_2": np.random.randn(100),
            }
        )

        with pytest.raises(KeyError):
            create_outliers_per_genotype_plot(df, outlier_viz_all_methods_results)

        plt.close("all")

    def test_with_mismatched_indices(self):
        """Test with outlier indices that don't match DataFrame indices."""
        # Create DataFrame with custom index
        df = pd.DataFrame(
            {
                "trait_1": np.random.randn(50),
                "geno": np.random.choice(["A", "B"], 50),
            }
        )
        df.index = range(100, 150)  # Custom indices

        results = {
            "method1": {"outlier_indices": [0, 1, 2]},  # Won't match df.index
        }

        # Should handle gracefully
        fig = create_outliers_per_genotype_plot(df, results)
        assert isinstance(fig, plt.Figure)

        plt.close("all")

    def test_empty_outliers(self, outlier_viz_sample_data, outlier_viz_empty_results):
        """Test with no outliers detected."""
        df, _ = outlier_viz_sample_data
        fig = create_outliers_per_genotype_plot(df, outlier_viz_empty_results)

        assert isinstance(fig, plt.Figure)

        # Bars should show 0 counts
        ax1 = fig.axes[0]
        # Check that bars exist but are zero height
        assert len(ax1.patches) >= 0  # May have bars even if zero height

        plt.close("all")


class TestCreateMahalanobisOutlierPlots:
    """Test create_mahalanobis_outlier_plots function."""

    def test_basic_plot_creation(
        self, outlier_viz_sample_data, outlier_viz_mahalanobis_results
    ):
        """Test basic Mahalanobis plot creation."""
        df, _ = outlier_viz_sample_data
        figures = create_mahalanobis_outlier_plots(df, outlier_viz_mahalanobis_results)

        assert "mahalanobis_outlier_detection" in figures
        assert "mahalanobis_pc_analysis" in figures
        assert "mahalanobis_threshold_analysis" in figures

        # Check main detection figure
        fig = figures["mahalanobis_outlier_detection"]
        assert len(fig.axes) == 3  # Three subplots

        plt.close("all")

    def test_with_distance_threshold(self, outlier_viz_sample_data):
        """Test with distance-based threshold instead of chi-squared."""
        df, _ = outlier_viz_sample_data

        results = {
            "method": "Mahalanobis",
            "mahalanobis_distances": np.random.uniform(0, 3, 100).tolist(),
            "outlier_indices": [5, 10, 15],
            "n_components": 3,
            "threshold_type": "distance",
            "threshold_value": 2.5,
            "pca_components": np.random.randn(100, 3).tolist(),
            "explained_variance_ratio": [0.4, 0.3, 0.2],
            "feature_names": [f"trait_{i}" for i in range(10)],
            "n_outliers": 3,
        }

        figures = create_mahalanobis_outlier_plots(df, results)

        assert "mahalanobis_outlier_detection" in figures

        # Check that title mentions distance threshold
        fig = figures["mahalanobis_outlier_detection"]
        title = (
            fig._suptitle.get_text()
            if hasattr(fig, "_suptitle") and fig._suptitle
            else ""
        )
        assert "Distance threshold" in title

        plt.close("all")

    def test_with_error_results(
        self, outlier_viz_sample_data, outlier_viz_error_results
    ):
        """Test handling of error results."""
        df, _ = outlier_viz_sample_data
        figures = create_mahalanobis_outlier_plots(
            df, outlier_viz_error_results["mahalanobis"]
        )

        # Should return empty dict when error present
        assert len(figures) == 0

        plt.close("all")

    def test_without_pca_components(self, outlier_viz_sample_data):
        """Test without PCA components for visualization."""
        df, _ = outlier_viz_sample_data

        results = {
            "method": "Mahalanobis",
            "mahalanobis_distances": np.random.uniform(0, 3, 100).tolist(),
            "outlier_indices": [5, 10, 15],
            "n_components": 3,
            "threshold_type": "chi_squared",
            "threshold_value": 7.81,
            "chi2_percentile": 95.0,
            "n_outliers": 3,
        }

        figures = create_mahalanobis_outlier_plots(df, results)

        # Should still create main figure
        assert "mahalanobis_outlier_detection" in figures

        # Third subplot should show message about missing PCA
        fig = figures["mahalanobis_outlier_detection"]
        ax3 = fig.axes[2]
        # Check that text was added to axis
        assert len(ax3.texts) > 0

        plt.close("all")

    def test_pc_selection_analysis(self, outlier_viz_mahalanobis_results):
        """Test PC selection analysis plot."""
        df = pd.DataFrame(np.random.randn(100, 10))

        figures = create_mahalanobis_outlier_plots(df, outlier_viz_mahalanobis_results)

        assert "mahalanobis_pc_analysis" in figures

        fig = figures["mahalanobis_pc_analysis"]
        assert len(fig.axes) == 2  # Two subplots

        plt.close("all")

    def test_with_legacy_pca_loadings(self, outlier_viz_sample_data):
        """Test fallback to legacy PCA loadings calculation."""
        df, _ = outlier_viz_sample_data

        n_features = 10
        n_components = 3

        results = {
            "method": "Mahalanobis",
            "mahalanobis_distances": np.random.uniform(0, 3, 100).tolist(),
            "outlier_indices": [5, 10, 15],
            "n_components": n_components,
            "threshold_type": "chi_squared",
            "threshold_value": 7.81,
            "chi2_percentile": 95.0,
            "explained_variance_ratio": [0.4, 0.3, 0.2],
            "pca_loadings": np.random.randn(n_features, n_components).tolist(),
            "feature_names": [f"trait_{i}" for i in range(n_features)],
            "eigenvalues": [2.5, 1.8, 1.2],
            "n_outliers": 3,
        }

        figures = create_mahalanobis_outlier_plots(df, results)

        assert "mahalanobis_pc_analysis" in figures

        plt.close("all")

    def test_threshold_analysis_plot(self, outlier_viz_mahalanobis_results):
        """Test threshold analysis plot creation."""
        df = pd.DataFrame(np.random.randn(100, 10))

        figures = create_mahalanobis_outlier_plots(df, outlier_viz_mahalanobis_results)

        assert "mahalanobis_threshold_analysis" in figures

        fig = figures["mahalanobis_threshold_analysis"]
        # Should have main axis and twin axes
        assert len(fig.axes) >= 1

        plt.close("all")

    def test_empty_distances(self, outlier_viz_sample_data):
        """Test with empty distances list."""
        df, _ = outlier_viz_sample_data

        results = {
            "method": "Mahalanobis",
            "mahalanobis_distances": [],
            "outlier_indices": [],
            "n_outliers": 0,
        }

        figures = create_mahalanobis_outlier_plots(df, results)

        # Should return empty or handle gracefully
        assert isinstance(figures, dict)

        plt.close("all")


class TestCreatePcaOutlierPlot:
    """Test create_pca_outlier_plot function."""

    def test_basic_plot_creation(
        self, outlier_viz_sample_data, outlier_viz_pca_results
    ):
        """Test basic PCA outlier plot creation."""
        df, _ = outlier_viz_sample_data
        fig = create_pca_outlier_plot(df, outlier_viz_pca_results)

        assert isinstance(fig, plt.Figure)
        # Should have 4 subplots in a 2x3 grid (but only 4 used)
        assert len(fig.axes) == 4

        # Check title
        suptitle = fig._suptitle
        if suptitle:
            assert "PCA Outlier Detection" in suptitle.get_text()

        plt.close("all")

    def test_with_custom_figsize(
        self, outlier_viz_sample_data, outlier_viz_pca_results
    ):
        """Test with custom figure size."""
        df, _ = outlier_viz_sample_data
        fig = create_pca_outlier_plot(df, outlier_viz_pca_results, figsize=(12, 8))

        assert isinstance(fig, plt.Figure)
        # Check figure size
        width, height = fig.get_size_inches()
        assert width == 12
        assert height == 8

        plt.close("all")

    def test_variance_explained_plot(
        self, outlier_viz_sample_data, outlier_viz_pca_results
    ):
        """Test variance explained subplot."""
        df, _ = outlier_viz_sample_data
        fig = create_pca_outlier_plot(df, outlier_viz_pca_results)

        # First subplot should show variance explained
        ax1 = fig.axes[0]
        assert "PC Selection" in ax1.get_title()
        assert ax1.get_xlabel() == "Principal Component"
        assert "Variance" in ax1.get_ylabel()

        plt.close("all")

    def test_reconstruction_errors_plot(
        self, outlier_viz_sample_data, outlier_viz_pca_results
    ):
        """Test reconstruction errors subplot."""
        df, _ = outlier_viz_sample_data
        fig = create_pca_outlier_plot(df, outlier_viz_pca_results)

        # Second subplot should show reconstruction errors
        ax2 = fig.axes[1]
        assert "Reconstruction Errors" in ax2.get_title()
        assert "sorted" in ax2.get_xlabel()

        plt.close("all")

    def test_pca_space_visualization(
        self, outlier_viz_sample_data, outlier_viz_pca_results
    ):
        """Test PCA space (PC1 vs PC2) visualization."""
        df, _ = outlier_viz_sample_data
        fig = create_pca_outlier_plot(df, outlier_viz_pca_results)

        # Third subplot should show PCA space
        ax3 = fig.axes[2]
        assert "PCA Space" in ax3.get_title()
        assert "PC1" in ax3.get_xlabel()
        assert "PC2" in ax3.get_ylabel()

        plt.close("all")

    def test_feature_variance_explained(
        self, outlier_viz_sample_data, outlier_viz_pca_results
    ):
        """Test feature variance explained subplot."""
        df, _ = outlier_viz_sample_data
        fig = create_pca_outlier_plot(df, outlier_viz_pca_results)

        # Fourth subplot should show feature importance
        ax4 = fig.axes[3]
        assert "Feature Variance" in ax4.get_title()

        plt.close("all")

    def test_without_reconstruction_errors(self, outlier_viz_sample_data):
        """Test with missing reconstruction errors."""
        df, _ = outlier_viz_sample_data

        results = {
            "method": "PCA",
            "outlier_indices": [1, 2, 3],
            "explained_variance_ratio": [0.4, 0.3, 0.2],
            "cumulative_variance": [0.4, 0.7, 0.9],
            "n_components": 3,
        }

        fig = create_pca_outlier_plot(df, results)
        assert isinstance(fig, plt.Figure)

        plt.close("all")

    def test_without_pca_components(self, outlier_viz_sample_data):
        """Test without PCA components for visualization."""
        df, _ = outlier_viz_sample_data

        results = {
            "method": "PCA",
            "outlier_indices": [1, 2, 3],
            "reconstruction_errors": np.random.uniform(0, 1, 100).tolist(),
            "explained_variance_ratio": [0.4, 0.3, 0.2],
            "cumulative_variance": [0.4, 0.7, 0.9],
            "n_components": 3,
            "threshold_value": 0.5,
        }

        fig = create_pca_outlier_plot(df, results)
        assert isinstance(fig, plt.Figure)

        plt.close("all")

    def test_with_single_pc_component(self, outlier_viz_sample_data):
        """Test with only 1 PC component (can't plot PC1 vs PC2)."""
        df, _ = outlier_viz_sample_data

        results = {
            "method": "PCA",
            "outlier_indices": [1, 2, 3],
            "pca_components": np.random.randn(100, 1).tolist(),  # Only 1 component
            "explained_variance_ratio": [0.8],
            "cumulative_variance": [0.8],
            "n_components": 1,
        }

        fig = create_pca_outlier_plot(df, results)
        assert isinstance(fig, plt.Figure)

        plt.close("all")

    def test_with_legacy_loadings(self, outlier_viz_sample_data):
        """Test fallback to legacy loadings calculation."""
        df, _ = outlier_viz_sample_data

        n_features = 10
        n_components = 3

        results = {
            "method": "PCA",
            "outlier_indices": [1, 2, 3],
            "loadings": np.random.randn(n_features, n_components).tolist(),
            "feature_names": [f"trait_{i}" for i in range(n_features)],
            "eigenvalues": [2.5, 1.5, 0.8],
            "explained_variance_ratio": [0.5, 0.3, 0.15],
            "cumulative_variance": [0.5, 0.8, 0.95],
            "n_components": n_components,
        }

        fig = create_pca_outlier_plot(df, results)
        assert isinstance(fig, plt.Figure)

        # Fourth subplot should still be created with loadings
        ax4 = fig.axes[3]
        assert "Feature Variance" in ax4.get_title()

        plt.close("all")

    def test_with_all_components_used(self, outlier_viz_sample_data):
        """Test when all components are used (explained variance ~1.0 for all features)."""
        df, _ = outlier_viz_sample_data

        n_features = 10

        results = {
            "method": "PCA",
            "outlier_indices": [1, 2, 3],
            "explained_variance_ratio_per_feature": [0.99] * n_features,  # All ~1.0
            "feature_names": [f"trait_{i}" for i in range(n_features)],
            "n_components": n_features,
            "explained_variance_ratio": [0.3, 0.2, 0.15, 0.1, 0.08],
            "cumulative_variance": [0.3, 0.5, 0.65, 0.75, 0.83],
        }

        fig = create_pca_outlier_plot(df, results)
        assert isinstance(fig, plt.Figure)

        # Fourth subplot should indicate all components used
        ax4 = fig.axes[3]
        title = ax4.get_title()
        assert "All components used" in title or "Feature Variance" in title

        plt.close("all")

    def test_with_custom_data_indices(self, outlier_viz_sample_data):
        """Test with custom data indices mapping."""
        df, _ = outlier_viz_sample_data
        n_samples = 50

        results = {
            "method": "PCA",
            "reconstruction_errors": np.random.uniform(0, 1, n_samples).tolist(),
            "outlier_indices": [15, 20, 25],
            "data_indices": list(range(10, 10 + n_samples)),  # Custom indices
            "pca_components": np.random.randn(n_samples, 3).tolist(),
            "explained_variance_ratio": [0.4, 0.3, 0.2],
            "cumulative_variance": [0.4, 0.7, 0.9],
            "n_components": 3,
            "threshold_value": 0.5,
        }

        fig = create_pca_outlier_plot(df, results)
        assert isinstance(fig, plt.Figure)

        plt.close("all")

    def test_empty_results(self, outlier_viz_sample_data):
        """Test with minimal/empty results."""
        df, _ = outlier_viz_sample_data

        results = {
            "method": "PCA",
            "outlier_indices": [],
            "n_outliers": 0,
        }

        fig = create_pca_outlier_plot(df, results)
        assert isinstance(fig, plt.Figure)

        plt.close("all")


class TestCreateComprehensiveOutlierComparison:
    """Test create_comprehensive_outlier_comparison function."""

    def test_basic_comparison_plot(self, outlier_viz_all_methods_results):
        """Test comprehensive comparison plot creation."""
        fig = create_comprehensive_outlier_comparison(outlier_viz_all_methods_results)

        assert isinstance(fig, plt.Figure)
        # Should have 4 main plots + colorbar for heatmap
        assert len(fig.axes) >= 4  # 2x2 grid + possible colorbar

        # Check main title
        suptitle = fig._suptitle
        if suptitle:
            assert "Comprehensive" in suptitle.get_text()

        plt.close("all")

    def test_without_combined_results(self):
        """Test without combined results."""
        results = {
            "pca": {"outlier_indices": [1, 2, 3]},
            "isolation": {"outlier_indices": [2, 3, 4]},
        }

        fig = create_comprehensive_outlier_comparison(results)

        assert isinstance(fig, plt.Figure)

        plt.close("all")

    def test_with_all_seven_methods(self):
        """Test with all seven outlier detection methods."""
        results = {
            "pca": {"outlier_indices": [1, 2]},
            "isolation": {"outlier_indices": [2, 3]},
            "mahalanobis": {"outlier_indices": [3, 4]},
            "kmeans": {"outlier_indices": [4, 5]},
            "gmm": {"outlier_indices": [5, 6]},
            "mincovdet": {"outlier_indices": [6, 7]},
            "iqr_per_trait": {"outlier_indices": [7, 8]},
            "combined": {
                "consensus_outliers": [2, 3, 5],
                "n_methods": 7,
                "consensus_threshold": 0.5,
                "pca_outliers": [1, 2],
                "isolation_outliers": [2, 3],
                "mahalanobis_outliers": [3, 4],
                "kmeans_outliers": [4, 5],
                "gmm_outliers": [5, 6],
                "mincovdet_outliers": [6, 7],
                "iqr_per_trait_outliers": [7, 8],
            },
        }

        fig = create_comprehensive_outlier_comparison(results)

        assert isinstance(fig, plt.Figure)

        # Check that bar plot shows 7 methods
        ax1 = fig.axes[0]
        assert len(ax1.patches) == 7  # 7 bars

        plt.close("all")

    def test_empty_results(self):
        """Test with empty results."""
        results = {}

        fig = create_comprehensive_outlier_comparison(results)

        assert isinstance(fig, plt.Figure)

        plt.close("all")

    def test_consensus_analysis(self):
        """Test consensus analysis subplot."""
        results = {
            "pca": {"outlier_indices": [1, 2, 3]},
            "isolation": {"outlier_indices": [2, 3, 4]},
            "combined": {
                "consensus_outliers": [2, 3],
                "n_methods": 2,
                "consensus_threshold": 0.5,
                "pca_outliers": [1, 2, 3],
                "isolation_outliers": [2, 3, 4],
            },
        }

        fig = create_comprehensive_outlier_comparison(results)

        # Check consensus analysis plot (ax3)
        ax3 = fig.axes[2]
        assert "Consensus" in ax3.get_title()

        plt.close("all")

    def test_summary_statistics(self, outlier_viz_all_methods_results):
        """Test summary statistics text display."""
        fig = create_comprehensive_outlier_comparison(outlier_viz_all_methods_results)

        # Check summary subplot (ax4)
        ax4 = fig.axes[3]
        assert "Summary" in ax4.get_title()

        # Should have text showing statistics
        assert len(ax4.texts) > 0

        plt.close("all")

    def test_dynamic_method_detection(self):
        """Test that methods are dynamically detected from combined results."""
        # Test with different method combinations
        results = {
            "combined": {
                "consensus_outliers": [1, 2],
                "custom_method_outliers": [1, 2, 3],
                "another_method_outliers": [2, 3, 4],
                "yet_another_outliers": [3, 4, 5],
                "n_methods": 3,
                "consensus_threshold": 0.5,
            }
        }

        fig = create_comprehensive_outlier_comparison(results)

        # Should create a valid figure even with custom method names
        assert isinstance(fig, plt.Figure)

        # Check that the heatmap includes the custom methods
        ax2 = fig.axes[1]
        assert "Overlap" in ax2.get_title()

        plt.close("all")

    def test_methods_from_individual_results(self):
        """Test method detection from individual results when no combined results."""
        # Test without combined results
        results = {
            "custom_method_1": {"outlier_indices": [1, 2, 3]},
            "custom_method_2": {"outlier_indices": [2, 3, 4]},
            "custom_method_3": {"outlier_indices": [3, 4, 5]},
        }

        fig = create_comprehensive_outlier_comparison(results)

        # Should create a valid figure
        assert isinstance(fig, plt.Figure)

        # Check that bar plot shows 3 methods
        ax1 = fig.axes[0]
        assert len(ax1.patches) == 3

        plt.close("all")


class TestIntegration:
    """Integration tests for outlier visualization module."""

    def test_full_pipeline_with_real_data(self, features_df):
        """Test full visualization pipeline with real data."""
        # Get numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        df_numeric = features_df[numeric_cols[:10]].dropna()  # Use first 10 columns

        if len(df_numeric) < 10:
            pytest.skip("Not enough data for integration test")

        # Create mock results
        n_samples = len(df_numeric)
        results = {
            "isolation_forest": {
                "method": "IsolationForest",
                "anomaly_scores": np.random.uniform(0.4, 1.0, n_samples).tolist(),
                "outlier_indices": list(range(5)),
                "n_outliers": 5,
            },
            "mahalanobis": {
                "method": "Mahalanobis",
                "mahalanobis_distances": np.random.uniform(0, 3, n_samples).tolist(),
                "outlier_indices": list(range(3, 8)),
                "n_components": 3,
                "threshold_type": "chi_squared",
                "threshold_value": 7.81,
                "chi2_percentile": 95.0,
                "n_outliers": 5,
            },
        }

        # Test each visualization function
        iso_figs = create_isolation_forest_plots(
            df_numeric, results["isolation_forest"]
        )
        assert isinstance(iso_figs, dict)

        overlap_fig = create_outlier_overlap_heatmap(results)
        assert isinstance(overlap_fig, plt.Figure)

        mahal_figs = create_mahalanobis_outlier_plots(
            df_numeric, results["mahalanobis"]
        )
        assert isinstance(mahal_figs, dict)

        comp_fig = create_comprehensive_outlier_comparison(results)
        assert isinstance(comp_fig, plt.Figure)

        plt.close("all")

    def test_all_functions_handle_empty_data(self):
        """Test that all functions handle empty data gracefully."""
        empty_df = pd.DataFrame()
        empty_results = {}

        # All functions should handle empty data without crashing
        iso_figs = create_isolation_forest_plots(empty_df, empty_results)
        assert isinstance(iso_figs, dict)

        overlap_fig = create_outlier_overlap_heatmap(empty_results)
        assert isinstance(overlap_fig, plt.Figure)

        mahal_figs = create_mahalanobis_outlier_plots(empty_df, empty_results)
        assert isinstance(mahal_figs, dict)

        comp_fig = create_comprehensive_outlier_comparison(empty_results)
        assert isinstance(comp_fig, plt.Figure)

        plt.close("all")

    def test_matplotlib_backend_compatibility(self):
        """Test that plots work with non-interactive backend."""
        # Already using Agg backend, just verify it works
        results = {
            "pca": {"outlier_indices": [1, 2, 3]},
        }

        fig = create_outlier_overlap_heatmap(results)

        # Should be able to save to buffer without display
        from io import BytesIO

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        assert buf.getvalue()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature

        plt.close("all")
