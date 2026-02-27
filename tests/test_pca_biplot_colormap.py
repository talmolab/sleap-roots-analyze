"""Tests for PCA biplot discrete coloring with integer genotype IDs.

Task coverage (fix-numeric-genotype-colormap):
  1.1a  Integer genotype column → N distinct scatter collections (one per genotype)
  1.1b  Integer genotype column → no colorbar
  1.1c  Integer genotype column → legend labels match str(int_id)
  1.1d  String genotype column → N distinct scatter collections (regression guard)
  1.1e  String genotype column → legend labels match original string values (regression guard)
  1.1f  Float color_by column → continuous colorbar retained (regression guard)
  1.1g  genotypes_to_color filter works when genotype column is integer
  1.1h  highlight_genotypes works when genotype column is integer
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from sleap_roots_analyze.visualization import create_pca_biplot


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pca_results():
    """Minimal PCA results dict for biplot testing."""
    np.random.seed(0)
    n_samples = 30
    n_features = 5
    n_components = 3

    X = np.random.randn(n_samples, n_components)
    loadings = np.random.randn(n_features, n_components)
    loadings, _ = np.linalg.qr(loadings)
    eigenvalues = np.array([3.0, 2.0, 1.0])

    return {
        "transformed_data": X,
        "loadings": loadings,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": eigenvalues / eigenvalues.sum(),
        "n_components_selected": n_components,
        "feature_names": [f"t{i}" for i in range(n_features)],
    }


@pytest.fixture
def df_with_int_genotype(pca_results):
    """DataFrame whose Genotype column is int64 (USDA-style accession IDs)."""
    n = len(pca_results["transformed_data"])
    accession_ids = [12305183, 12306047, 12315028, 12315892, 12316828]
    return pd.DataFrame(
        {
            "Genotype": np.random.choice(accession_ids, n),
            "trait_a": np.random.randn(n),
        }
    )


@pytest.fixture
def df_with_str_genotype(pca_results):
    """DataFrame whose Genotype column is string (object dtype)."""
    n = len(pca_results["transformed_data"])
    return pd.DataFrame(
        {
            "Genotype": np.random.choice(["GEN_A", "GEN_B", "GEN_C"], n),
            "trait_a": np.random.randn(n),
        }
    )


@pytest.fixture
def df_with_float_col(pca_results):
    """DataFrame with a float column as color_by (continuous measurement)."""
    n = len(pca_results["transformed_data"])
    return pd.DataFrame(
        {
            "Genotype": np.random.choice(["A", "B"], n),
            "trait_a": np.random.randn(n),  # float, many unique values
        }
    )


TRAIT_NAMES = ["t0", "t1", "t2", "t3", "t4"]


# =============================================================================
# Tests
# =============================================================================


class TestIntegerGenotypeBiplot:
    """Task 1.1a–1.1c: integer Genotype column → discrete tab10 colors."""

    def test_produces_n_scatter_collections(self, pca_results, df_with_int_genotype):
        """Task 1.1a: One PathCollection per unique integer genotype ID."""
        n_unique = df_with_int_genotype["Genotype"].nunique()
        fig = create_pca_biplot(
            pca_results,
            df_with_int_genotype,
            TRAIT_NAMES,
            color_by="Genotype",
        )
        ax = fig.axes[0]
        scatter_collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
        assert len(scatter_collections) == n_unique
        plt.close(fig)

    def test_no_colorbar(self, pca_results, df_with_int_genotype):
        """Task 1.1b: No continuous colorbar — only main axes."""
        fig = create_pca_biplot(
            pca_results,
            df_with_int_genotype,
            TRAIT_NAMES,
            color_by="Genotype",
        )
        assert len(fig.axes) == 1, "Expected no colorbar (one axis only)"
        plt.close(fig)

    def test_legend_labels_are_string_representations(
        self, pca_results, df_with_int_genotype
    ):
        """Task 1.1c: Legend labels are str(int_id) for each unique value."""
        int_ids = df_with_int_genotype["Genotype"].unique()
        expected_labels = {str(i) for i in int_ids}

        fig = create_pca_biplot(
            pca_results,
            df_with_int_genotype,
            TRAIT_NAMES,
            color_by="Genotype",
        )
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None, "Expected a categorical legend"
        legend_texts = {t.get_text() for t in legend.get_texts()}
        assert legend_texts == expected_labels
        plt.close(fig)


class TestStringGenotypeBiplotRegression:
    """Task 1.1d–1.1e: string Genotype column behavior unchanged."""

    def test_produces_n_scatter_collections(self, pca_results, df_with_str_genotype):
        """Task 1.1d: String genotype → one PathCollection per genotype."""
        n_unique = df_with_str_genotype["Genotype"].nunique()
        fig = create_pca_biplot(
            pca_results,
            df_with_str_genotype,
            TRAIT_NAMES,
            color_by="Genotype",
        )
        ax = fig.axes[0]
        scatter_collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
        assert len(scatter_collections) == n_unique
        plt.close(fig)

    def test_legend_labels_are_original_strings(
        self, pca_results, df_with_str_genotype
    ):
        """Task 1.1e: String genotype legend labels unchanged."""
        expected = set(df_with_str_genotype["Genotype"].unique())
        fig = create_pca_biplot(
            pca_results,
            df_with_str_genotype,
            TRAIT_NAMES,
            color_by="Genotype",
        )
        ax = fig.axes[0]
        legend_texts = {t.get_text() for t in ax.get_legend().get_texts()}
        assert legend_texts == expected
        plt.close(fig)


class TestFloatColorByRegression:
    """Task 1.1f: float color_by → continuous colorbar retained."""

    def test_float_color_by_still_produces_colorbar(
        self, pca_results, df_with_float_col
    ):
        """Task 1.1f: Float column with many unique values retains continuous colormap."""
        fig = create_pca_biplot(
            pca_results,
            df_with_float_col,
            TRAIT_NAMES,
            color_by="trait_a",
        )
        assert len(fig.axes) > 1, "Expected colorbar axis for float color_by"
        plt.close(fig)


class TestIntegerGenotypeWithFiltering:
    """Task 1.1g–1.1h: genotypes_to_color and highlight_genotypes work with int IDs."""

    def test_genotypes_to_color_filter_with_int_ids(
        self, pca_results, df_with_int_genotype
    ):
        """Task 1.1g: genotypes_to_color selects a subset when IDs are integers."""
        all_ids = list(df_with_int_genotype["Genotype"].unique())
        # Pass integer values (as they appear in the data) as strings
        # to match what the legend will show after the cast
        selected_str = [str(all_ids[0]), str(all_ids[1])]

        fig = create_pca_biplot(
            pca_results,
            df_with_int_genotype,
            TRAIT_NAMES,
            color_by="Genotype",
            genotypes_to_color=selected_str,
        )
        ax = fig.axes[0]
        legend_texts = {t.get_text() for t in ax.get_legend().get_texts()}
        # Selected IDs appear by name; others appear as "Other"
        for s in selected_str:
            assert s in legend_texts
        assert "Other" in legend_texts
        plt.close(fig)

    def test_highlight_genotypes_with_int_ids(self, pca_results, df_with_int_genotype):
        """Task 1.1h: highlight_genotypes does not raise when IDs are integers."""
        highlight_str = [str(df_with_int_genotype["Genotype"].iloc[0])]
        fig = create_pca_biplot(
            pca_results,
            df_with_int_genotype,
            TRAIT_NAMES,
            color_by="Genotype",
            highlight_genotypes=highlight_str,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
