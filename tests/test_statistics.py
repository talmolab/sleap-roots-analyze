"""Tests for statistics module."""

import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

from src.sleap_roots_analyze.statistics import (
    calculate_trait_statistics,
    perform_anova_by_genotype,
    calculate_heritability_estimates,
    identify_high_heritability_traits,
    analyze_heritability_thresholds,
)


class TestCalculateTraitStatistics:
    """Tests for calculate_trait_statistics function."""

    def test_basic_statistics(self):
        """Test calculation of basic statistics for traits."""
        df = pd.DataFrame(
            {
                "trait1": [1, 2, 3, 4, 5],
                "trait2": [10, 20, 30, 40, 50],
            }
        )
        trait_cols = ["trait1", "trait2"]

        stats = calculate_trait_statistics(df, trait_cols)

        assert "trait1" in stats
        assert "trait2" in stats
        assert stats["trait1"]["mean"] == 3.0
        assert stats["trait1"]["std"] > 0
        assert stats["trait2"]["mean"] == 30.0

    def test_with_nan_values(self):
        """Test handling of NaN values."""
        df = pd.DataFrame(
            {
                "trait1": [1, 2, np.nan, 4, 5],
            }
        )
        trait_cols = ["trait1"]

        stats = calculate_trait_statistics(df, trait_cols)

        assert stats["trait1"]["count"] == 4  # NaN excluded
        assert stats["trait1"]["mean"] == 3.0

    def test_empty_column(self):
        """Test handling of all NaN column."""
        df = pd.DataFrame(
            {
                "trait1": [np.nan, np.nan, np.nan],
            }
        )
        trait_cols = ["trait1"]

        stats = calculate_trait_statistics(df, trait_cols)

        assert stats["trait1"]["error"] == "No valid data"


class TestPerformAnovaByGenotype:
    """Tests for perform_anova_by_genotype function."""

    def test_basic_anova(self):
        """Test basic ANOVA functionality."""
        df = pd.DataFrame(
            {
                "geno": ["G1", "G1", "G2", "G2", "G3", "G3"],
                "trait1": [1, 2, 5, 6, 9, 10],
            }
        )
        trait_cols = ["trait1"]

        results = perform_anova_by_genotype(df, trait_cols)

        assert "trait1" in results
        assert "f_statistic" in results["trait1"]
        assert "p_value" in results["trait1"]
        assert results["trait1"]["n_groups"] == 3

    def test_insufficient_groups(self):
        """Test with insufficient groups for ANOVA."""
        df = pd.DataFrame(
            {
                "geno": ["G1", "G1", "G1"],
                "trait1": [1, 2, 3],
            }
        )
        trait_cols = ["trait1"]

        results = perform_anova_by_genotype(df, trait_cols)

        assert results["error"] == "Need at least 2 genotypes for ANOVA"

    def test_missing_genotype_column(self):
        """Test with missing genotype column."""
        df = pd.DataFrame(
            {
                "trait1": [1, 2, 3],
            }
        )
        trait_cols = ["trait1"]

        results = perform_anova_by_genotype(df, trait_cols)

        assert results["error"] == "Genotype column 'geno' not found"


class TestCalculateHeritabilityEstimates:
    """Tests for calculate_heritability_estimates function."""

    def test_basic_heritability(self):
        """Test basic heritability calculation."""
        # Create test data with genotype effects
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "geno": np.repeat(["G1", "G2", "G3"], 10),
                "rep": np.tile(range(1, 11), 3),
                "trait1": np.random.normal(10, 1, 30),
            }
        )
        # Add genotype effects
        df.loc[df["geno"] == "G1", "trait1"] += 2
        df.loc[df["geno"] == "G3", "trait1"] -= 2

        trait_cols = ["trait1"]

        results = calculate_heritability_estimates(df, trait_cols)

        assert "trait1" in results
        assert "heritability" in results["trait1"]
        assert 0 <= results["trait1"]["heritability"] <= 1

    def test_anova_based_method(self):
        """Test forcing ANOVA-based method."""
        df = pd.DataFrame(
            {
                "geno": np.repeat(["G1", "G2"], 10),
                "rep": np.tile(range(1, 11), 2),
                "trait1": np.random.normal(10, 1, 20),
            }
        )
        trait_cols = ["trait1"]

        results = calculate_heritability_estimates(
            df, trait_cols, force_method="anova_based"
        )

        assert results["trait1"]["model_type"] == "anova_based"

    def test_no_variance_trait(self):
        """Test trait with no variance."""
        df = pd.DataFrame(
            {
                "geno": ["G1", "G1", "G2", "G2"],
                "rep": [1, 2, 1, 2],
                "trait1": [5, 5, 5, 5],  # No variance
            }
        )
        trait_cols = ["trait1"]

        results = calculate_heritability_estimates(df, trait_cols)

        assert results["trait1"]["heritability"] == 0.0
        assert results["trait1"]["model_type"] == "no_variance"

    def test_with_filtering_disabled(self):
        """Test that filtering is disabled by default."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002", "BC003", "BC004"],
                "geno": ["G1", "G1", "G2", "G2"],
                "rep": [1, 2, 1, 2],
                "trait1": [1, 2, 5, 6],
            }
        )
        trait_cols = ["trait1"]

        results = calculate_heritability_estimates(df, trait_cols)

        # Should return just dictionary, not tuple
        assert isinstance(results, dict)
        assert "trait1" in results

    def test_with_filtering_enabled(self):
        """Test optional filtering of low heritability traits."""
        # Create data with varying heritability
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "Barcode": [f"BC{i:03d}" for i in range(30)],
                "geno": np.repeat(["G1", "G2", "G3"], 10),
                "rep": np.tile(range(1, 11), 3),
                "trait1": np.random.normal(10, 1, 30),  # High h2
                "trait2": np.random.uniform(0, 20, 30),  # Low h2
            }
        )
        # Add strong genotype effects to trait1
        df.loc[df["geno"] == "G1", "trait1"] += 5
        df.loc[df["geno"] == "G3", "trait1"] -= 5

        trait_cols = ["trait1", "trait2"]

        # Test with filtering enabled
        results = calculate_heritability_estimates(
            df, trait_cols, remove_low_h2=True, h2_threshold=0.3
        )

        # Should return tuple of 4 elements
        assert isinstance(results, tuple)
        assert len(results) == 4

        h2_results, df_filtered, removed_traits, removal_details = results

        assert isinstance(h2_results, dict)
        assert isinstance(df_filtered, pd.DataFrame)
        assert isinstance(removed_traits, list)
        assert isinstance(removal_details, dict)

        # Check that low h2 traits were removed
        assert "trait2" in removed_traits or len(removed_traits) == 0
        assert len(df_filtered.columns) <= len(df.columns)

    def test_filtering_with_custom_threshold(self):
        """Test filtering with custom heritability threshold."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002", "BC003", "BC004"],
                "geno": ["G1", "G1", "G2", "G2"],
                "rep": [1, 2, 1, 2],
                "trait1": [1, 2, 5, 6],
                "trait2": [10, 10, 10, 10],  # No variance
            }
        )
        trait_cols = ["trait1", "trait2"]

        # Use very low threshold to keep most traits
        results = calculate_heritability_estimates(
            df, trait_cols, remove_low_h2=True, h2_threshold=0.001
        )

        h2_results, df_filtered, removed_traits, removal_details = results

        # trait2 with zero variance should still be removed
        assert "trait2" in removed_traits
        assert "trait1" not in removed_traits


class TestIdentifyHighHeritabilityTraits:
    """Tests for identify_high_heritability_traits function."""

    def test_identify_high_h2(self):
        """Test identification of high heritability traits."""
        heritability_results = {
            "trait1": {"heritability": 0.8},
            "trait2": {"heritability": 0.3},
            "trait3": {"heritability": 0.6},
        }

        high_h2 = identify_high_heritability_traits(heritability_results, threshold=0.5)

        assert "trait1" in high_h2
        assert "trait3" in high_h2
        assert "trait2" not in high_h2

    def test_with_invalid_results(self):
        """Test handling of invalid heritability results."""
        heritability_results = {
            "trait1": {"heritability": 0.8},
            "trait2": {"error": "Failed"},
            "trait3": 0.5,  # Invalid format
        }

        high_h2 = identify_high_heritability_traits(heritability_results)

        assert "trait1" in high_h2
        assert "trait2" not in high_h2
        assert "trait3" not in high_h2


class TestAnalyzeHeritabilityThresholds:
    """Tests for analyze_heritability_thresholds function."""

    def test_threshold_analysis(self):
        """Test heritability threshold analysis."""
        heritability_results = {
            "trait1": {"heritability": 0.8},
            "trait2": {"heritability": 0.3},
            "trait3": {"heritability": 0.6},
            "trait4": {"heritability": 0.1},
        }

        analysis = analyze_heritability_thresholds(
            heritability_results, thresholds=np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        )

        assert "thresholds" in analysis
        assert "traits_retained" in analysis
        assert "fraction_retained" in analysis

        # At threshold 0.0, all traits retained
        assert analysis["traits_retained"][0] == 4
        # At threshold 0.5, only trait1 and trait3 retained
        assert analysis["traits_retained"][2] == 2
        # At threshold 1.0, no traits retained
        assert analysis["traits_retained"][4] == 0

    def test_with_nan_values(self):
        """Test handling of NaN heritability values."""
        heritability_results = {
            "trait1": {"heritability": 0.8},
            "trait2": {"heritability": np.nan},
            "trait3": {"error": "Failed"},
        }

        analysis = analyze_heritability_thresholds(heritability_results)

        # Only trait1 has valid heritability
        assert analysis["total_traits"] == 1
