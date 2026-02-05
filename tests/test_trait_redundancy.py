"""Tests for trait redundancy reduction functions.

TDD tests for hierarchical clustering-based trait redundancy reduction.
These tests define expected behavior before implementation.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def synthetic_clustered_traits():
    """Create synthetic data with known cluster structure.

    Creates 10 traits in 3 clusters:
    - Cluster A: traits A1, A2, A3 (highly correlated, r ≈ 0.95)
    - Cluster B: traits B1, B2, B3 (highly correlated, r ≈ 0.95)
    - Cluster C: traits C1, C2, C3, C4 (highly correlated, r ≈ 0.95)

    Cross-cluster correlations are low (r ≈ 0.1).

    Returns:
        pd.DataFrame: 20 genotypes × 10 traits
    """
    np.random.seed(42)
    n_genotypes = 20

    # Base signals for each cluster (independent)
    base_a = np.random.randn(n_genotypes)
    base_b = np.random.randn(n_genotypes)
    base_c = np.random.randn(n_genotypes)

    # Cluster A traits: base_a + small noise
    noise_scale = 0.1  # Small noise = high correlation
    df = pd.DataFrame(
        {
            # Cluster A (3 traits)
            "A1": base_a + np.random.randn(n_genotypes) * noise_scale,
            "A2": base_a + np.random.randn(n_genotypes) * noise_scale,
            "A3": base_a + np.random.randn(n_genotypes) * noise_scale,
            # Cluster B (3 traits)
            "B1": base_b + np.random.randn(n_genotypes) * noise_scale,
            "B2": base_b + np.random.randn(n_genotypes) * noise_scale,
            "B3": base_b + np.random.randn(n_genotypes) * noise_scale,
            # Cluster C (4 traits)
            "C1": base_c + np.random.randn(n_genotypes) * noise_scale,
            "C2": base_c + np.random.randn(n_genotypes) * noise_scale,
            "C3": base_c + np.random.randn(n_genotypes) * noise_scale,
            "C4": base_c + np.random.randn(n_genotypes) * noise_scale,
        },
        index=[f"G{i}" for i in range(n_genotypes)],
    )
    return df


@pytest.fixture
def independent_traits():
    """Create data with all independent traits (no clusters expected).

    Returns:
        pd.DataFrame: 20 genotypes × 5 independent traits (r ≈ 0)
    """
    np.random.seed(123)
    n_genotypes = 20

    df = pd.DataFrame(
        {
            "Ind1": np.random.randn(n_genotypes),
            "Ind2": np.random.randn(n_genotypes),
            "Ind3": np.random.randn(n_genotypes),
            "Ind4": np.random.randn(n_genotypes),
            "Ind5": np.random.randn(n_genotypes),
        },
        index=[f"G{i}" for i in range(n_genotypes)],
    )
    return df


@pytest.fixture
def traits_with_edge_cases():
    """Create data with edge cases for clustering.

    Contains:
    - Normal traits (A, B)
    - Constant trait (all same value) - should be singleton
    - All-NaN trait - should be singleton
    - High-variance trait (for representative selection)
    """
    np.random.seed(456)
    n_genotypes = 15

    base = np.random.randn(n_genotypes)

    df = pd.DataFrame(
        {
            "NormalA": base + np.random.randn(n_genotypes) * 0.1,
            "NormalB": base + np.random.randn(n_genotypes) * 0.1,
            "Constant": np.ones(n_genotypes) * 5.0,  # Zero variance
            "AllNaN": np.full(n_genotypes, np.nan),  # All NaN
            "HighVar": base * 10,  # High variance (10x)
            "LowVar": base * 0.1,  # Low variance (0.1x)
        },
        index=[f"G{i}" for i in range(n_genotypes)],
    )
    return df


@pytest.fixture
def perfect_correlation_traits():
    """Create data with perfect correlations (r = 1.0).

    Returns:
        pd.DataFrame: 10 genotypes × 3 traits where A and B are identical
    """
    np.random.seed(789)
    n_genotypes = 10

    base = np.random.randn(n_genotypes)

    df = pd.DataFrame(
        {
            "TraitA": base,
            "TraitB": base,  # Identical to A (r = 1.0)
            "TraitC": np.random.randn(n_genotypes),  # Independent
        },
        index=[f"G{i}" for i in range(n_genotypes)],
    )
    return df


# ============================================================================
# TESTS: cluster_correlated_traits()
# ============================================================================


class TestClusterCorrelatedTraits:
    """Tests for cluster_correlated_traits function."""

    def test_clusters_perfectly_correlated_traits(self, perfect_correlation_traits):
        """Test that perfectly correlated traits are grouped together.

        WHEN df contains 3 traits where traits A and B have r=1.0 and trait C is independent
        THEN cluster_correlated_traits returns 2 clusters: one with [A, B], one with [C]
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
        )

        clusters = cluster_correlated_traits(
            perfect_correlation_traits, threshold=0.8, linkage="complete"
        )

        # Should have 2 clusters
        assert len(clusters) == 2

        # Find which cluster contains TraitA
        cluster_with_a = None
        cluster_with_c = None
        for cluster_id, traits in clusters.items():
            if "TraitA" in traits:
                cluster_with_a = traits
            if "TraitC" in traits:
                cluster_with_c = traits

        # TraitA and TraitB should be in same cluster
        assert cluster_with_a is not None
        assert set(cluster_with_a) == {"TraitA", "TraitB"}

        # TraitC should be alone
        assert cluster_with_c is not None
        assert set(cluster_with_c) == {"TraitC"}

    def test_clusters_synthetic_data_correctly(self, synthetic_clustered_traits):
        """Test clustering with synthetic 3-cluster data.

        WHEN df contains 10 traits in 3 known clusters
        THEN cluster_correlated_traits returns approximately 3 clusters
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
        )

        clusters = cluster_correlated_traits(
            synthetic_clustered_traits, threshold=0.8, linkage="complete"
        )

        # Should have approximately 3 clusters (may vary slightly due to noise)
        assert 2 <= len(clusters) <= 4, f"Expected ~3 clusters, got {len(clusters)}"

        # Total traits should equal original count
        all_traits = []
        for traits in clusters.values():
            all_traits.extend(traits)
        assert len(all_traits) == 10

    def test_threshold_sensitivity(self, synthetic_clustered_traits):
        """Test that higher threshold produces more clusters.

        WHEN threshold increases
        THEN cluster count increases (more stringent grouping)
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
        )

        clusters_low = cluster_correlated_traits(
            synthetic_clustered_traits, threshold=0.5, linkage="complete"
        )
        clusters_high = cluster_correlated_traits(
            synthetic_clustered_traits, threshold=0.95, linkage="complete"
        )

        # Higher threshold = more clusters (stricter grouping)
        assert len(clusters_high) >= len(clusters_low)

    def test_deterministic_output(self, synthetic_clustered_traits):
        """Test that output is deterministic (same input = same output).

        WHEN cluster_correlated_traits is called twice with identical inputs
        THEN output cluster assignments are identical
        AND trait ordering within clusters is deterministic (alphabetical)
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
        )

        clusters1 = cluster_correlated_traits(
            synthetic_clustered_traits, threshold=0.8, linkage="complete"
        )
        clusters2 = cluster_correlated_traits(
            synthetic_clustered_traits, threshold=0.8, linkage="complete"
        )

        # Should be identical
        assert len(clusters1) == len(clusters2)

        # Convert to comparable format (sorted lists)
        def normalize_clusters(clusters):
            return sorted([sorted(traits) for traits in clusters.values()])

        assert normalize_clusters(clusters1) == normalize_clusters(clusters2)

        # Traits within each cluster should be alphabetically sorted
        for traits in clusters1.values():
            assert traits == sorted(traits), "Traits within cluster should be sorted"

    def test_single_trait_returns_singleton(self):
        """Test that single-trait DataFrame returns one cluster.

        WHEN df contains only one trait
        THEN cluster_correlated_traits returns one cluster containing that trait
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
        )

        df = pd.DataFrame(
            {"SingleTrait": np.random.randn(10)}, index=[f"G{i}" for i in range(10)]
        )

        clusters = cluster_correlated_traits(df, threshold=0.8, linkage="complete")

        assert len(clusters) == 1
        assert list(clusters.values())[0] == ["SingleTrait"]

    def test_empty_dataframe_returns_empty(self):
        """Test that empty DataFrame returns empty dict.

        WHEN df is empty
        THEN cluster_correlated_traits returns empty dict
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
        )

        df = pd.DataFrame()

        clusters = cluster_correlated_traits(df, threshold=0.8, linkage="complete")

        assert clusters == {}

    def test_constant_trait_is_singleton(self, traits_with_edge_cases):
        """Test that constant trait (zero variance) is placed in singleton cluster.

        WHEN df contains a trait with zero variance (all genotypes have same value)
        THEN that trait is placed in its own singleton cluster
        AND other traits are clustered normally
        AND no error is raised
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
        )

        clusters = cluster_correlated_traits(
            traits_with_edge_cases, threshold=0.8, linkage="complete"
        )

        # Find cluster containing "Constant"
        constant_cluster = None
        for traits in clusters.values():
            if "Constant" in traits:
                constant_cluster = traits
                break

        # Constant trait should be alone (can't correlate with others)
        assert constant_cluster is not None
        assert constant_cluster == ["Constant"]

    def test_all_nan_trait_is_singleton(self, traits_with_edge_cases):
        """Test that all-NaN trait is placed in singleton cluster.

        WHEN df contains a trait with all NaN values
        THEN that trait is placed in its own singleton cluster
        AND other traits are clustered normally
        AND no error is raised
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
        )

        clusters = cluster_correlated_traits(
            traits_with_edge_cases, threshold=0.8, linkage="complete"
        )

        # Find cluster containing "AllNaN"
        nan_cluster = None
        for traits in clusters.values():
            if "AllNaN" in traits:
                nan_cluster = traits
                break

        # AllNaN trait should be alone
        assert nan_cluster is not None
        assert nan_cluster == ["AllNaN"]

    def test_independent_traits_remain_separate(self, independent_traits):
        """Test that independent traits (low correlation) remain separate.

        WHEN df contains all independent traits (r ≈ 0)
        THEN each trait remains in its own cluster
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
        )

        clusters = cluster_correlated_traits(
            independent_traits, threshold=0.8, linkage="complete"
        )

        # Each trait should be its own cluster (no pairs have r >= 0.8)
        assert len(clusters) == 5

    def test_complete_linkage_prevents_weak_link_clustering(self):
        """Test that complete linkage requires all pairs to meet threshold.

        WHEN linkage is "complete" and traits A-B have r=0.9, B-C have r=0.9, but A-C have r=0.5
        THEN A, B, C are NOT all in the same cluster
        AND complete linkage prevents weak-link clustering
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
        )

        np.random.seed(999)
        n = 20

        # Create traits where A-B correlated, B-C correlated, but A-C not
        base_a = np.random.randn(n)
        base_c = np.random.randn(n)

        # B is mix of A and C (so B correlates with both, but A doesn't correlate with C)
        base_b = 0.7 * base_a + 0.7 * base_c

        df = pd.DataFrame(
            {
                "TraitA": base_a,
                "TraitB": base_b,
                "TraitC": base_c,
            },
            index=[f"G{i}" for i in range(n)],
        )

        clusters = cluster_correlated_traits(df, threshold=0.7, linkage="complete")

        # With complete linkage, not all three should be in same cluster
        # because A-C correlation is weak
        max_cluster_size = max(len(traits) for traits in clusters.values())
        # At threshold 0.7, we expect either 2 or 3 clusters
        assert max_cluster_size <= 2 or len(clusters) >= 2

    @pytest.mark.parametrize("linkage", ["complete", "average", "single"])
    def test_all_linkage_methods_work(self, synthetic_clustered_traits, linkage):
        """Test that all supported linkage methods execute without error."""
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
        )

        clusters = cluster_correlated_traits(
            synthetic_clustered_traits, threshold=0.8, linkage=linkage
        )

        # Should return valid clusters
        assert isinstance(clusters, dict)
        assert len(clusters) > 0


# ============================================================================
# TESTS: select_cluster_representatives()
# ============================================================================


class TestSelectClusterRepresentatives:
    """Tests for select_cluster_representatives function."""

    def test_selects_highest_variance_trait(self):
        """Test that highest variance trait is selected as representative.

        WHEN cluster contains traits with variances [0.5, 0.8, 0.3]
        THEN trait with variance 0.8 is selected as representative
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            select_cluster_representatives,
        )

        np.random.seed(42)
        n = 20

        # Create traits with different variances
        df = pd.DataFrame(
            {
                "LowVar": np.random.randn(n) * 0.5,
                "HighVar": np.random.randn(n) * 1.5,
                "MedVar": np.random.randn(n) * 1.0,
            },
            index=[f"G{i}" for i in range(n)],
        )

        clusters = {0: ["LowVar", "HighVar", "MedVar"]}

        representatives = select_cluster_representatives(df, clusters)

        assert len(representatives) == 1
        assert representatives[0] == "HighVar"

    def test_alphabetical_tiebreaking(self):
        """Test alphabetical tie-breaking when variances are equal.

        WHEN cluster contains traits "Beta" and "Alpha" with identical variance
        THEN "Alpha" is selected as representative (alphabetically first)
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            select_cluster_representatives,
        )

        np.random.seed(42)
        n = 20

        # Create identical variance traits
        values = np.random.randn(n)
        df = pd.DataFrame(
            {
                "Beta": values.copy(),
                "Alpha": values.copy(),  # Identical variance
                "Gamma": values.copy(),
            },
            index=[f"G{i}" for i in range(n)],
        )

        clusters = {0: ["Alpha", "Beta", "Gamma"]}

        representatives = select_cluster_representatives(df, clusters)

        assert representatives[0] == "Alpha"

    def test_singleton_cluster_returns_single_trait(self):
        """Test that singleton cluster returns its only trait.

        WHEN cluster contains only one trait
        THEN that trait is returned as representative
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            select_cluster_representatives,
        )

        df = pd.DataFrame(
            {"OnlyTrait": np.random.randn(10)}, index=[f"G{i}" for i in range(10)]
        )

        clusters = {0: ["OnlyTrait"]}

        representatives = select_cluster_representatives(df, clusters)

        assert representatives == ["OnlyTrait"]

    def test_multiple_clusters_multiple_representatives(
        self, synthetic_clustered_traits
    ):
        """Test that each cluster gets exactly one representative."""
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
            select_cluster_representatives,
        )

        clusters = cluster_correlated_traits(
            synthetic_clustered_traits, threshold=0.8, linkage="complete"
        )

        representatives = select_cluster_representatives(
            synthetic_clustered_traits, clusters
        )

        # One representative per cluster
        assert len(representatives) == len(clusters)

        # All representatives are unique
        assert len(set(representatives)) == len(representatives)

        # All representatives are valid trait names
        for rep in representatives:
            assert rep in synthetic_clustered_traits.columns

    def test_handles_nan_variance(self, traits_with_edge_cases):
        """Test handling of trait with NaN variance (all NaN values).

        WHEN a trait has NaN variance (all NaN values)
        THEN that trait is not selected as representative
        AND next highest variance trait is selected
        """
        from sleap_roots_analyze.cross_experiment_analysis import (
            select_cluster_representatives,
        )

        df = traits_with_edge_cases

        # Create cluster with AllNaN and a normal trait
        # AllNaN has undefined variance, so NormalA should be selected
        clusters = {0: ["AllNaN", "NormalA"]}

        representatives = select_cluster_representatives(df, clusters)

        assert representatives[0] == "NormalA"

    def test_preserves_trait_names_correctly(self):
        """Test that trait names are preserved exactly."""
        from sleap_roots_analyze.cross_experiment_analysis import (
            select_cluster_representatives,
        )

        df = pd.DataFrame(
            {
                "Trait_With_Underscore": np.random.randn(10),
                "trait-with-dash": np.random.randn(10),
                "CamelCaseTrait": np.random.randn(10),
            },
            index=[f"G{i}" for i in range(10)],
        )

        clusters = {
            0: ["Trait_With_Underscore"],
            1: ["trait-with-dash"],
            2: ["CamelCaseTrait"],
        }

        representatives = select_cluster_representatives(df, clusters)

        assert set(representatives) == {
            "Trait_With_Underscore",
            "trait-with-dash",
            "CamelCaseTrait",
        }

    def test_empty_clusters_returns_empty_list(self):
        """Test that empty clusters dict returns empty list."""
        from sleap_roots_analyze.cross_experiment_analysis import (
            select_cluster_representatives,
        )

        df = pd.DataFrame({"A": [1, 2, 3]}, index=["G1", "G2", "G3"])
        clusters = {}

        representatives = select_cluster_representatives(df, clusters)

        assert representatives == []


# ============================================================================
# TESTS: CrossPlatformConfig trait reduction parameters
# ============================================================================


class TestCrossPlatformConfigTraitReduction:
    """Tests for trait reduction configuration parameters."""

    def test_default_config_no_reduction(self, tmp_path):
        """Test that default config preserves existing behavior (no reduction).

        WHEN user creates CrossPlatformConfig without trait reduction parameters
        THEN trait_reduction_method defaults to "none"
        """
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        # Create minimal config
        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        config = CrossPlatformConfig(
            exp1_data_path=str(exp1_path),
            exp1_name="Exp1",
            exp1_genotype_col="genotype",
            exp2_data_path=str(exp2_path),
            exp2_name="Exp2",
            exp2_genotype_col="genotype",
        )

        assert config.trait_reduction_method == "none"

    def test_clustering_reduction_config(self, tmp_path):
        """Test configuration with clustering-based reduction enabled."""
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        config = CrossPlatformConfig(
            exp1_data_path=str(exp1_path),
            exp1_name="Exp1",
            exp1_genotype_col="genotype",
            exp2_data_path=str(exp2_path),
            exp2_name="Exp2",
            exp2_genotype_col="genotype",
            trait_reduction_method="clustering",
            trait_reduction_target="both",
            trait_clustering_threshold=0.85,
            trait_clustering_linkage="average",
        )

        assert config.trait_reduction_method == "clustering"
        assert config.trait_reduction_target == "both"
        assert config.trait_clustering_threshold == 0.85
        assert config.trait_clustering_linkage == "average"

    def test_invalid_reduction_method_raises_error(self, tmp_path):
        """Test that invalid reduction method raises validation error.

        WHEN user sets trait_reduction_method to "invalid_method"
        THEN configuration validation fails with error listing valid options
        """
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        with pytest.raises(
            ValueError, match=r"trait_reduction_method.*none.*clustering"
        ):
            CrossPlatformConfig(
                exp1_data_path=str(exp1_path),
                exp1_name="Exp1",
                exp1_genotype_col="genotype",
                exp2_data_path=str(exp2_path),
                exp2_name="Exp2",
                exp2_genotype_col="genotype",
                trait_reduction_method="pca",  # Not implemented
            )

    def test_invalid_threshold_zero_raises_error(self, tmp_path):
        """Test that threshold of 0 raises validation error.

        WHEN user sets trait_clustering_threshold to 0
        THEN configuration validation fails with error indicating valid range (0, 1]
        """
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        with pytest.raises(ValueError, match=r"trait_clustering_threshold.*\(0, 1\]"):
            CrossPlatformConfig(
                exp1_data_path=str(exp1_path),
                exp1_name="Exp1",
                exp1_genotype_col="genotype",
                exp2_data_path=str(exp2_path),
                exp2_name="Exp2",
                exp2_genotype_col="genotype",
                trait_reduction_method="clustering",
                trait_reduction_target="both",
                trait_clustering_threshold=0.0,
            )

    def test_invalid_threshold_negative_raises_error(self, tmp_path):
        """Test that negative threshold raises validation error."""
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        with pytest.raises(ValueError, match=r"trait_clustering_threshold.*\(0, 1\]"):
            CrossPlatformConfig(
                exp1_data_path=str(exp1_path),
                exp1_name="Exp1",
                exp1_genotype_col="genotype",
                exp2_data_path=str(exp2_path),
                exp2_name="Exp2",
                exp2_genotype_col="genotype",
                trait_reduction_method="clustering",
                trait_reduction_target="both",
                trait_clustering_threshold=-0.5,
            )

    def test_invalid_threshold_above_one_raises_error(self, tmp_path):
        """Test that threshold > 1 raises validation error.

        WHEN user sets trait_clustering_threshold to 1.5
        THEN configuration validation fails with error indicating valid range (0, 1]
        """
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        with pytest.raises(ValueError, match=r"trait_clustering_threshold.*\(0, 1\]"):
            CrossPlatformConfig(
                exp1_data_path=str(exp1_path),
                exp1_name="Exp1",
                exp1_genotype_col="genotype",
                exp2_data_path=str(exp2_path),
                exp2_name="Exp2",
                exp2_genotype_col="genotype",
                trait_reduction_method="clustering",
                trait_reduction_target="both",
                trait_clustering_threshold=1.5,
            )

    def test_threshold_of_one_is_valid(self, tmp_path):
        """Test that threshold of exactly 1.0 is valid (boundary)."""
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        config = CrossPlatformConfig(
            exp1_data_path=str(exp1_path),
            exp1_name="Exp1",
            exp1_genotype_col="genotype",
            exp2_data_path=str(exp2_path),
            exp2_name="Exp2",
            exp2_genotype_col="genotype",
            trait_reduction_method="clustering",
            trait_reduction_target="both",
            trait_clustering_threshold=1.0,  # Edge case: only perfect correlations
        )

        assert config.trait_clustering_threshold == 1.0

    def test_invalid_linkage_method_raises_error(self, tmp_path):
        """Test that invalid linkage method raises validation error.

        WHEN user sets trait_clustering_linkage to "ward" (unsupported)
        THEN configuration validation fails with error listing valid options
        """
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        with pytest.raises(
            ValueError, match=r"trait_clustering_linkage.*complete.*average.*single"
        ):
            CrossPlatformConfig(
                exp1_data_path=str(exp1_path),
                exp1_name="Exp1",
                exp1_genotype_col="genotype",
                exp2_data_path=str(exp2_path),
                exp2_name="Exp2",
                exp2_genotype_col="genotype",
                trait_reduction_method="clustering",
                trait_reduction_target="both",
                trait_clustering_linkage="ward",  # Not supported
            )


# ============================================================================
# TESTS: ReduceTraitRedundancyStep
# ============================================================================


class TestReduceTraitRedundancyStep:
    """Integration tests for ReduceTraitRedundancyStep pipeline step."""

    @pytest.fixture
    def mock_prev_result(self, synthetic_clustered_traits, tmp_path):
        """Create mock StepResult from LoadCrossPlatformDataStep."""
        from sleap_roots_analyze.pipeline.core import StepResult

        # Create exp1 and exp2 DataFrames
        np.random.seed(42)
        n_genotypes = 20

        exp1_df = pd.DataFrame(
            {
                "genotype": [f"G{i}" for i in range(n_genotypes)],
                "Trait1": np.random.randn(n_genotypes),
                "Trait2": np.random.randn(n_genotypes),
            }
        )

        # Use synthetic clustered data for exp2 (add genotype column)
        exp2_df = synthetic_clustered_traits.copy()
        exp2_df["genotype"] = exp2_df.index
        exp2_df = exp2_df.reset_index(drop=True)

        return StepResult(
            data={
                "exp1_df": exp1_df,
                "exp2_df": exp2_df,
                "common_genotypes": [f"G{i}" for i in range(n_genotypes)],
            },
            metadata={
                "exp1_name": "Exp1",
                "exp2_name": "Exp2",
                "exp1_trait_names": ["Trait1", "Trait2"],
                "exp2_trait_names": list(synthetic_clustered_traits.columns),
            },
        )

    def test_method_none_is_noop(self, mock_prev_result, tmp_path):
        """Test that method 'none' passes data through unchanged.

        WHEN trait_reduction_method is "none"
        THEN step returns data unchanged
        AND no trait_clusters.csv is generated
        """
        from sleap_roots_analyze.pipeline.steps.reduce_trait_redundancy import (
            ReduceTraitRedundancyStep,
        )
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        config = CrossPlatformConfig(
            exp1_data_path=str(exp1_path),
            exp1_name="Exp1",
            exp1_genotype_col="genotype",
            exp2_data_path=str(exp2_path),
            exp2_name="Exp2",
            exp2_genotype_col="genotype",
            trait_reduction_method="none",
        )

        step = ReduceTraitRedundancyStep()
        result = step.execute(
            data=mock_prev_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=mock_prev_result,
        )

        # Data should be unchanged
        original_traits = mock_prev_result.metadata["exp2_trait_names"]
        result_traits = result.metadata["exp2_trait_names"]
        assert result_traits == original_traits

        # Metadata should indicate no reduction
        assert result.metadata["trait_reduction_method"] == "none"
        assert result.metadata["exp2_original_traits"] == len(original_traits)
        assert result.metadata["exp2_reduced_traits"] == len(result_traits)

        # No cluster file should be generated
        assert not any("trait_clusters.csv" in str(f) for f in result.files_generated)

    def test_clustering_reduces_traits(self, mock_prev_result, tmp_path):
        """Test that clustering reduces trait count.

        WHEN trait_reduction_method is "clustering" and trait_reduction_target is "exp2"
        THEN exp2 trait count is reduced
        AND metadata includes reduction statistics
        """
        from sleap_roots_analyze.pipeline.steps.reduce_trait_redundancy import (
            ReduceTraitRedundancyStep,
        )
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        config = CrossPlatformConfig(
            exp1_data_path=str(exp1_path),
            exp1_name="Exp1",
            exp1_genotype_col="genotype",
            exp2_data_path=str(exp2_path),
            exp2_name="Exp2",
            exp2_genotype_col="genotype",
            trait_reduction_method="clustering",
            trait_reduction_target="exp2",
            trait_clustering_threshold=0.8,
        )

        step = ReduceTraitRedundancyStep()
        result = step.execute(
            data=mock_prev_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=mock_prev_result,
        )

        # Traits should be reduced for exp2
        original_count = len(mock_prev_result.metadata["exp2_trait_names"])
        reduced_count = len(result.metadata["exp2_trait_names"])
        assert reduced_count < original_count

        # Metadata should include statistics
        assert result.metadata["exp2_original_traits"] == original_count
        assert result.metadata["exp2_reduced_traits"] == reduced_count
        assert result.metadata["exp2_n_clusters"] >= 1

    def test_produces_cluster_membership_file(self, mock_prev_result, tmp_path):
        """Test that clustering produces traceable cluster membership file.

        WHEN ReduceTraitRedundancyStep executes with clustering enabled
        THEN exp2_trait_clusters.csv is generated with columns:
          - trait: Original trait name
          - cluster_id: Integer cluster assignment
          - is_representative: Boolean indicating if this trait was selected
          - variance: Trait variance used for selection
        """
        from sleap_roots_analyze.pipeline.steps.reduce_trait_redundancy import (
            ReduceTraitRedundancyStep,
        )
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        config = CrossPlatformConfig(
            exp1_data_path=str(exp1_path),
            exp1_name="Exp1",
            exp1_genotype_col="genotype",
            exp2_data_path=str(exp2_path),
            exp2_name="Exp2",
            exp2_genotype_col="genotype",
            trait_reduction_method="clustering",
            trait_reduction_target="exp2",
            trait_clustering_threshold=0.8,
        )

        step = ReduceTraitRedundancyStep()
        result = step.execute(
            data=mock_prev_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=mock_prev_result,
        )

        # Check file was generated
        assert any("exp2_trait_clusters.csv" in str(f) for f in result.files_generated)

        # Read and verify the file
        cluster_file = tmp_path / "exp2_trait_clusters.csv"
        assert cluster_file.exists()

        cluster_df = pd.read_csv(cluster_file)

        # Check required columns
        assert "trait" in cluster_df.columns
        assert "cluster_id" in cluster_df.columns
        assert "is_representative" in cluster_df.columns
        assert "variance" in cluster_df.columns

        # All original traits should be present
        original_traits = set(mock_prev_result.metadata["exp2_trait_names"])
        assert set(cluster_df["trait"]) == original_traits

        # Each cluster should have exactly one representative
        for cluster_id in cluster_df["cluster_id"].unique():
            cluster_rows = cluster_df[cluster_df["cluster_id"] == cluster_id]
            assert cluster_rows["is_representative"].sum() == 1

    def test_downstream_steps_receive_reduced_traits(self, mock_prev_result, tmp_path):
        """Test that output is correctly formatted for downstream steps.

        WHEN ReduceTraitRedundancyStep completes with clustering on exp2
        THEN exp2_trait_names in output metadata contains only representatives
        AND exp2_df contains only representative columns (plus genotype)
        """
        from sleap_roots_analyze.pipeline.steps.reduce_trait_redundancy import (
            ReduceTraitRedundancyStep,
        )
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        config = CrossPlatformConfig(
            exp1_data_path=str(exp1_path),
            exp1_name="Exp1",
            exp1_genotype_col="genotype",
            exp2_data_path=str(exp2_path),
            exp2_name="Exp2",
            exp2_genotype_col="genotype",
            trait_reduction_method="clustering",
            trait_reduction_target="exp2",
            trait_clustering_threshold=0.8,
        )

        step = ReduceTraitRedundancyStep()
        result = step.execute(
            data=mock_prev_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=mock_prev_result,
        )

        # Get reduced trait names
        reduced_traits = result.metadata["exp2_trait_names"]

        # exp2_df should only have genotype + reduced traits
        exp2_df = result.data["exp2_df"]
        expected_cols = set(reduced_traits) | {"genotype"}
        assert set(exp2_df.columns) == expected_cols

    def test_clustering_both_experiments(self, mock_prev_result, tmp_path):
        """Test that clustering works when targeting both experiments.

        WHEN trait_reduction_target is "both"
        THEN both exp1 and exp2 are clustered
        AND separate cluster membership files are generated for each
        AND dendrograms are generated for each
        """
        from sleap_roots_analyze.pipeline.steps.reduce_trait_redundancy import (
            ReduceTraitRedundancyStep,
        )
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        config = CrossPlatformConfig(
            exp1_data_path=str(exp1_path),
            exp1_name="Exp1",
            exp1_genotype_col="genotype",
            exp2_data_path=str(exp2_path),
            exp2_name="Exp2",
            exp2_genotype_col="genotype",
            trait_reduction_method="clustering",
            trait_reduction_target="both",
            trait_clustering_threshold=0.8,
        )

        step = ReduceTraitRedundancyStep()
        result = step.execute(
            data=mock_prev_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=mock_prev_result,
        )

        # Check both experiments have cluster files
        assert any("exp1_trait_clusters.csv" in str(f) for f in result.files_generated)
        assert any("exp2_trait_clusters.csv" in str(f) for f in result.files_generated)

        # Check both experiments have dendrograms
        assert any(
            "exp1_trait_clustering_dendrogram.png" in str(f)
            for f in result.files_generated
        )
        assert any(
            "exp2_trait_clustering_dendrogram.png" in str(f)
            for f in result.files_generated
        )

        # Check both experiments have heatmaps
        assert any(
            "exp1_trait_cluster_heatmap.png" in str(f) for f in result.files_generated
        )
        assert any(
            "exp2_trait_cluster_heatmap.png" in str(f) for f in result.files_generated
        )

        # Check metadata includes both experiment stats
        assert "exp1_original_traits" in result.metadata
        assert "exp1_reduced_traits" in result.metadata
        assert "exp1_n_clusters" in result.metadata
        assert "exp2_original_traits" in result.metadata
        assert "exp2_reduced_traits" in result.metadata
        assert "exp2_n_clusters" in result.metadata

    def test_clustering_only_exp1(self, mock_prev_result, tmp_path):
        """Test that clustering works when targeting only exp1.

        WHEN trait_reduction_target is "exp1"
        THEN only exp1 is clustered
        AND exp2 data is passed through unchanged
        """
        from sleap_roots_analyze.pipeline.steps.reduce_trait_redundancy import (
            ReduceTraitRedundancyStep,
        )
        from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

        exp1_path = tmp_path / "exp1.csv"
        exp2_path = tmp_path / "exp2.csv"
        exp1_path.touch()
        exp2_path.touch()

        config = CrossPlatformConfig(
            exp1_data_path=str(exp1_path),
            exp1_name="Exp1",
            exp1_genotype_col="genotype",
            exp2_data_path=str(exp2_path),
            exp2_name="Exp2",
            exp2_genotype_col="genotype",
            trait_reduction_method="clustering",
            trait_reduction_target="exp1",
            trait_clustering_threshold=0.8,
        )

        step = ReduceTraitRedundancyStep()
        result = step.execute(
            data=mock_prev_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=mock_prev_result,
        )

        # Check only exp1 has cluster files
        assert any("exp1_trait_clusters.csv" in str(f) for f in result.files_generated)
        assert not any(
            "exp2_trait_clusters.csv" in str(f) for f in result.files_generated
        )

        # exp2 traits should be unchanged
        original_exp2_traits = mock_prev_result.metadata["exp2_trait_names"]
        result_exp2_traits = result.metadata["exp2_trait_names"]
        assert result_exp2_traits == original_exp2_traits

        # exp1 traits should be reduced
        original_exp1_traits = mock_prev_result.metadata["exp1_trait_names"]
        result_exp1_traits = result.metadata["exp1_trait_names"]
        assert len(result_exp1_traits) <= len(original_exp1_traits)
