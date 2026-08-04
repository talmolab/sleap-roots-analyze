"""Tests for PCAAnalysisStep."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pca import select_top_features_from_pca
from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    PCAConfig,
    QCPipelineConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import PCAAnalysisStep


@pytest.fixture
def config():
    """Create test config with PCA settings."""
    return QCPipelineConfig(
        pipeline_name="test_pca",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path="test.csv"),
        pca=PCAConfig(
            n_components=2,
            standardize=True,
            n_top_features=5,
            feature_selection_strategy="top_absolute",
        ),
    )


@pytest.fixture
def sample_data():
    """Create sample data with traits."""
    np.random.seed(42)
    n_samples = 50
    return pd.DataFrame(
        {
            "Barcode": [f"sample_{i}" for i in range(n_samples)],
            "geno": [f"geno_{i % 5}" for i in range(n_samples)],
            "rep": [i % 3 + 1 for i in range(n_samples)],
            "trait1": np.random.randn(n_samples),
            "trait2": np.random.randn(n_samples) * 2,
            "trait3": np.random.randn(n_samples) * 0.5,
            "trait4": np.random.randn(n_samples) + 5,
            "trait5": np.random.randn(n_samples) - 2,
            "trait6": np.random.randn(n_samples) * 3,
        }
    )


@pytest.fixture
def prev_result(sample_data):
    """Create previous step result."""
    trait_cols = ["trait1", "trait2", "trait3", "trait4", "trait5", "trait6"]
    return StepResult(
        data=sample_data,
        metadata={
            "trait_cols": trait_cols,
            "metadata_cols": ["Barcode", "geno", "rep"],
        },
    )


class TestPCAAnalysisStep:
    """Test suite for PCAAnalysisStep."""

    def test_pca_basic_execution(self, config, sample_data, prev_result, tmp_path):
        """Test basic PCA step execution."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Check result structure
        assert isinstance(result, StepResult)
        assert isinstance(result.data, pd.DataFrame)
        assert result.data.equals(sample_data)

        # Check metadata
        assert "pca_results" in result.metadata
        assert "top_features" in result.metadata
        assert "n_pca_components" in result.metadata
        assert "pca_explained_variance" in result.metadata

        # Check PCA results
        assert result.metadata["n_pca_components"] == 2
        assert len(result.metadata["top_features"]) == 5

    def test_pca_output_files(self, config, sample_data, prev_result, tmp_path):
        """Test that PCA outputs are saved correctly."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        pca_dir = tmp_path / "data" / "pca"
        assert pca_dir.exists()

        # Check all expected files
        assert (pca_dir / "pc_scores.csv").exists()
        assert (pca_dir / "loadings.csv").exists()
        assert (pca_dir / "explained_variance.csv").exists()
        assert (pca_dir / "top_features.csv").exists()

        # Validate file contents
        pc_scores = pd.read_csv(pca_dir / "pc_scores.csv", index_col=0)
        assert pc_scores.shape == (50, 2)  # n_samples x n_components
        assert list(pc_scores.columns) == ["PC1", "PC2"]

        loadings = pd.read_csv(pca_dir / "loadings.csv", index_col=0)
        assert loadings.shape == (6, 2)  # n_traits x n_components

        explained_var = pd.read_csv(pca_dir / "explained_variance.csv", index_col=0)
        assert explained_var.shape == (2, 3)  # n_components x 3 metrics
        assert "explained_variance" in explained_var.columns
        assert "explained_variance_ratio" in explained_var.columns
        assert "cumulative_variance_ratio" in explained_var.columns

        top_features = pd.read_csv(pca_dir / "top_features.csv")
        assert len(top_features) == 5

    def test_pca_with_variance_threshold(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test PCA with variance threshold instead of fixed components."""
        config.pca.n_components = 0.95  # 95% variance threshold

        step = PCAAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should select components to reach 95% variance
        n_components = result.metadata["n_pca_components"]
        explained_var = result.metadata["pca_explained_variance"]

        assert n_components >= 1
        assert explained_var >= 0.95

    def test_pca_different_feature_selection_strategies(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test different feature selection strategies."""
        strategies = ["top_absolute", "extreme", "top_contribution", "top_variance"]

        for strategy in strategies:
            config.pca.feature_selection_strategy = strategy
            step = PCAAnalysisStep()

            run_dir = tmp_path / strategy
            run_dir.mkdir(parents=True, exist_ok=True)

            result = step.execute(
                data=sample_data,
                config=config,
                run_dir=run_dir,
                prev_result=prev_result,
            )

            if strategy == "extreme":
                # "extreme" ignores n_top_features entirely (issue #206): always
                # at most 1-most-positive + 1-most-negative per retained PC.
                assert (
                    len(result.metadata["top_features"]) <= 2 * config.pca.n_components
                )
            else:
                # Other strategies return at least the requested number
                # (some may return more).
                assert len(result.metadata["top_features"]) >= config.pca.n_top_features
            assert all(
                f in prev_result.metadata["trait_cols"]
                for f in result.metadata["top_features"]
            )

    def test_pca_without_standardization(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test PCA without standardization."""
        config.pca.standardize = False

        step = PCAAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should still complete successfully
        assert "pca_results" in result.metadata
        assert result.metadata["n_pca_components"] == 2

    def test_pca_metadata_propagation(self, config, sample_data, prev_result, tmp_path):
        """Test that previous metadata is propagated."""
        prev_result.metadata["custom_field"] = "custom_value"

        step = PCAAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Previous metadata should be retained
        assert "custom_field" in result.metadata
        assert result.metadata["custom_field"] == "custom_value"
        assert "trait_cols" in result.metadata

        # New metadata should be added
        assert "pca_results" in result.metadata
        assert "top_features" in result.metadata

    def test_pca_with_more_features_than_samples(self, config, prev_result, tmp_path):
        """Test PCA when requesting more features than exist."""
        config.pca.n_top_features = 100  # More than 6 traits

        # Create data with only 3 samples (less than features)
        small_data = pd.DataFrame(
            {
                "Barcode": ["s1", "s2", "s3"],
                "geno": ["g1", "g2", "g3"],
                "rep": [1, 1, 1],
                "trait1": [1, 2, 3],
                "trait2": [2, 3, 4],
                "trait3": [3, 4, 5],
                "trait4": [4, 5, 6],
                "trait5": [5, 6, 7],
                "trait6": [6, 7, 8],
            }
        )

        prev_result.data = small_data
        step = PCAAnalysisStep()

        result = step.execute(
            data=small_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should return at most the number of available features
        assert len(result.metadata["top_features"]) <= 6

    def test_pca_different_n_components(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test PCA with different numbers of components."""
        for n_comp in [1, 3, 5]:
            config.pca.n_components = n_comp
            step = PCAAnalysisStep()

            run_dir = tmp_path / f"pca_{n_comp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = step.execute(
                data=sample_data,
                config=config,
                run_dir=run_dir,
                prev_result=prev_result,
            )

            # Check correct number of components
            assert result.metadata["n_pca_components"] == n_comp

            # Check saved files have correct dimensions
            pca_dir = run_dir / "data" / "pca"
            pc_scores = pd.read_csv(pca_dir / "pc_scores.csv", index_col=0)
            assert pc_scores.shape[1] == n_comp

    def test_pca_top_features_selection(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test selecting different numbers of top features."""
        for n_features in [1, 3, 6]:
            config.pca.n_top_features = n_features
            step = PCAAnalysisStep()

            run_dir = tmp_path / f"features_{n_features}"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = step.execute(
                data=sample_data,
                config=config,
                run_dir=run_dir,
                prev_result=prev_result,
            )

            assert len(result.metadata["top_features"]) == n_features

    def test_pca_preserves_data_unchanged(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test that PCA step doesn't modify input data."""
        original_data = sample_data.copy()

        step = PCAAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Data should be unchanged
        pd.testing.assert_frame_equal(result.data, original_data)

    def test_pca_with_existing_data_pca_dir(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test that PCA step handles existing data/pca directory."""
        pca_dir = tmp_path / "data" / "pca"
        pca_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy file
        (pca_dir / "old_file.txt").write_text("old content")

        step = PCAAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should complete successfully and create new files
        assert (pca_dir / "pc_scores.csv").exists()
        assert (pca_dir / "loadings.csv").exists()


class TestPCADataOrganization:
    """Test that PCA outputs are saved to data/pca/ subdirectory (VIZ-OUTPUT-001)."""

    def test_pca_outputs_saved_to_data_pca_directory(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test that PCA step saves CSVs to data/pca/ subdirectory."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # PCA outputs should be in data/pca/, not pca/
        data_pca_dir = tmp_path / "data" / "pca"
        assert data_pca_dir.exists(), "data/pca/ directory should exist"

        # All PCA CSV files should be in data/pca/
        assert (data_pca_dir / "pc_scores.csv").exists()
        assert (data_pca_dir / "loadings.csv").exists()
        assert (data_pca_dir / "explained_variance.csv").exists()
        assert (data_pca_dir / "top_features.csv").exists()

        # Old path should NOT exist
        old_pca_dir = tmp_path / "pca"
        assert not old_pca_dir.exists(), "pca/ at root should not exist"

    def test_pca_output_files_content(self, config, sample_data, prev_result, tmp_path):
        """Test that PCA output file contents are correct in new location."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        data_pca_dir = tmp_path / "data" / "pca"

        # Validate file contents
        pc_scores = pd.read_csv(data_pca_dir / "pc_scores.csv", index_col=0)
        assert pc_scores.shape == (50, 2)  # n_samples x n_components
        assert list(pc_scores.columns) == ["PC1", "PC2"]

        loadings = pd.read_csv(data_pca_dir / "loadings.csv", index_col=0)
        assert loadings.shape == (6, 2)  # n_traits x n_components

        explained_var = pd.read_csv(
            data_pca_dir / "explained_variance.csv", index_col=0
        )
        assert explained_var.shape == (2, 3)  # n_components x 3 metrics

        top_features = pd.read_csv(data_pca_dir / "top_features.csv")
        assert len(top_features) == 5


class TestPCAZeroVarianceHandling:
    """Test PCA step handling of zero-variance (constant) traits.

    Addresses GitHub Issue #74: PCA pipeline crashes with shape mismatch
    when trait columns contain constant values.
    """

    @pytest.fixture
    def config_zv(self):
        """Config for zero-variance tests (request fewer top features)."""
        return QCPipelineConfig(
            pipeline_name="test_pca_zv",
            columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
            data=DataConfig(csv_path="test.csv"),
            pca=PCAConfig(
                n_components=2,
                standardize=True,
                n_top_features=3,
                feature_selection_strategy="top_absolute",
            ),
        )

    @pytest.fixture
    def data_with_some_zero_variance(self):
        """Dataset where 4 of 8 traits are constant (zero variance).

        Mimics the Alfalfa GWAS scenario from Issue #74: lateral root count
        at Day 0 yields identical values across all seedlings.
        """
        np.random.seed(42)
        n_samples = 50
        return pd.DataFrame(
            {
                "Barcode": [f"sample_{i}" for i in range(n_samples)],
                "geno": [f"geno_{i % 5}" for i in range(n_samples)],
                "rep": [i % 3 + 1 for i in range(n_samples)],
                # 4 variable traits
                "primary_root_length": np.random.normal(15.0, 3.0, n_samples),
                "lateral_root_density": np.random.normal(2.5, 0.8, n_samples),
                "network_area": np.random.normal(120.0, 25.0, n_samples),
                "root_depth": np.random.normal(8.0, 1.5, n_samples),
                # 4 constant traits (zero variance)
                "lateral_count_day0": np.full(n_samples, 1.0),
                "lateral_length_day0": np.full(n_samples, 0.0),
                "secondary_count_day0": np.full(n_samples, 0.0),
                "tip_angle_day0": np.full(n_samples, 45.0),
            }
        )

    @pytest.fixture
    def prev_result_zv(self, data_with_some_zero_variance):
        """Previous step result with mixed variable/constant traits."""
        trait_cols = [
            "primary_root_length",
            "lateral_root_density",
            "network_area",
            "root_depth",
            "lateral_count_day0",
            "lateral_length_day0",
            "secondary_count_day0",
            "tip_angle_day0",
        ]
        return StepResult(
            data=data_with_some_zero_variance,
            metadata={
                "trait_cols": trait_cols,
                "metadata_cols": ["Barcode", "geno", "rep"],
            },
        )

    @pytest.fixture
    def data_all_zero_variance(self):
        """Dataset where ALL traits are constant."""
        n_samples = 30
        return pd.DataFrame(
            {
                "Barcode": [f"sample_{i}" for i in range(n_samples)],
                "geno": [f"geno_{i % 3}" for i in range(n_samples)],
                "rep": [i % 3 + 1 for i in range(n_samples)],
                "trait_a": np.full(n_samples, 5.0),
                "trait_b": np.full(n_samples, 10.0),
                "trait_c": np.full(n_samples, 0.0),
            }
        )

    @pytest.fixture
    def prev_result_all_zv(self, data_all_zero_variance):
        """Previous step result where all traits are constant."""
        return StepResult(
            data=data_all_zero_variance,
            metadata={
                "trait_cols": ["trait_a", "trait_b", "trait_c"],
                "metadata_cols": ["Barcode", "geno", "rep"],
            },
        )

    @pytest.fixture
    def data_majority_zero_variance(self):
        """Dataset where >50% of traits are constant (triggers warning)."""
        np.random.seed(42)
        n_samples = 50
        return pd.DataFrame(
            {
                "Barcode": [f"sample_{i}" for i in range(n_samples)],
                "geno": [f"geno_{i % 5}" for i in range(n_samples)],
                "rep": [i % 3 + 1 for i in range(n_samples)],
                # 2 variable traits (< 50%)
                "variable_trait1": np.random.normal(10.0, 2.0, n_samples),
                "variable_trait2": np.random.normal(5.0, 1.0, n_samples),
                # 5 constant traits (> 50%)
                "const_a": np.full(n_samples, 1.0),
                "const_b": np.full(n_samples, 2.0),
                "const_c": np.full(n_samples, 3.0),
                "const_d": np.full(n_samples, 4.0),
                "const_e": np.full(n_samples, 5.0),
            }
        )

    @pytest.fixture
    def prev_result_majority_zv(self, data_majority_zero_variance):
        """Previous step result where >50% traits are constant."""
        return StepResult(
            data=data_majority_zero_variance,
            metadata={
                "trait_cols": [
                    "variable_trait1",
                    "variable_trait2",
                    "const_a",
                    "const_b",
                    "const_c",
                    "const_d",
                    "const_e",
                ],
                "metadata_cols": ["Barcode", "geno", "rep"],
            },
        )

    # --- Task 1: Shape mismatch crash tests ---

    def test_pca_with_zero_variance_traits(
        self, config_zv, data_with_some_zero_variance, prev_result_zv, tmp_path
    ):
        """PCA step completes when some traits have zero variance.

        Regression test for Issue #74: shape mismatch between loadings
        matrix and trait_cols when zero-variance traits are filtered.
        """
        step = PCAAnalysisStep()

        # Should NOT raise ValueError about shape mismatch
        result = step.execute(
            data=data_with_some_zero_variance,
            config=config_zv,
            run_dir=tmp_path,
            prev_result=prev_result_zv,
        )

        assert isinstance(result, StepResult)
        assert "pca_results" in result.metadata

        # Loadings should have rows matching the 4 variable traits, not 8 total
        pca_dir = tmp_path / "data" / "pca"
        loadings = pd.read_csv(pca_dir / "loadings.csv", index_col=0)
        assert (
            loadings.shape[0] == 4
        ), f"Expected 4 rows (variable traits only), got {loadings.shape[0]}"

    def test_pca_with_all_zero_variance_traits(
        self, config_zv, data_all_zero_variance, prev_result_all_zv, tmp_path
    ):
        """PCA step raises ValueError when ALL traits are constant."""
        step = PCAAnalysisStep()

        with pytest.raises(ValueError, match="zero variance"):
            step.execute(
                data=data_all_zero_variance,
                config=config_zv,
                run_dir=tmp_path,
                prev_result=prev_result_all_zv,
            )

    # --- Task 2: Metadata and logging tests ---

    def test_pca_zero_variance_metadata_tracking(
        self, config_zv, data_with_some_zero_variance, prev_result_zv, tmp_path
    ):
        """Metadata tracks which traits were excluded due to zero variance."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=data_with_some_zero_variance,
            config=config_zv,
            run_dir=tmp_path,
            prev_result=prev_result_zv,
        )

        # New metadata keys must be present
        assert "excluded_zero_variance_traits" in result.metadata
        assert "n_traits_after_filtering" in result.metadata

        excluded = result.metadata["excluded_zero_variance_traits"]
        n_after = result.metadata["n_traits_after_filtering"]

        # Exactly 4 constant traits should be excluded
        assert len(excluded) == 4
        assert set(excluded) == {
            "lateral_count_day0",
            "lateral_length_day0",
            "secondary_count_day0",
            "tip_angle_day0",
        }
        # 4 variable traits should remain
        assert n_after == 4

    def test_pca_zero_variance_warning_threshold(
        self, config_zv, data_majority_zero_variance, prev_result_majority_zv, tmp_path
    ):
        """UserWarning emitted when >50% of traits are zero-variance."""
        step = PCAAnalysisStep()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            step.execute(
                data=data_majority_zero_variance,
                config=config_zv,
                run_dir=tmp_path,
                prev_result=prev_result_majority_zv,
            )

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1, "Expected UserWarning for >50% zero-variance"
        assert "zero variance" in str(user_warnings[0].message).lower()

    def test_pca_no_warning_below_threshold(
        self, config_zv, data_with_some_zero_variance, prev_result_zv, tmp_path
    ):
        """No UserWarning when <=50% of traits are zero-variance."""
        step = PCAAnalysisStep()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            step.execute(
                data=data_with_some_zero_variance,
                config=config_zv,
                run_dir=tmp_path,
                prev_result=prev_result_zv,
            )

        # 4 of 8 traits = 50%, which is NOT >50%, so no warning
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        zero_var_warnings = [
            x for x in user_warnings if "zero variance" in str(x.message).lower()
        ]
        assert (
            len(zero_var_warnings) == 0
        ), "Should not warn when <=50% traits are zero-variance"

    # --- Task 3: Feature selection correctness tests ---

    def test_pca_top_features_use_filtered_names(
        self, config_zv, data_with_some_zero_variance, prev_result_zv, tmp_path
    ):
        """Top features should be actual feature names used in PCA, not original indices."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=data_with_some_zero_variance,
            config=config_zv,
            run_dir=tmp_path,
            prev_result=prev_result_zv,
        )

        top_features = result.metadata["top_features"]
        variable_traits = {
            "primary_root_length",
            "lateral_root_density",
            "network_area",
            "root_depth",
        }
        constant_traits = {
            "lateral_count_day0",
            "lateral_length_day0",
            "secondary_count_day0",
            "tip_angle_day0",
        }

        # All top features must be from variable traits (actually used in PCA)
        for feat in top_features:
            assert feat in variable_traits, (
                f"Top feature '{feat}' is not a variable trait — "
                f"may be indexing into original trait_cols instead of filtered names"
            )
        # No constant trait should appear in top features
        for feat in top_features:
            assert feat not in constant_traits

    def test_pca_loadings_csv_index_matches_features(
        self, config_zv, data_with_some_zero_variance, prev_result_zv, tmp_path
    ):
        """Saved loadings.csv index must match the features actually used in PCA."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=data_with_some_zero_variance,
            config=config_zv,
            run_dir=tmp_path,
            prev_result=prev_result_zv,
        )

        pca_dir = tmp_path / "data" / "pca"
        loadings = pd.read_csv(pca_dir / "loadings.csv", index_col=0)

        variable_traits = {
            "primary_root_length",
            "lateral_root_density",
            "network_area",
            "root_depth",
        }

        # Index should be exactly the 4 variable trait names
        assert set(loadings.index) == variable_traits, (
            f"Loadings index {set(loadings.index)} does not match "
            f"expected variable traits {variable_traits}"
        )


class TestPCATraitNamesMetadataPropagation:
    """Regression tests for Issue #80.

    `PCAAnalysisStep` must update `metadata["trait_names"]`/
    `valid_trait_names` to the post-filter feature set (matching the
    "traits still in play" contract every other filtering step maintains),
    and preserve the pre-filter list under `original_trait_names`.
    """

    @pytest.fixture
    def config_interleaved(self):
        """Config for the interleaved zero-variance fixture (2 real features)."""
        return QCPipelineConfig(
            pipeline_name="test_pca_trait_names_interleaved",
            data=DataConfig(csv_path="test.csv"),
            pca=PCAConfig(
                n_components=1,
                standardize=True,
                n_top_features=2,
                feature_selection_strategy="top_absolute",
            ),
        )

    @pytest.fixture
    def prev_result_interleaved(self, pca_constant_feature_data):
        """Previous step result using the interleaved constant/variable fixture.

        `pca_constant_feature_data` (tests/fixtures.py) has columns
        [constant1, constant2, variable1, variable2, constant3] — the
        excluded traits are NOT confined to the end of the list, so a
        buggy prefix-slice (`trait_names[:n_features]`) would wrongly
        yield [constant1, constant2] instead of the correct
        [variable1, variable2].
        """
        trait_cols = list(pca_constant_feature_data.columns)
        return StepResult(
            data=pca_constant_feature_data,
            metadata={"trait_cols": trait_cols},
        )

    def test_trait_names_reflects_post_filter_feature_set(
        self,
        config_interleaved,
        pca_constant_feature_data,
        prev_result_interleaved,
        tmp_path,
    ):
        """metadata['trait_names']/'valid_trait_names' equal the filtered feature_names."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=pca_constant_feature_data,
            config=config_interleaved,
            run_dir=tmp_path,
            prev_result=prev_result_interleaved,
        )

        feature_names = result.metadata["pca_results"]["feature_names"]
        assert feature_names == ["variable1", "variable2"]

        assert result.metadata["trait_names"] == feature_names
        assert result.metadata["valid_trait_names"] == feature_names

        # Guard against the specific bug this regression targets: a naive
        # prefix slice of the original (unfiltered) list would silently
        # substitute the wrong names instead of raising an error.
        original_trait_cols = list(pca_constant_feature_data.columns)
        naive_prefix_slice = original_trait_cols[: len(feature_names)]
        assert result.metadata["trait_names"] != naive_prefix_slice

    def test_original_trait_names_preserves_pre_filter_list(
        self,
        config_interleaved,
        pca_constant_feature_data,
        prev_result_interleaved,
        tmp_path,
    ):
        """metadata['original_trait_names'] preserves the full pre-filter list, in order."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=pca_constant_feature_data,
            config=config_interleaved,
            run_dir=tmp_path,
            prev_result=prev_result_interleaved,
        )

        assert result.metadata["original_trait_names"] == list(
            pca_constant_feature_data.columns
        )

    def test_trait_names_unchanged_when_nothing_excluded(
        self, config, sample_data, tmp_path
    ):
        """Verify trait_names/valid_trait_names/original_trait_names are unchanged.

        When no trait is zero-variance, all three equal the input trait list.

        Seeds `prev_result.metadata` with `trait_names`/`valid_trait_names`
        already set (as the real upstream `StatisticalAnalysisStep` does),
        rather than only `trait_cols` — this is a pure passthrough guard, not
        a red test, so it must reflect the metadata shape PCAAnalysisStep
        actually receives in production.
        """
        trait_cols = ["trait1", "trait2", "trait3", "trait4", "trait5", "trait6"]
        prev_result_with_trait_names = StepResult(
            data=sample_data,
            metadata={
                "trait_names": trait_cols,
                "valid_trait_names": trait_cols,
                "metadata_cols": ["Barcode", "geno", "rep"],
            },
        )
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_with_trait_names,
        )

        assert result.metadata["trait_names"] == trait_cols
        assert result.metadata["valid_trait_names"] == trait_cols
        assert result.metadata["original_trait_names"] == trait_cols


class TestPCAAnalysisStepFeatureSelectionResolution:
    """Regression tests for issues #203 and #206.

    `PCAAnalysisStep` must always pass `pc_indices` explicitly (scoped to
    every retained PC, never relying on `select_top_features_from_pca()`'s
    `[0, 1]` default), must ignore `n_top_features` entirely for
    `"extreme"`, and must resolve a `< 1` `n_top_features` under
    `"top_variance"` via `select_n_features_by_variance()`.
    """

    @pytest.fixture
    def config_3pc(self):
        """Config selecting exactly 3 PCs (beyond the buggy [0, 1] default)."""
        return QCPipelineConfig(
            pipeline_name="test_pca_3pc",
            columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
            data=DataConfig(csv_path="test.csv"),
            pca=PCAConfig(
                n_components=3,
                standardize=True,
                n_top_features=2,
                feature_selection_strategy="extreme",
            ),
        )

    @pytest.mark.parametrize(
        "strategy", ["extreme", "top_absolute", "top_contribution"]
    )
    def test_pc_indices_always_passed_scoped_to_n_components(
        self, config_3pc, sample_data, prev_result, tmp_path, strategy
    ):
        """pc_indices is always explicit and scoped to every retained PC (#203)."""
        config_3pc.pca.feature_selection_strategy = strategy
        step = PCAAnalysisStep()

        with mock.patch(
            "sleap_roots_analyze.pipeline.steps.pca_analysis.select_top_features_from_pca",
            wraps=select_top_features_from_pca,
        ) as spy:
            step.execute(
                data=sample_data,
                config=config_3pc,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        spy.assert_called_once()
        assert spy.call_args.kwargs["pc_indices"] == [0, 1, 2]

    def test_extreme_uses_n_features_to_select_1_ignoring_n_top_features(
        self, config, sample_data, prev_result, tmp_path
    ):
        """The "extreme" strategy always requests 1-per-direction, regardless of n_top_features."""
        config.pca.feature_selection_strategy = "extreme"
        config.pca.n_top_features = 999  # Arbitrary stale value; must be ignored.
        step = PCAAnalysisStep()

        with mock.patch(
            "sleap_roots_analyze.pipeline.steps.pca_analysis.select_top_features_from_pca",
            wraps=select_top_features_from_pca,
        ) as spy:
            step.execute(
                data=sample_data,
                config=config,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        assert spy.call_args.kwargs["n_features_to_select"] == 1

    def test_extreme_single_retained_pc(
        self, config, sample_data, prev_result, tmp_path
    ):
        """A single retained PC under "extreme" yields at most 2 features."""
        config.pca.n_components = 1
        config.pca.feature_selection_strategy = "extreme"
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        assert len(result.metadata["top_features"]) <= 2

    def test_extreme_logs_info_that_n_top_features_is_ignored(
        self, config, sample_data, prev_result, tmp_path, caplog
    ):
        """A logger.info notice fires for "extreme", not for other strategies."""
        config.pca.feature_selection_strategy = "extreme"
        step = PCAAnalysisStep()

        with caplog.at_level(logging.INFO):
            step.execute(
                data=sample_data,
                config=config,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        assert any(
            "n_top_features" in record.message and "extreme" in record.message
            for record in caplog.records
        )

    def test_no_ignored_notice_for_non_extreme_strategies(
        self, config, sample_data, prev_result, tmp_path, caplog
    ):
        """No "ignored" notice fires when n_top_features is actually read."""
        config.pca.feature_selection_strategy = "top_absolute"
        step = PCAAnalysisStep()

        with caplog.at_level(logging.INFO):
            step.execute(
                data=sample_data,
                config=config,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        assert not any("n_top_features" in record.message for record in caplog.records)

    def test_top_variance_threshold_resolves_feature_count(
        self, config, sample_data, prev_result, tmp_path
    ):
        """A < 1 threshold selects a count meeting but not overshooting it."""
        config.pca.feature_selection_strategy = "top_variance"
        config.pca.n_top_features = 0.8
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        n_selected = len(result.metadata["top_features"])
        feature_contributions = result.metadata["pca_results"]["feature_contributions"]
        cumulative = feature_contributions["fractional_contribution"].cumsum()

        assert cumulative.iloc[n_selected - 1] >= 0.8
        assert n_selected == 1 or cumulative.iloc[n_selected - 2] < 0.8

    @pytest.mark.parametrize("threshold", [0, -0.5])
    def test_top_variance_non_positive_threshold_selects_one_feature(
        self, config, sample_data, prev_result, tmp_path, threshold
    ):
        """A threshold <= 0 selects exactly 1 feature, no exception."""
        config.pca.feature_selection_strategy = "top_variance"
        config.pca.n_top_features = threshold
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        assert len(result.metadata["top_features"]) == 1

    def test_top_variance_1_0_is_a_count_not_a_100_percent_threshold(
        self, config, sample_data, prev_result, tmp_path
    ):
        """n_top_features == 1.0 selects exactly 1 feature (count branch)."""
        config.pca.feature_selection_strategy = "top_variance"
        config.pca.n_top_features = 1.0
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        assert len(result.metadata["top_features"]) == 1

    def test_top_variance_count_unchanged_for_n_top_features_ge_1(
        self, config, sample_data, prev_result, tmp_path
    ):
        """n_top_features >= 1 preserves the existing fixed-count behavior."""
        config.pca.feature_selection_strategy = "top_variance"
        config.pca.n_top_features = 5
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        assert len(result.metadata["top_features"]) == 5

    def test_count_selection_rounds_rather_than_truncates_near_whole_numbers(
        self, config, sample_data, prev_result, tmp_path
    ):
        """A value certified as "whole" must resolve to that same integer here.

        validate_*_config() only guarantees "whole" within tolerance; this
        step must not silently truncate down to a different neighbor.
        """
        config.pca.feature_selection_strategy = "top_absolute"
        config.pca.n_top_features = 5.0 - 1e-15  # int() truncates to 4; round() gives 5
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        assert len(result.metadata["top_features"]) == 5
