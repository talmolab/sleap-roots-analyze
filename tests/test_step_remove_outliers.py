"""Tests for RemoveOutliersStep."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    QCPipelineConfig,
)
from sleap_roots_analyze.pipeline.config import OutlierRemovalConfig
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import RemoveOutliersStep


@pytest.fixture
def config():
    """Create test config with outlier removal settings."""
    return QCPipelineConfig(
        pipeline_name="test_removal",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path="test.csv"),
        outlier_removal=OutlierRemovalConfig(
            strategy="single",
            method="mahalanobis",
            min_methods=2,
        ),
    )


@pytest.fixture
def sample_data():
    """Create sample data."""
    return pd.DataFrame(
        {
            "Barcode": [f"sample_{i}" for i in range(10)],
            "geno": [f"geno_{i%3}" for i in range(10)],
            "rep": [i % 2 + 1 for i in range(10)],
            "trait1": range(10),
            "trait2": range(10, 20),
            "trait3": range(20, 30),
        }
    )


@pytest.fixture
def outlier_results_three_methods():
    """Create outlier results from three methods."""
    return {
        "mahalanobis": {
            "outlier_indices": [0, 5, 9],  # 3 outliers
            "distances": [10.0] * 10,
        },
        "pca": {
            "outlier_indices": [0, 1, 9],  # 3 outliers, overlap with mahalanobis
            "distances": [8.0] * 10,
        },
        "isolation_forest": {
            "outlier_indices": [9],  # 1 outlier, overlap with both
            "distances": [5.0] * 10,
        },
    }


@pytest.fixture
def prev_result_with_outliers(sample_data, outlier_results_three_methods):
    """Create previous step result with outlier detection results."""
    return StepResult(
        data=sample_data,
        metadata={
            "outlier_results": outlier_results_three_methods,
            "methods_run": ["mahalanobis", "pca", "isolation_forest"],
            "trait_names": ["trait1", "trait2", "trait3"],
            "valid_trait_names": ["trait1", "trait2", "trait3"],
        },
    )


class TestRemoveOutliersStep:
    """Test suite for RemoveOutliersStep."""

    def test_single_method_removal(
        self, config, sample_data, prev_result_with_outliers, tmp_path
    ):
        """Test removing outliers from a single method."""
        config.outlier_removal.strategy = "single"
        config.outlier_removal.method = "mahalanobis"

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_with_outliers,
        )

        # Should remove samples 0, 5, 9 (flagged by mahalanobis)
        assert result.metadata["outliers_removed"] == 3
        assert result.metadata["samples_final"] == 7
        assert len(result.data) == 7

        # Check that correct samples were removed
        assert 0 not in result.data.index
        assert 5 not in result.data.index
        assert 9 not in result.data.index

    def test_consensus_removal(
        self, config, sample_data, prev_result_with_outliers, tmp_path
    ):
        """Test consensus removal (flagged by ALL methods)."""
        config.outlier_removal.strategy = "consensus"

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_with_outliers,
        )

        # Only sample 9 is flagged by ALL three methods
        assert result.metadata["outliers_removed"] == 1
        assert result.metadata["samples_final"] == 9
        assert len(result.data) == 9
        assert 9 not in result.data.index

    def test_subset_removal_min_2(
        self, config, sample_data, prev_result_with_outliers, tmp_path
    ):
        """Test subset removal (flagged by at least 2 methods)."""
        config.outlier_removal.strategy = "subset"
        config.outlier_removal.min_methods = 2

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_with_outliers,
        )

        # Samples 0 and 9 are flagged by 2+ methods
        # 0: mahalanobis + pca = 2
        # 9: mahalanobis + pca + isolation_forest = 3
        assert result.metadata["outliers_removed"] == 2
        assert result.metadata["samples_final"] == 8
        assert 0 not in result.data.index
        assert 9 not in result.data.index

    def test_subset_removal_min_3(
        self, config, sample_data, prev_result_with_outliers, tmp_path
    ):
        """Test subset removal (flagged by all 3 methods)."""
        config.outlier_removal.strategy = "subset"
        config.outlier_removal.min_methods = 3

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_with_outliers,
        )

        # Only sample 9 is flagged by 3 methods
        assert result.metadata["outliers_removed"] == 1
        assert result.metadata["samples_final"] == 9
        assert 9 not in result.data.index

    def test_no_methods_run(self, config, sample_data, tmp_path):
        """Test when no outlier detection methods were run."""
        prev_result = StepResult(
            data=sample_data,
            metadata={
                "outlier_results": {},
                "methods_run": [],  # Empty list
                "trait_names": ["trait1", "trait2", "trait3"],
                "valid_trait_names": ["trait1", "trait2", "trait3"],
            },
        )

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should return data unchanged
        assert result.metadata["outliers_removed"] == 0
        assert result.metadata["samples_before"] == 10
        assert result.metadata["samples_after"] == 10
        assert len(result.data) == 10
        pd.testing.assert_frame_equal(result.data, sample_data)

    def test_no_outliers_detected(self, config, sample_data, tmp_path):
        """Test when methods run but found no outliers."""
        prev_result = StepResult(
            data=sample_data,
            metadata={
                "outlier_results": {
                    "mahalanobis": {"outlier_indices": [], "distances": [1.0] * 10}
                },
                "methods_run": ["mahalanobis"],
                "trait_names": ["trait1", "trait2", "trait3"],
                "valid_trait_names": ["trait1", "trait2", "trait3"],
            },
        )

        config.outlier_removal.strategy = "single"
        config.outlier_removal.method = "mahalanobis"

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should not remove any samples
        assert result.metadata["outliers_removed"] == 0
        assert result.metadata["samples_final"] == 10
        assert len(result.data) == 10

    def test_invalid_method_name(
        self, config, sample_data, prev_result_with_outliers, tmp_path
    ):
        """Test error when specified method not in results."""
        config.outlier_removal.strategy = "single"
        config.outlier_removal.method = "nonexistent_method"

        step = RemoveOutliersStep()
        with pytest.raises(
            ValueError, match="Method 'nonexistent_method' not found in outlier results"
        ):
            step.execute(
                data=sample_data,
                config=config,
                run_dir=tmp_path,
                prev_result=prev_result_with_outliers,
            )

    def test_invalid_strategy(
        self, config, sample_data, prev_result_with_outliers, tmp_path
    ):
        """Test error with invalid removal strategy."""
        config.outlier_removal.strategy = "invalid_strategy"

        step = RemoveOutliersStep()
        with pytest.raises(ValueError, match="Invalid outlier removal strategy"):
            step.execute(
                data=sample_data,
                config=config,
                run_dir=tmp_path,
                prev_result=prev_result_with_outliers,
            )

    def test_output_files_generated(
        self, config, sample_data, prev_result_with_outliers, tmp_path
    ):
        """Test that all output files are created."""
        config.outlier_removal.strategy = "single"
        config.outlier_removal.method = "mahalanobis"

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_with_outliers,
        )

        # Check output files exist
        assert (tmp_path / "07_data_outliers_removed.csv").exists()
        assert (tmp_path / "07_removed_outliers_detail.csv").exists()
        assert (tmp_path / "07_outlier_removal_log.json").exists()

        # Verify file contents
        clean_data = pd.read_csv(tmp_path / "07_data_outliers_removed.csv")
        assert len(clean_data) == 7

        removed_data = pd.read_csv(tmp_path / "07_removed_outliers_detail.csv")
        assert len(removed_data) == 3
        assert "removal_reason" in removed_data.columns
        assert "mahalanobis_outlier" in removed_data.columns

    def test_removal_detail_columns(
        self, config, sample_data, prev_result_with_outliers, tmp_path
    ):
        """Test that removal details include per-method flags."""
        config.outlier_removal.strategy = "subset"
        config.outlier_removal.min_methods = 2

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_with_outliers,
        )

        removed_data = pd.read_csv(tmp_path / "07_removed_outliers_detail.csv")

        # Check that per-method columns exist
        assert "mahalanobis_outlier" in removed_data.columns
        assert "pca_outlier" in removed_data.columns
        assert "isolation_forest_outlier" in removed_data.columns

        # Sample 0: flagged by mahalanobis and pca
        sample_0_row = removed_data[removed_data["Barcode"] == "sample_0"].iloc[0]
        assert sample_0_row["mahalanobis_outlier"] == True
        assert sample_0_row["pca_outlier"] == True
        assert sample_0_row["isolation_forest_outlier"] == False

    def test_removal_log_contents(
        self, config, sample_data, prev_result_with_outliers, tmp_path
    ):
        """Test that removal log contains correct information."""
        config.outlier_removal.strategy = "subset"
        config.outlier_removal.min_methods = 2

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_with_outliers,
        )

        log = result.metadata["removal_log"]

        assert log["strategy"] == "subset"
        assert log["methods_considered"] == [
            "mahalanobis",
            "pca",
            "isolation_forest",
        ]
        assert log["original_samples"] == 10
        assert log["outliers_removed"] == 2
        assert log["final_samples"] == 8
        assert log["removal_fraction"] == 0.2
        assert "sample_0" in log["removed_sample_barcodes"]
        assert "sample_9" in log["removed_sample_barcodes"]
        assert "min_methods_required" in log

    def test_metadata_propagation(
        self, config, sample_data, prev_result_with_outliers, tmp_path
    ):
        """Test that trait metadata is propagated."""
        config.outlier_removal.strategy = "single"
        config.outlier_removal.method = "mahalanobis"

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_with_outliers,
        )

        assert "trait_names" in result.metadata
        assert "valid_trait_names" in result.metadata
        assert result.metadata["trait_names"] == ["trait1", "trait2", "trait3"]

    def test_empty_removal_file_when_no_outliers(self, config, sample_data, tmp_path):
        """Test that empty removal file is created when no outliers removed."""
        prev_result = StepResult(
            data=sample_data,
            metadata={
                "outlier_results": {
                    "mahalanobis": {"outlier_indices": [], "distances": [1.0] * 10}
                },
                "methods_run": ["mahalanobis"],
                "trait_names": ["trait1", "trait2", "trait3"],
            },
        )

        config.outlier_removal.strategy = "single"
        config.outlier_removal.method = "mahalanobis"

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Empty removal detail file should still exist
        removed_data = pd.read_csv(tmp_path / "07_removed_outliers_detail.csv")
        assert len(removed_data) == 0
        assert "Barcode" in removed_data.columns
        assert "removal_reason" in removed_data.columns
        assert "mahalanobis_outlier" in removed_data.columns

    def test_data_not_modified_inplace(
        self, config, sample_data, prev_result_with_outliers, tmp_path
    ):
        """Test that input data is not modified in place."""
        original_data = sample_data.copy()

        config.outlier_removal.strategy = "single"
        config.outlier_removal.method = "mahalanobis"

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_with_outliers,
        )

        # Original data should be unchanged
        pd.testing.assert_frame_equal(sample_data, original_data)

    def test_trait_names_fallback(self, config, sample_data, tmp_path):
        """Test fallback from valid_trait_names to trait_names."""
        prev_result = StepResult(
            data=sample_data,
            metadata={
                "outlier_results": {
                    "mahalanobis": {"outlier_indices": [0], "distances": [10.0] * 10}
                },
                "methods_run": ["mahalanobis"],
                # Only trait_names, no valid_trait_names
                "trait_names": ["trait1", "trait2", "trait3"],
            },
        )

        config.outlier_removal.strategy = "single"
        config.outlier_removal.method = "mahalanobis"

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should successfully use trait_names fallback
        assert result.metadata["trait_names"] == ["trait1", "trait2", "trait3"]

    def test_consensus_with_one_method(self, config, sample_data, tmp_path):
        """Test consensus strategy with only one method (should behave like single)."""
        prev_result = StepResult(
            data=sample_data,
            metadata={
                "outlier_results": {
                    "mahalanobis": {
                        "outlier_indices": [0, 5],
                        "distances": [10.0] * 10,
                    }
                },
                "methods_run": ["mahalanobis"],
                "trait_names": ["trait1", "trait2", "trait3"],
            },
        )

        config.outlier_removal.strategy = "consensus"

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # With only one method, consensus = all samples flagged by that method
        assert result.metadata["outliers_removed"] == 2
        assert 0 not in result.data.index
        assert 5 not in result.data.index

    def test_subset_removal_min_1(self, config, sample_data, tmp_path):
        """Test subset removal with min_methods=1 (union of all methods)."""
        outlier_results = {
            "method1": {"outlier_indices": [0, 1], "distances": [10.0] * 10},
            "method2": {"outlier_indices": [2, 3], "distances": [8.0] * 10},
        }

        prev_result = StepResult(
            data=sample_data,
            metadata={
                "outlier_results": outlier_results,
                "methods_run": ["method1", "method2"],
                "trait_names": ["trait1", "trait2", "trait3"],
            },
        )

        config.outlier_removal.strategy = "subset"
        config.outlier_removal.min_methods = 1

        step = RemoveOutliersStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should remove union of all flagged samples: 0, 1, 2, 3
        assert result.metadata["outliers_removed"] == 4
        assert result.metadata["samples_final"] == 6
        for idx in [0, 1, 2, 3]:
            assert idx not in result.data.index