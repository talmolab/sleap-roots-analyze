"""Tests for grouped pipeline execution functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config import get_default_qc_config
from sleap_roots_analyze.pipeline.utils import run_grouped_pipelines


def test_run_grouped_pipelines_splits_by_group(tmp_path):
    """Test that run_grouped_pipelines creates separate outputs per group."""
    # Create test data with plant_age_days column
    data = {
        "Barcode": ["p1", "p2", "p3", "p4", "p5", "p6"],
        "Genotype": ["A", "B", "A", "B", "A", "B"],
        "Replicate": [1, 1, 2, 2, 3, 3],
        "plant_age_days": [7, 7, 14, 14, 21, 21],
        "trait1": [10.0, 12.0, 15.0, 18.0, 20.0, 22.0],
        "trait2": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "trait3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
    df = pd.DataFrame(data)

    # Save to CSV
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    # Create config with group_by
    config = get_default_qc_config()
    config.data.csv_path = str(csv_path)
    config.data.group_by = "plant_age_days"
    config.columns.barcode = "Barcode"
    config.columns.genotype = "Genotype"
    config.columns.replicate = "Replicate"
    config.cleanup.min_samples_per_trait = 2  # Set to match test data
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    # Run grouped pipelines (validate=False to skip strict validation)
    from sleap_roots_analyze.pipeline.pipelines.qc_pipeline import QCPipeline

    # Mock pipeline.run() to avoid full pipeline execution in unit test
    with patch.object(QCPipeline, "run", return_value={"mock_task": "mock_result"}):
        results = run_grouped_pipelines(
            config=config,
            output_dir=tmp_path / "outputs",
            pipeline_class=QCPipeline,
            validate=False,  # Skip validation for test
        )

    # Should return dict with 3 groups
    assert isinstance(results, dict)
    assert len(results) == 3
    assert 7 in results
    assert 14 in results
    assert 21 in results

    # Each group should have successful results
    for group_value, group_results in results.items():
        assert "pipeline" in group_results
        assert "output_dir" in group_results

        # Check output directory exists and has correct naming
        output_dir = Path(group_results["output_dir"])
        assert output_dir.exists()
        assert f"plant_age_days_{group_value}_" in output_dir.name


def test_run_grouped_pipelines_without_group_by_runs_once(tmp_path):
    """Test that without group_by, pipeline runs normally once."""
    # Create test data
    data = {
        "Barcode": ["p1", "p2", "p3", "p4"],
        "Genotype": ["A", "B", "A", "B"],
        "Replicate": [1, 1, 2, 2],
        "trait1": [10.0, 12.0, 15.0, 18.0],
        "trait2": [5.0, 6.0, 7.0, 8.0],
        "trait3": [1.0, 2.0, 3.0, 4.0],
    }
    df = pd.DataFrame(data)

    # Save to CSV
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    # Create config WITHOUT group_by
    config = get_default_qc_config()
    config.data.csv_path = str(csv_path)
    # No group_by set
    config.columns.barcode = "Barcode"
    config.columns.genotype = "Genotype"
    config.columns.replicate = "Replicate"
    config.cleanup.min_samples_per_trait = 2
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    from sleap_roots_analyze.pipeline.pipelines.qc_pipeline import QCPipeline

    # Mock pipeline.run()
    with patch.object(QCPipeline, "run", return_value={"mock_task": "mock_result"}):
        results = run_grouped_pipelines(
            config=config,
            output_dir=tmp_path / "outputs",
            pipeline_class=QCPipeline,
            validate=False,
        )

    # Should return dict with single None key (no grouping)
    assert isinstance(results, dict)
    assert len(results) == 1
    assert None in results

    # Check single output directory was created
    output_dir = Path(results[None]["output_dir"])
    assert output_dir.exists()
    # Should NOT have group-specific naming
    assert "plant_age_days" not in output_dir.name


def test_run_grouped_pipelines_filters_small_groups(tmp_path):
    """Test that groups with insufficient samples are skipped."""
    # Create data with one small group (only 1 sample for day_21)
    data = {
        "Barcode": ["p1", "p2", "p3", "p4", "p5"],
        "Genotype": ["A", "B", "A", "B", "A"],
        "Replicate": [1, 1, 2, 2, 3],
        "plant_age_days": [7, 7, 14, 14, 21],  # day_21 has only 1 sample
        "trait1": [10.0, 12.0, 15.0, 18.0, 20.0],
        "trait2": [5.0, 6.0, 7.0, 8.0, 9.0],
        "trait3": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    df = pd.DataFrame(data)

    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    # Config with min_samples_per_trait = 2
    config = get_default_qc_config()
    config.data.csv_path = str(csv_path)
    config.data.group_by = "plant_age_days"
    config.columns.barcode = "Barcode"
    config.columns.genotype = "Genotype"
    config.columns.replicate = "Replicate"
    config.cleanup.min_samples_per_trait = 2
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    from sleap_roots_analyze.pipeline.pipelines.qc_pipeline import QCPipeline

    # Mock pipeline.run()
    with patch.object(QCPipeline, "run", return_value={"mock_task": "mock_result"}):
        results = run_grouped_pipelines(
            config=config,
            output_dir=tmp_path / "outputs",
            pipeline_class=QCPipeline,
            validate=False,
        )

    # Should only have 2 groups (day_21 skipped)
    assert len(results) == 2
    assert 7 in results
    assert 14 in results
    assert 21 not in results  # Skipped due to insufficient samples


def test_run_grouped_pipelines_output_directory_naming(tmp_path):
    """Test that output directories are named correctly with group values."""
    data = {
        "Barcode": ["p1", "p2", "p3", "p4"],
        "Genotype": ["A", "B", "A", "B"],
        "Replicate": [1, 1, 2, 2],
        "experiment_id": ["exp1", "exp1", "exp2", "exp2"],  # String values
        "trait1": [10.0, 12.0, 15.0, 18.0],
        "trait2": [5.0, 6.0, 7.0, 8.0],
        "trait3": [1.0, 2.0, 3.0, 4.0],
    }
    df = pd.DataFrame(data)

    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    config = get_default_qc_config()
    config.data.csv_path = str(csv_path)
    config.data.group_by = "experiment_id"
    config.columns.barcode = "Barcode"
    config.columns.genotype = "Genotype"
    config.columns.replicate = "Replicate"
    config.cleanup.min_samples_per_trait = 2
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    from sleap_roots_analyze.pipeline.pipelines.qc_pipeline import QCPipeline

    # Mock pipeline.run()
    with patch.object(QCPipeline, "run", return_value={"mock_task": "mock_result"}):
        results = run_grouped_pipelines(
            config=config,
            output_dir=tmp_path / "outputs",
            pipeline_class=QCPipeline,
            validate=False,
        )

    # Check directory naming includes group column and value
    for group_value, group_results in results.items():
        output_dir = Path(group_results["output_dir"])
        assert f"experiment_id_{group_value}_" in output_dir.name

        # Should also have timestamp
        assert len(output_dir.name.split("_")) >= 3  # column_value_timestamp
