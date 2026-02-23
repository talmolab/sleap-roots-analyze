"""Tests for group_by column validation in config validation."""

from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config import QCPipelineConfig
from sleap_roots_analyze.pipeline.config.utils import validate_qc_config


def test_validate_qc_config_group_by_column_missing(tmp_path):
    """Validation fails when group_by column not in CSV."""
    # Create CSV without 'age' column
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("barcode,genotype,replicate,trait1\np1,A,1,1.0\n")

    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = str(csv_path)
    config.data.group_by = "age"  # Column doesn't exist
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    with pytest.raises(ValueError) as exc_info:
        validate_qc_config(config, check_files=True)

    assert "group_by column 'age' not found" in str(exc_info.value)
    assert "Available columns:" in str(exc_info.value)


def test_validate_qc_config_group_by_validation_skipped_when_check_files_false(
    tmp_path,
):
    """Validation skipped when check_files=False."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("barcode,genotype,replicate,trait1\np1,A,1,1.0\n")

    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = str(csv_path)
    config.data.group_by = "nonexistent"  # Would fail if validated
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    # Should not raise when check_files=False
    validate_qc_config(config, check_files=False)


def test_validate_qc_config_group_by_validation_skipped_when_csv_missing(tmp_path):
    """Validation skipped when CSV file doesn't exist yet."""
    csv_path = tmp_path / "nonexistent.csv"  # Doesn't exist

    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = str(csv_path)
    config.data.group_by = "age"
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    # Should not raise even with check_files=True (CSV doesn't exist)
    # This allows creating configs before data is available
    validate_qc_config(config, check_files=True)


def test_validate_qc_config_group_by_null_skips_validation(tmp_path):
    """No validation when group_by is null."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("barcode,genotype,replicate,trait1\np1,A,1,1.0\n")

    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = str(csv_path)
    config.data.group_by = None  # No grouping
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    # Should not attempt validation when group_by is None
    validate_qc_config(config, check_files=True)


def test_validate_qc_config_group_by_exists_passes(tmp_path):
    """Validation passes when group_by column exists in CSV."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("barcode,genotype,replicate,age,trait1\np1,A,1,7,1.0\n")

    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = str(csv_path)
    config.data.group_by = "age"  # Column exists
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    # Should not raise
    validate_qc_config(config, check_files=True)


def test_validate_qc_config_group_by_with_empty_csv_skips(tmp_path):
    """Validation skipped for empty CSV (will fail at runtime anyway)."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")  # Empty file

    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = str(csv_path)
    config.data.group_by = "age"
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    # Should not raise - empty CSV will fail at runtime
    validate_qc_config(config, check_files=True)
