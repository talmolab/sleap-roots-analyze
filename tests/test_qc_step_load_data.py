"""Tests for LoadDataStep."""

from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import ColumnConfig, DataConfig, PipelineConfig
from sleap_roots_analyze.pipeline.steps import LoadDataStep


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    data = {
        "Barcode": ["plant1", "plant2", "plant3"],
        "geno": ["A", "B", "A"],
        "rep": [1, 1, 2],
        "trait1": [10.5, 12.3, 11.8],
        "trait2": [5.2, 6.1, 5.8],
        "QC_flag": ["pass", "pass", "fail"],  # Should be excluded (metadata)
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def config(sample_csv):
    """Create a test configuration."""
    return PipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path=str(sample_csv)),
    )


def test_load_data_step_initialization():
    """Test LoadDataStep initialization."""
    step = LoadDataStep()
    assert step.step_name == "LoadData"
    assert step.description == "Load and validate trait data from CSV"


def test_load_data_step_execute(config, tmp_path):
    """Test LoadDataStep execution."""
    step = LoadDataStep()
    result = step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    # Check data
    assert isinstance(result.data, pd.DataFrame)
    assert len(result.data) == 3
    assert "Barcode" in result.data.columns
    assert "trait1" in result.data.columns

    # Check metadata
    assert result.metadata["samples"] == 3
    assert result.metadata["trait_columns"] == 2  # trait1, trait2
    assert result.metadata["metadata_columns"] == 4  # Barcode, geno, rep, QC_flag
    assert "trait1" in result.metadata["trait_column_names"]
    assert "trait2" in result.metadata["trait_column_names"]
    assert "Barcode" in result.metadata["metadata_column_names"]
    assert "QC_flag" in result.metadata["metadata_column_names"]

    # Check files generated
    assert len(result.files_generated) == 2
    assert (tmp_path / "00_data_loaded.csv").exists()
    assert (tmp_path / "00_data_inspection.csv").exists()


def test_load_data_step_inspection_file(config, tmp_path):
    """Test that inspection file contains correct information."""
    step = LoadDataStep()
    step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    inspection = pd.read_csv(tmp_path / "00_data_inspection.csv")
    assert len(inspection) == 6  # All columns
    assert "column_name" in inspection.columns
    assert "column_type" in inspection.columns
    assert "dtype" in inspection.columns

    # Check trait vs metadata classification
    trait_rows = inspection[inspection["column_type"] == "trait"]
    assert len(trait_rows) == 2  # trait1, trait2
    assert set(trait_rows["column_name"]) == {"trait1", "trait2"}

    metadata_rows = inspection[inspection["column_type"] == "metadata"]
    assert len(metadata_rows) == 4  # Barcode, geno, rep, QC_flag


def test_load_data_step_with_additional_exclude(sample_csv, tmp_path):
    """Test LoadDataStep with additional_exclude_cols."""
    config = PipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path=str(sample_csv), additional_exclude_cols=["trait2"]),
    )

    step = LoadDataStep()
    result = step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    # trait2 should now be excluded
    assert result.metadata["trait_columns"] == 1  # Only trait1
    assert "trait1" in result.metadata["trait_column_names"]
    assert "trait2" not in result.metadata["trait_column_names"]


def test_load_data_step_missing_file(tmp_path):
    """Test LoadDataStep with missing CSV file."""
    config = PipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(),
        data=DataConfig(csv_path="nonexistent.csv"),
    )

    step = LoadDataStep()
    with pytest.raises(FileNotFoundError):
        step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)


def test_load_data_step_missing_required_columns(tmp_path):
    """Test LoadDataStep with missing required columns."""
    # Create CSV without required columns
    data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    df = pd.DataFrame(data)
    csv_path = tmp_path / "bad_data.csv"
    df.to_csv(csv_path, index=False)

    config = PipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path=str(csv_path)),
    )

    step = LoadDataStep()
    with pytest.raises(ValueError, match="Missing required columns"):
        step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)
