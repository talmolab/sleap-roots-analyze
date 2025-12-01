"""Tests for LoadRootCoreDataStep."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import (
    CoreQCConfig,
    QCPipelineConfig,
    RootCoreConfig,
    RootCoreSourceConfig,
)
from sleap_roots_analyze.pipeline.steps.load_root_core_data import LoadRootCoreDataStep


@pytest.fixture
def biomass_csv(tmp_path):
    """Create a temporary biomass CSV file."""
    data = {
        "Plot": [1, 1, 1, 2, 2, 2],
        "Rep": [1, 1, 1, 1, 1, 1],
        "geno": ["GH_7386", "GH_7386", "GH_7386", "GH_7420", "GH_7420", "GH_7420"],
        "Core_Replicate": [1, 2, 3, 1, 2, 3],
        "0-30": [1.2, 1.5, 1.1, 2.1, 2.3, 2.0],
        "30-60": [0.8, 0.9, 0.7, 1.5, 1.6, 1.4],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "biomass.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def counting_csv(tmp_path):
    """Create a temporary counting CSV file."""
    np.random.seed(42)
    data = {
        "Plot": [1, 1, 1, 2, 2, 2],
        "Rep": [1, 1, 1, 1, 1, 1],
        "geno": ["GH_7386", "GH_7386", "GH_7386", "GH_7420", "GH_7420", "GH_7420"],
        "core_n": [1, 2, 3, 1, 2, 3],
        "c_0_10_1": np.random.randint(10, 50, 6).tolist(),
        "c_0_10_2": np.random.randint(10, 50, 6).tolist(),
        "c_10_20_1": np.random.randint(5, 30, 6).tolist(),
        "c_10_20_2": np.random.randint(5, 30, 6).tolist(),
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "counting.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def config_biomass(biomass_csv):
    """Create configuration for biomass data only."""
    return QCPipelineConfig(
        pipeline_name="test_load_biomass",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path=str(biomass_csv),
                    data_type="biomass",
                    depth_column_prefix="RootDW",
                    value_column_name="Root_DW_g",
                    depth_mapping={"0-30": 15.0, "30-60": 45.0},
                )
            ],
            core_qc=CoreQCConfig(enabled=False),
        ),
    )


@pytest.fixture
def config_both(biomass_csv, counting_csv):
    """Create configuration for both biomass and counting data."""
    return QCPipelineConfig(
        pipeline_name="test_load_both",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path=str(biomass_csv),
                    data_type="biomass",
                    depth_column_prefix="RootDW",
                    value_column_name="Root_DW_g",
                    depth_mapping={"0-30": 15.0, "30-60": 45.0},
                ),
                RootCoreSourceConfig(
                    csv_path=str(counting_csv),
                    data_type="counting",
                    depth_column_prefix="RootCount",
                    value_column_name="Root_Count",
                ),
            ],
            core_qc=CoreQCConfig(enabled=False),
        ),
    )


def test_load_biomass_data(config_biomass, tmp_path):
    """Test loading biomass data with required columns."""
    step = LoadRootCoreDataStep()
    result = step.execute(data=None, config=config_biomass, run_dir=tmp_path)

    # Check result structure
    assert "biomass" in result.data
    assert isinstance(result.data["biomass"], pd.DataFrame)

    # Check data shape
    df = result.data["biomass"]
    assert len(df) == 6  # 2 plots × 3 cores
    assert "sample_id" in df.columns

    # Check sample identifiers created correctly
    expected_ids = [
        "plot1_rep1_GH_7386_core1",
        "plot1_rep1_GH_7386_core2",
        "plot1_rep1_GH_7386_core3",
        "plot2_rep1_GH_7420_core1",
        "plot2_rep1_GH_7420_core2",
        "plot2_rep1_GH_7420_core3",
    ]
    assert df["sample_id"].tolist() == expected_ids

    # Check metadata
    assert result.metadata["total_samples"] == 6
    assert len(result.metadata["sources"]) == 1
    assert result.metadata["sources"][0]["data_type"] == "biomass"

    # Check files generated
    assert len(result.files_generated) == 2  # CSV + JSON
    assert (tmp_path / "00a_root_core_biomass_loaded.csv").exists()
    assert (tmp_path / "00a_root_core_metadata.json").exists()


def test_load_both_sources(config_both, tmp_path):
    """Test loading both biomass and counting data."""
    step = LoadRootCoreDataStep()
    result = step.execute(data=None, config=config_both, run_dir=tmp_path)

    # Check both data types loaded
    assert "biomass" in result.data
    assert "counting" in result.data

    # Check data shapes
    assert len(result.data["biomass"]) == 6
    assert len(result.data["counting"]) == 6

    # Check metadata
    assert result.metadata["total_samples"] == 12  # 6 + 6
    assert len(result.metadata["sources"]) == 2

    # Check files generated
    assert len(result.files_generated) == 3  # 2 CSVs + 1 JSON
    assert (tmp_path / "00a_root_core_biomass_loaded.csv").exists()
    assert (tmp_path / "00a_root_core_counting_loaded.csv").exists()


def test_missing_required_columns(config_biomass, tmp_path):
    """Test error when required columns are missing."""
    # Create CSV missing 'Rep' column
    bad_csv = tmp_path / "bad_biomass.csv"
    data = {
        "Plot": [1, 2],
        "geno": ["A", "B"],
        "Core_Replicate": [1, 1],
        "0-30": [1.0, 2.0],
    }
    pd.DataFrame(data).to_csv(bad_csv, index=False)

    # Update config to use bad CSV
    config_biomass.root_core.sources[0].csv_path = str(bad_csv)

    step = LoadRootCoreDataStep()
    with pytest.raises(ValueError, match="Missing required columns"):
        step.execute(data=None, config=config_biomass, run_dir=tmp_path)


def test_file_not_found(config_biomass, tmp_path):
    """Test error when CSV file doesn't exist."""
    config_biomass.root_core.sources[0].csv_path = "nonexistent.csv"

    step = LoadRootCoreDataStep()
    with pytest.raises(FileNotFoundError):
        step.execute(data=None, config=config_biomass, run_dir=tmp_path)


def test_invalid_data_type(biomass_csv, tmp_path):
    """Test error with invalid data_type."""
    config = QCPipelineConfig(
        pipeline_name="test_invalid",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path=str(biomass_csv),
                    data_type="invalid_type",
                    depth_column_prefix="Test",
                    value_column_name="Value",
                )
            ]
        ),
    )

    step = LoadRootCoreDataStep()
    with pytest.raises(ValueError, match="Invalid data_type"):
        step.execute(data=None, config=config, run_dir=tmp_path)


def test_no_sources_configured(tmp_path):
    """Test error when no sources are configured."""
    config = QCPipelineConfig(
        pipeline_name="test_no_sources", root_core=RootCoreConfig(sources=[])
    )

    step = LoadRootCoreDataStep()
    with pytest.raises(ValueError, match="No root core sources configured"):
        step.execute(data=None, config=config, run_dir=tmp_path)
