"""Tests for LoadAboveGroundTraitsStep."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import QCPipelineConfig, RootCoreConfig
from sleap_roots_analyze.pipeline.config.components import MergeTraitsConfig
from sleap_roots_analyze.pipeline.steps.load_above_ground import (
    LoadAboveGroundTraitsStep,
)


@pytest.fixture
def above_ground_data():
    """Create sample above-ground trait data."""
    data = []
    for plot in [1, 2, 3]:
        for rep in [1, 2]:
            for geno in ["GH_7386", "GH_7420", "Control"]:
                data.append(
                    {
                        "Plot": plot,
                        "Rep": rep,
                        "geno": geno,
                        "Height_cm": np.random.uniform(20.0, 40.0),
                        "Biomass_g": np.random.uniform(10.0, 30.0),
                        "Yield_g": np.random.uniform(5.0, 15.0),
                    }
                )
    return pd.DataFrame(data)


@pytest.fixture
def config_with_merge(tmp_path):
    """Create config with merge_traits settings."""
    return QCPipelineConfig(
        pipeline_name="test_load_above_ground",
        root_core=RootCoreConfig(
            sources=[],
            merge_traits=MergeTraitsConfig(
                above_ground_csv=str(tmp_path / "above_ground.csv"),
                join_keys=["Plot", "Rep", "geno"],
                join_type="inner",
                duplicate_strategy="fail",
            ),
        ),
    )


def test_load_above_ground_basic(above_ground_data, config_with_merge, tmp_path):
    """Test basic loading of above-ground traits."""
    # Save data to CSV
    csv_path = tmp_path / "above_ground.csv"
    above_ground_data.to_csv(csv_path, index=False)

    step = LoadAboveGroundTraitsStep()
    result = step.execute(data=None, config=config_with_merge, run_dir=tmp_path)

    # Check result
    assert isinstance(result.data, pd.DataFrame)
    assert len(result.data) == len(above_ground_data)

    # Check columns preserved
    expected_cols = ["Plot", "Rep", "geno", "Height_cm", "Biomass_g", "Yield_g"]
    for col in expected_cols:
        assert col in result.data.columns

    # Check metadata
    assert "csv_path" in result.metadata
    assert "num_samples" in result.metadata
    assert "num_traits" in result.metadata
    assert result.metadata["num_samples"] == 18  # 3 plots × 2 reps × 3 genos


def test_load_above_ground_validates_join_keys(
    above_ground_data, config_with_merge, tmp_path
):
    """Test that join keys are validated."""
    # Remove a join key from data
    above_ground_data_missing = above_ground_data.drop(columns=["Rep"])

    # Save data to CSV
    csv_path = tmp_path / "above_ground.csv"
    above_ground_data_missing.to_csv(csv_path, index=False)

    step = LoadAboveGroundTraitsStep()

    # Should raise error about missing join key
    with pytest.raises(ValueError, match="Missing join keys"):
        step.execute(data=None, config=config_with_merge, run_dir=tmp_path)


def test_load_above_ground_file_not_found(config_with_merge, tmp_path):
    """Test error handling when CSV file doesn't exist."""
    # Config points to non-existent file
    step = LoadAboveGroundTraitsStep()

    with pytest.raises(FileNotFoundError):
        step.execute(data=None, config=config_with_merge, run_dir=tmp_path)


def test_load_above_ground_no_merge_config(tmp_path):
    """Test error when merge_traits config is missing."""
    # Config without merge_traits
    config = QCPipelineConfig(
        pipeline_name="test_no_merge", root_core=RootCoreConfig(sources=[])
    )

    step = LoadAboveGroundTraitsStep()

    with pytest.raises(ValueError, match="merge_traits configuration is required"):
        step.execute(data=None, config=config, run_dir=tmp_path)


def test_load_above_ground_preserves_dtypes(
    above_ground_data, config_with_merge, tmp_path
):
    """Test that numeric columns are preserved with correct dtypes."""
    # Ensure Plot/Rep are integers
    above_ground_data["Plot"] = above_ground_data["Plot"].astype(int)
    above_ground_data["Rep"] = above_ground_data["Rep"].astype(int)

    # Save data to CSV
    csv_path = tmp_path / "above_ground.csv"
    above_ground_data.to_csv(csv_path, index=False)

    step = LoadAboveGroundTraitsStep()
    result = step.execute(data=None, config=config_with_merge, run_dir=tmp_path)

    # Check dtypes (Plot/Rep will be float after CSV load, which is expected)
    assert pd.api.types.is_numeric_dtype(result.data["Plot"])
    assert pd.api.types.is_numeric_dtype(result.data["Rep"])
    assert pd.api.types.is_string_dtype(result.data["geno"])
    assert pd.api.types.is_float_dtype(result.data["Height_cm"])


def test_load_above_ground_with_extra_columns(
    above_ground_data, config_with_merge, tmp_path
):
    """Test loading data with extra metadata columns."""
    # Add extra columns
    above_ground_data["Date"] = "2024-07-01"
    above_ground_data["Location"] = "Field A"

    # Save data to CSV
    csv_path = tmp_path / "above_ground.csv"
    above_ground_data.to_csv(csv_path, index=False)

    step = LoadAboveGroundTraitsStep()
    result = step.execute(data=None, config=config_with_merge, run_dir=tmp_path)

    # Check extra columns are preserved
    assert "Date" in result.data.columns
    assert "Location" in result.data.columns


def test_load_above_ground_empty_file(config_with_merge, tmp_path):
    """Test error handling for empty CSV file."""
    # Create empty CSV with just headers
    csv_path = tmp_path / "above_ground.csv"
    pd.DataFrame(columns=["Plot", "Rep", "geno"]).to_csv(csv_path, index=False)

    step = LoadAboveGroundTraitsStep()

    with pytest.raises(ValueError, match="Above-ground CSV is empty"):
        step.execute(data=None, config=config_with_merge, run_dir=tmp_path)


def test_load_above_ground_duplicate_samples(config_with_merge, tmp_path):
    """Test warning for duplicate Plot-Rep-geno combinations."""
    # Create data with duplicates
    data = pd.DataFrame(
        {
            "Plot": [1, 1, 2],
            "Rep": [1, 1, 1],
            "geno": ["A", "A", "B"],
            "Height_cm": [20.0, 25.0, 30.0],
        }
    )

    # Save data to CSV
    csv_path = tmp_path / "above_ground.csv"
    data.to_csv(csv_path, index=False)

    step = LoadAboveGroundTraitsStep()

    # Should raise error about duplicates
    with pytest.raises(ValueError, match="Duplicate samples found"):
        step.execute(data=None, config=config_with_merge, run_dir=tmp_path)
