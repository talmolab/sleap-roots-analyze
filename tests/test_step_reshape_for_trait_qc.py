"""Tests for ReshapeForTraitQCStep."""

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
from sleap_roots_analyze.pipeline.steps.reshape_for_trait_qc import (
    ReshapeForTraitQCStep,
)


@pytest.fixture
def biomass_aggregated_data():
    """Create aggregated long-format biomass data."""
    data = []
    for plot in [1, 2]:
        for rep in [1, 2]:
            for geno in ["GH_7386", "GH_7420"]:
                for depth in [15.0, 45.0]:
                    data.append(
                        {
                            "Plot": plot,
                            "Rep": rep,
                            "geno": geno,
                            "Depth_cm": depth,
                            "Root_DW_g": np.random.uniform(1.0, 3.0),
                        }
                    )
    return pd.DataFrame(data)


@pytest.fixture
def counting_aggregated_data():
    """Create aggregated long-format counting data."""
    np.random.seed(42)
    data = []
    for plot in [1, 2]:
        for rep in [1, 2]:
            for geno in ["GH_7386", "GH_7420"]:
                for depth in [0.0, 5.0, 10.0, 15.0]:
                    data.append(
                        {
                            "Plot": plot,
                            "Rep": rep,
                            "geno": geno,
                            "Depth_cm": depth,
                            "Root_Count": np.random.randint(10, 50),
                        }
                    )
    return pd.DataFrame(data)


@pytest.fixture
def config_biomass():
    """Create configuration for biomass reshape."""
    return QCPipelineConfig(
        pipeline_name="test_reshape_biomass",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path="dummy.csv",
                    data_type="biomass",
                    depth_column_prefix="RootDW",
                    value_column_name="Root_DW_g",
                )
            ],
            core_qc=CoreQCConfig(enabled=False),
        ),
    )


@pytest.fixture
def config_counting():
    """Create configuration for counting reshape."""
    return QCPipelineConfig(
        pipeline_name="test_reshape_counting",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path="dummy.csv",
                    data_type="counting",
                    depth_column_prefix="RootCount",
                    value_column_name="Root_Count",
                )
            ],
            core_qc=CoreQCConfig(enabled=False),
        ),
    )


def test_reshape_biomass_with_prefix(biomass_aggregated_data, config_biomass, tmp_path):
    """Test reshaping biomass data with RootDW prefix."""
    input_data = {"biomass": biomass_aggregated_data}

    step = ReshapeForTraitQCStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    # Check result is a DataFrame (merged output)
    assert isinstance(result.data, pd.DataFrame)
    df_wide = result.data

    # Check shape: one row per Plot-Rep-geno combination
    num_unique_samples = biomass_aggregated_data.groupby(
        ["Plot", "Rep", "geno"]
    ).ngroups
    assert len(df_wide) == num_unique_samples

    # Check column names have prefix
    assert "RootDW_15cm" in df_wide.columns
    assert "RootDW_45cm" in df_wide.columns

    # Check index columns preserved
    assert "Plot" in df_wide.columns
    assert "Rep" in df_wide.columns
    assert "geno" in df_wide.columns

    # Check Depth_cm and value columns are gone
    assert "Depth_cm" not in df_wide.columns
    assert "Root_DW_g" not in df_wide.columns

    # Check files generated
    assert (tmp_path / "00e_root_core_biomass_wide.csv").exists()
    assert (tmp_path / "00e_root_core_merged.csv").exists()
    assert (tmp_path / "00e_reshape_metadata.json").exists()

    # Check metadata
    source_meta = result.metadata["sources"][0]
    assert source_meta["prefix"] == "RootDW"
    assert source_meta["num_depths"] == 2
    assert set(source_meta["depth_columns"]) == {"RootDW_15cm", "RootDW_45cm"}


def test_reshape_counting_with_prefix(
    counting_aggregated_data, config_counting, tmp_path
):
    """Test reshaping counting data with RootCount prefix."""
    input_data = {"counting": counting_aggregated_data}

    step = ReshapeForTraitQCStep()
    result = step.execute(data=input_data, config=config_counting, run_dir=tmp_path)

    # Check result is a DataFrame (merged output)
    assert isinstance(result.data, pd.DataFrame)
    df_wide = result.data

    # Check column names have prefix
    expected_cols = [
        "RootCount_0cm",
        "RootCount_5cm",
        "RootCount_10cm",
        "RootCount_15cm",
    ]
    for col in expected_cols:
        assert col in df_wide.columns

    # Check metadata
    source_meta = result.metadata["sources"][0]
    assert source_meta["prefix"] == "RootCount"
    assert source_meta["num_depths"] == 4
    assert set(source_meta["depth_columns"]) == set(expected_cols)


def test_reshape_both_sources(
    biomass_aggregated_data, counting_aggregated_data, config_biomass, tmp_path
):
    """Test reshaping both biomass and counting data."""
    # Add counting source to config
    config_biomass.root_core.sources.append(
        RootCoreSourceConfig(
            csv_path="dummy2.csv",
            data_type="counting",
            depth_column_prefix="RootCount",
            value_column_name="Root_Count",
        )
    )

    input_data = {
        "biomass": biomass_aggregated_data,
        "counting": counting_aggregated_data,
    }

    step = ReshapeForTraitQCStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    # Check result is a merged DataFrame with columns from both sources
    assert isinstance(result.data, pd.DataFrame)
    df_merged = result.data

    # Check both prefixes present in columns
    assert "RootDW_15cm" in df_merged.columns
    assert "RootDW_45cm" in df_merged.columns
    assert "RootCount_0cm" in df_merged.columns
    assert "RootCount_5cm" in df_merged.columns

    # Check files generated
    assert (tmp_path / "00e_root_core_biomass_wide.csv").exists()
    assert (tmp_path / "00e_root_core_counting_wide.csv").exists()
    assert (tmp_path / "00e_root_core_merged.csv").exists()

    # Check metadata
    assert len(result.metadata["sources"]) == 2
    assert "merged_shape" in result.metadata


def test_reshape_preserves_metadata_columns(
    biomass_aggregated_data, config_biomass, tmp_path
):
    """Test that metadata columns are preserved during reshape."""
    # Add metadata columns
    biomass_aggregated_data["Cid"] = "C123"
    biomass_aggregated_data["Sid"] = "S456"

    input_data = {"biomass": biomass_aggregated_data}

    step = ReshapeForTraitQCStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    df_wide = result.data

    # Check metadata columns preserved
    assert "Cid" in df_wide.columns
    assert "Sid" in df_wide.columns
    assert df_wide["Cid"].iloc[0] == "C123"


def test_reshape_column_naming_format(
    biomass_aggregated_data, config_biomass, tmp_path
):
    """Test that column names follow the expected format."""
    input_data = {"biomass": biomass_aggregated_data}

    step = ReshapeForTraitQCStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    df_wide = result.data

    # Check column name format: {prefix}_{depth}cm
    depth_cols = [col for col in df_wide.columns if col.startswith("RootDW_")]
    assert len(depth_cols) == 2

    # All should end with 'cm'
    assert all(col.endswith("cm") for col in depth_cols)

    # Should have integer depth values
    for col in depth_cols:
        depth_str = col.replace("RootDW_", "").replace("cm", "")
        assert depth_str.isdigit() or "." in depth_str  # int or float


def test_reshape_float_depths(config_biomass, tmp_path):
    """Test reshaping with non-integer depth values."""
    # Create data with float depths
    data = []
    for plot in [1]:
        for rep in [1]:
            for geno in ["A"]:
                for depth in [15.5, 45.7]:
                    data.append(
                        {
                            "Plot": plot,
                            "Rep": rep,
                            "geno": geno,
                            "Depth_cm": depth,
                            "Root_DW_g": 1.0,
                        }
                    )

    df = pd.DataFrame(data)
    input_data = {"biomass": df}

    step = ReshapeForTraitQCStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    df_wide = result.data

    # Check columns exist with float depths
    assert "RootDW_15.5cm" in df_wide.columns
    assert "RootDW_45.7cm" in df_wide.columns


def test_reshape_single_depth(config_biomass, tmp_path):
    """Test reshaping with only one depth."""
    # Create data with single depth
    data = []
    for plot in [1, 2]:
        for rep in [1]:
            for geno in ["A", "B"]:
                data.append(
                    {
                        "Plot": plot,
                        "Rep": rep,
                        "geno": geno,
                        "Depth_cm": 15.0,
                        "Root_DW_g": 1.5,
                    }
                )

    df = pd.DataFrame(data)
    input_data = {"biomass": df}

    step = ReshapeForTraitQCStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    df_wide = result.data

    # Check only one depth column
    assert "RootDW_15cm" in df_wide.columns
    depth_cols = [col for col in df_wide.columns if col.startswith("RootDW_")]
    assert len(depth_cols) == 1

    # Check metadata
    source_meta = result.metadata["sources"][0]
    assert source_meta["num_depths"] == 1



def test_barcode_with_float_columns(config_biomass, tmp_path):
    """Test that Barcode column doesn't contain '.0' when Plot/Rep are floats.
    
    This regression test ensures that when Plot and Rep columns are float64
    (as happens when loading from CSV), the Barcode column format is correct
    (e.g., "1-1" not "1.0-1.0").
    """
    # Create data with float Plot/Rep columns (simulates CSV loading)
    data = []
    for plot in [1.0, 2.0]:  # float64
        for rep in [1.0, 2.0]:  # float64
            for geno in ["GH_7386", "GH_7420"]:
                for depth in [15.0, 45.0]:
                    data.append(
                        {
                            "Plot": plot,
                            "Rep": rep,
                            "geno": geno,
                            "Depth_cm": depth,
                            "Root_DW_g": np.random.uniform(1.0, 3.0),
                        }
                    )
    
    df = pd.DataFrame(data)
    
    # Verify columns are float
    assert df["Plot"].dtype == np.float64
    assert df["Rep"].dtype == np.float64
    
    input_data = {"biomass": df}
    
    step = ReshapeForTraitQCStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)
    
    df_wide = result.data
    
    # Check Barcode column exists
    assert "Barcode" in df_wide.columns
    
    # Check Barcode format is correct (no ".0" decimals)
    expected_barcodes = {
        "1-1", "1-2", "2-1", "2-2"
    }
    actual_barcodes = set(df_wide["Barcode"].unique())
    assert actual_barcodes == expected_barcodes
    
    # Explicitly verify no ".0" appears in any barcode
    for barcode in df_wide["Barcode"]:
        assert ".0" not in barcode, f"Found '.0' in Barcode: {barcode}"
