"""Tests for TransformDepthDataStep."""

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
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps.transform_depth_data import (
    TransformDepthDataStep,
)


@pytest.fixture
def biomass_wide_data():
    """Create wide-format biomass data."""
    return pd.DataFrame(
        {
            "Plot": [1, 1, 1],
            "Rep": [1, 1, 1],
            "geno": ["GH_7386"] * 3,
            "Core_Replicate": [1, 2, 3],
            "sample_id": [
                "plot1_rep1_GH_7386_core1",
                "plot1_rep1_GH_7386_core2",
                "plot1_rep1_GH_7386_core3",
            ],
            "0-30": [1.2, 1.5, 1.1],
            "30-60": [0.8, 0.9, 0.7],
        }
    )


@pytest.fixture
def counting_wide_data():
    """Create wide-format counting data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Plot": [1, 1, 1],
            "Rep": [1, 1, 1],
            "geno": ["GH_7386"] * 3,
            "core_n": [1, 2, 3],
            "sample_id": [
                "plot1_rep1_GH_7386_core1",
                "plot1_rep1_GH_7386_core2",
                "plot1_rep1_GH_7386_core3",
            ],
            "c_0_10_1": [15, 20, 18],
            "c_0_10_2": [22, 25, 20],
            "c_10_20_1": [10, 12, 11],
            "c_10_20_2": [8, 9, 7],
        }
    )


@pytest.fixture
def config_biomass():
    """Create configuration for biomass transformation."""
    return QCPipelineConfig(
        pipeline_name="test_transform_biomass",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path="dummy.csv",
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
def config_counting():
    """Create configuration for counting transformation."""
    return QCPipelineConfig(
        pipeline_name="test_transform_counting",
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


@pytest.fixture
def config_both():
    """Create configuration for both data types."""
    return QCPipelineConfig(
        pipeline_name="test_transform_both",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path="dummy1.csv",
                    data_type="biomass",
                    depth_column_prefix="RootDW",
                    value_column_name="Root_DW_g",
                    depth_mapping={"0-30": 15.0, "30-60": 45.0},
                ),
                RootCoreSourceConfig(
                    csv_path="dummy2.csv",
                    data_type="counting",
                    depth_column_prefix="RootCount",
                    value_column_name="Root_Count",
                ),
            ],
            core_qc=CoreQCConfig(enabled=False),
        ),
    )


def test_transform_biomass_manual_mapping(biomass_wide_data, config_biomass, tmp_path):
    """Test transforming biomass data with manual depth mapping."""
    input_data = {"biomass": biomass_wide_data}

    step = TransformDepthDataStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    # Check result structure
    assert "biomass" in result.data
    df_long = result.data["biomass"]

    # Check shape: 3 cores × 2 depths = 6 rows
    assert len(df_long) == 6

    # Check required columns
    assert "Depth_cm" in df_long.columns
    assert "Root_DW_g" in df_long.columns

    # Check depth values
    assert set(df_long["Depth_cm"].unique()) == {15.0, 45.0}

    # Check metadata columns preserved
    assert "Plot" in df_long.columns
    assert "Rep" in df_long.columns
    assert "geno" in df_long.columns
    assert "sample_id" in df_long.columns

    # Check files generated
    assert (tmp_path / "00b_root_core_biomass_long.csv").exists()
    assert (tmp_path / "00b_depth_mapping.json").exists()


def test_transform_counting_auto_parse(counting_wide_data, config_counting, tmp_path):
    """Test transforming counting data with automatic depth parsing."""
    input_data = {"counting": counting_wide_data}

    step = TransformDepthDataStep()
    result = step.execute(data=input_data, config=config_counting, run_dir=tmp_path)

    # Check result structure
    assert "counting" in result.data
    df_long = result.data["counting"]

    # Check shape: 3 cores × 4 depth columns = 12 rows
    assert len(df_long) == 12

    # Check required columns
    assert "Depth_cm" in df_long.columns
    assert "Root_Count" in df_long.columns

    # Check depth values calculated correctly
    # c_0_10_1 -> 0.0, c_0_10_2 -> 5.0, c_10_20_1 -> 10.0, c_10_20_2 -> 15.0
    expected_depths = {0.0, 5.0, 10.0, 15.0}
    assert set(df_long["Depth_cm"].unique()) == expected_depths

    # Check depth mapping in metadata
    depth_mapping = result.metadata["depth_mappings"]["counting"]
    assert depth_mapping["c_0_10_1"] == 0.0
    assert depth_mapping["c_0_10_2"] == 5.0
    assert depth_mapping["c_10_20_1"] == 10.0
    assert depth_mapping["c_10_20_2"] == 15.0


def test_transform_both_sources(
    biomass_wide_data, counting_wide_data, config_both, tmp_path
):
    """Test transforming both biomass and counting data."""
    input_data = {"biomass": biomass_wide_data, "counting": counting_wide_data}

    step = TransformDepthDataStep()
    result = step.execute(data=input_data, config=config_both, run_dir=tmp_path)

    # Check both data types transformed
    assert "biomass" in result.data
    assert "counting" in result.data

    # Check shapes
    assert len(result.data["biomass"]) == 6  # 3 cores × 2 depths
    assert len(result.data["counting"]) == 12  # 3 cores × 4 depths

    # Check files generated
    assert (tmp_path / "00b_root_core_biomass_long.csv").exists()
    assert (tmp_path / "00b_root_core_counting_long.csv").exists()
    assert (tmp_path / "00b_depth_mapping.json").exists()


def test_biomass_missing_depth_mapping(biomass_wide_data, tmp_path):
    """Test error when biomass data lacks depth_mapping."""
    config = QCPipelineConfig(
        pipeline_name="test_missing_mapping",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path="dummy.csv",
                    data_type="biomass",
                    depth_column_prefix="RootDW",
                    value_column_name="Root_DW_g",
                    depth_mapping=None,  # Missing!
                )
            ]
        ),
    )

    input_data = {"biomass": biomass_wide_data}
    step = TransformDepthDataStep()

    with pytest.raises(ValueError, match="Biomass data requires depth_mapping"):
        step.execute(data=input_data, config=config, run_dir=tmp_path)


def test_counting_no_matching_columns(config_counting, tmp_path):
    """Test error when counting data has no depth columns."""
    # Create data without c_<start>_<end>_<subcore> pattern
    bad_data = pd.DataFrame(
        {
            "Plot": [1],
            "Rep": [1],
            "geno": ["A"],
            "core_n": [1],
            "value1": [10],
            "value2": [20],
        }
    )

    input_data = {"counting": bad_data}
    step = TransformDepthDataStep()

    with pytest.raises(ValueError, match="No depth columns found"):
        step.execute(data=input_data, config=config_counting, run_dir=tmp_path)


def test_depth_column_pattern_parsing():
    """Test depth calculation formula."""
    step = TransformDepthDataStep()

    # Test different patterns
    test_cases = [
        ("c_0_10_1", 0.0),  # start + (1-1) * (10-0)/2 = 0
        ("c_0_10_2", 5.0),  # start + (2-1) * (10-0)/2 = 5
        ("c_10_20_1", 10.0),  # start + (1-1) * (20-10)/2 = 10
        ("c_10_20_2", 15.0),  # start + (2-1) * (20-10)/2 = 15
        ("c_20_40_1", 20.0),  # start + (1-1) * (40-20)/2 = 20
        ("c_20_40_2", 30.0),  # start + (2-1) * (40-20)/2 = 30
    ]

    import re

    pattern = re.compile(r"c_(\d+)_(\d+)_(\d+)")
    for col_name, expected_depth in test_cases:
        match = pattern.match(col_name)
        start, end, subcore = map(int, match.groups())
        calculated_depth = start + (subcore - 1) * (end - start) / 2
        assert calculated_depth == expected_depth, f"Failed for {col_name}"


def test_metadata_columns_preserved(biomass_wide_data, config_biomass, tmp_path):
    """Test that all metadata columns are preserved during transformation."""
    # Add extra metadata columns
    biomass_wide_data["Cid"] = ["C1", "C2", "C3"]
    biomass_wide_data["Sid"] = ["S1", "S2", "S3"]

    input_data = {"biomass": biomass_wide_data}
    step = TransformDepthDataStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    df_long = result.data["biomass"]

    # Check all metadata columns preserved
    assert "Cid" in df_long.columns
    assert "Sid" in df_long.columns
    assert "Plot" in df_long.columns
    assert "Rep" in df_long.columns
