"""Tests for AggregateCoresStep."""

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
from sleap_roots_analyze.pipeline.steps.aggregate_cores import AggregateCoresStep


@pytest.fixture
def biomass_long_data():
    """Create long-format biomass data with 3 cores per Plot-Rep-Depth."""
    data = []
    # Plot 1: GH_7386, Plot 2: GH_7420 (each plot has different geno)
    plot_geno_map = {1: "GH_7386", 2: "GH_7420"}
    for plot in [1, 2]:
        geno = plot_geno_map[plot]
        for rep in [1]:
            for core in [1, 2, 3]:
                for depth in [15.0, 45.0]:
                    data.append(
                        {
                            "Plot": plot,
                            "Rep": rep,
                            "geno": geno,
                            "Core_Replicate": core,
                            "sample_id": f"plot{plot}_rep{rep}_{geno}_core{core}",
                            "Depth_cm": depth,
                            "Root_DW_g": np.random.uniform(1.0, 3.0),
                        }
                    )
    return pd.DataFrame(data)


@pytest.fixture
def counting_long_data():
    """Create long-format counting data with 3 cores per Plot-Rep-Depth."""
    np.random.seed(42)
    data = []
    plot_geno_map = {1: "GH_7386", 2: "GH_7420"}
    for plot in [1, 2]:
        geno = plot_geno_map[plot]
        for rep in [1]:
            for core in [1, 2, 3]:
                for depth in [0.0, 5.0, 10.0, 15.0]:
                    data.append(
                        {
                            "Plot": plot,
                            "Rep": rep,
                            "geno": geno,
                            "core_n": core,
                            "sample_id": f"plot{plot}_rep{rep}_{geno}_core{core}",
                            "Depth_cm": depth,
                            "Root_Count": np.random.randint(5, 50),
                        }
                    )
    return pd.DataFrame(data)


@pytest.fixture
def config_biomass():
    """Create configuration for biomass aggregation."""
    return QCPipelineConfig(
        pipeline_name="test_aggregate_biomass",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path="dummy.csv",
                    data_type="biomass",
                    depth_column_prefix="RootDW",
                    value_column_name="Root_DW_g",
                    aggregation_method="mean",
                )
            ],
            core_qc=CoreQCConfig(enabled=False),
        ),
    )


@pytest.fixture
def config_counting():
    """Create configuration for counting aggregation."""
    return QCPipelineConfig(
        pipeline_name="test_aggregate_counting",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path="dummy.csv",
                    data_type="counting",
                    depth_column_prefix="RootCount",
                    value_column_name="Root_Count",
                    aggregation_method="median",
                )
            ],
            core_qc=CoreQCConfig(enabled=False),
        ),
    )


def test_aggregate_biomass_mean(biomass_long_data, config_biomass, tmp_path):
    """Test aggregating biomass data using mean."""
    input_data = {"biomass": biomass_long_data}

    step = AggregateCoresStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    # Check result structure
    assert "biomass" in result.data
    df_agg = result.data["biomass"]

    # Check shape: 2 plots × 2 genos × 1 rep × 2 depths = 8 rows
    # Actually: Plot 1 has GH_7386, Plot 2 has GH_7420, so 2 unique Plot-Rep-geno-Depth combos × 2 depths × 2 plots = 8
    # Let me recalculate: 2 genos total across 2 plots, each with 2 depths = 4 unique Plot-Rep-geno combinations × 2 depths = 8 rows
    # Actually simpler: each unique (Plot, Rep, geno, Depth_cm) becomes one row
    num_unique_groups = biomass_long_data.groupby(
        ["Plot", "Rep", "geno", "Depth_cm"]
    ).ngroups
    assert len(df_agg) == num_unique_groups

    # Check required columns
    assert "Plot" in df_agg.columns
    assert "Rep" in df_agg.columns
    assert "geno" in df_agg.columns
    assert "Depth_cm" in df_agg.columns
    assert "Root_DW_g" in df_agg.columns

    # Check that sample_id and Core_Replicate are NOT in aggregated data
    assert "sample_id" not in df_agg.columns
    assert "Core_Replicate" not in df_agg.columns

    # Check files generated
    assert (tmp_path / "00c_root_core_biomass_aggregated.csv").exists()
    assert (tmp_path / "00c_aggregation_metadata.json").exists()

    # Check metadata
    assert len(result.metadata["sources"]) == 1
    source_meta = result.metadata["sources"][0]
    assert source_meta["aggregation_method"] == "mean"
    assert source_meta["max_cores_per_group"] == 3


@pytest.mark.parametrize("replicate_value", [None, "rep", "block"])
def test_rep_aggregation_invariant_to_columns_replicate(
    biomass_long_data, config_biomass, tmp_path, replicate_value
):
    """Root-core aggregation keys on hardcoded "Rep", not columns.replicate (issue #142).

    Making columns.replicate optional must not change field/root-core behavior:
    the "Rep" column is a separate, hardcoded field.
    """
    config_biomass.columns.replicate = replicate_value
    input_data = {"biomass": biomass_long_data}

    step = AggregateCoresStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)
    df_agg = result.data["biomass"]

    # Aggregation still groups on hardcoded "Rep" regardless of columns.replicate.
    num_unique_groups = biomass_long_data.groupby(
        ["Plot", "Rep", "geno", "Depth_cm"]
    ).ngroups
    assert len(df_agg) == num_unique_groups
    assert "Rep" in df_agg.columns


def test_aggregate_counting_median(counting_long_data, config_counting, tmp_path):
    """Test aggregating counting data using median."""
    input_data = {"counting": counting_long_data}

    step = AggregateCoresStep()
    result = step.execute(data=input_data, config=config_counting, run_dir=tmp_path)

    # Check result structure
    assert "counting" in result.data
    df_agg = result.data["counting"]

    # Check that aggregation happened
    num_unique_groups = counting_long_data.groupby(
        ["Plot", "Rep", "geno", "Depth_cm"]
    ).ngroups
    assert len(df_agg) == num_unique_groups

    # Check metadata shows median was used
    source_meta = result.metadata["sources"][0]
    assert source_meta["aggregation_method"] == "median"


def test_aggregate_both_sources(
    biomass_long_data, counting_long_data, config_biomass, tmp_path
):
    """Test aggregating both biomass and counting data."""
    # Add counting source to config
    config_biomass.root_core.sources.append(
        RootCoreSourceConfig(
            csv_path="dummy2.csv",
            data_type="counting",
            depth_column_prefix="RootCount",
            value_column_name="Root_Count",
            aggregation_method="mean",
        )
    )

    input_data = {"biomass": biomass_long_data, "counting": counting_long_data}

    step = AggregateCoresStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    # Check both data types aggregated
    assert "biomass" in result.data
    assert "counting" in result.data

    # Check files generated
    assert (tmp_path / "00c_root_core_biomass_aggregated.csv").exists()
    assert (tmp_path / "00c_root_core_counting_aggregated.csv").exists()

    # Check metadata
    assert len(result.metadata["sources"]) == 2


def test_aggregate_preserves_metadata_columns(
    biomass_long_data, config_biomass, tmp_path
):
    """Test that metadata columns are preserved during aggregation."""
    # Add metadata columns
    biomass_long_data["Cid"] = "C123"
    biomass_long_data["Sid"] = "S456"

    input_data = {"biomass": biomass_long_data}

    step = AggregateCoresStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    df_agg = result.data["biomass"]

    # Check metadata columns preserved (using 'first' aggregation)
    assert "Cid" in df_agg.columns
    assert "Sid" in df_agg.columns
    assert df_agg["Cid"].iloc[0] == "C123"


def test_aggregate_metadata_statistics(biomass_long_data, config_biomass, tmp_path):
    """Test that aggregation metadata includes statistics."""
    input_data = {"biomass": biomass_long_data}

    step = AggregateCoresStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    source_meta = result.metadata["sources"][0]

    # Check required statistics
    assert "total_groups" in source_meta
    assert "min_cores_per_group" in source_meta
    assert "max_cores_per_group" in source_meta
    assert "mean_cores_per_group" in source_meta

    # All groups should have 3 cores
    assert source_meta["min_cores_per_group"] == 3
    assert source_meta["max_cores_per_group"] == 3
    assert source_meta["mean_cores_per_group"] == 3.0


def test_aggregate_invalid_method(biomass_long_data, tmp_path):
    """Test error with invalid aggregation method."""
    config = QCPipelineConfig(
        pipeline_name="test_invalid_agg",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path="dummy.csv",
                    data_type="biomass",
                    depth_column_prefix="RootDW",
                    value_column_name="Root_DW_g",
                    aggregation_method="invalid_method",
                )
            ]
        ),
    )

    input_data = {"biomass": biomass_long_data}
    step = AggregateCoresStep()

    with pytest.raises(ValueError, match="Invalid aggregation_method"):
        step.execute(data=input_data, config=config, run_dir=tmp_path)


def test_aggregate_with_unequal_cores(config_biomass, tmp_path):
    """Test aggregation when some groups have different numbers of cores."""
    # Create data with unequal cores (one group has only 2 cores)
    data = []
    for plot, num_cores in [(1, 3), (2, 2)]:  # Plot 2 has only 2 cores
        for rep in [1]:
            for depth in [15.0, 45.0]:
                for core in range(1, num_cores + 1):
                    data.append(
                        {
                            "Plot": plot,
                            "Rep": rep,
                            "geno": f"geno{plot}",
                            "Core_Replicate": core,
                            "Depth_cm": depth,
                            "Root_DW_g": 1.0 + core,
                        }
                    )

    df = pd.DataFrame(data)
    input_data = {"biomass": df}

    step = AggregateCoresStep()
    result = step.execute(data=input_data, config=config_biomass, run_dir=tmp_path)

    # Check metadata reflects unequal cores
    source_meta = result.metadata["sources"][0]
    assert source_meta["min_cores_per_group"] == 2
    assert source_meta["max_cores_per_group"] == 3
