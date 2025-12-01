"""Tests for QCCoreLevelStep."""

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
from sleap_roots_analyze.pipeline.steps.qc_core_level import QCCoreLevelStep


@pytest.fixture
def biomass_with_outliers():
    """Create biomass data with one clear outlier core."""
    np.random.seed(42)
    data = []

    # Normal cores: values around 2.0 ± 0.5
    for core in [1, 2]:
        for depth in [15.0, 45.0]:
            data.append(
                {
                    "Plot": 1,
                    "Rep": 1,
                    "geno": "GH_7386",
                    "Core_Replicate": core,
                    "Depth_cm": depth,
                    "Root_DW_g": np.random.normal(2.0, 0.5),
                }
            )

    # Outlier core: values around 10.0 (5x higher)
    for depth in [15.0, 45.0]:
        data.append(
            {
                "Plot": 1,
                "Rep": 1,
                "geno": "GH_7386",
                "Core_Replicate": 3,
                "Depth_cm": depth,
                "Root_DW_g": 10.0 + np.random.normal(0, 0.5),
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def biomass_with_missing_data():
    """Create biomass data with one core having excessive NaNs."""
    data = []

    # Normal cores
    for core in [1, 2]:
        for depth in [15.0, 45.0]:
            data.append(
                {
                    "Plot": 1,
                    "Rep": 1,
                    "geno": "GH_7386",
                    "Core_Replicate": core,
                    "Depth_cm": depth,
                    "Root_DW_g": 2.0,
                }
            )

    # Core with >50% missing (2 of 2 depths are NaN = 100% missing)
    data.append(
        {
            "Plot": 1,
            "Rep": 1,
            "geno": "GH_7386",
            "Core_Replicate": 3,
            "Depth_cm": 15.0,
            "Root_DW_g": np.nan,
        }
    )
    data.append(
        {
            "Plot": 1,
            "Rep": 1,
            "geno": "GH_7386",
            "Core_Replicate": 3,
            "Depth_cm": 45.0,
            "Root_DW_g": np.nan,
        }
    )

    return pd.DataFrame(data)


@pytest.fixture
def config_qc_enabled():
    """Create configuration with core QC enabled."""
    return QCPipelineConfig(
        pipeline_name="test_core_qc",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path="dummy.csv",
                    data_type="biomass",
                    depth_column_prefix="RootDW",
                    value_column_name="Root_DW_g",
                )
            ],
            core_qc=CoreQCConfig(
                enabled=True,
                outlier_method="mahalanobis",
                contamination=0.3,  # Expect 30% outliers
                max_missing_proportion=0.5,
                remove_outliers=True,
            ),
        ),
    )


@pytest.fixture
def config_qc_disabled():
    """Create configuration with core QC disabled."""
    return QCPipelineConfig(
        pipeline_name="test_no_qc",
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


def test_qc_disabled_passthrough(biomass_with_outliers, config_qc_disabled, tmp_path):
    """Test that QC disabled mode passes data through unchanged."""
    input_data = {"biomass": biomass_with_outliers}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config_qc_disabled, run_dir=tmp_path)

    # Data should be unchanged
    assert "biomass" in result.data
    assert len(result.data["biomass"]) == len(biomass_with_outliers)

    # Metadata should indicate QC was disabled
    assert result.metadata["qc_enabled"] is False

    # No files generated
    assert len(result.files_generated) == 0


def test_detect_mahalanobis_outlier(biomass_with_outliers, config_qc_enabled, tmp_path):
    """Test detection of outlier core using Mahalanobis distance."""
    input_data = {"biomass": biomass_with_outliers}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config_qc_enabled, run_dir=tmp_path)

    # Check outlier detected
    assert result.metadata["total_outliers"] > 0

    # Check outlier removed (since remove_outliers=True)
    assert result.metadata["total_removed"] > 0

    # Data should be smaller
    df_qc = result.data["biomass"]
    assert len(df_qc) < len(biomass_with_outliers)

    # Files generated
    assert (tmp_path / "00c_root_core_biomass_qc.csv").exists()
    assert (tmp_path / "00c_core_qc_metadata.json").exists()


def test_detect_missing_data_outlier(
    biomass_with_missing_data, config_qc_enabled, tmp_path
):
    """Test detection of core with excessive missing data."""
    input_data = {"biomass": biomass_with_missing_data}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config_qc_enabled, run_dir=tmp_path)

    # Check outlier detected
    assert result.metadata["total_outliers"] > 0

    # Check outlier list contains missing_data reason
    source_meta = result.metadata["sources"][0]
    outlier_reasons = [o["reasons"] for o in source_meta["outlier_list"]]
    assert any("missing_data" in str(reasons) for reasons in outlier_reasons)


def test_qc_with_remove_false(biomass_with_outliers, tmp_path):
    """Test QC with remove_outliers=False (flag only)."""
    config = QCPipelineConfig(
        pipeline_name="test_flag_only",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path="dummy.csv",
                    data_type="biomass",
                    depth_column_prefix="RootDW",
                    value_column_name="Root_DW_g",
                )
            ],
            core_qc=CoreQCConfig(
                enabled=True,
                contamination=0.3,
                remove_outliers=False,  # Just flag, don't remove
            ),
        ),
    )

    input_data = {"biomass": biomass_with_outliers}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config, run_dir=tmp_path)

    # Outliers detected but not removed
    assert result.metadata["total_outliers"] > 0
    assert result.metadata["total_removed"] == 0

    # Data size unchanged
    assert len(result.data["biomass"]) == len(biomass_with_outliers)

    # But outlier flags should be present
    df_qc = result.data["biomass"]
    assert "outlier_flag" in df_qc.columns
    assert "outlier_reason" in df_qc.columns
    assert df_qc["outlier_flag"].any()


def test_qc_with_normal_data(tmp_path):
    """Test QC with all normal cores (no outliers)."""
    # Create config with low contamination for normal data
    config = QCPipelineConfig(
        pipeline_name="test_normal_data",
        root_core=RootCoreConfig(
            sources=[
                RootCoreSourceConfig(
                    csv_path="dummy.csv",
                    data_type="biomass",
                    depth_column_prefix="RootDW",
                    value_column_name="Root_DW_g",
                )
            ],
            core_qc=CoreQCConfig(
                enabled=True,
                contamination=0.05,  # Very low contamination for normal data
                remove_outliers=True,
            ),
        ),
    )

    # Create normal data with identical values (no variation = no outliers possible)
    data = []
    for core in [1, 2, 3]:
        for depth in [15.0, 45.0]:
            data.append(
                {
                    "Plot": 1,
                    "Rep": 1,
                    "geno": "GH_7386",
                    "Core_Replicate": core,
                    "Depth_cm": depth,
                    "Root_DW_g": 2.0,  # Identical values - no outliers
                }
            )

    df = pd.DataFrame(data)
    input_data = {"biomass": df}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config, run_dir=tmp_path)

    # No outliers should be detected
    assert result.metadata["total_outliers"] == 0
    assert result.metadata["total_removed"] == 0

    # Data unchanged
    assert len(result.data["biomass"]) == len(df)


def test_qc_multiple_groups(config_qc_enabled, tmp_path):
    """Test QC with multiple Plot-Rep-geno groups."""
    np.random.seed(42)
    data = []

    # Group 1: Normal cores
    for core in [1, 2, 3]:
        for depth in [15.0, 45.0]:
            data.append(
                {
                    "Plot": 1,
                    "Rep": 1,
                    "geno": "A",
                    "Core_Replicate": core,
                    "Depth_cm": depth,
                    "Root_DW_g": 2.0 + np.random.normal(0, 0.2),
                }
            )

    # Group 2: One outlier core
    for core in [1, 2]:
        for depth in [15.0, 45.0]:
            data.append(
                {
                    "Plot": 2,
                    "Rep": 1,
                    "geno": "B",
                    "Core_Replicate": core,
                    "Depth_cm": depth,
                    "Root_DW_g": 3.0 + np.random.normal(0, 0.2),
                }
            )
    # Outlier in group 2
    for depth in [15.0, 45.0]:
        data.append(
            {
                "Plot": 2,
                "Rep": 1,
                "geno": "B",
                "Core_Replicate": 3,
                "Depth_cm": depth,
                "Root_DW_g": 15.0,
            }
        )

    df = pd.DataFrame(data)
    input_data = {"biomass": df}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config_qc_enabled, run_dir=tmp_path)

    # Should detect outlier in group 2 only
    assert result.metadata["total_outliers"] >= 1

    # Check outlier list contains correct group info
    source_meta = result.metadata["sources"][0]
    outlier_groups = [o["group"] for o in source_meta["outlier_list"]]
    assert any("2" in str(group) and "B" in str(group) for group in outlier_groups)


def test_qc_metadata_structure(biomass_with_outliers, config_qc_enabled, tmp_path):
    """Test that QC metadata has expected structure."""
    input_data = {"biomass": biomass_with_outliers}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config_qc_enabled, run_dir=tmp_path)

    # Check top-level metadata
    assert "sources" in result.metadata
    assert "total_outliers" in result.metadata
    assert "total_removed" in result.metadata

    # Check source-level metadata
    source_meta = result.metadata["sources"][0]
    assert "data_type" in source_meta
    assert "outliers_detected" in source_meta
    assert "outliers_removed" in source_meta
    assert "outlier_list" in source_meta
    assert "samples_before" in source_meta
    assert "samples_after" in source_meta

    # Check outlier list structure
    if source_meta["outlier_list"]:
        outlier = source_meta["outlier_list"][0]
        assert "core_id" in outlier
        assert "group" in outlier
        assert "reasons" in outlier
