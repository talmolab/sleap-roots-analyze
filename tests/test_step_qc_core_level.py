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
    """Create configuration with core QC enabled.

    NOTE: Core QC only performs missing data detection. Statistical outlier detection
    is not used due to insufficient sample sizes (need 30+, have ~3).
    """
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


def test_biomass_outliers_not_detected_statistically(
    biomass_with_outliers, config_qc_enabled, tmp_path
):
    """Test that statistical outliers are NOT detected at core level.

    This test verifies that cores with extreme values are NOT flagged by statistical
    methods, because core-level QC only performs missing data detection. Statistical
    outlier detection requires 30+ samples but core-level data has only ~3 cores.
    """
    input_data = {"biomass": biomass_with_outliers}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config_qc_enabled, run_dir=tmp_path)

    # No outliers detected (because statistical detection is disabled)
    assert result.metadata["total_flagged"] == 0
    assert result.metadata["total_removed"] == 0

    # Data size unchanged
    assert len(result.data["biomass"]) == len(biomass_with_outliers)

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

    # Check core flagged for missing data
    assert result.metadata["total_flagged"] > 0

    # Check flagged cores list contains missing_data reason
    source_meta = result.metadata["sources"][0]
    flagged_reasons = [c["reason"] for c in source_meta["flagged_cores_list"]]
    assert any("missing_data" in str(reason) for reason in flagged_reasons)


def test_qc_with_remove_false(biomass_with_missing_data, tmp_path):
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
                remove_outliers=False,  # Just flag, don't remove
                max_missing_proportion=0.5,
            ),
        ),
    )

    input_data = {"biomass": biomass_with_missing_data}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config, run_dir=tmp_path)

    # Cores flagged but not removed
    assert result.metadata["total_flagged"] > 0
    assert result.metadata["total_removed"] == 0

    # Data size unchanged
    assert len(result.data["biomass"]) == len(biomass_with_missing_data)

    # But outlier flags should be present
    df_qc = result.data["biomass"]
    assert "outlier_flag" in df_qc.columns
    assert "outlier_reason" in df_qc.columns
    assert df_qc["outlier_flag"].any()


def test_qc_with_normal_data(tmp_path):
    """Test QC with all normal cores (no missing data)."""
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
                max_missing_proportion=0.5,
                remove_outliers=True,
            ),
        ),
    )

    # Create normal data with no missing values
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
                    "Root_DW_g": 2.0,
                }
            )

    df = pd.DataFrame(data)
    input_data = {"biomass": df}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config, run_dir=tmp_path)

    # No cores should be flagged
    assert result.metadata["total_flagged"] == 0
    assert result.metadata["total_removed"] == 0

    # Data unchanged
    assert len(result.data["biomass"]) == len(df)


def test_qc_multiple_groups(config_qc_enabled, tmp_path):
    """Test QC with multiple Plot-Rep-geno groups with missing data."""
    data = []

    # Group 1: All normal cores (no missing data)
    for core in [1, 2, 3]:
        for depth in [15.0, 45.0]:
            data.append(
                {
                    "Plot": 1,
                    "Rep": 1,
                    "geno": "A",
                    "Core_Replicate": core,
                    "Depth_cm": depth,
                    "Root_DW_g": 2.0,
                }
            )

    # Group 2: One core with excessive missing data
    for core in [1, 2]:
        for depth in [15.0, 45.0]:
            data.append(
                {
                    "Plot": 2,
                    "Rep": 1,
                    "geno": "B",
                    "Core_Replicate": core,
                    "Depth_cm": depth,
                    "Root_DW_g": 3.0,
                }
            )
    # Core 3 in group 2 has all NaN (100% missing)
    for depth in [15.0, 45.0]:
        data.append(
            {
                "Plot": 2,
                "Rep": 1,
                "geno": "B",
                "Core_Replicate": 3,
                "Depth_cm": depth,
                "Root_DW_g": np.nan,
            }
        )

    df = pd.DataFrame(data)
    input_data = {"biomass": df}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config_qc_enabled, run_dir=tmp_path)

    # Should flag core in group 2 only
    assert result.metadata["total_flagged"] >= 1

    # Check flagged cores list contains correct group info
    source_meta = result.metadata["sources"][0]
    flagged_groups = [c["group"] for c in source_meta["flagged_cores_list"]]
    assert any("2" in str(group) and "B" in str(group) for group in flagged_groups)


def test_qc_metadata_structure(biomass_with_missing_data, config_qc_enabled, tmp_path):
    """Test that QC metadata has expected structure."""
    input_data = {"biomass": biomass_with_missing_data}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config_qc_enabled, run_dir=tmp_path)

    # Check top-level metadata
    assert "sources" in result.metadata
    assert "total_flagged" in result.metadata
    assert "total_removed" in result.metadata

    # Check source-level metadata
    source_meta = result.metadata["sources"][0]
    assert "data_type" in source_meta
    assert "cores_flagged" in source_meta
    assert "cores_removed" in source_meta
    assert "flagged_cores_list" in source_meta
    assert "samples_before" in source_meta
    assert "samples_after" in source_meta

    # Check flagged cores list structure
    if source_meta["flagged_cores_list"]:
        flagged_core = source_meta["flagged_cores_list"][0]
        assert "core_id" in flagged_core
        assert "group" in flagged_core
        assert "reason" in flagged_core
