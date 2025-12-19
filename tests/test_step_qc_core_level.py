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


# ============================================================================
# Value Outlier Detection Tests
# ============================================================================


@pytest.fixture
def config_value_outliers_enabled():
    """Create configuration with value outlier detection enabled."""
    return QCPipelineConfig(
        pipeline_name="test_value_qc",
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
                detect_value_outliers=True,  # Enable value outlier detection
                max_deviation_from_median=0.30,  # 30% threshold
                min_cores_after_qc=1,
            ),
        ),
    )


@pytest.fixture
def gh_7371_data():
    """Create GH_7371 Rep 1 real-world case data.

    Cores: [0.7636g, 0.7071g, 0.3132g]
    Core 2 has 56% deviation from median - measurement error
    """
    data = []
    values = {1: 0.7636, 2: 0.7071, 3: 0.3132}  # Core 3 is the outlier (56% deviation)

    for core, value in values.items():
        for depth in [15.0, 45.0]:
            data.append(
                {
                    "Plot": 5,
                    "Rep": 1,
                    "geno": "GH_7371",
                    "Core_Replicate": core,
                    "Depth_cm": depth,
                    "Root_DW_g": value,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def normal_variation_data():
    """Create data with normal biological variation (~10%).

    Cores: [0.72g, 0.68g, 0.75g]
    All within 10% of median - should NOT be flagged
    """
    data = []
    values = {1: 0.72, 2: 0.68, 3: 0.75}

    for core, value in values.items():
        for depth in [15.0, 45.0]:
            data.append(
                {
                    "Plot": 1,
                    "Rep": 1,
                    "geno": "GH_7386",
                    "Core_Replicate": core,
                    "Depth_cm": depth,
                    "Root_DW_g": value,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def all_outliers_data():
    """Create data where all cores are outliers (safety test).

    Cores: [0.1g, 0.5g, 0.9g]
    All differ significantly - safety should keep at least 1
    """
    data = []
    values = {1: 0.1, 2: 0.5, 3: 0.9}

    for core, value in values.items():
        for depth in [15.0, 45.0]:
            data.append(
                {
                    "Plot": 1,
                    "Rep": 1,
                    "geno": "GH_7386",
                    "Core_Replicate": core,
                    "Depth_cm": depth,
                    "Root_DW_g": value,
                }
            )

    return pd.DataFrame(data)


def test_gh_7371_real_world_case(gh_7371_data, config_value_outliers_enabled, tmp_path):
    """Test GH_7371 Rep 1 real-world case: core 2 (56% deviation) is flagged."""
    input_data = {"biomass": gh_7371_data}

    step = QCCoreLevelStep()
    result = step.execute(
        data=input_data, config=config_value_outliers_enabled, run_dir=tmp_path
    )

    # Core 2 (0.3132g, 56% deviation) should be flagged
    assert result.metadata["total_flagged"] > 0

    # Check flagged_by_method breakdown
    source_meta = result.metadata["sources"][0]
    assert "flagged_by_method" in source_meta
    assert source_meta["flagged_by_method"]["value_outlier"] > 0

    # Check flagged cores list contains value_deviation reason
    flagged_cores = source_meta["flagged_cores_list"]
    value_outliers = [c for c in flagged_cores if "value_deviation" in c["reason"]]
    assert len(value_outliers) > 0

    # Check metadata includes deviation percentage
    outlier = value_outliers[0]
    assert "deviation_pct" in outlier
    assert outlier["deviation_pct"] > 0.50  # Should be ~56%
    assert "value" in outlier
    assert "median" in outlier
    assert "threshold" in outlier
    assert outlier["threshold"] == 0.30


def test_normal_variation_not_flagged(
    normal_variation_data, config_value_outliers_enabled, tmp_path
):
    """Test that normal biological variation (~10%) is NOT flagged."""
    input_data = {"biomass": normal_variation_data}

    step = QCCoreLevelStep()
    result = step.execute(
        data=input_data, config=config_value_outliers_enabled, run_dir=tmp_path
    )

    # No cores should be flagged (all within 30% threshold)
    source_meta = result.metadata["sources"][0]
    assert source_meta["flagged_by_method"]["value_outlier"] == 0

    # Data unchanged
    assert len(result.data["biomass"]) == len(normal_variation_data)


def test_safety_keeps_minimum_cores(all_outliers_data, tmp_path):
    """Test safety mechanism: keeps at least min_cores_after_qc cores."""
    config = QCPipelineConfig(
        pipeline_name="test_safety",
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
                remove_outliers=True,
                detect_value_outliers=True,
                max_deviation_from_median=0.20,  # Strict threshold
                min_cores_after_qc=1,  # Keep at least 1 core
            ),
        ),
    )

    input_data = {"biomass": all_outliers_data}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config, run_dir=tmp_path)

    # Not all cores should be removed (safety prevents empty groups)
    assert len(result.data["biomass"]) > 0

    # At least 1 core per depth should remain (1 core × 2 depths = 2 rows minimum)
    assert len(result.data["biomass"]) >= 2


def test_median_zero_edge_case(config_value_outliers_enabled, tmp_path):
    """Test edge case: median = 0 (skip group to avoid division by zero)."""
    data = []
    values = {1: 0.0, 2: 0.0, 3: 0.0}  # All zeros → median = 0

    for core, value in values.items():
        for depth in [15.0]:
            data.append(
                {
                    "Plot": 1,
                    "Rep": 1,
                    "geno": "GH_7386",
                    "Core_Replicate": core,
                    "Depth_cm": depth,
                    "Root_DW_g": value,
                }
            )

    df = pd.DataFrame(data)
    input_data = {"biomass": df}

    step = QCCoreLevelStep()
    result = step.execute(
        data=input_data, config=config_value_outliers_enabled, run_dir=tmp_path
    )

    # Should not crash (division by zero avoided)
    assert len(result.data["biomass"]) == len(df)

    # No value outliers flagged (group skipped)
    source_meta = result.metadata["sources"][0]
    assert source_meta["flagged_by_method"]["value_outlier"] == 0


def test_insufficient_cores_edge_case(config_value_outliers_enabled, tmp_path):
    """Test edge case: only 1-2 cores available (skip value detection)."""
    data = []
    # Only 1 core (need at least 2 for median comparison)
    for depth in [15.0, 45.0]:
        data.append(
            {
                "Plot": 1,
                "Rep": 1,
                "geno": "GH_7386",
                "Core_Replicate": 1,
                "Depth_cm": depth,
                "Root_DW_g": 2.0,
            }
        )

    df = pd.DataFrame(data)
    input_data = {"biomass": df}

    step = QCCoreLevelStep()
    result = step.execute(
        data=input_data, config=config_value_outliers_enabled, run_dir=tmp_path
    )

    # Should not crash
    assert len(result.data["biomass"]) == len(df)

    # No value outliers flagged (insufficient data)
    source_meta = result.metadata["sources"][0]
    assert source_meta["flagged_by_method"]["value_outlier"] == 0


def test_combined_missing_and_value_outliers(config_value_outliers_enabled, tmp_path):
    """Test combined detection: missing data + value outliers."""
    data = []

    # Core 1: Normal
    for depth in [15.0, 45.0]:
        data.append(
            {
                "Plot": 1,
                "Rep": 1,
                "geno": "GH_7386",
                "Core_Replicate": 1,
                "Depth_cm": depth,
                "Root_DW_g": 0.75,
            }
        )

    # Core 2: Missing data (NaN)
    for depth in [15.0, 45.0]:
        data.append(
            {
                "Plot": 1,
                "Rep": 1,
                "geno": "GH_7386",
                "Core_Replicate": 2,
                "Depth_cm": depth,
                "Root_DW_g": np.nan,
            }
        )

    # Core 3: Extreme value outlier
    for depth in [15.0, 45.0]:
        data.append(
            {
                "Plot": 1,
                "Rep": 1,
                "geno": "GH_7386",
                "Core_Replicate": 3,
                "Depth_cm": depth,
                "Root_DW_g": 0.2,  # 73% deviation from 0.75
            }
        )

    df = pd.DataFrame(data)
    input_data = {"biomass": df}

    step = QCCoreLevelStep()
    result = step.execute(
        data=input_data, config=config_value_outliers_enabled, run_dir=tmp_path
    )

    # Both types of outliers should be flagged
    source_meta = result.metadata["sources"][0]
    assert source_meta["flagged_by_method"]["missing_data"] > 0
    assert source_meta["flagged_by_method"]["value_outlier"] > 0

    # Both should be removed
    assert result.metadata["total_flagged"] >= 2
    assert result.metadata["total_removed"] >= 2


def test_backward_compatibility_disabled(gh_7371_data, config_qc_enabled, tmp_path):
    """Test backward compatibility: detect_value_outliers=False (default)."""
    input_data = {"biomass": gh_7371_data}

    step = QCCoreLevelStep()
    result = step.execute(data=input_data, config=config_qc_enabled, run_dir=tmp_path)

    # Value outlier detection should NOT run (backward compatible)
    source_meta = result.metadata["sources"][0]
    assert "flagged_by_method" in source_meta
    assert source_meta["flagged_by_method"]["value_outlier"] == 0

    # Core 2 should NOT be flagged (detection disabled)
    assert result.metadata["total_flagged"] == 0


def test_value_outlier_metadata_structure(
    gh_7371_data, config_value_outliers_enabled, tmp_path
):
    """Test that value outlier metadata has expected structure."""
    input_data = {"biomass": gh_7371_data}

    step = QCCoreLevelStep()
    result = step.execute(
        data=input_data, config=config_value_outliers_enabled, run_dir=tmp_path
    )

    source_meta = result.metadata["sources"][0]

    # Check flagged_by_method breakdown exists
    assert "flagged_by_method" in source_meta
    assert "missing_data" in source_meta["flagged_by_method"]
    assert "value_outlier" in source_meta["flagged_by_method"]

    # Check value outlier metadata fields
    flagged_cores = source_meta["flagged_cores_list"]
    value_outliers = [c for c in flagged_cores if "value_deviation" in c["reason"]]

    if value_outliers:
        outlier = value_outliers[0]
        # Required fields for value outliers
        assert "value" in outlier
        assert "median" in outlier
        assert "deviation_pct" in outlier
        assert "threshold" in outlier
        assert "core_id" in outlier
        assert "group" in outlier
        assert "reason" in outlier


def test_per_group_detection_independence(config_value_outliers_enabled, tmp_path):
    """Test that value detection is performed independently per group."""
    data = []

    # Group 1 (Plot 1): Low values [0.3, 0.32, 0.35] - all normal within group
    values_g1 = {1: 0.3, 2: 0.32, 3: 0.35}
    for core, value in values_g1.items():
        data.append(
            {
                "Plot": 1,
                "Rep": 1,
                "geno": "A",
                "Core_Replicate": core,
                "Depth_cm": 15.0,
                "Root_DW_g": value,
            }
        )

    # Group 2 (Plot 2): High values [0.7, 0.72, 0.75] - all normal within group
    values_g2 = {1: 0.7, 2: 0.72, 3: 0.75}
    for core, value in values_g2.items():
        data.append(
            {
                "Plot": 2,
                "Rep": 1,
                "geno": "B",
                "Core_Replicate": core,
                "Depth_cm": 15.0,
                "Root_DW_g": value,
            }
        )

    df = pd.DataFrame(data)
    input_data = {"biomass": df}

    step = QCCoreLevelStep()
    result = step.execute(
        data=input_data, config=config_value_outliers_enabled, run_dir=tmp_path
    )

    # No cores should be flagged (both groups have normal within-group variation)
    source_meta = result.metadata["sources"][0]
    assert source_meta["flagged_by_method"]["value_outlier"] == 0

    # Data unchanged (per-group analysis prevents cross-group contamination)
    assert len(result.data["biomass"]) == len(df)


def test_different_thresholds(gh_7371_data, tmp_path):
    """Test sensitivity to different deviation thresholds."""
    # Test with aggressive threshold (20%)
    config_20 = QCPipelineConfig(
        pipeline_name="test_20pct",
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
                detect_value_outliers=True,
                max_deviation_from_median=0.20,  # 20% threshold (aggressive)
            ),
        ),
    )

    # Test with very conservative threshold (40%)
    config_40 = QCPipelineConfig(
        pipeline_name="test_40pct",
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
                detect_value_outliers=True,
                max_deviation_from_median=0.40,  # 40% threshold (very conservative)
            ),
        ),
    )

    input_data = {"biomass": gh_7371_data}
    step = QCCoreLevelStep()

    # Both thresholds should detect GH_7371 core 2 (56% deviation)
    result_20 = step.execute(data=input_data, config=config_20, run_dir=tmp_path)
    result_40 = step.execute(data=input_data, config=config_40, run_dir=tmp_path)

    # GH_7371 core 2 (56%) should be flagged by both 20% and 40% thresholds
    assert result_20.metadata["total_flagged"] > 0
    assert result_40.metadata["total_flagged"] > 0
