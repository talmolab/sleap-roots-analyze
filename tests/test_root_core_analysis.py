"""Tests for root_core_analysis module."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from sleap_roots_analyze.root_core_analysis import (
    create_sample_identifier,
    validate_unique_identifiers,
    melt_depth_data,
    aggregate_by_replicate,
)


class TestCreateSampleIdentifier:
    """Tests for create_sample_identifier function."""

    def test_create_identifier_default_columns(self, create_test_root_core_data):
        """Test creating identifiers with default column configuration."""
        df = create_test_root_core_data
        result = create_sample_identifier(df)

        # Check it returns a Series
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

        # Check format matches expected pattern
        expected_first = "plot1_rep1_GH_7386_core1"
        assert result.iloc[0] == expected_first

        expected_last = "plot2_rep1_GH_7418_core3"
        assert result.iloc[-1] == expected_last

    def test_create_identifier_custom_columns(self, create_test_root_core_data):
        """Test creating identifiers with custom column list."""
        df = create_test_root_core_data
        result = create_sample_identifier(df, columns=["Plot", "geno", "core_n"])

        # Should only include specified columns
        expected_first = "plot1_GH_7386_core1"
        assert result.iloc[0] == expected_first

    def test_create_identifier_custom_prefix_map(self, create_test_root_core_data):
        """Test creating identifiers with custom prefix mapping."""
        df = create_test_root_core_data
        prefix_map = {"Plot": "p", "Rep": "r", "geno": "g", "core_n": "c"}
        result = create_sample_identifier(df, prefix_map=prefix_map)

        expected_first = "p1_r1_gGH_7386_c1"
        assert result.iloc[0] == expected_first

    def test_create_identifier_missing_column(self, create_test_root_core_data):
        """Test error handling when required column is missing."""
        df = create_test_root_core_data.drop(columns=["Rep"])

        with pytest.raises(ValueError, match="Missing required column"):
            create_sample_identifier(df)

    def test_create_identifier_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=["Plot", "Rep", "geno", "core_n"])
        result = create_sample_identifier(df)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_create_identifier_with_float_columns(self):
        """Test that float columns (Rep, Plot, core_n) are converted to int without decimals."""
        # Simulate data loaded from CSV where numeric columns become float
        df = pd.DataFrame(
            {
                "Plot": [1.0, 2.0, 1.0],  # float64
                "Rep": [1.0, 1.0, 2.0],  # float64
                "geno": ["Control", "GH_7386", "Control"],
                "core_n": [1.0, 2.0, 1.0],  # float64
            }
        )

        result = create_sample_identifier(df)

        # Should NOT have ".0" in the identifiers
        assert result.iloc[0] == "plot1_rep1_Control_core1"
        assert result.iloc[1] == "plot2_rep1_GH_7386_core2"
        assert result.iloc[2] == "plot1_rep2_Control_core1"

        # Verify no ".0" appears in any identifier
        assert not any(".0" in id_str for id_str in result)


class TestValidateUniqueIdentifiers:
    """Tests for validate_unique_identifiers function."""

    def test_all_identifiers_unique(self, create_test_root_core_data):
        """Test when all identifiers are unique."""
        df = create_test_root_core_data
        df["sample_id"] = create_sample_identifier(df)

        is_unique, duplicates = validate_unique_identifiers(df, "sample_id")

        assert is_unique is True
        assert duplicates == []

    def test_duplicate_identifiers_detected(self):
        """Test detection of duplicate identifiers."""
        df = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2", "sample1", "sample3", "sample2"],
                "value": [1, 2, 3, 4, 5],
            }
        )

        is_unique, duplicates = validate_unique_identifiers(df, "sample_id")

        assert is_unique is False
        assert set(duplicates) == {"sample1", "sample2"}

    def test_empty_dataframe_is_unique(self):
        """Test that empty DataFrame is considered unique (vacuously true)."""
        df = pd.DataFrame(columns=["sample_id"])

        is_unique, duplicates = validate_unique_identifiers(df, "sample_id")

        assert is_unique is True
        assert duplicates == []

    def test_single_row_is_unique(self):
        """Test that single row is always unique."""
        df = pd.DataFrame({"sample_id": ["sample1"]})

        is_unique, duplicates = validate_unique_identifiers(df, "sample_id")

        assert is_unique is True
        assert duplicates == []


class TestMeltDepthData:
    """Tests for melt_depth_data function."""

    def test_melt_with_depth_parsing(self, create_test_root_core_data):
        """Test melting data with automatic depth calculation."""
        df = create_test_root_core_data
        id_vars = ["Plot", "geno", "Rep", "core_n"]

        result = melt_depth_data(
            df, id_vars=id_vars, depth_prefix="c_", parse_depth=True
        )

        # Check it's in long format
        assert "Depth" in result.columns
        assert "Root_Count" in result.columns
        assert "Depth_cm" in result.columns

        # Check all id_vars are preserved
        for col in id_vars:
            assert col in result.columns

        # Check we have the right number of rows
        # 6 rows * 8 depth columns = 48 rows
        assert len(result) == 6 * 8

    def test_depth_calculation_first_subcore_0_10(self, create_test_root_core_data):
        """Test depth calculation for c_0_10_1 (should be 0.0 cm)."""
        df = create_test_root_core_data
        result = melt_depth_data(df, id_vars=["Plot"], parse_depth=True)

        # Filter to c_0_10_1
        c_0_10_1 = result[result["Depth"] == "c_0_10_1"]
        assert (c_0_10_1["Depth_cm"] == 0.0).all()

    def test_depth_calculation_second_subcore_0_10(self, create_test_root_core_data):
        """Test depth calculation for c_0_10_2 (should be 5.0 cm)."""
        df = create_test_root_core_data
        result = melt_depth_data(df, id_vars=["Plot"], parse_depth=True)

        # Filter to c_0_10_2
        c_0_10_2 = result[result["Depth"] == "c_0_10_2"]
        assert (c_0_10_2["Depth_cm"] == 5.0).all()

    def test_depth_calculation_first_subcore_30_40(self, create_test_root_core_data):
        """Test depth calculation for c_30_40_1 (should be 30.0 cm)."""
        df = create_test_root_core_data
        result = melt_depth_data(df, id_vars=["Plot"], parse_depth=True)

        # Filter to c_30_40_1
        c_30_40_1 = result[result["Depth"] == "c_30_40_1"]
        assert (c_30_40_1["Depth_cm"] == 30.0).all()

    def test_depth_calculation_second_subcore_30_40(self, create_test_root_core_data):
        """Test depth calculation for c_30_40_2 (should be 35.0 cm)."""
        df = create_test_root_core_data
        result = melt_depth_data(df, id_vars=["Plot"], parse_depth=True)

        # Filter to c_30_40_2
        c_30_40_2 = result[result["Depth"] == "c_30_40_2"]
        assert (c_30_40_2["Depth_cm"] == 35.0).all()

    def test_melt_without_depth_parsing(self, create_test_root_core_data):
        """Test melting without depth calculation."""
        df = create_test_root_core_data
        result = melt_depth_data(df, id_vars=["Plot"], parse_depth=False)

        # Should not have Depth_cm column
        assert "Depth_cm" not in result.columns
        assert "Depth" in result.columns
        assert "Root_Count" in result.columns

    def test_melt_preserves_nan_values(self):
        """Test that NaN values are preserved during melting."""
        df = pd.DataFrame(
            {"Plot": [1, 2], "c_0_10_1": [50, np.nan], "c_0_10_2": [np.nan, 60]}
        )

        result = melt_depth_data(df, id_vars=["Plot"], parse_depth=False)

        # NaN values should be in the output
        assert result["Root_Count"].isna().sum() == 2

    def test_melt_with_custom_prefix(self):
        """Test melting with custom depth column prefix."""
        df = pd.DataFrame(
            {
                "Plot": [1, 2],
                "depth_0_10_1": [50, 60],
                "depth_0_10_2": [55, 65],
                "other_column": [100, 200],
            }
        )

        result = melt_depth_data(
            df, id_vars=["Plot"], depth_prefix="depth_", parse_depth=False
        )

        # Should only melt columns starting with 'depth_'
        assert len(result) == 4  # 2 rows * 2 depth columns
        assert "other_column" not in result["Depth"].values


class TestAggregateByReplicate:
    """Tests for aggregate_by_replicate function."""

    def test_aggregate_mean_by_plot(self, create_test_root_core_data):
        """Test aggregation using mean across cores within a plot."""
        df = create_test_root_core_data

        # First melt the data
        df_melted = melt_depth_data(
            df, id_vars=["Plot", "geno", "Rep", "core_n"], parse_depth=True
        )

        # Create plot_rep identifier
        df_melted["plot_rep"] = (
            "plot"
            + df_melted["Plot"].astype(str)
            + "_rep"
            + df_melted["Rep"].astype(str)
        )

        # Aggregate by plot_rep, geno, and depth
        result = aggregate_by_replicate(
            df_melted,
            group_cols=["plot_rep", "geno", "Depth_cm"],
            value_col="Root_Count",
            agg_func="mean",
        )

        # Check we have one row per unique group
        assert len(result) == 2 * 8  # 2 plots * 8 depth levels

        # Check group columns are preserved
        assert "plot_rep" in result.columns
        assert "geno" in result.columns
        assert "Depth_cm" in result.columns
        assert "Root_Count" in result.columns

        # Verify known aggregation
        # For plot1, c_0_10_1: mean of [78, 89, 87] = 84.666...
        plot1_0_10_1 = result[
            (result["plot_rep"] == "plot1_rep1") & (result["Depth_cm"] == 0.0)
        ]
        assert len(plot1_0_10_1) == 1
        expected_mean = (78 + 89 + 87) / 3
        assert np.isclose(plot1_0_10_1["Root_Count"].iloc[0], expected_mean)

    def test_aggregate_median(self, create_test_root_core_data):
        """Test aggregation using median instead of mean."""
        df = create_test_root_core_data
        df_melted = melt_depth_data(df, id_vars=["Plot", "core_n"], parse_depth=True)

        result = aggregate_by_replicate(
            df_melted,
            group_cols=["Plot", "Depth_cm"],
            value_col="Root_Count",
            agg_func="median",
        )

        # For plot1, c_0_10_1: median of [78, 89, 87] = 87
        plot1_0_10_1 = result[(result["Plot"] == 1) & (result["Depth_cm"] == 0.0)]
        assert plot1_0_10_1["Root_Count"].iloc[0] == 87

    def test_aggregate_handles_nan(self):
        """Test that aggregation handles NaN values correctly."""
        df = pd.DataFrame(
            {
                "plot_rep": ["plot1_rep1", "plot1_rep1", "plot1_rep1"],
                "Depth_cm": [0.0, 0.0, 0.0],
                "Root_Count": [50, np.nan, 60],
            }
        )

        result = aggregate_by_replicate(
            df,
            group_cols=["plot_rep", "Depth_cm"],
            value_col="Root_Count",
            agg_func="mean",
        )

        # Mean should ignore NaN: (50 + 60) / 2 = 55
        assert result["Root_Count"].iloc[0] == 55

    def test_aggregate_preserves_all_group_columns(self):
        """Test that all specified group columns are in the output."""
        df = pd.DataFrame(
            {
                "plot_rep": ["plot1_rep1"] * 3,
                "Plot": [1] * 3,
                "geno": ["GH_7386"] * 3,
                "Depth_cm": [0.0] * 3,
                "Root_Count": [50, 60, 70],
            }
        )

        result = aggregate_by_replicate(
            df,
            group_cols=["plot_rep", "Plot", "geno", "Depth_cm"],
            value_col="Root_Count",
            agg_func="mean",
        )

        # All group columns should be present
        assert "plot_rep" in result.columns
        assert "Plot" in result.columns
        assert "geno" in result.columns
        assert "Depth_cm" in result.columns
