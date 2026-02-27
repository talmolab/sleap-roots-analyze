"""Tests for data grouping functionality."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.utils import (
    filter_valid_groups,
    split_data_by_group,
)


def test_split_data_by_group_single_column():
    """Test splitting data by a single grouping column."""
    # Create sample data with plant_age_days column
    data = {
        "Barcode": ["p1", "p2", "p3", "p4", "p5", "p6"],
        "Genotype": ["A", "B", "A", "B", "A", "B"],
        "plant_age_days": [7, 7, 14, 14, 21, 21],
        "trait1": [10.0, 12.0, 15.0, 18.0, 20.0, 22.0],
        "trait2": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    }
    df = pd.DataFrame(data)

    # Split by plant_age_days
    groups = split_data_by_group(df, group_by_column="plant_age_days")

    # Should return dict with 3 groups
    assert isinstance(groups, dict)
    assert len(groups) == 3
    assert 7 in groups
    assert 14 in groups
    assert 21 in groups

    # Check group contents
    assert len(groups[7]) == 2  # p1, p2
    assert len(groups[14]) == 2  # p3, p4
    assert len(groups[21]) == 2  # p5, p6

    # Verify data integrity
    assert groups[7]["Barcode"].tolist() == ["p1", "p2"]
    assert groups[14]["plant_age_days"].unique().tolist() == [14]
    assert groups[21]["trait1"].tolist() == [20.0, 22.0]


def test_split_data_by_group_preserves_all_columns():
    """Test that splitting preserves all columns in each group."""
    data = {
        "Barcode": ["p1", "p2"],
        "Genotype": ["A", "B"],
        "Replicate": [1, 1],
        "plant_age_days": [7, 14],
        "trait1": [10.0, 15.0],
        "trait2": [5.0, 7.0],
    }
    df = pd.DataFrame(data)

    groups = split_data_by_group(df, group_by_column="plant_age_days")

    # Each group should have all columns
    for group_value, group_df in groups.items():
        assert set(group_df.columns) == set(df.columns)


def test_split_data_by_group_string_values():
    """Test splitting by a column with string values."""
    data = {
        "Barcode": ["p1", "p2", "p3"],
        "experiment_id": ["exp1", "exp2", "exp1"],
        "trait1": [10.0, 12.0, 11.0],
    }
    df = pd.DataFrame(data)

    groups = split_data_by_group(df, group_by_column="experiment_id")

    assert len(groups) == 2
    assert "exp1" in groups
    assert "exp2" in groups
    assert len(groups["exp1"]) == 2
    assert len(groups["exp2"]) == 1


def test_split_data_by_group_missing_column_raises_error():
    """Test that splitting by non-existent column raises KeyError."""
    data = {
        "Barcode": ["p1", "p2"],
        "trait1": [10.0, 12.0],
    }
    df = pd.DataFrame(data)

    with pytest.raises(KeyError, match="nonexistent_column"):
        split_data_by_group(df, group_by_column="nonexistent_column")


def test_split_data_by_group_empty_dataframe():
    """Test splitting an empty DataFrame returns empty dict."""
    df = pd.DataFrame(columns=["Barcode", "plant_age_days", "trait1"])

    groups = split_data_by_group(df, group_by_column="plant_age_days")

    assert groups == {}


def test_split_data_by_group_single_group():
    """Test that data with single unique value returns one group."""
    data = {
        "Barcode": ["p1", "p2", "p3"],
        "plant_age_days": [7, 7, 7],
        "trait1": [10.0, 12.0, 11.0],
    }
    df = pd.DataFrame(data)

    groups = split_data_by_group(df, group_by_column="plant_age_days")

    assert len(groups) == 1
    assert 7 in groups
    assert len(groups[7]) == 3


# Group validation tests


def test_filter_valid_groups_removes_small_groups():
    """Test that groups with too few samples are filtered out."""
    # Create groups with different sizes
    groups = {
        7: pd.DataFrame({"Barcode": ["p1", "p2"], "trait1": [10.0, 12.0]}),  # 2 samples
        14: pd.DataFrame(
            {"Barcode": ["p3", "p4", "p5"], "trait1": [15.0, 18.0, 20.0]}
        ),  # 3 samples
        21: pd.DataFrame(
            {
                "Barcode": ["p6", "p7", "p8", "p9", "p10", "p11"],
                "trait1": [20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
            }
        ),  # 6 samples
    }

    # Filter with min_samples=3
    valid_groups, skipped_groups = filter_valid_groups(
        groups, min_samples=3, group_column_name="plant_age_days"
    )

    # Should keep groups with >= 3 samples
    assert len(valid_groups) == 2
    assert 14 in valid_groups
    assert 21 in valid_groups
    assert 7 not in valid_groups

    # Should track skipped groups
    assert len(skipped_groups) == 1
    assert 7 in skipped_groups
    assert skipped_groups[7] == 2  # Had 2 samples


def test_filter_valid_groups_keeps_all_when_sufficient():
    """Test that all groups are kept when all meet minimum."""
    groups = {
        7: pd.DataFrame(
            {"Barcode": ["p1", "p2", "p3", "p4", "p5"], "trait1": [1, 2, 3, 4, 5]}
        ),
        14: pd.DataFrame(
            {"Barcode": ["p6", "p7", "p8", "p9", "p10"], "trait1": [6, 7, 8, 9, 10]}
        ),
    }

    valid_groups, skipped_groups = filter_valid_groups(
        groups, min_samples=3, group_column_name="plant_age_days"
    )

    assert len(valid_groups) == 2
    assert len(skipped_groups) == 0


def test_filter_valid_groups_logs_warnings(caplog):
    """Test that warnings are logged for skipped groups."""
    groups = {
        7: pd.DataFrame({"Barcode": ["p1", "p2"], "trait1": [10.0, 12.0]}),  # Too small
        14: pd.DataFrame(
            {"Barcode": ["p3", "p4", "p5", "p6"], "trait1": [15.0, 18.0, 20.0, 22.0]}
        ),
    }

    with caplog.at_level(logging.WARNING):
        valid_groups, skipped_groups = filter_valid_groups(
            groups, min_samples=3, group_column_name="plant_age_days"
        )

    # Check warning was logged
    assert len(caplog.records) == 1
    assert "Skipping group" in caplog.text
    assert "plant_age_days=7" in caplog.text
    assert "2 samples < 3 minimum" in caplog.text


def test_filter_valid_groups_empty_dict():
    """Test filtering empty group dict returns empty results."""
    valid_groups, skipped_groups = filter_valid_groups(
        {}, min_samples=3, group_column_name="plant_age_days"
    )

    assert valid_groups == {}
    assert skipped_groups == {}
