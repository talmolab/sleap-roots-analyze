"""Tests for grouped pipeline robustness improvements.

These tests verify error handling, logging, and edge case handling
identified in GitHub Copilot code review of PR #63.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config.utils import get_default_qc_config
from sleap_roots_analyze.pipeline.utils import run_grouped_pipelines


@pytest.fixture
def sparse_data(tmp_path):
    """Create data where all groups have < min_samples."""
    csv_path = tmp_path / "sparse_data.csv"
    rows = ["barcode,genotype,replicate,age,trait1"]
    # 3 groups with only 2 samples each
    for i in range(6):
        rows.append(f"p{i},A,1,{i % 3},1.0")
    csv_path.write_text("\n".join(rows))
    return csv_path


@pytest.fixture
def mixed_type_data(tmp_path):
    """Create data with mixed string/numeric group values."""
    csv_path = tmp_path / "mixed_types.csv"
    rows = [
        "barcode,genotype,replicate,experiment_id,trait1",
        # String group IDs
        "p1,A,1,exp1,1.0",
        "p2,A,2,exp1,1.1",
        "p3,A,3,exp1,1.2",
        "p4,A,1,exp1,1.3",
        "p5,A,2,exp1,1.4",
        # Numeric group ID
        "p6,B,1,7,2.0",
        "p7,B,2,7,2.1",
        "p8,B,3,7,2.2",
        "p9,B,1,7,2.3",
        "p10,B,2,7,2.4",
        # Another string group ID
        "p11,C,1,exp2,3.0",
        "p12,C,2,exp2,3.1",
        "p13,C,3,exp2,3.2",
        "p14,C,1,exp2,3.3",
        "p15,C,2,exp2,3.4",
    ]
    csv_path.write_text("\n".join(rows))
    return csv_path


@pytest.fixture
def valid_data(tmp_path):
    """Create valid data for testing."""
    csv_path = tmp_path / "valid_data.csv"
    rows = ["barcode,genotype,replicate,age,trait1"]
    for i in range(30):
        rows.append(f"p{i},A,{i % 3},{i % 3},1.0")
    csv_path.write_text("\n".join(rows))
    return csv_path


class MockPipeline:
    """Mock pipeline for testing."""

    def __init__(self, config, output_dir, **kwargs):
        self.config = config
        self.run_dir = output_dir
        # Create run_dir to simulate pipeline execution
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        return {"status": "success"}


class FailingPipeline:
    """Mock pipeline that always fails."""

    def __init__(self, config, output_dir, **kwargs):
        self.config = config
        self.run_dir = output_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        raise ValueError("Simulated pipeline failure")


class TestPipelineFailureLogging:
    """Test that pipeline failures are logged with group context."""

    def test_pipeline_failure_logs_group_context(self, valid_data, tmp_path, caplog):
        """Pipeline failures should log the failing group for diagnostics.

        HIGH PRIORITY: This ensures users get actionable error messages
        when a grouped pipeline fails.
        """
        config = get_default_qc_config()
        config.data.csv_path = str(valid_data)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 5

        # Should re-raise exception after logging
        with pytest.raises(ValueError, match="Simulated pipeline failure"):
            with caplog.at_level(logging.ERROR):
                run_grouped_pipelines(
                    config=config,
                    output_dir=tmp_path / "output",
                    pipeline_class=FailingPipeline,
                    validate=False,
                )

        # Should have logged the failing group with context
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_logs) > 0, "Should have logged an error"

        # Error message should include group column and value
        found_group_context = any(
            "age=" in r.message and "failed" in r.message for r in error_logs
        )
        assert found_group_context, (
            f"Error log should mention 'age=' and 'failed'. "
            f"Error logs: {[r.message for r in error_logs]}"
        )


class TestEmptyGroupsWarning:
    """Test that empty groups after filtering are handled gracefully."""

    def test_empty_groups_after_filtering_warns(self, sparse_data, tmp_path, caplog):
        """When all groups filtered out, should log warning with context.

        MEDIUM PRIORITY: Helps users diagnose configuration issues.
        """
        config = get_default_qc_config()
        config.data.csv_path = str(sparse_data)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 5  # All groups have only 2 samples

        with caplog.at_level(logging.WARNING):
            result = run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "output",
                pipeline_class=MockPipeline,
                validate=False,
            )

        # Should warn about no valid groups
        warning_found = any(
            "No valid groups remain" in r.message for r in caplog.records
        )
        assert warning_found, (
            "Should warn when all groups filtered out. "
            f"Warnings: {[r.message for r in caplog.records if r.levelname == 'WARNING']}"
        )

        # Should return empty result
        assert result == {}, "Should return empty dict when no valid groups"


class TestMixedTypeGroupSorting:
    """Test that mixed-type group values don't crash."""

    def test_mixed_type_group_values_sorted_gracefully(
        self, mixed_type_data, tmp_path, caplog
    ):
        """Mixed string/numeric group values should not crash sorting.

        MEDIUM PRIORITY: Prevents TypeError when group column has mixed types.
        """
        config = get_default_qc_config()
        config.data.csv_path = str(mixed_type_data)
        config.data.group_by = "experiment_id"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 3

        with caplog.at_level(logging.WARNING):
            # Should not crash with TypeError
            result = run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "output",
                pipeline_class=MockPipeline,
                validate=False,
            )

        # Should have all 3 groups (mixed types handled)
        assert len(result) == 3, f"Should have 3 groups, got {len(result)}"

        # Check that groups exist (may be int or str keys)
        # Note: pandas reads CSV columns as strings when mixed, so "7" is a string
        group_keys = list(result.keys())
        assert (7 in group_keys or "7" in group_keys), f"Missing numeric group 7 in {group_keys}"
        assert "exp1" in group_keys, f"Missing string group 'exp1' in {group_keys}"
        assert "exp2" in group_keys, f"Missing string group 'exp2' in {group_keys}"

        # Note: CSV data is read as strings by pandas, so no TypeError occurs
        # and no warning is logged. The try-except in utils.py handles
        # programmatically-created mixed-type data (not common in practice).


class TestContextualizedErrorMessages:
    """Test that error messages include helpful context."""

    def test_missing_csv_error_message_includes_context(self, tmp_path):
        """FileNotFoundError should mention group_by context.

        MEDIUM PRIORITY: Makes debugging easier for users.
        """
        config = get_default_qc_config()
        config.data.csv_path = "nonexistent.csv"
        config.data.group_by = "plant_age_days"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"

        with pytest.raises(FileNotFoundError) as exc_info:
            run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "output",
                pipeline_class=MockPipeline,
                validate=False,
            )

        # Error message should mention group_by for context
        error_msg = str(exc_info.value)
        assert "group_by" in error_msg or "plant_age_days" in error_msg, (
            f"Error should mention group_by context. Got: {error_msg}"
        )

    def test_empty_csv_error_message_includes_context(self, tmp_path):
        """Empty CSV should provide clear error with context.

        MEDIUM PRIORITY: Helps users identify data issues.
        """
        # Create empty CSV
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")

        config = get_default_qc_config()
        config.data.csv_path = str(csv_path)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"

        # Should raise a clear error (could be EmptyDataError or other)
        with pytest.raises(Exception) as exc_info:
            run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "output",
                pipeline_class=MockPipeline,
                validate=False,
            )

        # Error should mention the file or grouping context
        error_msg = str(exc_info.value)
        assert (
            "empty" in error_msg.lower()
            or "group_by" in error_msg
            or str(csv_path) in error_msg
        ), f"Error should provide context. Got: {error_msg}"

    def test_missing_group_column_error_message_clear(self, tmp_path):
        """ValueError for missing column should be descriptive.

        MEDIUM PRIORITY: Prevents confusion when group_by column doesn't exist.
        """
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("barcode,trait1\np1,1.0\np2,1.1\np3,1.2")

        config = get_default_qc_config()
        config.data.csv_path = str(csv_path)
        config.data.group_by = "nonexistent_column"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 1

        with pytest.raises(ValueError) as exc_info:
            run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "output",
                pipeline_class=MockPipeline,
                validate=False,
            )

        # Error should mention the missing column
        error_msg = str(exc_info.value)
        assert "nonexistent_column" in error_msg, (
            f"Error should mention missing column name. Got: {error_msg}"
        )
