"""Tests for NaN handling in grouped pipeline execution.

These tests verify that NaN group values are handled explicitly (not silently dropped)
with full traceability.

Bug: split_data_by_group used `df[df[col] == value]` which doesn't match NaN rows.
Samples with missing group labels were silently excluded without warning or tracing.

Fix: Explicitly check for NaN values, log warnings, save dropped samples to CSV,
and track in pipeline metadata.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config.utils import get_default_qc_config
from sleap_roots_analyze.pipeline.pipelines.qc_pipeline import QCPipeline
from sleap_roots_analyze.pipeline.utils import run_grouped_pipelines, split_data_by_group


@pytest.fixture
def test_data_with_nans(tmp_path):
    """Create test CSV with NaN values in group column."""
    csv_path = tmp_path / "data.csv"
    rows = [
        "barcode,genotype,replicate,age,trait1,trait2",
        # age=7 group (10 samples)
        "p0,A,1,7,1.0,2.0",
        "p1,A,2,7,1.1,2.1",
        "p2,A,3,7,1.2,2.2",
        "p3,A,1,7,1.3,2.3",
        "p4,A,2,7,1.4,2.4",
        "p5,A,3,7,1.5,2.5",
        "p6,A,1,7,1.6,2.6",
        "p7,A,2,7,1.7,2.7",
        "p8,A,3,7,1.8,2.8",
        "p9,A,1,7,1.9,2.9",
        # age=14 group (10 samples)
        "p10,B,2,14,3.0,4.0",
        "p11,B,3,14,3.1,4.1",
        "p12,B,1,14,3.2,4.2",
        "p13,B,2,14,3.3,4.3",
        "p14,B,3,14,3.4,4.4",
        "p15,B,1,14,3.5,4.5",
        "p16,B,2,14,3.6,4.6",
        "p17,B,3,14,3.7,4.7",
        "p18,B,1,14,3.8,4.8",
        "p19,B,2,14,3.9,4.9",
        # NaN age (missing values - 5 samples)
        "p20,C,3,,5.0,6.0",
        "p21,C,1,,5.1,6.1",
        "p22,C,2,,5.2,6.2",
        "p23,C,3,,5.3,6.3",
        "p24,C,1,,5.4,6.4",
    ]
    csv_path.write_text("\n".join(rows))
    return csv_path


class TestSplitDataByGroupNaNHandling:
    """Test split_data_by_group NaN handling directly."""

    def test_nan_group_values_logged_and_dropped_by_default(
        self, test_data_with_nans, caplog
    ):
        """Samples with NaN group values should be logged and dropped (default behavior)."""
        df = pd.read_csv(test_data_with_nans)
        assert df["age"].isna().sum() == 5, "Test setup: should have 5 NaN rows"

        # Split by age column
        with caplog.at_level(logging.WARNING):
            groups = split_data_by_group(df, group_by_column="age")

        # Should log warning about NaN values
        warning_found = any(
            "Dropping 5/25 samples with missing 'age' values" in record.message
            for record in caplog.records
        )
        assert warning_found, (
            "Should log warning when dropping NaN values.\n"
            f"Log records: {[r.message for r in caplog.records]}"
        )

        # NaN rows should NOT be in any group
        all_group_rows = sum(len(group_df) for group_df in groups.values())
        assert all_group_rows == 20, (
            "Only 20 non-NaN rows should be in groups (5 NaN rows dropped)"
        )

        # Verify specific groups
        assert len(groups) == 2, "Should have 2 groups (7.0 and 14.0)"

        # Groups might be float or string depending on pandas version
        age_7_group = groups.get(7.0) or groups.get(7) or groups.get("7.0")
        age_14_group = groups.get(14.0) or groups.get(14) or groups.get("14.0")

        assert age_7_group is not None, "Should have age=7 group"
        assert age_14_group is not None, "Should have age=14 group"
        assert len(age_7_group) == 10
        assert len(age_14_group) == 10

    def test_nan_handling_with_treat_as_group_option(self, test_data_with_nans):
        """Optional: NaN values can be treated as their own group."""
        df = pd.read_csv(test_data_with_nans)

        # Split with handle_na="treat_as_group"
        groups = split_data_by_group(
            df,
            group_by_column="age",
            handle_na="treat_as_group"
        )

        # Should have 3 groups: 7.0, 14.0, and NaN
        assert len(groups) == 3, (
            "Should have 3 groups when treating NaN as a group"
        )

        # Check for NaN group (pandas represents NaN key in groupby)
        has_nan_group = any(pd.isna(k) for k in groups.keys())
        assert has_nan_group, "Should have a group for NaN values"

        # All 25 rows should be in groups (none dropped)
        all_group_rows = sum(len(group_df) for group_df in groups.values())
        assert all_group_rows == 25


class TestGroupedPipelineNaNTraceability:
    """Test that dropped NaN samples are fully traceable."""

    def test_dropped_nan_samples_saved_to_csv(
        self, test_data_with_nans, tmp_path, caplog
    ):
        """Dropped NaN samples must be saved to a CSV file."""
        config = get_default_qc_config()
        config.data.csv_path = str(test_data_with_nans)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 5

        # Mock pipeline class (skip actual execution)
        class MockPipeline:
            def __init__(self, config, output_dir, **kwargs):
                self.config = config
                self.run_dir = output_dir

            def run(self):
                return {"status": "success"}

        with caplog.at_level(logging.WARNING):
            result = run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "output",
                pipeline_class=MockPipeline,
                validate=False,
            )

        output_dir = tmp_path / "output"

        # Check for dropped samples CSV
        dropped_csv = output_dir / "00_dropped_samples_missing_age.csv"
        assert dropped_csv.exists(), (
            "Dropped samples CSV must be saved for traceability"
        )

        # Verify dropped CSV contains exactly the NaN rows
        dropped_df = pd.read_csv(dropped_csv)
        assert len(dropped_df) == 5, "Should have 5 dropped rows"

        # Verify correct barcodes
        expected_barcodes = {"p20", "p21", "p22", "p23", "p24"}
        actual_barcodes = set(dropped_df["barcode"])
        assert actual_barcodes == expected_barcodes, (
            f"Dropped CSV should contain barcodes {expected_barcodes}, "
            f"but has {actual_barcodes}"
        )

        # Should log warning about dropped samples
        warning_found = any(
            "Dropped 5/25 samples" in record.message
            and "age" in record.message
            for record in caplog.records
        )
        assert warning_found, "Should log warning about dropped samples"

    def test_dropped_samples_metadata_file_created(
        self, test_data_with_nans, tmp_path
    ):
        """Metadata file explaining dropped samples must be created."""
        config = get_default_qc_config()
        config.data.csv_path = str(test_data_with_nans)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 5

        class MockPipeline:
            def __init__(self, config, output_dir, **kwargs):
                self.config = config
                self.run_dir = output_dir

            def run(self):
                return {"status": "success"}

        run_grouped_pipelines(
            config=config,
            output_dir=tmp_path / "output",
            pipeline_class=MockPipeline,
            validate=False,
        )

        output_dir = tmp_path / "output"

        # Check for metadata file
        metadata_file = output_dir / "00_dropped_samples_missing_age.txt"
        assert metadata_file.exists(), (
            "Metadata file explaining dropped samples must be created"
        )

        # Verify metadata content
        metadata_text = metadata_file.read_text()
        assert "5/25 samples" in metadata_text
        assert "missing 'age'" in metadata_text or "Missing values in" in metadata_text

        # Should list the dropped barcodes
        for barcode in ["p20", "p21", "p22", "p23", "p24"]:
            assert barcode in metadata_text, (
                f"Metadata should list dropped barcode {barcode}"
            )

    def test_dropped_samples_tracked_in_summary(
        self, test_data_with_nans, tmp_path
    ):
        """Pipeline summary must include dropped sample metadata."""
        config = get_default_qc_config()
        config.data.csv_path = str(test_data_with_nans)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 5

        class MockPipeline:
            def __init__(self, config, output_dir, **kwargs):
                self.config = config
                self.run_dir = output_dir

            def run(self):
                # Return result that mimics real pipeline
                return {
                    "status": "success",
                    "metadata": {},
                }

        result = run_grouped_pipelines(
            config=config,
            output_dir=tmp_path / "output",
            pipeline_class=MockPipeline,
            validate=False,
        )

        # Each group result should have dropped_samples metadata
        for group_label, group_result in result.items():
            assert "dropped_samples" in group_result, (
                f"Group {group_label} result should have dropped_samples metadata"
            )

            dropped_info = group_result["dropped_samples"]
            assert dropped_info["column"] == "age"
            assert dropped_info["count"] == 5
            assert dropped_info["total"] == 25
            assert dropped_info["fraction"] == 5 / 25

            # Check CSV and metadata paths exist
            assert dropped_info["csv_path"] is not None
            assert Path(dropped_info["csv_path"]).exists()
            assert dropped_info["metadata_path"] is not None
            assert Path(dropped_info["metadata_path"]).exists()

    def test_no_dropped_samples_when_no_nans(self, tmp_path):
        """When there are no NaN values, no dropped sample files should be created."""
        # Create CSV without NaN values
        csv_path = tmp_path / "data_no_nans.csv"
        rows = [
            "barcode,genotype,replicate,age,trait1",
            "p1,A,1,7,1.0",
            "p2,A,2,7,1.1",
            "p3,A,3,7,1.2",
            "p4,B,1,14,2.0",
            "p5,B,2,14,2.1",
            "p6,B,3,14,2.2",
            "p7,C,1,21,3.0",
            "p8,C,2,21,3.1",
            "p9,C,3,21,3.2",
            "p10,C,1,21,3.3",
        ]
        csv_path.write_text("\n".join(rows))

        config = get_default_qc_config()
        config.data.csv_path = str(csv_path)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 3

        class MockPipeline:
            def __init__(self, config, output_dir, **kwargs):
                self.config = config
                self.run_dir = output_dir

            def run(self):
                return {"status": "success"}

        result = run_grouped_pipelines(
            config=config,
            output_dir=tmp_path / "output",
            pipeline_class=MockPipeline,
            validate=False,
        )

        output_dir = tmp_path / "output"

        # Should NOT create dropped samples files
        dropped_files = list(output_dir.glob("00_dropped_samples_*.csv"))
        assert len(dropped_files) == 0, (
            "Should not create dropped samples files when there are no NaN values"
        )

        # Dropped samples metadata should indicate 0 dropped
        for group_label, group_result in result.items():
            dropped_info = group_result["dropped_samples"]
            assert dropped_info["count"] == 0
            assert dropped_info["csv_path"] is None
