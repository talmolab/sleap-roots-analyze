"""Tests for grouped pipeline configuration persistence and reproducibility.

These tests verify that grouped pipelines save configs with valid,
persistent CSV paths that enable reproducibility.

Bug: Grouped pipelines were writing group CSVs to temp files, updating
config.csv_path to point to the temp file, then deleting it. This made
saved configs non-reproducible.

Fix: Write group CSVs to the output directory as persistent files.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config.utils import (
    get_default_qc_config,
    load_qc_config,
    save_qc_config,
)
from sleap_roots_analyze.pipeline.pipelines.qc_pipeline import QCPipeline
from sleap_roots_analyze.pipeline.utils import run_grouped_pipelines


@pytest.fixture
def grouped_test_data(tmp_path):
    """Create test CSV with multiple groups."""
    csv_path = tmp_path / "data.csv"
    rows = ["barcode,genotype,replicate,age,trait1,trait2"]

    # Group age=7 (10 samples)
    for i in range(10):
        rows.append(f"p{i},A,{i % 3},7,{1.0 + i * 0.1},{2.0 + i * 0.1}")

    # Group age=14 (10 samples)
    for i in range(10, 20):
        rows.append(f"p{i},B,{i % 3},14,{3.0 + i * 0.1},{4.0 + i * 0.1}")

    # Group age=21 (10 samples)
    for i in range(20, 30):
        rows.append(f"p{i},C,{i % 3},21,{5.0 + i * 0.1},{6.0 + i * 0.1}")

    csv_path.write_text("\n".join(rows))
    return csv_path


class TestGroupedPipelineConfigPersistence:
    """Test that grouped pipelines save configs with valid, persistent CSV paths."""

    pytestmark = pytest.mark.slow

    def test_saved_config_csv_path_exists(self, grouped_test_data, tmp_path):
        """Saved config.yaml must reference an existing CSV file.

        Bug: Previously, config pointed to deleted temp file.
        Fix: Config now points to persistent file in output directory.
        """
        config = get_default_qc_config()
        config.data.csv_path = str(grouped_test_data)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 5

        # Run grouped pipelines
        result = run_grouped_pipelines(
            config=config,
            output_dir=tmp_path / "output",
            pipeline_class=QCPipeline,
            validate=False,
        )

        # For each group, verify saved config points to existing file
        assert len(result) == 3, "Should have 3 groups (ages 7, 14, 21)"

        for group_label, group_result in result.items():
            # Extract run_dir from result structure
            run_dir = Path(group_result["output_dir"])
            saved_config_path = run_dir / "config.yaml"
            assert (
                saved_config_path.exists()
            ), f"Group {group_label} should have saved config"

            # Load saved config
            saved_config = load_qc_config(saved_config_path)

            # CRITICAL: csv_path must exist
            csv_file = Path(saved_config.data.csv_path)
            assert csv_file.exists(), (
                f"Saved config for group {group_label} references non-existent CSV: "
                f"{csv_file}\n"
                f"This breaks reproducibility - cannot re-run using saved config."
            )

            # CSV should be in the group's output directory (not /tmp)
            assert csv_file.parent == run_dir, (
                f"CSV should be in output directory {run_dir}, "
                f"but is in {csv_file.parent}"
            )

    def test_saved_config_is_reproducible(self, grouped_test_data, tmp_path):
        """Saved config can be used to re-run the exact same analysis.

        This is the ultimate reproducibility test: can we re-run using
        the saved config.yaml without modification?
        """
        config = get_default_qc_config()
        config.data.csv_path = str(grouped_test_data)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 5

        # Run pipeline once
        result1 = run_grouped_pipelines(
            config=config,
            output_dir=tmp_path / "run1",
            pipeline_class=QCPipeline,
            validate=False,
        )

        # Pick one group's saved config
        group_result = list(result1.values())[0]
        run_dir = Path(group_result["output_dir"])
        saved_config_path = run_dir / "config.yaml"

        # Load the saved config
        saved_config = load_qc_config(saved_config_path)

        # CRITICAL: Re-run using ONLY the saved config
        # This is what a user would do to reproduce results
        rerun_output_dir = tmp_path / "rerun"
        pipeline = QCPipeline(
            config=saved_config,
            output_dir=rerun_output_dir,
            validate=False,
        )
        result2 = pipeline.run()

        # Should succeed (config is valid and data file exists)
        # Check that pipeline completed and created output directory
        assert result2 is not None, "Pipeline should return results"
        assert pipeline.run_dir.exists(), (
            "Re-running with saved config failed to create output directory. "
            "This means saved configs are not reproducible."
        )
        # Verify some expected outputs exist
        assert (
            pipeline.run_dir / "pipeline_summary.json"
        ).exists(), "Pipeline should create summary file"

    def test_input_csv_preserved_in_output_directory(self, grouped_test_data, tmp_path):
        """Each group's input CSV is saved in its output directory.

        The group-specific CSV should be saved as a numbered step file
        (e.g., 00_input_data_age_7.csv) for full traceability.
        """
        config = get_default_qc_config()
        config.data.csv_path = str(grouped_test_data)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 5

        # Run grouped pipelines
        result = run_grouped_pipelines(
            config=config,
            output_dir=tmp_path / "output",
            pipeline_class=QCPipeline,
            validate=False,
        )

        for group_label, group_result in result.items():
            # Check for input CSV in output dir
            run_dir = Path(group_result["output_dir"])
            input_csvs = list(run_dir.glob("00_input_data_*.csv"))

            assert len(input_csvs) == 1, (
                f"Expected 1 input CSV in {run_dir}, " f"found {len(input_csvs)}"
            )

            input_csv = input_csvs[0]

            # Verify filename follows pattern
            assert "00_input_data_" in input_csv.name
            assert str(group_label) in input_csv.name

            # Verify it contains only the group's data
            df = pd.read_csv(input_csv)

            # group_label is the age value (7, 14, or 21)
            expected_age = group_label

            # All rows should belong to this group
            assert all(
                df["age"] == expected_age
            ), f"Input CSV should contain only age={expected_age} rows"

            # Should have exactly 10 samples per group
            assert len(df) == 10

    def test_no_temporary_files_in_tmp_dir(
        self, grouped_test_data, tmp_path, monkeypatch
    ):
        """Verify that no temp files are created (bug regression test).

        Previously, the code created temp CSVs in /tmp and then deleted them.
        This test ensures we're no longer using temp files.
        """
        import tempfile

        # Track all temp files created during execution
        original_mkstemp = tempfile.mkstemp
        temp_files_created = []

        def tracking_mkstemp(*args, **kwargs):
            fd, path = original_mkstemp(*args, **kwargs)
            temp_files_created.append(path)
            return fd, path

        monkeypatch.setattr(tempfile, "mkstemp", tracking_mkstemp)

        config = get_default_qc_config()
        config.data.csv_path = str(grouped_test_data)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 5

        # Run grouped pipelines
        run_grouped_pipelines(
            config=config,
            output_dir=tmp_path / "output",
            pipeline_class=QCPipeline,
            validate=False,
        )

        # Should NOT have created any temp files for group CSVs
        # (temp files might be created for other reasons, so we check filenames)
        group_csv_temps = [
            f for f in temp_files_created if "group" in f.lower() or "csv" in f.lower()
        ]

        assert len(group_csv_temps) == 0, (
            f"Should not create temp files for group CSVs. " f"Found: {group_csv_temps}"
        )
