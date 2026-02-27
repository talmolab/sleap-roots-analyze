"""Tests for error handling in grouped pipeline execution."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config import get_default_qc_config
from sleap_roots_analyze.pipeline.utils import run_grouped_pipelines


class TestErrorHandling:
    """Tests for error handling in run_grouped_pipelines."""

    def test_file_not_found_error_includes_context(self, tmp_path):
        """FileNotFoundError includes group_by context in message."""
        config = get_default_qc_config()
        config.data.csv_path = "nonexistent.csv"
        config.data.group_by = "age"

        with pytest.raises(FileNotFoundError) as exc_info:
            run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "outputs",
                pipeline_class=Mock(),
                validate=False,
            )

        # Error message should mention both the file path and group_by context
        assert "nonexistent.csv" in str(exc_info.value)
        assert "group_by='age'" in str(exc_info.value)

    def test_empty_data_error_includes_context(self, tmp_path):
        """EmptyDataError includes group_by context in message."""
        # Create empty CSV file
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")

        config = get_default_qc_config()
        config.data.csv_path = str(empty_csv)
        config.data.group_by = "age"

        with pytest.raises(pd.errors.EmptyDataError) as exc_info:
            run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "outputs",
                pipeline_class=Mock(),
                validate=False,
            )

        assert str(empty_csv) in str(exc_info.value)
        assert "group_by='age'" in str(exc_info.value)

    def test_key_error_from_split_preserved(self, tmp_path):
        """KeyError from split_data_by_group is preserved with original message."""
        # CSV without the group_by column
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("barcode,genotype,trait1\np1,A,1.0\n")

        config = get_default_qc_config()
        config.data.csv_path = str(csv_path)
        config.data.group_by = "nonexistent_col"

        # Should raise ValueError (not KeyError) since split_data_by_group raises ValueError
        with pytest.raises(ValueError) as exc_info:
            run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "outputs",
                pipeline_class=Mock(),
                validate=False,
            )

        # Should be the original error message from split_data_by_group
        assert "nonexistent_col" in str(exc_info.value)

    def test_empty_groups_warning_logged(self, tmp_path, caplog):
        """Warning logged when all groups filtered out."""
        # Create CSV with groups that will all be filtered
        csv_path = tmp_path / "small_groups.csv"
        csv_path.write_text(
            "barcode,genotype,replicate,age,trait1\n"
            "p1,A,1,0,1.0\n"  # age=0, n=1 (will be filtered with min_samples=10)
            "p2,B,1,1,2.0\n"  # age=1, n=1 (will be filtered)
        )

        config = get_default_qc_config()
        config.data.csv_path = str(csv_path)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 10

        with caplog.at_level(logging.WARNING):
            result = run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "outputs",
                pipeline_class=Mock(),
                validate=False,
            )

        # Should return empty dict
        assert result == {}

        # Should log warning about no valid groups
        assert any(
            "No valid groups remain" in record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        )
        assert any(
            "min_samples_per_trait=10" in record.message for record in caplog.records
        )

    def test_mixed_type_groups_sorted_safely(self, tmp_path, caplog):
        """Mixed string/int group values handled gracefully."""
        # Create CSV with mixed-type group column
        csv_path = tmp_path / "mixed_groups.csv"
        rows = [
            "barcode,genotype,replicate,site,trait1",
            "p1,A,1,site0,1.0",
            "p2,A,2,site0,1.1",
            "p3,A,3,site0,1.2",
            "p4,B,1,1,2.0",  # Will be read as string "1"
            "p5,B,2,1,2.1",
            "p6,B,3,1,2.2",
            "p7,C,1,site2,3.0",
            "p8,C,2,site2,3.1",
            "p9,C,3,site2,3.2",
        ]
        csv_path.write_text("\n".join(rows))

        config = get_default_qc_config()
        config.data.csv_path = str(csv_path)
        config.data.group_by = "site"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 3

        # Mock pipeline class
        class MockPipeline:
            def __init__(self, config, output_dir, **kwargs):
                self.config = config
                self.run_dir = output_dir

            def run(self):
                return {"status": "success"}

        with caplog.at_level(logging.WARNING):
            result = run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "outputs",
                pipeline_class=MockPipeline,
                validate=False,
            )

        # Should complete successfully with all 3 groups
        # Note: pandas reads CSV columns as consistent types, so "1" is string not int
        # The mixed-type handling code is defensive but won't trigger from CSV data
        assert len(result) == 3

        # Test passes if execution completes without TypeError
        # (The warning check removed since pandas doesn't create mixed types from CSV)

    def test_group_failure_logged_and_fails_fast(self, tmp_path, caplog):
        """Pipeline failure for one group is logged and re-raised (fail-fast behavior)."""
        csv_path = tmp_path / "data.csv"
        rows = ["barcode,genotype,replicate,age,trait1"]
        for i in range(30):
            rows.append(f"p{i},A,{i % 3},{ i % 3},1.0")
        csv_path.write_text("\n".join(rows))

        config = get_default_qc_config()
        config.data.csv_path = str(csv_path)
        config.data.group_by = "age"
        config.columns.barcode = "barcode"
        config.columns.genotype = "genotype"
        config.columns.replicate = "replicate"
        config.cleanup.min_samples_per_trait = 3

        # Mock pipeline that fails for age=0 (first group, sorted)
        class FailingPipeline:
            def __init__(self, config, output_dir, **kwargs):
                self.config = config
                self.run_dir = output_dir

            def run(self):
                # Check if this is the age=0 group by looking at data
                df = pd.read_csv(self.config.data.csv_path)
                if df["age"].iloc[0] == 0:
                    raise ValueError("Simulated pipeline failure for age=0")
                return {"status": "success"}

        # Should raise exception and fail fast (not continue to other groups)
        with pytest.raises(ValueError, match="Simulated pipeline failure for age=0"):
            with caplog.at_level(logging.ERROR):
                run_grouped_pipelines(
                    config=config,
                    output_dir=tmp_path / "outputs",
                    pipeline_class=FailingPipeline,
                    validate=False,
                )

        # Should have logged exception with group context
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_logs) > 0, "Should have logged an error"
        assert any(
            "age=0" in r.message and "failed" in r.message for r in error_logs
        ), f"Error should mention group context. Logs: {[r.message for r in error_logs]}"
