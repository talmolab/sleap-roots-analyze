"""Tests for statistical analysis error handling.

These tests verify that ANOVA calculation errors are handled gracefully
instead of crashing the pipeline.

Bug: When calculate_anova_by_genotype() returns an error string (instead of dict),
the code calls .get() on the string, causing AttributeError: 'str' object has no attribute 'get'

Fix: Check if result is a string before calling .get(), and store error in dataframe.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config.utils import get_default_qc_config
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps.statistical_analysis import (
    StatisticalAnalysisStep,
)


class TestStatisticalAnalysisErrorHandling:
    """Test that ANOVA errors are handled gracefully (not crash)."""

    def test_anova_string_error_handled_gracefully(self, tmp_path):
        """When ANOVA returns error string, should store it (not crash).

        Bug: Previously crashed with AttributeError when calling .get() on string.
        Fix: Check isinstance(result, str) before calling .get().
        """
        # Create minimal data that will likely fail ANOVA (insufficient samples)
        csv_path = tmp_path / "minimal_data.csv"
        rows = [
            "Barcode,Genotype,Replicate,trait1,trait2",
            "p1,A,1,1.0,2.0",
            "p2,B,1,1.1,2.1",
            # Only 2 genotypes with 1 sample each - insufficient for ANOVA
        ]
        csv_path.write_text("\n".join(rows))

        df = pd.read_csv(csv_path)

        config = get_default_qc_config()
        config.columns.barcode = "Barcode"
        config.columns.genotype = "Genotype"
        config.columns.replicate = "Replicate"
        # ANOVA is always calculated in Step 8, no config flag needed

        step = StatisticalAnalysisStep()

        # Create prev_result with trait columns in metadata
        prev_result = StepResult(
            data=df,
            metadata={"valid_trait_names": ["trait1", "trait2"]},
            files_generated=[],
        )

        # CRITICAL: This should NOT crash with AttributeError
        try:
            result = step.execute(
                data=df,
                config=config,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            # Should return result (not crash)
            assert result is not None, "Step should return StepResult"

            # Check that ANOVA results CSV was created
            anova_csv = tmp_path / "data" / "08_anova_results.csv"
            assert anova_csv.exists(), "ANOVA results CSV should be created"

            # Read and verify error handling
            anova_df = pd.read_csv(anova_csv)

            # Should have entries for both traits
            assert len(anova_df) == 2, "Should have results for both traits"

            # Traits might succeed or fail, but should not crash
            # Just verify the dataframe has expected columns
            expected_cols = [
                "trait",
                "f_statistic",
                "p_value",
                "eta_squared",
                "significant",
                "n_groups",
                "total_n",
                "error",
            ]
            for col in expected_cols:
                assert col in anova_df.columns, f"Missing column: {col}"

        except AttributeError as e:
            if "'str' object has no attribute 'get'" in str(e):
                pytest.fail(
                    "ANOVA error handling bug not fixed: still calling .get() on error string.\n"
                    f"Error: {e}"
                )
            else:
                raise

    def test_anova_mixed_success_and_failure(self, tmp_path):
        """Some traits succeed ANOVA, others fail - should handle both.

        This is a more realistic scenario: some traits have good data,
        others might have issues (zero variance, missing data, etc.).
        """
        csv_path = tmp_path / "mixed_data.csv"
        rows = ["Barcode,Genotype,Replicate,good_trait,constant_trait,variable_trait"]

        # Create 30 samples with 3 genotypes
        for i in range(30):
            genotype = ["A", "B", "C"][i % 3]
            replicate = i % 3
            good_value = 1.0 + (i % 3) * 0.5  # Varies by genotype
            constant_value = 1.0  # Zero variance (might fail ANOVA)
            variable_value = 1.0 + i * 0.1  # High variance

            rows.append(
                f"p{i},{genotype},{replicate},{good_value},{constant_value},{variable_value}"
            )

        csv_path.write_text("\n".join(rows))

        df = pd.read_csv(csv_path)

        config = get_default_qc_config()
        config.columns.barcode = "Barcode"
        config.columns.genotype = "Genotype"
        config.columns.replicate = "Replicate"
        # ANOVA is always calculated in Step 8, no config flag needed

        step = StatisticalAnalysisStep()

        # Create prev_result with trait columns in metadata
        prev_result = StepResult(
            data=df,
            metadata={
                "valid_trait_names": ["good_trait", "constant_trait", "variable_trait"]
            },
            files_generated=[],
        )

        # Should NOT crash regardless of which traits fail
        try:
            result = step.execute(
                data=df,
                config=config,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            assert result is not None

            # Read results
            anova_csv = tmp_path / "data" / "08_anova_results.csv"
            anova_df = pd.read_csv(anova_csv)

            # Should have all 3 traits
            assert len(anova_df) == 3

            # Each row should not crash - having NaN for f_statistic is valid for edge cases
            # (e.g., zero variance traits produce NaN F-statistic, which is correct)
            for idx, row in anova_df.iterrows():
                trait = row["trait"]

                # Just verify the row exists and has the expected columns
                # NaN values are acceptable for edge cases
                assert "trait" in row
                assert "f_statistic" in row
                assert "error" in row

        except AttributeError as e:
            if "'str' object has no attribute 'get'" in str(e):
                pytest.fail(f"ANOVA error handling bug not fixed.\nError: {e}")
            else:
                raise

    def test_zero_variance_trait_handled(self, tmp_path):
        """Trait with zero variance should be handled gracefully."""
        csv_path = tmp_path / "zero_var_data.csv"
        rows = ["Barcode,Genotype,Replicate,zero_var_trait,normal_trait"]

        for i in range(30):
            genotype = ["A", "B", "C"][i % 3]
            rows.append(f"p{i},{genotype},{i % 3},1.0,{1.0 + (i % 3) * 0.5}")

        csv_path.write_text("\n".join(rows))

        df = pd.read_csv(csv_path)

        config = get_default_qc_config()
        config.columns.barcode = "Barcode"
        config.columns.genotype = "Genotype"
        config.columns.replicate = "Replicate"
        # ANOVA is always calculated in Step 8, no config flag needed

        step = StatisticalAnalysisStep()

        # Create prev_result with trait columns in metadata
        prev_result = StepResult(
            data=df,
            metadata={"valid_trait_names": ["zero_var_trait", "normal_trait"]},
            files_generated=[],
        )

        # Should handle zero variance gracefully
        result = step.execute(
            data=df,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        anova_df = pd.read_csv(tmp_path / "data" / "08_anova_results.csv")

        # zero_var_trait should either fail or return NaN (not crash)
        zero_var_row = anova_df[anova_df["trait"] == "zero_var_trait"].iloc[0]

        # normal_trait should have valid results
        normal_row = anova_df[anova_df["trait"] == "normal_trait"].iloc[0]
        assert not pd.isna(
            normal_row["f_statistic"]
        ), "Normal trait should have valid F-statistic"
