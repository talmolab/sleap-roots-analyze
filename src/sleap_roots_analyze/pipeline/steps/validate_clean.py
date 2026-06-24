"""Step 3: Validate that data is clean (no NaNs in trait columns)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from sleap_roots_analyze.data_cleanup import (
    build_clean_validation_report,
    _format_nan_validation_error,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class ValidateCleanStep(BaseStep):
    """Validate that cleaned data has no NaN values in trait columns.

    This is a validation-only step that ensures the cleanup process
    (Step 2) successfully removed all NaN values. It doesn't modify the data.

    Outputs:
        - 03_validation_report.json: Validation results
    """

    def __init__(self):
        """Initialize ValidateCleanStep."""
        super().__init__(
            step_name="ValidateClean",
            description="Validate cleaned data has no NaN values",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the validation step.

        Args:
            data: DataFrame from previous step (CleanupTraitsStep).
            config: Pipeline configuration (not used in validation).
            run_dir: Directory to save outputs.
            prev_result: Result from CleanupTraitsStep (contains trait names).

        Returns:
            StepResult with validation results.

        Raises:
            ValueError: If NaN values are found in trait columns.
        """
        df = data

        # Get valid trait columns from previous step
        trait_cols = prev_result.metadata["valid_trait_names"]

        # Build the validation report (single source of truth, shared with the
        # public clean_traits_for_analysis entry point).
        validation_report = build_clean_validation_report(df, trait_cols)
        total_nans = validation_report["nan_values_in_traits"]
        total_metadata_nans = validation_report["nan_values_in_metadata"]

        # Save validation report
        files = []
        files.append(
            self.save_json(validation_report, "03_validation_report.json", run_dir)
        )

        # Raise error if validation fails
        if total_nans > 0:
            raise ValueError(_format_nan_validation_error(validation_report))

        # Create metadata
        metadata = {
            "validation_passed": True,
            "total_nans_in_traits": 0,
            "total_nans_in_metadata": int(total_metadata_nans),
            "samples": len(df),
            "trait_columns": len(trait_cols),
            "trait_names": trait_cols,  # Primary key (standardized)
            "valid_trait_names": trait_cols,  # For consistency
        }

        return StepResult(data=df, metadata=metadata, files_generated=files)
