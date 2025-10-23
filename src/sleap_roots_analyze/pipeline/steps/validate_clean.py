"""Step 3: Validate that data is clean (no NaNs in trait columns)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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

        # Check for NaNs in trait columns
        nan_counts = df[trait_cols].isna().sum()
        total_nans = nan_counts.sum()

        # Check for NaNs in metadata columns too (just for reporting)
        metadata_cols = [col for col in df.columns if col not in trait_cols]
        metadata_nan_counts = df[metadata_cols].isna().sum()
        total_metadata_nans = metadata_nan_counts.sum()

        # Create validation report
        validation_report = {
            "validation_passed": total_nans == 0,
            "total_samples": len(df),
            "total_trait_columns": len(trait_cols),
            "total_metadata_columns": len(metadata_cols),
            "nan_values_in_traits": int(total_nans),
            "nan_values_in_metadata": int(total_metadata_nans),
            "trait_nan_counts": {
                trait: int(count) for trait, count in nan_counts.items() if count > 0
            },
            "metadata_nan_counts": {
                col: int(count)
                for col, count in metadata_nan_counts.items()
                if count > 0
            },
        }

        # Save validation report
        files = []
        files.append(
            self.save_json(validation_report, "03_validation_report.json", run_dir)
        )

        # Raise error if validation fails
        if total_nans > 0:
            raise ValueError(
                f"Validation failed: {total_nans} NaN values found in trait columns!\n"
                f"Affected traits: {list(validation_report['trait_nan_counts'].keys())}"
            )

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
