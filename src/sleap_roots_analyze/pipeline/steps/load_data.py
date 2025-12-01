"""Step 1: Load and validate trait data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.data_cleanup import get_trait_columns, load_trait_data
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class LoadDataStep(BaseStep):
    """Load trait data from CSV and perform initial validation.

    This step:
    1. Loads the CSV file
    2. Identifies trait columns vs metadata columns
    3. Validates required columns exist
    4. Saves inspection information

    Outputs:
        - 00_data_loaded.csv: The loaded data
        - 00_data_inspection.csv: Summary of columns and data structure
    """

    def __init__(self):
        """Initialize LoadDataStep."""
        super().__init__(
            step_name="LoadData",
            description="Load and validate trait data from CSV",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the data loading step.

        Args:
            data: Pre-loaded DataFrame from root core processing (if provided), else None.
            config: Pipeline configuration with:
                - data.csv_path: Path to CSV file (if data is None)
                - columns.barcode: Barcode column name
                - columns.genotype: Genotype column name
                - columns.replicate: Replicate column name (optional)
                - data.additional_exclude_cols: Additional columns to exclude (optional)
            run_dir: Directory to save outputs.
            prev_result: Previous step result from root core processing (if applicable).

        Returns:
            StepResult with loaded DataFrame and metadata about columns.
        """
        # Check if data was provided from root core processing
        if data is not None and isinstance(data, pd.DataFrame):
            # Use pre-loaded data from root core processing
            df = data
            csv_path = "root_core_processing_output"
        else:
            # Load the data from CSV
            df = load_trait_data(
                config.data.csv_path,
                barcode_col=config.columns.barcode,
                genotype_col=config.columns.genotype,
                replicate_col=config.columns.replicate,
            )
            csv_path = Path(config.data.csv_path).as_posix()

        # Get additional columns to exclude if specified
        additional_exclude = config.data.additional_exclude_cols

        # Get trait columns (numeric, non-metadata)
        trait_cols = get_trait_columns(
            df,
            barcode_col=config.columns.barcode,
            genotype_col=config.columns.genotype,
            replicate_col=config.columns.replicate,
            additional_exclude=additional_exclude,
        )

        # Metadata columns are everything else
        metadata_cols = [col for col in df.columns if col not in trait_cols]

        # Reorder columns: metadata first, then traits (sorted)
        df = self.reorder_dataframe_columns(df, trait_cols)

        # Create inspection summary
        inspection = pd.DataFrame(
            {
                "column_name": df.columns,
                "column_type": [
                    "trait" if col in trait_cols else "metadata" for col in df.columns
                ],
                "dtype": [str(df[col].dtype) for col in df.columns],
                "non_null_count": [df[col].notna().sum() for col in df.columns],
                "null_count": [df[col].isna().sum() for col in df.columns],
            }
        )

        # Save outputs (save_dataframe is defined in BaseStep)
        files = []
        files.append(self.save_dataframe(df, "00_data_loaded.csv", run_dir))
        files.append(self.save_dataframe(inspection, "00_data_inspection.csv", run_dir))

        # Create metadata
        metadata = {
            "samples": len(df),
            "total_columns": len(df.columns),
            "trait_columns": len(trait_cols),
            "trait_column_names": trait_cols,  # Legacy
            "trait_names": trait_cols,  # Primary key (standardized)
            "valid_trait_names": trait_cols,  # For consistency
            "metadata_columns": len(metadata_cols),
            "metadata_column_names": metadata_cols,
            "csv_path": csv_path,
        }

        return StepResult(data=df, metadata=metadata, files_generated=files)
