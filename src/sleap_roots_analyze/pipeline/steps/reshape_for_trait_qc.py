"""Reshape aggregated data for trait-level QC with column prefixing (Step 0d)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class ReshapeForTraitQCStep(BaseStep):
    """Reshape long-format root data to wide format with prefixed column names.

    This step pivots the aggregated long-format data (one row per Plot-Rep-Depth)
    back to wide format (one row per Plot-Rep) with depth-specific columns.

    Column naming: {depth_column_prefix}_{depth}cm
    Example: RootDW_15cm, RootCount_5cm

    This prevents duplicate column names when merging with above-ground traits.

    Outputs:
        - 00d_root_core_biomass_wide.csv: Wide-format biomass (if present)
        - 00d_root_core_counting_wide.csv: Wide-format counting (if present)
        - 00d_reshape_metadata.json: Metadata about reshape operation
    """

    def __init__(self):
        """Initialize ReshapeForTraitQCStep."""
        super().__init__(
            step_name="ReshapeForTraitQC",
            description="Reshape root data to wide format with prefixes",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the reshape step.

        Args:
            data: Dict of aggregated long-format DataFrames.
            config: Pipeline configuration with root_core.sources list.
            run_dir: Directory to save outputs.
            prev_result: StepResult from previous step.

        Returns:
            StepResult with dict of wide-format DataFrames keyed by data_type.
        """
        reshaped_data = {}
        files = []
        metadata = {"sources": []}

        for source in config.root_core.sources:
            df_long = data[source.data_type]

            # Reshape to wide format with prefixes
            df_wide, reshape_metadata = self._pivot_to_wide(
                df_long,
                source.value_column_name,
                source.depth_column_prefix,
                source.data_type,
            )

            reshaped_data[source.data_type] = df_wide

            # Save wide-format data
            filename = f"00d_root_core_{source.data_type}_wide.csv"
            output_path = self.save_dataframe(df_wide, filename, run_dir)
            files.append(output_path)

            # Track metadata
            metadata["sources"].append(
                {
                    "data_type": source.data_type,
                    "prefix": source.depth_column_prefix,
                    "rows": len(df_wide),
                    **reshape_metadata,
                }
            )

        # Save metadata
        metadata_path = self.save_json(metadata, "00d_reshape_metadata.json", run_dir)
        files.append(metadata_path)

        return StepResult(data=reshaped_data, metadata=metadata, files_generated=files)

    def _pivot_to_wide(
        self,
        df: pd.DataFrame,
        value_column: str,
        prefix: str,
        data_type: str,
    ) -> tuple[pd.DataFrame, dict]:
        """Pivot long-format data to wide format with prefixed columns.

        Args:
            df: Long-format DataFrame with Depth_cm column.
            value_column: Name of the value column to pivot.
            prefix: Prefix for column names (e.g., 'RootDW', 'RootCount').
            data_type: Type of data for logging.

        Returns:
            Tuple of (wide-format DataFrame, metadata dict).
        """
        # Identify index columns (grouping variables)
        index_cols = ["Plot", "Rep", "geno"]

        # Identify metadata columns to preserve
        metadata_cols = [
            col
            for col in df.columns
            if col not in index_cols + ["Depth_cm", value_column]
        ]

        # Get unique depths
        depths = sorted(df["Depth_cm"].unique())

        # Pivot the value column
        df_pivot = df.pivot(
            index=index_cols, columns="Depth_cm", values=value_column
        ).reset_index()

        # Rename columns with prefix: {prefix}_{depth}cm
        new_cols = {}
        for col in df_pivot.columns:
            if col in index_cols:
                new_cols[col] = col
            else:
                # col is a depth value (float)
                depth_int = int(col) if col == int(col) else col
                new_cols[col] = f"{prefix}_{depth_int}cm"

        df_pivot = df_pivot.rename(columns=new_cols)

        # Add back metadata columns (take first value per Plot-Rep-geno)
        if metadata_cols:
            # Get metadata for each Plot-Rep-geno group (take first)
            df_meta = (
                df[index_cols + metadata_cols]
                .drop_duplicates(subset=index_cols)
                .reset_index(drop=True)
            )

            # Merge with pivoted data
            df_pivot = df_pivot.merge(df_meta, on=index_cols, how="left")

        # Create column names list for metadata
        depth_columns = [f"{prefix}_{int(d) if d == int(d) else d}cm" for d in depths]

        metadata = {
            "num_depths": len(depths),
            "depth_values": [float(d) for d in depths],
            "depth_columns": depth_columns,
            "num_samples": len(df_pivot),
        }

        return df_pivot, metadata
