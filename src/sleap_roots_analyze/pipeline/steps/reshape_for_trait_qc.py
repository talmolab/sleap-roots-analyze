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

    Multiple sources (e.g., biomass and counting) are merged into a single DataFrame.

    Outputs:
        - 00e_root_core_biomass_wide.csv: Wide-format biomass (if present)
        - 00e_root_core_counting_wide.csv: Wide-format counting (if present)
        - 00e_root_core_merged.csv: Merged DataFrame with all sources
        - 00e_reshape_metadata.json: Metadata about reshape operation
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
            StepResult with merged wide-format DataFrame (single table with all depths).
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
            filename = f"00e_root_core_{source.data_type}_wide.csv"
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

        # Merge all sources into a single DataFrame
        df_merged = self._merge_sources(reshaped_data)

        # Create Barcode column for compatibility with downstream steps
        # Format: Plot-Rep (e.g., "1-1", "2-3")
        # Convert to int first to avoid ".0" in output (handles float columns from CSV)
        df_merged["Barcode"] = (
            df_merged["Plot"].astype(int).astype(str)
            + "-"
            + df_merged["Rep"].astype(int).astype(str)
        )

        # Reorder columns: metadata first, then traits
        # Identify trait columns (contain depth info like "_Ncm")
        trait_cols = [
            col
            for col in df_merged.columns
            if any(pattern in col for pattern in ["cm", "DW", "Count"])
        ]
        # Metadata columns are everything else
        metadata_cols = [col for col in df_merged.columns if col not in trait_cols]

        # Reorder: metadata first, then traits (sorted)
        df_merged = df_merged[metadata_cols + sorted(trait_cols)]

        # Save merged data
        merged_path = self.save_dataframe(
            df_merged, "00e_root_core_merged.csv", run_dir
        )
        files.append(merged_path)
        metadata["merged_shape"] = df_merged.shape

        # Save metadata
        metadata_path = self.save_json(metadata, "00e_reshape_metadata.json", run_dir)
        files.append(metadata_path)

        return StepResult(data=df_merged, metadata=metadata, files_generated=files)

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

    def _merge_sources(self, reshaped_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple wide-format DataFrames into a single table.

        Args:
            reshaped_data: Dict of wide-format DataFrames keyed by data_type.

        Returns:
            Single merged DataFrame with all depth columns from all sources.
        """
        if len(reshaped_data) == 0:
            raise ValueError("No data sources to merge")

        if len(reshaped_data) == 1:
            # Only one source, return it directly
            return list(reshaped_data.values())[0]

        # Multiple sources - merge on Plot, Rep, geno
        # First, identify common metadata columns (non-trait columns)
        first_df = list(reshaped_data.values())[0]
        index_cols = ["Plot", "Rep", "geno"]

        # Metadata columns are those that don't contain depth info (no "_Ncm" pattern)
        metadata_cols = [
            col
            for col in first_df.columns
            if col not in index_cols
            and not any(c in col for c in ["cm", "DW", "Count"])
        ]

        df_merged = None
        for data_type, df in reshaped_data.items():
            if df_merged is None:
                df_merged = df
            else:
                # Separate trait columns (depth-specific) from metadata columns
                trait_cols = [
                    col
                    for col in df.columns
                    if col not in index_cols and col not in metadata_cols
                ]

                # Merge only index + trait columns to avoid duplicate metadata
                df_to_merge = df[index_cols + trait_cols]

                df_merged = df_merged.merge(df_to_merge, on=index_cols, how="outer")

        return df_merged
