"""Aggregate technical replicates (cores) to biological replicate level (Step 0c)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class AggregateCoresStep(BaseStep):
    """Aggregate technical replicates (cores) to biological replicate level.

    This step aggregates multiple cores per Plot-Rep-Depth combination using
    the specified aggregation method (mean, median, or custom function).

    Grouping: ['Plot', 'Rep', 'geno', 'Depth_cm']
    Result: One row per Plot-Rep-Depth with aggregated value

    Outputs:
        - 00c_root_core_biomass_aggregated.csv: Aggregated biomass (if present)
        - 00c_root_core_counting_aggregated.csv: Aggregated counting (if present)
        - 00c_aggregation_metadata.json: Metadata about aggregation
    """

    def __init__(self):
        """Initialize AggregateCoresStep."""
        super().__init__(
            step_name="AggregateCores",
            description="Aggregate cores to biological replicate level",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the core aggregation step.

        Args:
            data: Dict of long-format DataFrames from TransformDepthDataStep.
            config: Pipeline configuration with root_core.sources list.
            run_dir: Directory to save outputs.
            prev_result: StepResult from previous step.

        Returns:
            StepResult with dict of aggregated DataFrames keyed by data_type.
        """
        aggregated_data = {}
        files = []
        metadata = {"sources": []}

        for source in config.root_core.sources:
            df_long = data[source.data_type]

            # Aggregate using specified method
            df_agg, agg_metadata = self._aggregate_by_replicate(
                df_long,
                source.value_column_name,
                source.aggregation_method,
                source.data_type,
            )

            aggregated_data[source.data_type] = df_agg

            # Save aggregated data
            filename = f"00c_root_core_{source.data_type}_aggregated.csv"
            output_path = self.save_dataframe(df_agg, filename, run_dir)
            files.append(output_path)

            # Track metadata
            metadata["sources"].append(
                {
                    "data_type": source.data_type,
                    "aggregation_method": source.aggregation_method,
                    "rows_before": len(df_long),
                    "rows_after": len(df_agg),
                    **agg_metadata,
                }
            )

        # Save metadata
        metadata_path = self.save_json(
            metadata, "00c_aggregation_metadata.json", run_dir
        )
        files.append(metadata_path)

        return StepResult(
            data=aggregated_data, metadata=metadata, files_generated=files
        )

    def _aggregate_by_replicate(
        self,
        df: pd.DataFrame,
        value_column: str,
        agg_method: str,
        data_type: str,
    ) -> tuple[pd.DataFrame, dict]:
        """Aggregate cores to biological replicate level.

        Args:
            df: Long-format DataFrame with core-level data.
            value_column: Name of the value column to aggregate.
            agg_method: Aggregation method ('mean', 'median', or callable).
            data_type: Type of data for logging.

        Returns:
            Tuple of (aggregated DataFrame, metadata dict).
        """
        # Group by Plot, Rep, geno, Depth_cm
        group_cols = ["Plot", "Rep", "geno", "Depth_cm"]

        # Identify metadata columns to preserve (take first value per group)
        metadata_cols = [
            col
            for col in df.columns
            if col
            not in group_cols + [value_column, "sample_id", "Core_Replicate", "core_n"]
        ]

        # Determine aggregation function
        if agg_method == "mean":
            agg_func = "mean"
        elif agg_method == "median":
            agg_func = "median"
        elif callable(agg_method):
            agg_func = agg_method
        else:
            raise ValueError(
                f"Invalid aggregation_method: {agg_method}. Must be 'mean', 'median', or callable"
            )

        # Perform aggregation
        agg_dict = {value_column: agg_func}

        # Add metadata columns (take first value)
        for col in metadata_cols:
            agg_dict[col] = "first"

        df_agg = df.groupby(group_cols, as_index=False).agg(agg_dict)

        # Calculate aggregation statistics
        cores_per_group = df.groupby(group_cols).size()
        metadata = {
            "total_groups": len(df_agg),
            "min_cores_per_group": int(cores_per_group.min()),
            "max_cores_per_group": int(cores_per_group.max()),
            "mean_cores_per_group": float(cores_per_group.mean()),
        }

        return df_agg, metadata
