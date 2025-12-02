"""Transform depth data from wide to long format (Step 0b)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class TransformDepthDataStep(BaseStep):
    """Transform wide-format depth data to long format.

    This step handles two transformation strategies:
    1. Manual depth mapping (biomass): User-provided depth values
    2. Automatic depth parsing (counting): Parse from column names using pattern

    For counting data, depths are calculated from column pattern c_<start>_<end>_<subcore>:
    - depth_cm = start + (subcore_index - 1) * (end - start) / 2

    Outputs:
        - 00b_root_core_biomass_long.csv: Long-format biomass data (if present)
        - 00b_root_core_counting_long.csv: Long-format counting data (if present)
        - 00b_depth_mapping.json: Metadata about depth calculations
    """

    def __init__(self):
        """Initialize TransformDepthDataStep."""
        super().__init__(
            step_name="TransformDepthData",
            description="Transform depth data from wide to long format",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the depth data transformation step.

        Args:
            data: Dict of DataFrames from LoadRootCoreDataStep (keyed by data_type).
            config: Pipeline configuration with root_core.sources list.
            run_dir: Directory to save outputs.
            prev_result: StepResult from previous step.

        Returns:
            StepResult with dict of long-format DataFrames keyed by data_type.

        Raises:
            ValueError: If depth mapping is missing for biomass data.
        """
        transformed_data = {}
        files = []
        depth_mappings = {}

        for source in config.root_core.sources:
            df_wide = data[source.data_type]

            # Transform based on data type
            if source.data_type == "biomass":
                df_long, depth_map = self._transform_biomass(df_wide, source)
            else:  # counting
                df_long, depth_map = self._transform_counting(df_wide, source)

            transformed_data[source.data_type] = df_long
            depth_mappings[source.data_type] = depth_map

            # Save long-format data
            filename = f"00b_root_core_{source.data_type}_long.csv"
            output_path = self.save_dataframe(df_long, filename, run_dir)
            files.append(output_path)

        # Save depth mapping metadata
        metadata_path = self.save_json(
            depth_mappings, "00b_depth_mapping.json", run_dir
        )
        files.append(metadata_path)

        metadata = {
            "sources_transformed": len(transformed_data),
            "depth_mappings": depth_mappings,
        }

        return StepResult(
            data=transformed_data, metadata=metadata, files_generated=files
        )

    def _transform_biomass(
        self, df: pd.DataFrame, source: Any
    ) -> tuple[pd.DataFrame, dict]:
        """Transform biomass data using manual depth mapping.

        Args:
            df: Wide-format biomass DataFrame.
            source: Source configuration with depth_mapping.

        Returns:
            Tuple of (long-format DataFrame, depth mapping dict).

        Raises:
            ValueError: If depth_mapping is not provided.
        """
        if not source.depth_mapping:
            raise ValueError("Biomass data requires depth_mapping in configuration")

        # Identify metadata columns (keep everything except depth columns)
        depth_cols = list(source.depth_mapping.keys())
        id_cols = [col for col in df.columns if col not in depth_cols]

        # Melt to long format
        df_long = df.melt(
            id_vars=id_cols,
            value_vars=depth_cols,
            var_name="depth_column",
            value_name=source.value_column_name,
        )

        # Map depth column names to actual depths
        df_long["Depth_cm"] = df_long["depth_column"].map(source.depth_mapping)

        # Drop the intermediate depth_column
        df_long = df_long.drop(columns=["depth_column"])

        # Reorder columns: id_cols, Depth_cm, value_column
        column_order = id_cols + ["Depth_cm", source.value_column_name]
        df_long = df_long[column_order]

        return df_long, source.depth_mapping

    def _transform_counting(
        self, df: pd.DataFrame, source: Any
    ) -> tuple[pd.DataFrame, dict]:
        """Transform counting data using automatic depth parsing.

        Args:
            df: Wide-format counting DataFrame.
            source: Source configuration.

        Returns:
            Tuple of (long-format DataFrame, depth mapping dict).
        """
        # Find depth columns matching pattern c_<start>_<end>_<subcore>
        depth_pattern = re.compile(r"c_(\d+)_(\d+)_(\d+)")
        depth_cols = [col for col in df.columns if depth_pattern.match(col)]

        if not depth_cols:
            raise ValueError(
                f"No depth columns found matching pattern 'c_<start>_<end>_<subcore>'"
            )

        # Identify metadata columns
        id_cols = [col for col in df.columns if col not in depth_cols]

        # Melt to long format
        df_long = df.melt(
            id_vars=id_cols,
            value_vars=depth_cols,
            var_name="depth_column",
            value_name=source.value_column_name,
        )

        # Calculate depth from column name
        def parse_depth(col_name: str) -> float:
            match = depth_pattern.match(col_name)
            if not match:
                return None
            start, end, subcore = map(int, match.groups())
            # depth_cm = start + (subcore_index - 1) * (end - start) / 2
            return start + (subcore - 1) * (end - start) / 2

        df_long["Depth_cm"] = df_long["depth_column"].apply(parse_depth)

        # Create depth mapping for metadata
        depth_mapping = {col: parse_depth(col) for col in depth_cols}

        # Drop the intermediate depth_column
        df_long = df_long.drop(columns=["depth_column"])

        # Reorder columns
        column_order = id_cols + ["Depth_cm", source.value_column_name]
        df_long = df_long[column_order]

        return df_long, depth_mapping
