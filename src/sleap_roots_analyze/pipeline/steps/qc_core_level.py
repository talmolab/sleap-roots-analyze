"""Core-level quality control - detect outlier cores within replicates (Step 0c-insert)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Statistical outlier detection removed - insufficient samples at core level
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class QCCoreLevelStep(BaseStep):
    """Quality control for individual cores before aggregation.

    IMPORTANT: This step only performs missing data filtering. Statistical outlier
    detection (e.g., Mahalanobis) is NOT used because core-level data has insufficient
    samples (~3 cores per plot). Statistical methods require 30+ samples for reliability.

    Recommended approach:
    1. Disable this step (core_qc.enabled: false)
    2. Use median aggregation for robustness to outliers
    3. Perform outlier detection at trait level (Step 5) with 60+ plot samples

    If enabled, this step:
    1. Flags cores with excessive missing data (>50% NaN depths)
    2. Optionally removes flagged cores before aggregation

    Outputs:
        - 00c_root_core_biomass_qc.csv: QC'd biomass data (if present)
        - 00c_root_core_counting_qc.csv: QC'd counting data (if present)
        - 00c_core_qc_metadata.json: Metadata about cores flagged/removed
    """

    def __init__(self):
        """Initialize QCCoreLevelStep."""
        super().__init__(
            step_name="QCCoreLevel",
            description="Detect and remove outlier cores within replicates",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute core-level QC step.

        Args:
            data: Dict of long-format DataFrames from TransformDepthDataStep.
            config: Pipeline configuration with root_core.core_qc settings.
            run_dir: Directory to save outputs.
            prev_result: StepResult from previous step.

        Returns:
            StepResult with dict of QC'd DataFrames keyed by data_type.
        """
        if not config.root_core.core_qc.enabled:
            # QC disabled - pass through unchanged
            return StepResult(
                data=data,
                metadata={"qc_enabled": False},
                files_generated=[],
            )

        qc_config = config.root_core.core_qc
        qc_data = {}
        files = []
        metadata = {"sources": [], "total_flagged": 0, "total_removed": 0}

        for source in config.root_core.sources:
            df_long = data[source.data_type]

            # Perform core-level QC
            df_qc, qc_metadata = self._detect_outlier_cores(
                df_long,
                source.value_column_name,
                source.data_type,
                qc_config,  # Pass entire config object
            )

            qc_data[source.data_type] = df_qc

            # Save QC'd data
            filename = f"00c_root_core_{source.data_type}_qc.csv"
            output_path = self.save_dataframe(df_qc, filename, run_dir)
            files.append(output_path)

            # Track metadata
            metadata["sources"].append(
                {
                    "data_type": source.data_type,
                    **qc_metadata,
                }
            )
            metadata["total_flagged"] += qc_metadata["cores_flagged"]
            metadata["total_removed"] += qc_metadata["cores_removed"]

        # Save metadata
        metadata_path = self.save_json(metadata, "00c_core_qc_metadata.json", run_dir)
        files.append(metadata_path)

        return StepResult(data=qc_data, metadata=metadata, files_generated=files)

    def _detect_outlier_cores(
        self,
        df: pd.DataFrame,
        value_column: str,
        data_type: str,
        qc_config: Any,  # CoreQCConfig
    ) -> tuple[pd.DataFrame, dict]:
        """Flag cores with excessive missing data.

        NOTE: Statistical outlier detection is NOT performed at the core level due to
        insufficient sample sizes. Core-level data typically has only 3 cores per plot,
        but methods like Mahalanobis distance require 30+ samples for reliable detection.

        Instead, this method only flags cores with excessive missing depths. Users should:
        1. Use median aggregation for robustness to outliers (e.g., typos, miscounts)
        2. Perform statistical outlier detection at trait level (Step 5) with 60+ samples

        Args:
            df: Long-format DataFrame with core-level data.
            value_column: Name of the value column.
            data_type: Type of data for logging ('biomass' or 'counting').
            qc_config: CoreQCConfig with missing data threshold.

        Returns:
            Tuple of (QC'd DataFrame, metadata dict).
        """
        df = df.copy()

        # Create unique core identifier
        core_col = "Core_Replicate" if data_type == "biomass" else "core_n"
        df["core_id"] = (
            "plot"
            + df["Plot"].astype(str)
            + "_rep"
            + df["Rep"].astype(str)
            + "_"
            + df["geno"].astype(str)
            + "_core"
            + df[core_col].astype(str)
        )

        # Add outlier flag columns
        df["outlier_flag"] = False
        df["outlier_reason"] = ""

        # Group by Plot-Rep-geno
        group_cols = ["Plot", "Rep", "geno"]
        flagged_cores = []

        for group_key, group_df in df.groupby(group_cols):
            # Pivot: rows=cores, columns=depths
            core_depth_matrix = group_df.pivot(
                index="core_id", columns="Depth_cm", values=value_column
            )

            # Flag cores with excessive missing data
            missing_prop = core_depth_matrix.isna().sum(axis=1) / len(
                core_depth_matrix.columns
            )
            missing_outliers = missing_prop > qc_config.max_missing_proportion

            # Flag cores
            for core_id in core_depth_matrix.index:
                if missing_outliers.loc[core_id]:
                    reason = f"missing_data_{missing_prop.loc[core_id]:.2f}"
                    flagged_cores.append(
                        {
                            "core_id": core_id,
                            "group": f"{group_key}",
                            "reason": reason,
                        }
                    )

                    # Flag all rows for this core
                    mask = df["core_id"] == core_id
                    df.loc[mask, "outlier_flag"] = True
                    df.loc[mask, "outlier_reason"] = reason

        # Remove flagged cores if requested
        cores_flagged = len(flagged_cores)
        if qc_config.remove_outliers and cores_flagged > 0:
            df = df[~df["outlier_flag"]].copy()
            cores_removed = cores_flagged
        else:
            cores_removed = 0

        # Drop temporary core_id column
        if "core_id" in df.columns:
            df = df.drop(columns=["core_id"])

        # Calculate samples before/after
        if cores_removed > 0 and len(df) > 0:
            samples_per_core = df.groupby(group_cols + ["Depth_cm"]).size().iloc[0]
            samples_before = len(df) + (cores_removed * samples_per_core)
        else:
            samples_before = len(df)

        metadata = {
            "cores_flagged": cores_flagged,
            "cores_removed": cores_removed,
            "flagged_cores_list": flagged_cores,
            "samples_before": samples_before,
            "samples_after": len(df),
        }

        return df, metadata
