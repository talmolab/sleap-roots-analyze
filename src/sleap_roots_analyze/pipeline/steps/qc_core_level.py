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

    This step performs measurement error detection (quality control) at the core level,
    using per-group analysis to detect gross errors before aggregation.

    **Two Detection Methods:**

    1. **Missing Data Detection** (always enabled if core_qc.enabled=True):
       - Flags cores with excessive missing depths (>50% NaN)
       - Catches incomplete or damaged samples

    2. **Value Outlier Detection** (opt-in via detect_value_outliers=True):
       - Flags cores with anomalous values using percent deviation from median
       - Formula: |value - median| / median > threshold (e.g., 0.30 = 30%)
       - Per-group analysis (within Plot-Rep-Geno-Depth)
       - Conservative 30% default threshold minimizes false positives

    **Statistical Framework:**
    - QC paradigm: Not statistical hypothesis testing, but quality control
    - Sample size: Works with N=3 cores per group (non-parametric)
    - Independence: Per-group analysis respects nested structure
    - Transparent: Simple, interpretable criteria

    **Example (GH_7371 Rep 1):**
    - Cores: [0.76g, 0.71g, 0.31g]
    - Median: 0.71g
    - Deviations: [8%, 0%, 56%]
    - Result: Core 2 flagged (56% > 30%), removed before aggregation
    - Impact: Heritability improved from 0.27 to >0.50

    **When to Enable:**
    - Measurement errors expected (typos, damaged cores, sampling failures)
    - Median aggregation alone not sufficient (N=3 not robust enough)
    - Need to preserve heritability by removing gross errors

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
        """Detect outlier cores using missing data and/or value-based methods.

        This method combines two detection approaches:
        1. Missing data detection: Flags cores with excessive missing depths
        2. Value outlier detection: Flags cores with anomalous values (if enabled)

        Both methods use per-group analysis (within Plot-Rep-Geno-Depth) to respect
        the nested experimental structure.

        Args:
            df: Long-format DataFrame with core-level data.
            value_column: Name of the value column.
            data_type: Type of data for logging ('biomass' or 'counting').
            qc_config: CoreQCConfig with detection settings.

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

        # Group by Plot-Rep-geno-Depth for value outlier detection
        group_cols = ["Plot", "Rep", "geno", "Depth_cm"]
        flagged_cores = []
        flagged_by_method = {"missing_data": 0, "value_outlier": 0}

        # Method 1: Missing data detection (always performed)
        missing_flagged = self._flag_missing_data(
            df, value_column, data_type, qc_config
        )
        flagged_cores.extend(missing_flagged)
        flagged_by_method["missing_data"] = len(missing_flagged)

        # Mark missing data flags in DataFrame
        for core_info in missing_flagged:
            mask = df["core_id"] == core_info["core_id"]
            df.loc[mask, "outlier_flag"] = True
            df.loc[mask, "outlier_reason"] = core_info["reason"]

        # Method 2: Value outlier detection (if enabled)
        if qc_config.detect_value_outliers:
            value_flagged = self._detect_value_outliers_per_group(
                df, value_column, data_type, qc_config, group_cols
            )

            # Add value outliers that aren't already flagged by missing data
            for core_info in value_flagged:
                core_id = core_info["core_id"]
                if core_id not in [c["core_id"] for c in missing_flagged]:
                    flagged_cores.append(core_info)
                    flagged_by_method["value_outlier"] += 1

                    # Mark value outlier flags in DataFrame
                    mask = df["core_id"] == core_id
                    df.loc[mask, "outlier_flag"] = True
                    df.loc[mask, "outlier_reason"] = core_info["reason"]

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
            samples_per_core = (
                df.groupby(group_cols).size().iloc[0] if len(df) > 0 else 1
            )
            samples_before = len(df) + (cores_removed * samples_per_core)
        else:
            samples_before = len(df)

        metadata = {
            "cores_flagged": cores_flagged,
            "cores_removed": cores_removed,
            "flagged_by_method": flagged_by_method,
            "flagged_cores_list": flagged_cores,
            "samples_before": samples_before,
            "samples_after": len(df),
        }

        return df, metadata

    def _flag_missing_data(
        self,
        df: pd.DataFrame,
        value_column: str,
        data_type: str,
        qc_config: Any,
    ) -> list[dict]:
        """Flag cores with excessive missing data.

        Args:
            df: Long-format DataFrame with core-level data.
            value_column: Name of the value column.
            data_type: Type of data ('biomass' or 'counting').
            qc_config: CoreQCConfig with missing data threshold.

        Returns:
            List of dicts with flagged core info.
        """
        flagged_cores = []
        group_cols = ["Plot", "Rep", "geno"]

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

        return flagged_cores

    def _detect_value_outliers_per_group(
        self,
        df: pd.DataFrame,
        value_column: str,
        data_type: str,
        qc_config: Any,
        group_cols: list[str],
    ) -> list[dict]:
        """Detect cores with anomalous values using percent deviation from median.

        This method analyzes each (Plot, Rep, Geno, Depth) group independently to
        detect measurement errors while respecting the nested experimental structure.

        Formula: percent_deviation = |value - median| / median
        Flag if: percent_deviation > threshold (e.g., 0.30 = 30%)

        Edge cases:
        - Median = 0: Skip group (avoid division by zero)
        - N < 2: Skip group (need at least 2 cores for median comparison)
        - All flagged: Keep min_cores_after_qc closest to median (safety)

        Args:
            df: Long-format DataFrame with core-level data.
            value_column: Name of the value column.
            data_type: Type of data ('biomass' or 'counting').
            qc_config: CoreQCConfig with detection settings.
            group_cols: Columns defining groups (Plot, Rep, Geno, Depth).

        Returns:
            List of dicts with flagged core info including deviation percentages.
        """
        flagged_cores = []

        for group_key, group_df in df.groupby(group_cols):
            # Get values for this group (one value per core at this depth)
            values = group_df.groupby("core_id")[value_column].first()

            # Skip if insufficient data
            if len(values) < 2:
                continue

            # Remove already-flagged cores (missing data) from value analysis
            unflagged_mask = (
                ~df.loc[group_df.index, "outlier_flag"]
                .groupby(df.loc[group_df.index, "core_id"])
                .first()
            )
            unflagged_values = values[unflagged_mask]

            if len(unflagged_values) < 2:
                continue

            # Calculate median
            median_val = unflagged_values.median()

            # Skip if median is zero (avoid division by zero)
            if median_val == 0:
                continue

            # Calculate percent deviations
            deviations = (unflagged_values - median_val).abs() / median_val

            # Identify outliers
            outlier_mask = deviations > qc_config.max_deviation_from_median
            n_outliers = outlier_mask.sum()

            # Safety: Keep at least min_cores_after_qc cores
            if n_outliers >= len(unflagged_values) - qc_config.min_cores_after_qc + 1:
                # Too many would be removed - keep closest to median
                n_to_keep = qc_config.min_cores_after_qc
                keep_indices = deviations.nsmallest(n_to_keep).index
                outlier_mask = outlier_mask & ~outlier_mask.index.isin(keep_indices)

            # Record flagged cores
            for core_id in unflagged_values[outlier_mask].index:
                deviation_pct = deviations.loc[core_id]
                value = unflagged_values.loc[core_id]

                flagged_cores.append(
                    {
                        "core_id": core_id,
                        "group": f"{group_key}",
                        "reason": f"value_deviation_{deviation_pct:.2f}",
                        "value": float(value),
                        "median": float(median_val),
                        "deviation_pct": float(deviation_pct),
                        "threshold": qc_config.max_deviation_from_median,
                    }
                )

        return flagged_cores
