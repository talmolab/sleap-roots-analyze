"""Core-level quality control - detect outlier cores within replicates (Step 0c-insert)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class QCCoreLevelStep(BaseStep):
    """Detect and optionally remove outlier cores within biological replicates.

    This step performs quality control on individual cores before aggregation:
    1. Groups data by Plot-Rep-geno
    2. Detects outlier cores using Mahalanobis distance on depth profiles
    3. Flags cores with excessive missing data
    4. Optionally removes flagged cores before aggregation

    Outputs:
        - 00c_root_core_biomass_qc.csv: QC'd biomass data (if present)
        - 00c_root_core_counting_qc.csv: QC'd counting data (if present)
        - 00c_core_qc_metadata.json: Metadata about outliers detected/removed
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
        metadata = {"sources": [], "total_outliers": 0, "total_removed": 0}

        for source in config.root_core.sources:
            df_long = data[source.data_type]

            # Perform core-level QC
            df_qc, qc_metadata = self._detect_outlier_cores(
                df_long,
                source.value_column_name,
                source.data_type,
                qc_config.outlier_method,
                qc_config.contamination,
                qc_config.max_missing_proportion,
                qc_config.remove_outliers,
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
            metadata["total_outliers"] += qc_metadata["outliers_detected"]
            metadata["total_removed"] += qc_metadata["outliers_removed"]

        # Save metadata
        metadata_path = self.save_json(metadata, "00c_core_qc_metadata.json", run_dir)
        files.append(metadata_path)

        return StepResult(data=qc_data, metadata=metadata, files_generated=files)

    def _detect_outlier_cores(
        self,
        df: pd.DataFrame,
        value_column: str,
        data_type: str,
        outlier_method: str,
        contamination: float,
        max_missing_proportion: float,
        remove_outliers: bool,
    ) -> tuple[pd.DataFrame, dict]:
        """Detect outlier cores within Plot-Rep groups.

        Args:
            df: Long-format DataFrame with core-level data.
            value_column: Name of the value column.
            data_type: Type of data for logging.
            outlier_method: Method for outlier detection ('mahalanobis').
            contamination: Expected proportion of outliers.
            max_missing_proportion: Max proportion of missing depths allowed.
            remove_outliers: Whether to remove flagged outliers.

        Returns:
            Tuple of (QC'd DataFrame, metadata dict).
        """
        df = df.copy()

        # Create unique core identifier (for all rows)
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
        outlier_cores = []

        for group_key, group_df in df.groupby(group_cols):

            # Pivot: rows=cores, columns=depths
            core_depth_matrix = group_df.pivot(
                index="core_id", columns="Depth_cm", values=value_column
            )

            # Flag cores with excessive missing data
            missing_prop = core_depth_matrix.isna().sum(axis=1) / len(
                core_depth_matrix.columns
            )
            missing_outliers = missing_prop > max_missing_proportion

            # Detect outliers using Mahalanobis distance (on non-missing cores)
            valid_cores = ~missing_outliers
            if valid_cores.sum() >= 3:  # Need at least 3 cores for covariance
                X = core_depth_matrix.loc[valid_cores].fillna(
                    core_depth_matrix.loc[valid_cores].mean()
                )

                if len(X) > 0 and X.shape[1] > 0:
                    # Calculate Mahalanobis distances
                    cov = EmpiricalCovariance().fit(X)
                    mahal_dist = cov.mahalanobis(X)

                    # Threshold based on contamination
                    threshold = np.percentile(mahal_dist, (1 - contamination) * 100)
                    mahal_outliers = mahal_dist > threshold

                    # Map back to core IDs
                    mahal_outlier_cores = X.index[mahal_outliers].tolist()
                else:
                    mahal_outlier_cores = []
            else:
                mahal_outlier_cores = []

            # Combine outlier flags
            for core_id in core_depth_matrix.index:
                reasons = []

                if missing_outliers.loc[core_id]:
                    reasons.append(f"missing_data_{missing_prop.loc[core_id]:.2f}")

                if core_id in mahal_outlier_cores:
                    reasons.append("mahalanobis")

                if reasons:
                    outlier_cores.append(
                        {
                            "core_id": core_id,
                            "group": f"{group_key}",
                            "reasons": reasons,
                        }
                    )

                    # Flag all rows for this core
                    mask = df["core_id"] == core_id
                    df.loc[mask, "outlier_flag"] = True
                    df.loc[mask, "outlier_reason"] = "|".join(reasons)

        # Remove outliers if requested
        outliers_detected = len(outlier_cores)
        if remove_outliers and outliers_detected > 0:
            df = df[~df["outlier_flag"]].copy()
            outliers_removed = outliers_detected
        else:
            outliers_removed = 0

        # Drop temporary core_id column
        if "core_id" in df.columns:
            df = df.drop(columns=["core_id"])

        metadata = {
            "outliers_detected": outliers_detected,
            "outliers_removed": outliers_removed,
            "outlier_list": outlier_cores,
            "samples_before": len(df)
            + (
                outliers_removed
                * df.groupby(["Plot", "Rep", "geno", "Depth_cm"]).size().iloc[0]
                if outliers_removed > 0
                else 0
            ),
            "samples_after": len(df),
        }

        return df, metadata
