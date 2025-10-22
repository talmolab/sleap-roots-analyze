"""Step 7: Remove outliers based on detection results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class RemoveOutliersStep(BaseStep):
    """Remove outliers detected in the previous step.

    This step provides flexible outlier removal strategies:
    - Single method: Remove outliers identified by one specific method
    - Consensus: Remove samples flagged by ALL selected methods
    - Subset consensus: Remove samples flagged by at least N methods

    Outputs:
        - 07_data_outliers_removed.csv: Data after outlier removal
        - 07_removed_outliers_detail.csv: Details about removed samples
        - 07_outlier_removal_log.json: Summary of outlier removal
    """

    def __init__(self):
        """Initialize RemoveOutliersStep."""
        super().__init__(
            step_name="RemoveOutliers",
            description="Remove outliers based on detection results",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute outlier removal.

        Args:
            data: DataFrame from previous step (VisualizeOutliersStep).
            config: Pipeline configuration with:
                - outlier_removal.strategy: "single", "consensus", or "subset"
                - outlier_removal.method: Method name (for "single" strategy)
                - outlier_removal.min_methods: Min methods required (for "subset" strategy)
            run_dir: Directory to save outputs.
            prev_result: Result from DetectOutliersStep or VisualizeOutliersStep
                (contains outlier results).

        Returns:
            StepResult with DataFrame after outlier removal and metadata.
        """
        df = data.copy()

        # Get outlier results from previous step metadata
        outlier_results = prev_result.metadata["outlier_results"]
        methods_run = prev_result.metadata["methods_run"]

        # Get trait columns from previous step (try multiple keys for robustness)
        trait_cols = prev_result.metadata.get("valid_trait_names")
        if trait_cols is None:
            trait_cols = prev_result.metadata.get("trait_names", [])

        # Get removal strategy
        strategy = config.outlier_removal.strategy
        barcode_col = config.columns.barcode

        # Identify samples to remove based on strategy
        if strategy == "single":
            # Remove outliers from a single method
            method = config.outlier_removal.method
            if method not in outlier_results:
                raise ValueError(
                    f"Method '{method}' not found in outlier results. "
                    f"Available methods: {list(outlier_results.keys())}"
                )

            outlier_mask = outlier_results[method]["outlier_mask"]
            samples_to_remove = df.index[outlier_mask].tolist()
            removal_reason = f"flagged_by_{method}"

        elif strategy == "consensus":
            # Remove samples flagged by ALL methods
            # Start with all True, then AND with each method's mask
            consensus_mask = pd.Series(True, index=df.index)
            for method in methods_run:
                consensus_mask &= outlier_results[method]["outlier_mask"]

            samples_to_remove = df.index[consensus_mask].tolist()
            removal_reason = f"consensus_{len(methods_run)}methods"

        elif strategy == "subset":
            # Remove samples flagged by at least N methods
            min_methods = config.outlier_removal.min_methods

            # Count how many methods flagged each sample
            flag_counts = pd.Series(0, index=df.index)
            for method in methods_run:
                flag_counts += outlier_results[method]["outlier_mask"].astype(int)

            # Remove samples flagged by at least min_methods
            subset_mask = flag_counts >= min_methods
            samples_to_remove = df.index[subset_mask].tolist()
            removal_reason = f"flagged_by_{min_methods}_or_more_methods"

        else:
            raise ValueError(
                f"Invalid outlier removal strategy: {strategy}. "
                f"Must be 'single', 'consensus', or 'subset'."
            )

        # Create removal details DataFrame
        if samples_to_remove:
            removed_df = df.loc[samples_to_remove].copy()
            removed_df["removal_reason"] = removal_reason

            # Add per-method flags
            for method in methods_run:
                removed_df[f"{method}_outlier"] = outlier_results[method][
                    "outlier_mask"
                ][samples_to_remove].values

            # Remove outliers from main DataFrame
            df_clean = df.drop(samples_to_remove)
        else:
            # No outliers to remove
            removed_df = pd.DataFrame()
            df_clean = df

        # Create removal log
        removal_log = {
            "strategy": strategy,
            "methods_considered": methods_run,
            "original_samples": len(data),
            "outliers_removed": len(samples_to_remove),
            "final_samples": len(df_clean),
            "removal_fraction": len(samples_to_remove) / len(data),
            "removed_sample_barcodes": (
                df.loc[samples_to_remove, barcode_col].tolist()
                if samples_to_remove
                else []
            ),
        }

        # Add strategy-specific info
        if strategy == "single":
            removal_log["method_used"] = config.outlier_removal.method
        elif strategy == "subset":
            removal_log["min_methods_required"] = config.outlier_removal.min_methods

        # Save outputs
        files = []
        files.append(
            self.save_dataframe(df_clean, "07_data_outliers_removed.csv", run_dir)
        )

        if len(removed_df) > 0:
            files.append(
                self.save_dataframe(
                    removed_df, "07_removed_outliers_detail.csv", run_dir
                )
            )
        else:
            # Save empty file with headers
            empty_df = pd.DataFrame(
                columns=[barcode_col, "removal_reason"]
                + [f"{m}_outlier" for m in methods_run]
            )
            files.append(
                self.save_dataframe(empty_df, "07_removed_outliers_detail.csv", run_dir)
            )

        files.append(
            self.save_json(removal_log, "07_outlier_removal_log.json", run_dir)
        )

        # Create metadata
        metadata = {
            "samples_original": len(data),
            "outliers_removed": len(samples_to_remove),
            "samples_final": len(df_clean),
            "removal_strategy": strategy,
            "removal_log": removal_log,
            "trait_names": trait_cols,  # Pass through for next step
            "valid_trait_names": trait_cols,  # For consistency with other steps
        }

        return StepResult(data=df_clean, metadata=metadata, files_generated=files)
