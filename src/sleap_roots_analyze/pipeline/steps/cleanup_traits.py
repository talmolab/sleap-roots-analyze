"""Step 2: Clean up problematic traits and samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters
from sleap_roots_analyze.data_utils import sanitize_trait_names
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class CleanupTraitsStep(BaseStep):
    """Remove problematic traits and samples in optimal order.

    This step uses apply_data_cleanup_filters which optimally interleaves:
    1. Remove zero-inflated traits
    2. Remove traits with many NaNs
    3. Remove samples with many NaNs (after bad traits removed)
    4. Remove traits with insufficient samples (after sample removal)

    By removing bad traits first, we minimize the number of samples removed.

    Outputs:
        - 01_data_traits_cleaned.csv: Data after trait removal only
        - 01_removed_traits_detail.csv: Details about removed traits
        - 01_trait_cleanup_log.json: Summary of trait cleanup
        - 02_data_samples_cleaned.csv: Data after sample removal
        - 02_removed_samples_detail.csv: Details about removed samples
        - 02_cleanup_log.json: Complete cleanup log
    """

    def __init__(self):
        """Initialize CleanupTraitsStep."""
        super().__init__(
            step_name="CleanupTraits",
            description="Remove problematic traits and samples (optimal ordering)",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the cleanup step.

        Args:
            data: DataFrame from previous step (LoadDataStep).
            config: Pipeline configuration with:
                - cleanup.max_zeros_per_trait: Max fraction of zeros allowed per trait
                - cleanup.max_nans_per_trait: Max fraction of NaNs allowed per trait
                - cleanup.max_nan_fraction: Max fraction of NaNs allowed per sample
                - cleanup.min_samples_per_trait: Min samples required per trait
            run_dir: Directory to save outputs.
            prev_result: Result from LoadDataStep (contains trait column names).

        Returns:
            StepResult with cleaned DataFrame and metadata about cleanup.
        """
        df = data.copy()

        # Get trait columns from previous step
        trait_cols = prev_result.metadata["trait_column_names"]

        # Extract depth_range_mapping from previous step (if available from ReshapeForTraitQCStep)
        depth_range_mapping = prev_result.metadata.get("depth_range_mapping", None)

        # Sanitize trait names (abbreviate + custom replacements + depth ranges)
        df, trait_name_mapping = sanitize_trait_names(
            df=df,
            trait_cols=trait_cols,
            abbreviate=True,
            return_mapping=True,
            sanitize_metadata=True,
            genotype_col=config.columns.genotype,
            replicate_col=config.columns.replicate,
            barcode_col=config.columns.barcode,
            custom_replacements=config.cleanup.custom_replacements,
            depth_range_mapping=depth_range_mapping,
        )

        # Update column references to use sanitized names
        barcode_col = "Barcode"
        genotype_col = "Genotype"
        replicate_col = "Replicate"

        # Update trait columns list with sanitized names
        trait_cols = [trait_name_mapping.get(col, col) for col in trait_cols]

        # Apply cleanup filters (optimal interleaving of trait/sample removal)
        df_clean, cleanup_log = apply_data_cleanup_filters(
            df,
            trait_cols,
            max_zeros_per_trait=config.cleanup.max_zeros_per_trait,
            max_nans_per_trait=config.cleanup.max_nans_per_trait,
            max_nans_per_sample=config.cleanup.max_nan_fraction,
            min_samples_per_trait=config.cleanup.min_samples_per_trait,
            barcode_col=barcode_col,
            genotype_col=genotype_col,
            replicate_col=replicate_col,
        )

        # Extract trait-only state (after trait removal, before sample removal)
        # We need to reconstruct this from the log
        traits_after_cleanup = [
            col
            for col in trait_cols
            if col not in [t["trait"] for t in cleanup_log["removed_traits"]]
        ]

        # Get data after trait cleanup but before sample removal
        df_traits_cleaned = df[
            [
                col
                for col in df.columns
                if col in traits_after_cleanup
                or col in prev_result.metadata["metadata_column_names"]
            ]
        ]

        # Create removed traits detail DataFrame
        if cleanup_log["removed_traits"]:
            removed_traits_df = pd.DataFrame(cleanup_log["removed_traits"])
        else:
            removed_traits_df = pd.DataFrame(
                columns=[
                    "trait",
                    "reason",
                    "zero_fraction",
                    "nan_fraction",
                    "n_samples",
                ]
            )

        # Create removed samples detail DataFrame
        if cleanup_log.get("removed_samples_detail"):
            removed_samples_df = pd.DataFrame(cleanup_log["removed_samples_detail"])
        else:
            removed_samples_df = pd.DataFrame(
                columns=[
                    "sample_index",
                    "barcode",
                    "genotype",
                    "rep",
                    "nan_count",
                    "nan_fraction",
                    "nan_traits",
                    "removal_reason",
                ]
            )

        # Reorder columns before saving: metadata first, then traits (sorted)
        df_traits_cleaned = self.reorder_dataframe_columns(
            df_traits_cleaned, traits_after_cleanup
        )
        df_clean = self.reorder_dataframe_columns(df_clean, traits_after_cleanup)

        # Save outputs
        files = []

        # Trait cleanup outputs (01_*)
        files.append(
            self.save_dataframe(
                df_traits_cleaned, "01_data_traits_cleaned.csv", run_dir
            )
        )
        files.append(
            self.save_dataframe(
                removed_traits_df, "01_removed_traits_detail.csv", run_dir
            )
        )

        # Extract trait-only cleanup log
        trait_cleanup_log = {
            "original_samples": cleanup_log["original_samples"],
            "original_traits": cleanup_log["original_traits"],
            "removed_traits": cleanup_log["removed_traits"],
            "final_traits": len(traits_after_cleanup),
            "traits_retained_fraction": cleanup_log["traits_retained_fraction"],
            "trait_cleanup_steps": [
                step for step in cleanup_log["cleanup_steps"] if "trait" in step["step"]
            ],
        }
        files.append(
            self.save_json(trait_cleanup_log, "01_trait_cleanup_log.json", run_dir)
        )

        # Sample cleanup outputs (02_*)
        files.append(
            self.save_dataframe(df_clean, "02_data_samples_cleaned.csv", run_dir)
        )
        files.append(
            self.save_dataframe(
                removed_samples_df, "02_removed_samples_detail.csv", run_dir
            )
        )
        files.append(self.save_json(cleanup_log, "02_cleanup_log.json", run_dir))

        # Validate no NaNs remain in trait columns
        valid_trait_cols = [
            col for col in df_clean.columns if col in traits_after_cleanup
        ]
        nan_check = df_clean[valid_trait_cols].isna().sum().sum()
        if nan_check > 0:
            raise ValueError(
                f"Data cleanup failed: {nan_check} NaN values remain in trait columns after cleanup!"
            )

        # Create metadata
        # Only include trait name mapping if names actually changed
        names_changed = {
            old: new for old, new in trait_name_mapping.items() if old != new
        }

        metadata = {
            "samples_original": cleanup_log["original_samples"],
            "samples_removed": len(cleanup_log.get("removed_samples_detail", [])),
            "samples_final": cleanup_log["final_samples"],
            "traits_original": cleanup_log["original_traits"],
            "traits_removed": len(cleanup_log["removed_traits"]),
            "traits_final": cleanup_log["final_traits"],
            "trait_names": traits_after_cleanup,  # Primary key (standardized)
            "valid_trait_names": traits_after_cleanup,  # For consistency
            "trait_name_mapping": names_changed if names_changed else None,
            "cleanup_log": cleanup_log,
            "nan_validation_passed": True,
            # Add column mapping for downstream steps to use sanitized names
            "column_mapping": {
                "barcode": barcode_col,  # "barcode" -> "Barcode"
                "genotype": genotype_col,  # "genotype" -> "Genotype"
                "replicate": replicate_col,  # "replicate" -> "Replicate"
            },
        }

        return StepResult(data=df_clean, metadata=metadata, files_generated=files)
