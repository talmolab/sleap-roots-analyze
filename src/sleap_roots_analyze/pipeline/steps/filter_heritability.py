"""Step 9: Filter out low heritability traits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.data_cleanup import get_trait_columns
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.statistics import (
    analyze_heritability_thresholds,
    identify_high_heritability_traits,
)
from sleap_roots_analyze.visualization import create_heritability_threshold_plot


class FilterHeritabilityStep(BaseStep):
    """Filter out traits with low heritability.

    This step removes traits that have heritability below the configured threshold.
    Only runs if heritability filtering is enabled in the configuration.

    Outputs:
        - 09_data_high_heritability.csv: Data with only high heritability traits
        - 09_removed_traits.json: List of removed low heritability traits
        - 09_heritability_filter_summary.json: Summary of filtering
        - figures/09_heritability_threshold_analysis.png: Threshold analysis plot
    """

    def __init__(self):
        """Initialize FilterHeritabilityStep."""
        super().__init__(
            step_name="FilterHeritability",
            description="Filter out low heritability traits",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute heritability filtering.

        Args:
            data: DataFrame from previous step (StatisticalAnalysisStep).
            config: Pipeline configuration with:
                - heritability.enabled: Whether to filter by heritability
                - heritability.threshold: Minimum heritability threshold
                - columns.barcode, genotype, replicate: Column names
                - data.additional_exclude_cols: Additional columns to exclude
            run_dir: Directory to save outputs.
            prev_result: Result from StatisticalAnalysisStep (contains heritability results).

        Returns:
            StepResult with filtered DataFrame and metadata.
        """
        df = data.copy()

        # Get heritability results from previous step
        heritability_results = prev_result.metadata["heritability_results"]
        trait_cols = prev_result.metadata["trait_names"]

        # Check if heritability filtering is enabled
        if not config.heritability.enabled:
            # Skip filtering, return all traits
            summary = {
                "filtering_enabled": False,
                "threshold": config.heritability.threshold,
                "traits_original": len(trait_cols),
                "traits_retained": len(trait_cols),
                "traits_removed": 0,
                "removed_trait_names": [],
            }

            files = []
            files.append(
                self.save_dataframe(df, "09_data_high_heritability.csv", run_dir)
            )
            files.append(self.save_json([], "09_removed_traits.json", run_dir))
            files.append(
                self.save_json(summary, "09_heritability_filter_summary.json", run_dir)
            )

            metadata = {
                "filtering_enabled": False,
                "threshold": config.heritability.threshold,
                "traits_retained": trait_cols,
                "traits_removed": [],
                "summary": summary,
                "trait_names": trait_cols,  # Pass through for next step
                "valid_trait_names": trait_cols,  # For consistency with other steps
            }

            return StepResult(data=df, metadata=metadata, files_generated=files)

        # Heritability filtering is enabled
        threshold = config.heritability.threshold

        # Identify high heritability traits
        high_h2_traits = identify_high_heritability_traits(
            heritability_results, threshold=threshold
        )

        # Determine which traits to remove
        removed_traits = [t for t in trait_cols if t not in high_h2_traits]

        # Remove low heritability trait columns from DataFrame
        # But keep all metadata columns (automatically handled by dropping only trait columns)
        df_filtered = df.drop(columns=removed_traits).copy()

        # Verify the high heritability traits are still present
        final_traits = get_trait_columns(
            df_filtered,
            barcode_col=config.columns.barcode,
            genotype_col=config.columns.genotype,
            replicate_col=config.columns.replicate,
            additional_exclude=config.data.additional_exclude_cols,
        )

        # Create removal details
        removed_details = []
        for trait in removed_traits:
            trait_result = heritability_results.get(trait, {})
            removed_details.append(
                {
                    "trait": trait,
                    "heritability": trait_result.get("heritability"),
                    "var_genetic": trait_result.get("var_genetic"),
                    "var_residual": trait_result.get("var_residual"),
                    "model_type": trait_result.get("model_type"),
                    "reason": f"h2 < {threshold}",
                }
            )

        # Create summary
        summary = {
            "filtering_enabled": True,
            "threshold": threshold,
            "traits_original": len(trait_cols),
            "traits_retained": len(high_h2_traits),
            "traits_removed": len(removed_traits),
            "removed_trait_names": removed_traits,
            "retention_fraction": (
                len(high_h2_traits) / len(trait_cols) if len(trait_cols) > 0 else 0
            ),
            "mean_heritability_retained": (
                sum(
                    heritability_results[t]["heritability"]
                    for t in high_h2_traits
                    if "heritability" in heritability_results.get(t, {})
                    and heritability_results[t]["heritability"] is not None
                )
                / len(high_h2_traits)
                if high_h2_traits
                else None
            ),
        }

        # Generate heritability threshold analysis plot
        threshold_analysis = analyze_heritability_thresholds(heritability_results)
        fig = create_heritability_threshold_plot(
            threshold_analysis, current_threshold=threshold
        )
        threshold_plot_path = (
            run_dir / "figures" / "09_heritability_threshold_analysis.png"
        )
        threshold_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(threshold_plot_path, dpi=300, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)

        # Save outputs
        files = []
        files.append(
            self.save_dataframe(df_filtered, "09_data_high_heritability.csv", run_dir)
        )
        files.append(self.save_json(removed_details, "09_removed_traits.json", run_dir))
        files.append(
            self.save_json(summary, "09_heritability_filter_summary.json", run_dir)
        )
        files.append(threshold_plot_path)

        # Create metadata
        metadata = {
            "filtering_enabled": True,
            "threshold": threshold,
            "traits_retained": high_h2_traits,
            "traits_removed": removed_traits,
            "removed_details": removed_details,
            "summary": summary,
            "trait_names": final_traits,  # Pass through for next step
            "valid_trait_names": final_traits,  # For consistency with other steps
        }

        return StepResult(data=df_filtered, metadata=metadata, files_generated=files)
