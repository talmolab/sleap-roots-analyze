"""Step 9: Filter out low heritability traits."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

from sleap_roots_analyze.data_cleanup import get_trait_columns
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.statistics import (
    analyze_heritability_thresholds,
    compare_trait_heritabilities,
    identify_high_heritability_traits,
)
from sleap_roots_analyze.visualization import (
    create_heritability_threshold_plot,
    create_trait_by_genotype_boxplots,
    create_variance_decomposition_plot,
)
from sleap_roots_analyze.viz_utils import calculate_figure_size, calculate_barplot_size


class FilterHeritabilityStep(BaseStep):
    """Filter out traits with low heritability.

    This step removes traits that have heritability below the configured threshold.
    Only runs if heritability filtering is enabled in the configuration.

    Outputs:
        - 09_data_high_heritability.csv: Data with only high heritability traits
        - 09_removed_traits.json: List of removed low heritability traits
        - 09_heritability_filter_summary.json: Summary of filtering
        - figures/09_heritability_threshold_analysis.png: Threshold analysis plot

    Optional Diagnostic Outputs (if generate_diagnostics=True):
        - 09_heritability_diagnostics.csv: Variance component comparison for all traits
        - figures/09_variance_decomposition.png: 4-panel variance decomposition plot
        - figures/09_removed_traits_boxplots.png: Boxplots of removed traits by genotype
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

        # Get visualization config (different attribute names in QC vs Viz pipelines)
        viz_config = (
            config.visualization
            if hasattr(config, "visualization")
            else config.static_viz
        )

        # Get heritability results from previous step
        heritability_results = prev_result.metadata["heritability_results"]
        trait_cols = prev_result.metadata["trait_names"]

        # Defense-in-depth: prevent silent trait removal when heritability_results
        # is empty but filtering is enabled (e.g., config mismatch where
        # statistics.calculate_heritability=False but heritability.enabled=True)
        if config.heritability.enabled and not heritability_results:
            heritability_summary = prev_result.metadata.get("summary", {}).get(
                "heritability_summary", {}
            )
            was_skipped = heritability_summary.get("skipped", False)

            if was_skipped:
                guard_reason = "heritability_calculation_disabled"
                logger.warning(
                    "Heritability filtering is enabled but heritability calculation "
                    "was disabled (statistics.calculate_heritability=False). "
                    "Skipping filtering to prevent silent data loss. "
                    "Set heritability.enabled=False or enable heritability calculation."
                )
            else:
                guard_reason = "unexpected_empty_results"
                logger.warning(
                    "Heritability filtering is enabled but heritability_results is "
                    "empty. Skipping filtering to prevent silent data loss."
                )

            summary = {
                "filtering_enabled": True,
                "threshold": config.heritability.threshold,
                "traits_original": len(trait_cols),
                "traits_retained": len(trait_cols),
                "traits_removed": 0,
                "removed_trait_names": [],
                "guard_activated": True,
                "guard_reason": guard_reason,
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
                **prev_result.metadata,
                "filtering_enabled": True,
                "threshold": config.heritability.threshold,
                "traits_retained": trait_cols,
                "traits_removed": [],
                "summary": summary,
                "trait_names": trait_cols,
                "valid_trait_names": trait_cols,
                "guard_activated": True,
                "guard_reason": guard_reason,
            }

            return StepResult(data=df, metadata=metadata, files_generated=files)

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
                **prev_result.metadata,  # Preserve metadata from previous steps (e.g., pca_results)
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
            heritability_results=heritability_results, threshold=threshold
        )

        # Determine which traits to remove
        removed_traits = [t for t in trait_cols if t not in high_h2_traits]

        # Remove low heritability trait columns from DataFrame
        # But keep all metadata columns (automatically handled by dropping only trait columns)
        df_filtered = df.drop(columns=removed_traits).copy()

        # Verify the high heritability traits are still present
        # After CleanupTraitsStep (Step 02), columns are sanitized to standard names
        final_traits = get_trait_columns(
            df_filtered,
            barcode_col="Barcode",
            genotype_col="Genotype",
            replicate_col="Replicate",
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

        # Generate diagnostics if enabled
        diagnostic_results = None
        if config.heritability.generate_diagnostics and removed_traits:
            try:
                # Generate comparison DataFrame for all traits
                # After CleanupTraitsStep (Step 02), columns are sanitized to standard names
                comparison_df = compare_trait_heritabilities(
                    df=df,
                    traits=trait_cols,
                    heritability_results=heritability_results,
                    genotype_col="Genotype",
                    replicate_col="Replicate",
                    sort_by="heritability",  # Sort by heritability (lowest first)
                )

                # Save diagnostic comparison CSV
                diag_csv_path = run_dir / "09_heritability_diagnostics.csv"
                comparison_df.to_csv(diag_csv_path, index=False)

                # Create variance decomposition plot for all traits
                # Use adaptive sizing if enabled
                # Variance decomposition has traits on X-axis, so scale width
                if config.adaptive_sizing and config.adaptive_sizing.enabled:
                    from sleap_roots_analyze.viz_utils import calculate_barplot_size

                    # Calculate size for each subplot based on trait count
                    subplot_size = calculate_barplot_size(
                        n_items=len(comparison_df),  # Number of traits
                        config=config.adaptive_sizing,
                        orientation="vertical",  # Traits on X-axis
                        as_subplot=True,
                        n_subplots=4,  # 4 panels in 2x2 grid
                    )

                    # Total figure size for 2x2 grid
                    # Width scales with traits (2 subplots wide), height stays reasonable
                    var_figsize = (subplot_size[0] * 2, 8.0)
                else:
                    var_figsize = (14, 10)

                fig_var = create_variance_decomposition_plot(
                    comparison_df=comparison_df,
                    figsize=var_figsize,
                    output_path=None,  # Will save manually
                    threshold=config.heritability.threshold,
                )
                var_plot_path = run_dir / "figures" / "09_variance_decomposition.png"
                var_plot_path.parent.mkdir(parents=True, exist_ok=True)
                fig_var.savefig(
                    var_plot_path,
                    dpi=viz_config.dpi,
                    bbox_inches=viz_config.bbox_inches,
                    facecolor=getattr(viz_config, "facecolor", None),
                    edgecolor=getattr(viz_config, "edgecolor", None),
                    transparent=getattr(viz_config, "transparent", False),
                )
                plt.close(fig_var)

                # Create boxplots for removed traits (limit to top 10 if many)
                traits_to_plot = (
                    removed_traits[:10] if len(removed_traits) > 10 else removed_traits
                )
                adaptive_cfg = (
                    config.adaptive_sizing if config.adaptive_sizing.enabled else None
                )
                # After CleanupTraitsStep (Step 02), columns are sanitized to standard names
                fig_box = create_trait_by_genotype_boxplots(
                    df=df,
                    traits=traits_to_plot,
                    heritability_results=heritability_results,
                    genotype_col="Genotype",
                    output_path=None,  # Will save manually
                    adaptive_config=adaptive_cfg,
                )
                box_plot_path = run_dir / "figures" / "09_removed_traits_boxplots.png"
                fig_box.savefig(
                    box_plot_path,
                    dpi=viz_config.dpi,
                    bbox_inches=viz_config.bbox_inches,
                    facecolor=getattr(viz_config, "facecolor", None),
                    edgecolor=getattr(viz_config, "edgecolor", None),
                    transparent=getattr(viz_config, "transparent", False),
                )
                plt.close(fig_box)

                # Store diagnostic results for metadata
                diagnostic_results = {
                    "comparison_df": comparison_df.to_dict("records"),
                    "diagnostic_csv": str(diag_csv_path),
                    "variance_plot": str(var_plot_path),
                    "boxplot": str(box_plot_path),
                    "traits_analyzed": len(trait_cols),
                    "traits_removed_plotted": len(traits_to_plot),
                }

                logger.info(f"Generated heritability diagnostics:")
                logger.info(f"  - Comparison CSV: {diag_csv_path}")
                logger.info(f"  - Variance decomposition: {var_plot_path}")
                logger.info(f"  - Removed traits boxplots: {box_plot_path}")

            except Exception as e:
                # Log warning but don't fail the step
                logger.warning(f"Failed to generate heritability diagnostics: {e}")
                diagnostic_results = {"error": str(e), "status": "failed"}

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
        threshold_analysis = analyze_heritability_thresholds(
            heritability_results=heritability_results
        )

        # Use adaptive sizing if enabled
        if config.adaptive_sizing and config.adaptive_sizing.enabled:
            threshold_figsize = (10, 6)  # Keep fixed for this plot (single panel)
        else:
            threshold_figsize = (10, 6)

        fig = create_heritability_threshold_plot(
            threshold_analysis=threshold_analysis,
            current_threshold=threshold,
            figsize=threshold_figsize,
        )
        threshold_plot_path = (
            run_dir / "figures" / "09_heritability_threshold_analysis.png"
        )
        threshold_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            threshold_plot_path,
            dpi=viz_config.dpi,
            bbox_inches=viz_config.bbox_inches,
            facecolor=getattr(viz_config, "facecolor", None),
            edgecolor=getattr(viz_config, "edgecolor", None),
            transparent=getattr(viz_config, "transparent", False),
        )
        plt.close(fig)

        # Reorder columns before saving: metadata first, then traits (sorted)
        df_filtered = self.reorder_dataframe_columns(df_filtered, high_h2_traits)

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

        # Add diagnostic files if they were generated
        if diagnostic_results and "error" not in diagnostic_results:
            files.append(Path(diagnostic_results["diagnostic_csv"]))
            files.append(Path(diagnostic_results["variance_plot"]))
            files.append(Path(diagnostic_results["boxplot"]))

        # Create metadata
        metadata = {
            **prev_result.metadata,  # Preserve metadata from previous steps (e.g., pca_results)
            "filtering_enabled": True,
            "threshold": threshold,
            "traits_retained": high_h2_traits,
            "traits_removed": removed_traits,
            "removed_details": removed_details,
            "summary": summary,
            "trait_names": final_traits,  # Pass through for next step
            "valid_trait_names": final_traits,  # For consistency with other steps
        }

        # Add diagnostic results to metadata if they were generated
        if diagnostic_results:
            metadata["diagnostic_results"] = diagnostic_results

        return StepResult(data=df_filtered, metadata=metadata, files_generated=files)
