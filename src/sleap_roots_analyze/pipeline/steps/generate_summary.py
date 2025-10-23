"""Step 10: Generate complete pipeline summary."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.pipeline.utils import get_code_snapshot


class GenerateSummaryStep(BaseStep):
    """Generate complete QC pipeline summary.

    This final step aggregates all metadata from previous steps and creates
    a comprehensive summary of the entire pipeline run, including code snapshot
    for reproducibility.

    Outputs:
        - 10_final_data.csv: Final cleaned and QC'd data
        - 10_pipeline_summary.json: Complete pipeline summary with all metadata
        - code_snapshot/ (optional): Git info or code archive for reproducibility
    """

    def __init__(self):
        """Initialize GenerateSummaryStep."""
        super().__init__(
            step_name="GenerateSummary",
            description="Generate complete pipeline summary",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute summary generation.

        Args:
            data: DataFrame from previous step (FilterHeritabilityStep).
            config: Pipeline configuration.
            run_dir: Directory to save outputs.
            prev_result: Result from FilterHeritabilityStep.

        Returns:
            StepResult with final DataFrame and complete pipeline summary.
        """
        df = data.copy()

        # Get code snapshot for reproducibility
        code_snapshot = get_code_snapshot(run_dir, create_archive_if_dirty=True)

        # Aggregate metadata from all previous steps
        # This assumes we have access to all previous step results through the metadata chain
        # In practice, the orchestrator would pass this information

        # Get final trait list
        final_traits = prev_result.metadata.get("trait_names", [])

        # Build comprehensive summary
        summary = {
            "pipeline_info": {
                "pipeline_name": config.pipeline_name,
                "version": config.version,
                "run_timestamp": datetime.now().isoformat(),
                "run_directory": run_dir.as_posix(),
            },
            "code_snapshot": code_snapshot,
            "configuration": {
                "columns": {
                    "barcode": config.columns.barcode,
                    "genotype": config.columns.genotype,
                    "replicate": config.columns.replicate,
                },
                "data": {
                    "csv_path": Path(config.data.csv_path).as_posix(),
                    "image_dir": (
                        Path(config.data.image_dir).as_posix()
                        if config.data.image_dir
                        else None
                    ),
                    "additional_exclude_cols": config.data.additional_exclude_cols,
                    "traits_to_include": config.data.traits_to_include,
                    "traits_to_exclude": config.data.traits_to_exclude,
                },
                "cleanup": {
                    "max_nan_fraction": config.cleanup.max_nan_fraction,
                    "max_zeros_per_trait": config.cleanup.max_zeros_per_trait,
                    "max_nans_per_trait": config.cleanup.max_nans_per_trait,
                    "min_samples_per_trait": config.cleanup.min_samples_per_trait,
                },
                "outlier_detection": {
                    "traditional_methods": config.outlier_detection.traditional_methods,
                    "clustering_methods": config.outlier_detection.clustering_methods,
                },
                "outlier_removal": {
                    "strategy": config.outlier_removal.strategy,
                    "method": config.outlier_removal.method,
                    "min_methods": config.outlier_removal.min_methods,
                },
                "heritability": {
                    "enabled": config.heritability.enabled,
                    "threshold": config.heritability.threshold,
                },
            },
            "final_data": {
                "n_samples": len(df),
                "n_traits": len(final_traits),
                "n_genotypes": df[config.columns.genotype].nunique(),
                "trait_names": final_traits,
            },
            "steps_executed": {
                "01_load_data": "Loaded and validated CSV data",
                "02_cleanup_traits": "Removed problematic traits and samples",
                "03_validate_clean": "Validated no NaN values remain",
                "04_exploratory_analysis": "Generated EDA visualizations",
                "05_detect_outliers": "Detected outliers using configured methods",
                "06_visualize_outliers": "Created outlier visualizations",
                "07_remove_outliers": "Removed outliers based on strategy",
                "08_statistical_analysis": "Calculated ANOVA and heritability",
                "09_filter_heritability": (
                    "Filtered low heritability traits"
                    if config.heritability.enabled
                    else "Skipped (heritability filtering disabled)"
                ),
                "10_generate_summary": "Generated complete pipeline summary",
            },
        }

        # Try to include step-specific summaries from previous metadata
        # (This would be enhanced if we had access to all previous step results)
        if prev_result and prev_result.metadata:
            summary["step_summaries"] = {}

            # Add heritability filter summary if available
            if "summary" in prev_result.metadata:
                summary["step_summaries"]["heritability_filter"] = prev_result.metadata[
                    "summary"
                ]

        # Save outputs
        files = []
        files.append(self.save_dataframe(df, "10_final_data.csv", run_dir))
        files.append(self.save_json(summary, "10_pipeline_summary.json", run_dir))

        # Create metadata
        metadata = {
            "pipeline_complete": True,
            "summary": summary,
            "final_traits": final_traits,
            "final_n_samples": len(df),
            "final_n_traits": len(final_traits),
            "final_n_genotypes": df[config.columns.genotype].nunique(),
        }

        return StepResult(data=df, metadata=metadata, files_generated=files)
