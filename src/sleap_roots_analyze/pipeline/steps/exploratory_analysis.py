"""Step 4: Exploratory data analysis and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt

from sleap_roots_analyze.statistics import calculate_trait_statistics
from sleap_roots_analyze.visualization import (
    _generate_trait_boxplot_batches,
    _generate_trait_histogram_batches,
    create_correlation_heatmap,
    create_exploratory_summary_plots,
    create_trait_eda_plots,
)
from sleap_roots_analyze.viz_utils import calculate_correlation_matrix_size
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class ExploratoryAnalysisStep(BaseStep):
    """Perform exploratory data analysis on cleaned data.

    This step generates comprehensive visualizations and statistics:
    1. Trait distributions (histograms)
    2. Missing data patterns
    3. Trait ranges by genotype (boxplots)
    4. Sample counts per genotype
    5. Trait correlation heatmap
    6. Detailed EDA plots (NaN/zero/outlier fractions, variance)
    7. Batched trait visualizations (auto-generated for >16 traits)

    Outputs:
        - figures/*.png: All generated visualizations
        - figures/04_trait_histograms_batch_*.png: Batched histograms (if >16 traits)
        - figures/04_trait_boxplots_batch_*.png: Batched boxplots (if >16 traits)
        - trait_statistics.json: Comprehensive trait statistics
    """

    def __init__(self):
        """Initialize ExploratoryAnalysisStep."""
        super().__init__(
            step_name="ExploratoryAnalysis",
            description="Generate EDA visualizations and statistics",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the exploratory analysis step.

        Args:
            data: DataFrame from previous step (ValidateCleanStep).
            config: Pipeline configuration with:
                - columns.genotype: Genotype column name
                - cleanup.max_zeros_per_trait: Zero threshold for EDA plots
                - cleanup.max_nans_per_trait: NaN threshold for EDA plots
                - cleanup.min_samples_per_trait: Min samples for EDA plots
            run_dir: Directory to save outputs.
            prev_result: Result from ValidateCleanStep (contains trait names and cleanup log).

        Returns:
            StepResult with statistics and list of generated figure files.
        """
        df = data

        # Get trait columns and cleanup log from previous step
        trait_cols = prev_result.metadata["valid_trait_names"]
        cleanup_log = prev_result.metadata.get("cleanup_log", {})

        # Get sanitized column names from CleanupTraitsStep metadata
        column_mapping = prev_result.metadata.get("column_mapping", {})
        genotype_col = column_mapping.get("genotype", "Genotype")

        # Create figures directory
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # 1. Calculate trait statistics
        stats = calculate_trait_statistics(df=df, trait_cols=trait_cols)

        # Save and close each figure as soon as it's produced instead of
        # accumulating every figure in memory before saving any of them
        # (Issue #110: peak memory was the sum of every figure generated in
        # this step -- e.g. dozens of batched boxplot figures, a full
        # correlation heatmap, and histograms all held simultaneously --
        # rather than the size of the single largest one).
        files: list[Path] = []
        figure_names: list[str] = []

        def _save_and_close(name: str, fig: plt.Figure) -> None:
            fig_path = figures_dir / f"{name}.{config.visualization.figure_format}"
            try:
                fig.savefig(
                    fig_path,
                    dpi=config.visualization.dpi,
                    bbox_inches=config.visualization.bbox_inches,
                    facecolor=config.visualization.facecolor,
                    edgecolor=config.visualization.edgecolor,
                    transparent=config.visualization.transparent,
                )
            finally:
                # Always close, even if savefig raises, so a save failure
                # can't leak the figure back into the unbounded-accumulation
                # pattern this step exists to avoid (Issue #110).
                plt.close(fig)
            files.append(fig_path)
            figure_names.append(name)

        # 2. Create exploratory summary plots
        # Pass adaptive config if enabled
        adaptive_cfg = (
            config.adaptive_sizing if config.adaptive_sizing.enabled else None
        )
        summary_figures = create_exploratory_summary_plots(
            df=df,
            trait_cols=trait_cols,
            genotype_col=genotype_col,
            adaptive_config=adaptive_cfg,
        )
        for fig_name, fig in summary_figures.items():
            _save_and_close(fig_name, fig)

        # 3. Create detailed EDA plots with cleanup thresholds
        thresholds = {
            "zero": config.cleanup.max_zeros_per_trait,
            "nan": config.cleanup.max_nans_per_trait,
            "outlier": 0.1,  # IQR outliers for visualization only
        }

        eda_figures = create_trait_eda_plots(
            df=df,
            trait_cols=trait_cols,
            thresholds=thresholds,
            cleanup_log=cleanup_log,
            min_samples_per_trait=config.cleanup.min_samples_per_trait,
        )
        for fig_name, fig in eda_figures.items():
            _save_and_close(fig_name, fig)

        # 3.5. Add full correlation heatmap (separate from summary plots)
        # Use adaptive sizing if enabled
        if config.adaptive_sizing and config.adaptive_sizing.enabled:
            corr_figsize = calculate_correlation_matrix_size(
                n_traits=len(trait_cols), config=config.adaptive_sizing
            )
        else:
            corr_figsize = tuple(config.visualization.figsize)

        full_corr_fig = create_correlation_heatmap(
            df=df,
            trait_cols=trait_cols,
            figsize=corr_figsize,
        )
        _save_and_close("full_correlation_heatmap", full_corr_fig)

        # 4. Add batched trait visualizations if enabled and threshold exceeded.
        # Generate, save, and close one batch at a time (rather than collecting
        # the full list first) so at most one batch figure exists at a time.
        if (
            config.visualization.enable_batched_plots
            and len(trait_cols) > config.visualization.batched_plot_threshold
        ):
            for i, fig in enumerate(
                _generate_trait_histogram_batches(
                    df=df,
                    trait_cols=trait_cols,
                    batch_size=config.visualization.batch_size,
                )
            ):
                _save_and_close(f"04_trait_histograms_batch_{i + 1}", fig)

            for i, fig in enumerate(
                _generate_trait_boxplot_batches(
                    df=df,
                    trait_cols=trait_cols,
                    genotype_col=genotype_col,
                    batch_size=config.visualization.batch_size,
                )
            ):
                _save_and_close(f"04_trait_boxplots_batch_{i + 1}", fig)

        # Save statistics
        stats_path = self.save_json(stats, "trait_statistics.json", run_dir)
        files.append(stats_path)

        # Create metadata
        metadata = {
            "samples": len(df),
            "traits": len(trait_cols),
            "figures_generated": len(figure_names),
            "figure_names": figure_names,
            "statistics": stats,
            "trait_names": trait_cols,  # Primary key (standardized)
            "valid_trait_names": trait_cols,  # For consistency
        }

        return StepResult(data=df, metadata=metadata, files_generated=files)
