"""Step 00f: Visualize root depth profiles from aggregated core data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.depth_profile_plots import (
    plot_depth_profile_faceted,
    plot_depth_profile_replicates,
    plot_biomass_depth_barplot,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class VisualizeDepthProfilesStep(BaseStep):
    """Generate depth profile visualizations from root core data.

    This step creates faceted line plots and spaghetti plots showing root
    measurements vs depth for each genotype. It runs after core aggregation
    (Step 00d) and uses the long-format aggregated data.

    Outputs:
        For each data source (biomass, counting):
        - figures/00f_depth_profile_{source}_mean.png: Faceted line plots (mean ± SE)
        - figures/00f_depth_profile_{source}_reps.png: Faceted spaghetti plots (individual reps)
        - 00f_depth_profile_metadata.json: Metadata about generated plots

    This step only runs if:
    1. Root core processing is enabled (config.root_core exists)
    2. Depth profile generation is enabled (config.root_core.generate_depth_profiles)
    3. Aggregated CSV files exist from Step 00d
    """

    def __init__(self):
        """Initialize VisualizeDepthProfilesStep."""
        super().__init__(
            step_name="VisualizeDepthProfiles",
            description="Generate depth profile visualizations",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute depth profile visualization.

        Args:
            data: Dict of aggregated DataFrames from Step 00d (passed through).
            config: Pipeline configuration with:
                - root_core.sources: List of data sources (biomass, counting)
                - root_core.generate_depth_profiles: Whether to generate plots
                - visualization.dpi: DPI for saved figures
            run_dir: Directory to save outputs.
            prev_result: Result from Step 00d (AggregateCoresStep).

        Returns:
            StepResult with pass-through data and metadata about generated plots.
        """
        # Check if depth profile generation is enabled
        if not hasattr(config, "root_core") or not config.root_core:
            # No root core config - skip this step
            return StepResult(
                data=data,
                metadata={"skipped": True, "reason": "no_root_core_config"},
                files_generated=[],
            )

        if not config.root_core.generate_depth_profiles:
            # Depth profiles disabled - skip this step
            return StepResult(
                data=data,
                metadata={"skipped": True, "reason": "depth_profiles_disabled"},
                files_generated=[],
            )

        files = []
        metadata = {"sources": []}

        # Create figures directory
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Process each data source
        for source in config.root_core.sources:
            data_type = source.data_type
            value_col = source.value_column_name

            # Look for aggregated CSV from Step 00c (saved by aggregate_cores)
            csv_path = run_dir / f"00c_root_core_{data_type}_aggregated.csv"
            if not csv_path.exists():
                logger.warning(
                    f"Aggregated CSV not found for {data_type}: {csv_path}. Skipping depth profiles."
                )
                continue

            # Load the aggregated data
            df = pd.read_csv(csv_path)

            # Verify required columns exist
            required_cols = ["Depth_cm", value_col, "geno"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logger.warning(
                    f"Missing required columns for {data_type}: {missing}. Skipping."
                )
                continue

            logger.info(f"Generating depth profile plots for {data_type}...")

            try:
                # 1. Generate faceted line plot (mean ± SE)
                mean_plot_path = figures_dir / f"00f_depth_profile_{data_type}_mean.png"
                plot_depth_profile_faceted(
                    df=df,
                    x="Depth_cm",
                    y=value_col,
                    facet_col="geno",
                    errorbar="se",
                    col_wrap=4,
                    height=4,
                    output_path=mean_plot_path,  # Let function handle rotation
                )
                files.append(mean_plot_path)
                logger.info(f"  - Saved mean plot: {mean_plot_path}")

                # 2. Generate faceted spaghetti plot (individual replicates)
                # Check if we have a replicate identifier column
                rep_col = None
                for possible_rep in ["Rep", "plot_rep", "Replicate"]:
                    if possible_rep in df.columns:
                        rep_col = possible_rep
                        break

                if rep_col:
                    reps_plot_path = (
                        figures_dir / f"00f_depth_profile_{data_type}_reps.png"
                    )
                    plot_depth_profile_replicates(
                        df=df,
                        x="Depth_cm",
                        y=value_col,
                        facet_col="geno",
                        hue=rep_col,
                        col_wrap=4,
                        height=4,
                        alpha=0.6,
                        output_path=reps_plot_path,  # Let function handle rotation
                    )
                    files.append(reps_plot_path)
                    logger.info(f"  - Saved replicate plot: {reps_plot_path}")
                else:
                    logger.warning(
                        f"No replicate column found for {data_type}. Skipping replicate plot."
                    )
                    reps_plot_path = None

                # 3. Generate barplot for biomass data (grouped by depth interval)
                barplot_path = None
                if data_type == "biomass":
                    logger.info(f"  - Generating biomass barplot for {data_type}...")
                    barplot_path = (
                        figures_dir / f"00f_depth_profile_{data_type}_barplot.png"
                    )
                    
                    # Use same aggregation method as core aggregation for consistency
                    estimator = source.aggregation_method if hasattr(source, 'aggregation_method') else "median"
                    
                    plot_biomass_depth_barplot(
                        df=df,
                        depth_col="Depth_cm",
                        value_col=value_col,
                        genotype_col="geno",
                        include_points=True,  # Show individual plot points
                        estimator=estimator,
                        output_path=barplot_path,
                    )
                    files.append(barplot_path)
                    logger.info(f"  - Saved barplot: {barplot_path}")

                # Track metadata
                n_genotypes = df["geno"].nunique()
                n_depths = df["Depth_cm"].nunique()
                source_metadata = {
                    "data_type": data_type,
                    "value_column": value_col,
                    "n_genotypes": n_genotypes,
                    "n_depths": n_depths,
                    "mean_plot": str(mean_plot_path),
                    "reps_plot": str(reps_plot_path) if reps_plot_path else None,
                }
                if barplot_path:
                    source_metadata["barplot"] = str(barplot_path)
                metadata["sources"].append(source_metadata)

            except Exception as e:
                logger.error(f"Failed to generate depth profiles for {data_type}: {e}")
                metadata["sources"].append(
                    {
                        "data_type": data_type,
                        "error": str(e),
                        "status": "failed",
                    }
                )

        # Save metadata
        metadata_path = self.save_json(
            metadata, "00f_depth_profile_metadata.json", run_dir
        )
        files.append(metadata_path)

        # Pass through the data unchanged
        return StepResult(data=data, metadata=metadata, files_generated=files)
