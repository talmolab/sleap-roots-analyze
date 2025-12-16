"""Step: Visualize cross-platform correlations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd

from sleap_roots_analyze.cross_experiment_analysis import (
    create_correlation_summary_plot,
    create_joint_plot,
    create_genotype_boxplots,
    calculate_genotype_means,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class VisualizeCrossPlatformStep(BaseStep):
    """Visualize cross-platform trait correlations.

    This step:
    1. Creates correlation summary plot (4-panel visualization)
    2. Generates joint plots for top N correlations
    3. Creates genotype boxplots for top N correlations
    4. Saves all visualizations to files

    Outputs:
        - cross_platform_correlation_summary.png: 4-panel summary visualization
        - cross_platform_joint_*.png: Joint plots for top correlations
        - cross_platform_boxplot_*.png: Boxplots for top correlations
    """

    def __init__(self):
        """Initialize VisualizeCrossPlatformStep."""
        super().__init__(
            step_name="VisualizeCrossPlatform",
            description="Visualize cross-platform trait correlations",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the cross-platform visualization step.

        Args:
            data: Dictionary containing correlation_df, exp1_df, exp2_df
            config: CrossPlatformConfig with visualization parameters
            run_dir: Directory to save outputs
            prev_result: Previous step result with metadata

        Returns:
            StepResult with visualization files generated

        Raises:
            ValueError: If required data is missing
        """
        # Validate inputs
        if prev_result is None:
            raise ValueError("prev_result is required for visualization")

        # Get data from previous step
        correlation_df = data["correlation_df"]
        exp1_df = data["exp1_df"]
        exp2_df = data["exp2_df"]

        # Get experiment names from metadata
        exp1_name = prev_result.metadata.get("exp1_name", "Experiment 1")
        exp2_name = prev_result.metadata.get("exp2_name", "Experiment 2")

        files_generated = []

        # 1. Create correlation summary plot
        fig = create_correlation_summary_plot(
            correlation_df,
            correlation_col="correlation",
            pvalue_col="p_value",
            exp1_trait_col="exp1_trait",
            exp2_trait_col="exp2_trait",
            figsize=config.figsize_summary,
            top_n=config.top_n_correlations,
        )

        summary_output = run_dir / "cross_platform_correlation_summary.png"
        fig.savefig(summary_output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        files_generated.append(str(summary_output))

        # Calculate genotype means for joint plots and boxplots
        exp1_traits = prev_result.metadata.get("exp1_trait_names", [])
        exp2_traits = prev_result.metadata.get("exp2_trait_names", [])

        exp1_means = calculate_genotype_means(
            exp1_df,
            trait_cols=exp1_traits,
            genotype_col="genotype",
        )

        exp2_means = calculate_genotype_means(
            exp2_df,
            trait_cols=exp2_traits,
            genotype_col="genotype",
        )

        # 2. Create joint plots for top N correlations
        n_joint_plots = min(config.top_n_joint_plots, len(correlation_df))
        for i in range(n_joint_plots):
            row = correlation_df.iloc[i]
            trait1 = row["exp1_trait"]
            trait2 = row["exp2_trait"]
            corr = row["correlation"]

            fig = create_joint_plot(
                exp1_means,
                exp2_means,
                trait1,
                trait2,
                exp1_name=exp1_name,
                exp2_name=exp2_name,
                figsize=config.figsize_joint,
            )

            # Sanitize trait names for filename
            trait1_clean = trait1.replace("/", "_").replace("\\", "_").replace(" ", "_")
            trait2_clean = trait2.replace("/", "_").replace("\\", "_").replace(" ", "_")

            joint_output = (
                run_dir
                / f"cross_platform_joint_{i+1:02d}_{trait1_clean}_vs_{trait2_clean}.png"
            )
            fig.savefig(joint_output, dpi=300, bbox_inches="tight")
            plt.close(fig)
            files_generated.append(str(joint_output))

        # 3. Create genotype boxplots for top N correlations
        n_boxplots = min(config.top_n_boxplots, len(correlation_df))
        for i in range(n_boxplots):
            row = correlation_df.iloc[i]
            trait1 = row["exp1_trait"]
            trait2 = row["exp2_trait"]

            fig = create_genotype_boxplots(
                exp1_df,
                exp2_df,
                trait1,
                trait2,
                genotype_col="genotype",
                exp1_name=exp1_name,
                exp2_name=exp2_name,
                figsize=config.figsize_boxplot,
            )

            # Sanitize trait names for filename
            trait1_clean = trait1.replace("/", "_").replace("\\", "_").replace(" ", "_")
            trait2_clean = trait2.replace("/", "_").replace("\\", "_").replace(" ", "_")

            boxplot_output = (
                run_dir
                / f"cross_platform_boxplot_{i+1:02d}_{trait1_clean}_vs_{trait2_clean}.png"
            )
            fig.savefig(boxplot_output, dpi=300, bbox_inches="tight")
            plt.close(fig)
            files_generated.append(str(boxplot_output))

        # Prepare metadata
        metadata = {
            "plots_generated": len(files_generated),
            "summary_plots": 1,
            "joint_plots": n_joint_plots,
            "boxplots": n_boxplots,
        }

        return StepResult(
            data=data,  # Pass through the data
            metadata=metadata,
            files_generated=files_generated,
        )
