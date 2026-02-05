"""Step: Visualize cross-platform correlations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sleap_roots_analyze.cross_experiment_analysis import (
    create_correlation_summary_plot,
    create_joint_plot,
    create_genotype_boxplots,
    calculate_genotype_means,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


def _sanitize_trait_for_filename(trait: str) -> str:
    """Sanitize trait name for use in filenames.

    Replaces path separators and spaces with underscores while preserving
    other characters that are valid in trait names.

    Args:
        trait: Original trait name.

    Returns:
        Sanitized trait name safe for filenames.
    """
    return trait.replace("/", "_").replace("\\", "_").replace(" ", "_")


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

        # 1. Create correlation summary plot (uses Spearman as primary metric)
        # Pass significant_col to use FDR-corrected significance count
        fig = create_correlation_summary_plot(
            correlation_df,
            correlation_col="spearman_r",
            pvalue_col="spearman_p",
            exp1_trait_col="exp1_trait",
            exp2_trait_col="exp2_trait",
            figsize=config.figsize_summary,
            top_n=config.top_n_correlations,
            significant_col="significant_fdr",
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
        # Pass pre-computed correlation values from CSV (single source of truth)
        # This prevents discrepancies when min_samples_per_genotype filters genotypes
        # from the correlation calculation vs the plotting data
        n_joint_plots = min(config.top_n_joint_plots, len(correlation_df))
        for i in range(n_joint_plots):
            row = correlation_df.iloc[i]
            trait1 = row["exp1_trait"]
            trait2 = row["exp2_trait"]

            fig = create_joint_plot(
                exp1_means,
                exp2_means,
                trait1,
                trait2,
                exp1_name=exp1_name,
                exp2_name=exp2_name,
                figsize=config.figsize_joint,
                # Pass pre-computed values from CSV to ensure consistency
                # Both Spearman and Pearson are stored - single source of truth
                correlation=row["spearman_r"],
                p_value=row["spearman_p"],
                n_genotypes=row["n_genotypes"],
                pearson_r=row["pearson_r"],
                pearson_p=row["pearson_p"],
            )

            # Sanitize trait names for filename
            trait1_clean = _sanitize_trait_for_filename(trait1)
            trait2_clean = _sanitize_trait_for_filename(trait2)

            joint_output = (
                run_dir
                / f"cross_platform_joint_{i + 1:02d}_{trait1_clean}_vs_{trait2_clean}.png"
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
            trait1_clean = _sanitize_trait_for_filename(trait1)
            trait2_clean = _sanitize_trait_for_filename(trait2)

            boxplot_output = (
                run_dir
                / f"cross_platform_boxplot_{i + 1:02d}_{trait1_clean}_vs_{trait2_clean}.png"
            )
            fig.savefig(boxplot_output, dpi=300, bbox_inches="tight")
            plt.close(fig)
            files_generated.append(str(boxplot_output))

        # 4. Create representative heatmap when clustering is enabled
        # Check config directly since metadata may not be passed through all steps
        representative_heatmap_generated = False
        trait_reduction_method = getattr(config, "trait_reduction_method", "none")
        if trait_reduction_method == "clustering":
            heatmap_file = self._create_representative_heatmap(
                correlation_df,
                exp1_traits,
                exp2_traits,
                exp1_name,
                exp2_name,
                run_dir,
            )
            if heatmap_file:
                files_generated.append(str(heatmap_file))
                representative_heatmap_generated = True

        # Prepare metadata
        metadata = {
            "plots_generated": len(files_generated),
            "summary_plots": 1,
            "joint_plots": n_joint_plots,
            "boxplots": n_boxplots,
            "representative_heatmap": representative_heatmap_generated,
        }

        return StepResult(
            data=data,  # Pass through the data
            metadata=metadata,
            files_generated=files_generated,
        )

    def _create_representative_heatmap(
        self,
        correlation_df: pd.DataFrame,
        exp1_traits: list[str],
        exp2_traits: list[str],
        exp1_name: str,
        exp2_name: str,
        run_dir: Path,
    ) -> Optional[Path]:
        """Create heatmap of cross-platform correlations with significance annotations.

        Args:
            correlation_df: DataFrame with correlation results including spearman_r,
                spearman_p_adjusted, and significant_fdr columns.
            exp1_traits: List of exp1 trait names (representatives if clustered).
            exp2_traits: List of exp2 trait names (representatives if clustered).
            exp1_name: Display name for exp1.
            exp2_name: Display name for exp2.
            run_dir: Directory to save output.

        Returns:
            Path to saved heatmap file, or None if generation failed.
        """
        if correlation_df.empty:
            return None

        # Pivot correlation data to matrix form
        # Filter to only traits that exist in the data
        available_exp1 = set(correlation_df["exp1_trait"].unique())
        available_exp2 = set(correlation_df["exp2_trait"].unique())

        exp1_filtered = [t for t in exp1_traits if t in available_exp1]
        exp2_filtered = [t for t in exp2_traits if t in available_exp2]

        if not exp1_filtered or not exp2_filtered:
            return None

        # Filter to representative traits
        filtered_df = correlation_df[
            (correlation_df["exp1_trait"].isin(exp1_filtered))
            & (correlation_df["exp2_trait"].isin(exp2_filtered))
        ]

        if filtered_df.empty:
            return None

        # Create correlation matrix
        corr_matrix = filtered_df.pivot(
            index="exp1_trait", columns="exp2_trait", values="spearman_r"
        )

        # Create significance matrix for annotations
        if "significant_fdr" in filtered_df.columns:
            sig_matrix = filtered_df.pivot(
                index="exp1_trait", columns="exp2_trait", values="significant_fdr"
            )
        else:
            sig_matrix = None

        # Reorder to match trait order
        corr_matrix = corr_matrix.reindex(
            index=[t for t in exp1_filtered if t in corr_matrix.index],
            columns=[t for t in exp2_filtered if t in corr_matrix.columns],
        )
        if sig_matrix is not None:
            sig_matrix = sig_matrix.reindex(
                index=corr_matrix.index, columns=corr_matrix.columns
            )

        # Create figure
        n_exp1 = len(corr_matrix.index)
        n_exp2 = len(corr_matrix.columns)
        fig_width = max(10, n_exp2 * 0.8 + 3)
        fig_height = max(8, n_exp1 * 0.6 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create custom annotations with significance markers
        annot_matrix = corr_matrix.copy().astype(str)
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                val = corr_matrix.loc[row, col]
                if pd.isna(val):
                    annot_matrix.loc[row, col] = ""
                else:
                    # Add asterisk for FDR-significant correlations
                    if sig_matrix is not None and sig_matrix.loc[row, col]:
                        annot_matrix.loc[row, col] = f"{val:.2f}*"
                    else:
                        annot_matrix.loc[row, col] = f"{val:.2f}"

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=annot_matrix,
            fmt="",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
            linewidths=0.5,
            linecolor="white",
        )

        # Set labels
        ax.set_title(
            f"Cross-Platform Correlations: {exp1_name} vs {exp2_name}\n"
            f"(Representative traits, * = FDR significant)"
        )
        ax.set_xlabel(f"{exp2_name} Traits")
        ax.set_ylabel(f"{exp1_name} Traits")

        # Rotate labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

        plt.tight_layout()

        # Save figure
        output_path = run_dir / "cross_platform_representative_heatmap.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return output_path
