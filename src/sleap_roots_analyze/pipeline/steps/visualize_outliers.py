"""Step 6: Visualize outlier detection results."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt

from sleap_roots_analyze.outlier_visualization import (
    create_comprehensive_outlier_comparison,
    create_gmm_outlier_plots,
    create_hierarchical_outlier_plots,
    create_isolation_forest_plots,
    create_kmeans_outlier_plots,
    create_mahalanobis_outlier_plots,
    create_outlier_overlap_heatmap,
    create_outliers_per_genotype_plot,
    create_pca_outlier_plot,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class VisualizeOutliersStep(BaseStep):
    """Create visualizations for all outlier detection results.

    This step generates comprehensive visualizations for each outlier
    detection method that was run in the previous step, plus comparison
    plots showing overlap between methods.

    Outputs:
        - figures/outliers_*.png: Method-specific visualizations
        - figures/outlier_comparison.png: Comparison across all methods
        - figures/outlier_overlap_heatmap.png: Overlap heatmap
        - figures/outliers_per_genotype.png: Outliers by genotype
    """

    def __init__(self):
        """Initialize VisualizeOutliersStep."""
        super().__init__(
            step_name="VisualizeOutliers",
            description="Create visualizations for outlier detection results",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute outlier visualization.

        Args:
            data: DataFrame from previous step (DetectOutliersStep).
            config: Pipeline configuration with:
                - columns.genotype: Genotype column for per-genotype plots
                - visualization.dpi: Figure DPI
            run_dir: Directory to save outputs.
            prev_result: Result from DetectOutliersStep (contains outlier results).

        Returns:
            StepResult with list of generated figure files.
        """
        df = data

        # Get outlier results from previous step
        outlier_results = prev_result.metadata["outlier_results"]
        methods_run = prev_result.metadata["methods_run"]

        # Validate that outlier detection was performed
        # methods_run is guaranteed to be a list (empty list [] if no methods)
        if not methods_run:
            warnings.warn(
                "No outlier detection methods were run. Skipping outlier visualization. "
                "Configure outlier_detection.traditional_methods or clustering_methods to enable.",
                UserWarning,
                stacklevel=2,
            )
            # Return early with no files generated, but pass through outlier_results
            metadata = {
                "figures_generated": 0,
                "methods_visualized": [],
                "outlier_results": outlier_results,  # Pass through for next step
                "methods_run": methods_run,  # Pass through for next step
                "trait_names": prev_result.metadata.get("trait_names", []),
                "valid_trait_names": prev_result.metadata.get("valid_trait_names", []),
            }
            return StepResult(data=df, metadata=metadata, files_generated=[])

        # Create figures directory
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        files = []

        # Generate method-specific visualizations
        for method in methods_run:
            result = outlier_results[method]

            if method == "pca":
                fig = create_pca_outlier_plot(result, df)
                fig_path = figures_dir / "outliers_pca.png"
                fig.savefig(fig_path, dpi=config.visualization.dpi, bbox_inches="tight")
                plt.close(fig)
                files.append(fig_path)

            elif method == "isolation_forest":
                figs = create_isolation_forest_plots(result, df)
                for fig_name, fig in figs.items():
                    fig_path = figures_dir / f"outliers_if_{fig_name}.png"
                    fig.savefig(
                        fig_path, dpi=config.visualization.dpi, bbox_inches="tight"
                    )
                    plt.close(fig)
                    files.append(fig_path)

            elif method == "mahalanobis":
                figs = create_mahalanobis_outlier_plots(result, df)
                for fig_name, fig in figs.items():
                    fig_path = figures_dir / f"outliers_mahal_{fig_name}.png"
                    fig.savefig(
                        fig_path, dpi=config.visualization.dpi, bbox_inches="tight"
                    )
                    plt.close(fig)
                    files.append(fig_path)

            elif method == "kmeans":
                figs = create_kmeans_outlier_plots(result, df)
                for fig_name, fig in figs.items():
                    fig_path = figures_dir / f"outliers_kmeans_{fig_name}.png"
                    fig.savefig(
                        fig_path, dpi=config.visualization.dpi, bbox_inches="tight"
                    )
                    plt.close(fig)
                    files.append(fig_path)

            elif method == "gmm":
                figs = create_gmm_outlier_plots(result, df)
                for fig_name, fig in figs.items():
                    fig_path = figures_dir / f"outliers_gmm_{fig_name}.png"
                    fig.savefig(
                        fig_path, dpi=config.visualization.dpi, bbox_inches="tight"
                    )
                    plt.close(fig)
                    files.append(fig_path)

            elif method == "hierarchical":
                figs = create_hierarchical_outlier_plots(result, df)
                for fig_name, fig in figs.items():
                    fig_path = figures_dir / f"outliers_hierarchical_{fig_name}.png"
                    fig.savefig(
                        fig_path, dpi=config.visualization.dpi, bbox_inches="tight"
                    )
                    plt.close(fig)
                    files.append(fig_path)

        # Generate comparison plots (only if multiple methods were run)
        if len(methods_run) > 1:
            # Comprehensive comparison
            fig = create_comprehensive_outlier_comparison(outlier_results)
            fig_path = figures_dir / "outlier_comparison.png"
            fig.savefig(fig_path, dpi=config.visualization.dpi, bbox_inches="tight")
            plt.close(fig)
            files.append(fig_path)

            # Overlap heatmap
            fig = create_outlier_overlap_heatmap(outlier_results)
            fig_path = figures_dir / "outlier_overlap_heatmap.png"
            fig.savefig(fig_path, dpi=config.visualization.dpi, bbox_inches="tight")
            plt.close(fig)
            files.append(fig_path)

        # Per-genotype outlier counts (if genotype column exists)
        if config.columns.genotype in df.columns:
            fig = create_outliers_per_genotype_plot(
                df=df,
                all_outlier_results=outlier_results,
                genotype_col=config.columns.genotype,
            )
            fig_path = figures_dir / "outliers_per_genotype.png"
            fig.savefig(fig_path, dpi=config.visualization.dpi, bbox_inches="tight")
            plt.close(fig)
            files.append(fig_path)

        # Create metadata (pass through outlier_results for next step)
        metadata = {
            "methods_visualized": methods_run,
            "figures_generated": len(files),
            "comparison_plots": len(methods_run) > 1,
            "outlier_results": outlier_results,  # Pass through for RemoveOutliersStep
            "methods_run": methods_run,  # Pass through for RemoveOutliersStep
            "trait_names": prev_result.metadata.get("trait_names", []),  # Pass through
            "valid_trait_names": prev_result.metadata.get(
                "valid_trait_names", []
            ),  # Pass through
        }

        return StepResult(data=df, metadata=metadata, files_generated=files)
