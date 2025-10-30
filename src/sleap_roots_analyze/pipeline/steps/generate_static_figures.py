"""Step 9: Generate publication-quality static figures."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.visualization import (
    create_correlation_heatmap,
    create_feature_contribution_heatmap,
    create_heritability_plot,
    create_pc_genotype_boxplots,
    create_pca_biplot,
    create_pca_scree_plot,
    create_trait_boxplots_by_genotype_batched,
    create_trait_histograms_batched,
)

logger = logging.getLogger(__name__)


class GenerateStaticFiguresStep(BaseStep):
    """Generate publication-quality static figures.

    Creates static visualizations based on configuration:
    - PCA plots (scree, biplot, PC boxplots, feature contributions)
    - Trait distributions (histograms, boxplots by genotype)
    - Correlation heatmaps
    - Heritability plots
    - Genotype comparisons

    All figures are saved in configurable formats (PNG, PDF, SVG) with
    publication-quality DPI settings.

    Outputs:
        - static_figures/*.{png,pdf,svg}: Generated figures
        - 09_static_figures_manifest.json: List of generated files
    """

    def __init__(self):
        """Initialize GenerateStaticFiguresStep."""
        super().__init__(
            step_name="GenerateStaticFigures",
            description="Generate publication-quality static figures",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute static figure generation.

        Args:
            data: DataFrame with trait data.
            config: Pipeline configuration with static_viz settings.
            run_dir: Directory to save outputs.
            prev_result: Result from previous step (contains PCA, stats metadata).

        Returns:
            StepResult with DataFrame and list of generated figure paths.
        """
        if not config.static_viz.enabled:
            logger.info("Static visualization disabled, skipping")
            return StepResult(
                data=data,
                metadata=prev_result.metadata if prev_result else {},
            )

        logger.info("Generating static figures...")
        df = data.copy()

        # Create output directory
        figures_dir = run_dir / "static_figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Get metadata from previous steps
        metadata = prev_result.metadata if prev_result else {}
        trait_cols = metadata.get("trait_names", metadata.get("valid_trait_names", []))
        pca_results = metadata.get("pca_results")
        heritability_results = metadata.get("heritability_results")

        # Track generated files
        generated_files = []
        formats = config.static_viz.formats
        dpi = config.static_viz.dpi

        try:
            # 1. PCA Plots
            if config.static_viz.create_pca_plots and pca_results:
                logger.info("  Creating PCA plots...")
                generated_files.extend(
                    self._create_pca_plots(
                        df,
                        pca_results,
                        trait_cols,
                        config,
                        figures_dir,
                        formats,
                        dpi,
                    )
                )

            # 2. Trait Distribution Plots
            if config.static_viz.create_trait_distributions and trait_cols:
                logger.info("  Creating trait distribution plots...")
                generated_files.extend(
                    self._create_trait_distributions(
                        df, trait_cols, config, figures_dir, formats, dpi
                    )
                )

            # 3. Correlation Heatmap
            if config.static_viz.create_trait_correlations and trait_cols:
                logger.info("  Creating correlation heatmap...")
                generated_files.extend(
                    self._create_correlation_plot(
                        df, trait_cols, figures_dir, formats, dpi
                    )
                )

            # 4. Heritability Plots
            if config.static_viz.create_heritability_plots and heritability_results:
                logger.info("  Creating heritability plots...")
                generated_files.extend(
                    self._create_heritability_plots(
                        heritability_results, figures_dir, formats, dpi
                    )
                )

            # 5. Genotype Comparison Plots
            if config.static_viz.create_genotype_comparisons and trait_cols:
                logger.info("  Creating genotype comparison plots...")
                generated_files.extend(
                    self._create_genotype_comparisons(
                        df, trait_cols, config, figures_dir, formats, dpi
                    )
                )

            logger.info(f"Generated {len(generated_files)} static figures")

            # Save manifest
            manifest = {
                "total_figures": len(generated_files),
                "formats": formats,
                "dpi": dpi,
                "files": [str(f.relative_to(run_dir)) for f in generated_files],
            }
            manifest_file = self.save_json(
                manifest, "09_static_figures_manifest.json", run_dir
            )

            # Update metadata
            new_metadata = {
                **metadata,
                "static_figures": generated_files,
                "static_figures_manifest": manifest,
            }

            return StepResult(
                data=df,
                metadata=new_metadata,
                files_generated=[manifest_file] + generated_files,
            )

        except Exception as e:
            logger.error(f"Error generating static figures: {e}")
            raise

    def _create_pca_plots(
        self,
        df: pd.DataFrame,
        pca_results: dict,
        trait_cols: list,
        config: Any,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create PCA-related plots."""
        files = []

        # Scree plot
        fig = create_pca_scree_plot(pca_results)
        files.extend(self._save_figure(fig, "pca_scree_plot", output_dir, formats, dpi))
        plt.close(fig)

        # Biplot
        genotype_col = config.columns.genotype
        fig = create_pca_biplot(
            pca_results, df=df, genotype_col=genotype_col, n_features=10
        )
        files.extend(self._save_figure(fig, "pca_biplot", output_dir, formats, dpi))
        plt.close(fig)

        # Feature contribution heatmap
        fig = create_feature_contribution_heatmap(pca_results, top_n=20)
        files.extend(
            self._save_figure(
                fig, "pca_feature_contributions", output_dir, formats, dpi
            )
        )
        plt.close(fig)

        # PC boxplots by genotype
        if "pc_scores" in pca_results and genotype_col in df.columns:
            fig = create_pc_genotype_boxplots(
                pca_results["pc_scores"], df, genotype_col=genotype_col, n_pcs=3
            )
            files.extend(
                self._save_figure(fig, "pca_pc_boxplots", output_dir, formats, dpi)
            )
            plt.close(fig)

        return files

    def _create_trait_distributions(
        self,
        df: pd.DataFrame,
        trait_cols: list,
        config: Any,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create trait distribution plots."""
        files = []

        # Histograms (batched)
        fig = create_trait_histograms_batched(df, trait_cols, traits_per_figure=9)
        for i, subfig in enumerate(fig):
            files.extend(
                self._save_figure(
                    subfig, f"trait_histograms_batch{i+1}", output_dir, formats, dpi
                )
            )
            plt.close(subfig)

        # Boxplots by genotype (batched)
        genotype_col = config.columns.genotype
        if genotype_col in df.columns:
            fig = create_trait_boxplots_by_genotype_batched(
                df,
                trait_cols,
                genotype_col=genotype_col,
                traits_per_figure=6,
            )
            for i, subfig in enumerate(fig):
                files.extend(
                    self._save_figure(
                        subfig,
                        f"trait_boxplots_by_genotype_batch{i+1}",
                        output_dir,
                        formats,
                        dpi,
                    )
                )
                plt.close(subfig)

        return files

    def _create_correlation_plot(
        self,
        df: pd.DataFrame,
        trait_cols: list,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create correlation heatmap."""
        files = []

        fig = create_correlation_heatmap(df, trait_cols)
        files.extend(
            self._save_figure(fig, "trait_correlations", output_dir, formats, dpi)
        )
        plt.close(fig)

        return files

    def _create_heritability_plots(
        self,
        heritability_results: dict,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create heritability plots."""
        files = []

        fig = create_heritability_plot(heritability_results)
        files.extend(
            self._save_figure(fig, "heritability_estimates", output_dir, formats, dpi)
        )
        plt.close(fig)

        return files

    def _create_genotype_comparisons(
        self,
        df: pd.DataFrame,
        trait_cols: list,
        config: Any,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create genotype comparison plots (already covered by boxplots)."""
        # Genotype comparisons are already handled by boxplots_by_genotype
        # This is a placeholder for future additional comparisons
        return []

    def _save_figure(
        self,
        fig,
        basename: str,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Save figure in multiple formats."""
        files = []
        for fmt in formats:
            filepath = output_dir / f"{basename}.{fmt}"
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
            files.append(filepath)
        return files
