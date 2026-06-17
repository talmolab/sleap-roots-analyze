"""Step 9: Generate publication-quality static figures."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.visualization import (
    create_correlation_heatmap,
    create_feature_contribution_heatmap,
    create_feature_contribution_plot,
    create_genotype_image_grid,
    create_heritability_plot,
    create_pc_genotype_boxplots,
    create_pca_biplot,
    create_pca_scree_plot,
    create_phenotype_variation_plot,
    create_regression_plot,
    create_trait_boxplots_by_genotype_batched,
    create_trait_histograms_batched,
    create_umap_colored_by_top_traits,
    identify_extreme_genotypes_by_pc,
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

    Configuration:
        Control figure generation via config.static_viz:
        - enabled: Enable/disable step
        - formats: Output formats ['png', 'pdf', 'svg']
        - dpi: Resolution (default: 300)
        - create_pca_plots: Generate PCA visualizations
        - create_trait_distributions: Generate histograms/boxplots
        - create_trait_correlations: Generate correlation heatmaps
        - create_heritability_plots: Generate heritability plots
        - pca_biplot_top_features: Top features in biplot (default: 10)
        - pca_heatmap_features: Features in heatmap (default: 20)
        - pca_n_components: PCs in boxplots (default: 3)
        - histogram_batch_size: Traits per histogram (default: 9)
        - boxplot_batch_size: Traits per boxplot (default: 6)

    Outputs:
        - static_figures/*.{png,pdf,svg}: Generated figures
        - 09_static_figures_manifest.json: List of generated files

    Example:
        ```python
        config.static_viz.enabled = True
        config.static_viz.formats = ['png', 'pdf']
        config.static_viz.create_pca_plots = True
        config.static_viz.pca_biplot_top_features = 15

        step = GenerateStaticFiguresStep()
        result = step.execute(data, config, run_dir, prev_result)
        ```
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

        # Create output directory structure
        # NOTE: We use figures/ with subdirectories instead of flat static_figures/
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Get metadata from previous steps
        metadata = prev_result.metadata if prev_result else {}
        trait_cols = metadata.get("trait_names", metadata.get("valid_trait_names", []))
        pca_results = metadata.get("pca_results")
        umap_results = metadata.get("umap_results")
        heritability_results = metadata.get("heritability_results")
        image_paths = metadata.get("image_paths")

        # Track generated files
        generated_files = []
        formats = config.static_viz.formats
        # Filter out 'pdf' if save_pdf is False
        if not config.static_viz.save_pdf:
            formats = [f for f in formats if f.lower() != "pdf"]
        dpi = config.static_viz.dpi

        try:
            # 1. PCA Plots (saved to figures/pca/)
            if config.static_viz.create_pca_plots and pca_results:
                logger.info("  Creating PCA plots...")
                pca_dir = figures_dir / "pca"
                pca_dir.mkdir(parents=True, exist_ok=True)
                generated_files.extend(
                    self._create_pca_plots(
                        df,
                        pca_results,
                        trait_cols,
                        config,
                        pca_dir,
                        formats,
                        dpi,
                    )
                )

            # 2. UMAP Plots (saved to figures/umap/)
            if config.static_viz.create_umap_plots and umap_results and pca_results:
                logger.info("  Creating UMAP plots...")
                umap_dir = figures_dir / "umap"
                umap_dir.mkdir(parents=True, exist_ok=True)
                generated_files.extend(
                    self._create_umap_plots(
                        df,
                        umap_results,
                        pca_results,
                        trait_cols,
                        config,
                        umap_dir,
                        formats,
                        dpi,
                    )
                )

            # 3. Trait Distribution Plots (saved to figures/trait_histograms/ and figures/trait_boxplots/)
            if config.static_viz.create_trait_distributions and trait_cols:
                logger.info("  Creating trait distribution plots...")
                histograms_dir = figures_dir / "trait_histograms"
                histograms_dir.mkdir(parents=True, exist_ok=True)
                boxplots_dir = figures_dir / "trait_boxplots"
                boxplots_dir.mkdir(parents=True, exist_ok=True)
                generated_files.extend(
                    self._create_trait_distributions(
                        df,
                        trait_cols,
                        config,
                        histograms_dir,
                        boxplots_dir,
                        formats,
                        dpi,
                    )
                )

            # 4. Correlation Heatmap (saved to figures/overview/)
            if config.static_viz.create_trait_correlations and trait_cols:
                logger.info("  Creating correlation heatmap...")
                overview_dir = figures_dir / "overview"
                overview_dir.mkdir(parents=True, exist_ok=True)
                generated_files.extend(
                    self._create_correlation_plot(
                        df, trait_cols, config, overview_dir, formats, dpi
                    )
                )

            # 5. Heritability Plots (saved to figures/heritability/)
            if config.static_viz.create_heritability_plots and heritability_results:
                logger.info("  Creating heritability plots...")
                heritability_dir = figures_dir / "heritability"
                heritability_dir.mkdir(parents=True, exist_ok=True)
                generated_files.extend(
                    self._create_heritability_plots(
                        heritability_results, config, heritability_dir, formats, dpi
                    )
                )

            # 6. Phenotype Variation Plots (saved to figures/phenotype_variation/)
            if (
                config.static_viz.create_phenotype_variation_plots
                and heritability_results
                and trait_cols
            ):
                logger.info("  Creating phenotype variation plots...")
                variation_dir = figures_dir / "phenotype_variation"
                variation_dir.mkdir(parents=True, exist_ok=True)
                generated_files.extend(
                    self._create_phenotype_variation_plots(
                        df,
                        heritability_results,
                        config,
                        variation_dir,
                        formats,
                        dpi,
                    )
                )

            # 7. Genotype Comparison Plots (saved to figures/overview/)
            if config.static_viz.create_genotype_comparisons and trait_cols:
                logger.info("  Creating genotype comparison plots...")
                overview_dir = figures_dir / "overview"
                overview_dir.mkdir(parents=True, exist_ok=True)
                generated_files.extend(
                    self._create_genotype_comparisons(
                        df, trait_cols, config, overview_dir, formats, dpi
                    )
                )

            # 8. Regression Plots (saved to figures/overview/)
            if config.static_viz.regression_trait_pairs:
                logger.info("  Creating regression plots...")
                overview_dir = figures_dir / "overview"
                overview_dir.mkdir(parents=True, exist_ok=True)
                generated_files.extend(
                    self._create_regression_plots(
                        df, config, overview_dir, formats, dpi
                    )
                )

            # 9. Genotype Image Grids (saved to figures/extreme_genotypes/)
            # Task 10.20: Wire identify_extreme_genotypes_by_pc + create_genotype_image_grid
            if config.static_viz.create_genotype_image_grids:
                if image_paths is not None and pca_results:
                    logger.info("  Creating genotype image grids...")
                    extreme_dir = figures_dir / "extreme_genotypes"
                    extreme_dir.mkdir(parents=True, exist_ok=True)
                    generated_files.extend(
                        self._create_genotype_image_grids(
                            df,
                            pca_results,
                            image_paths,
                            config,
                            extreme_dir,
                            formats,
                            dpi,
                        )
                    )
                else:
                    logger.info(
                        "  Skipping genotype image grids: image paths not available"
                    )

            logger.info(f"Generated {len(generated_files)} static figures")

            # Save manifest
            manifest = {
                "total_figures": len(generated_files),
                "formats": formats,
                "dpi": dpi,
                "files": [f.relative_to(run_dir) for f in generated_files],
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
        # Pass the variance threshold from config to ensure plot matches PCA analysis
        variance_threshold = (
            config.pca.n_components if config.pca.n_components < 1 else 0.95
        )
        fig = create_pca_scree_plot(pca_results, variance_threshold=variance_threshold)
        files.extend(
            self._save_figure(
                fig,
                "pca_scree_plot",
                output_dir,
                formats,
                dpi,
                bbox_inches=config.static_viz.bbox_inches,
                transparent=config.static_viz.transparent,
            )
        )
        plt.close(fig)

        # Biplot
        # Use hardcoded sanitized column name (data from QC pipeline already sanitized)
        genotype_col = "Genotype"
        fig = create_pca_biplot(
            pca_results,
            df=df,
            trait_names=trait_cols,
            color_by=genotype_col,
            top_n_features=config.static_viz.pca_biplot_top_features,
            feature_selection=config.pca.feature_selection_strategy,  # Pass feature selection method from config
            genotypes_to_color=config.static_viz.genotypes_to_color,
            highlight_genotypes=config.static_viz.highlight_genotypes,
        )
        files.extend(
            self._save_figure(
                fig,
                "pca_biplot",
                output_dir,
                formats,
                dpi,
                bbox_inches=config.static_viz.bbox_inches,
                transparent=config.static_viz.transparent,
            )
        )
        plt.close(fig)

        # Feature contribution heatmap (returns tuple of 2 figures)
        variance_fig, loadings_fig = create_feature_contribution_heatmap(
            pca_results, n_features=config.static_viz.pca_heatmap_features
        )
        files.extend(
            self._save_figure(
                variance_fig,
                "pca_feature_variance",
                output_dir,
                formats,
                dpi,
                bbox_inches=config.static_viz.bbox_inches,
                transparent=config.static_viz.transparent,
            )
        )
        files.extend(
            self._save_figure(
                loadings_fig,
                "pca_feature_loadings",
                output_dir,
                formats,
                dpi,
                bbox_inches=config.static_viz.bbox_inches,
                transparent=config.static_viz.transparent,
            )
        )
        plt.close(variance_fig)
        plt.close(loadings_fig)

        # Feature contribution bar chart
        # Determine variance threshold: use explicit config, or inherit from pca.n_components
        if config.static_viz.feature_contribution_variance_threshold is not None:
            variance_threshold = (
                config.static_viz.feature_contribution_variance_threshold
            )
        elif config.pca.n_components < 1:
            # pca.n_components < 1 means it's a variance threshold
            variance_threshold = config.pca.n_components
        else:
            # Default fallback
            variance_threshold = 0.95

        contrib_fig = create_feature_contribution_plot(
            pca_results,
            trait_names=trait_cols,
            n_components=None,  # Use variance threshold instead
            variance_threshold=variance_threshold,
            top_n=config.static_viz.feature_contribution_top_n,
            feature_selection=config.pca.feature_selection_strategy,
        )
        files.extend(
            self._save_figure(
                contrib_fig,
                "pca_feature_contributions",
                output_dir,
                formats,
                dpi,
                bbox_inches=config.static_viz.bbox_inches,
                transparent=config.static_viz.transparent,
            )
        )
        plt.close(contrib_fig)

        # PC boxplots by genotype
        if "transformed_data" in pca_results and genotype_col in df.columns:
            # Use same variance threshold logic as feature contribution plot
            # If pca.n_components < 1, treat as variance threshold; else use explicit count
            if config.pca.n_components < 1:
                pc_variance_threshold = config.pca.n_components
                pc_n_components = None
            else:
                pc_variance_threshold = None
                pc_n_components = config.static_viz.pca_n_components

            # Calculate adaptive figsize based on genotype count and PC count
            n_genotypes = df[genotype_col].nunique()

            # Determine actual number of PCs that will be shown
            if pc_n_components is not None:
                n_pcs = min(pc_n_components, pca_results["transformed_data"].shape[1])
            else:
                # Use variance threshold to determine PC count
                cumvar = pca_results.get("cumulative_variance_ratio", [1.0])
                threshold = pc_variance_threshold or 0.95
                n_pcs = int(np.argmax(cumvar >= threshold) + 1)
                n_pcs = min(n_pcs, pca_results["transformed_data"].shape[1])

            # Use adaptive_sizing config if enabled
            if config.adaptive_sizing.enabled:
                sizing = config.adaptive_sizing
                # Width scales with genotype count (min ~0.25 inches per genotype)
                adaptive_width = max(
                    sizing.min_width,
                    min(sizing.max_width, sizing.base_width + n_genotypes * 0.25),
                )
                # Height scales with PC count (at least 3 inches per PC)
                adaptive_height = max(
                    sizing.min_height,
                    min(sizing.max_height, n_pcs * sizing.height_per_item * 1.5),
                )
                pc_figsize = (adaptive_width, adaptive_height)
            else:
                # Default sizing
                pc_figsize = (20, max(6, n_pcs * 3))

            fig = create_pc_genotype_boxplots(
                pca_results,
                df,
                genotype_col=genotype_col,
                n_components=pc_n_components,
                variance_threshold=pc_variance_threshold,
                highlight_genotypes=config.static_viz.highlight_genotypes,
                figsize=pc_figsize,
            )
            files.extend(
                self._save_figure(
                    fig,
                    "pca_pc_boxplots",
                    output_dir,
                    formats,
                    dpi,
                    bbox_inches=config.static_viz.bbox_inches,
                    transparent=config.static_viz.transparent,
                )
            )
            plt.close(fig)

        return files

    def _create_umap_plots(
        self,
        df: pd.DataFrame,
        umap_results: dict,
        pca_results: dict,
        trait_cols: list,
        config: Any,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create UMAP-related plots.

        Args:
            df: DataFrame with trait data.
            umap_results: UMAP analysis results from UMAPAnalysisStep.
            pca_results: PCA analysis results (for feature selection).
            trait_cols: List of trait column names.
            config: Pipeline configuration.
            output_dir: Directory to save plots (figures/umap/).
            formats: Output formats.
            dpi: DPI for figures.

        Returns:
            List of generated file paths.
        """
        files = []

        # Get clean indices from UMAP results to align data
        clean_indices = umap_results.get("clean_indices")
        if clean_indices is not None:
            df_clean = df.loc[clean_indices].copy()
        else:
            # Fallback: use rows without NaN in trait columns
            df_clean = df[trait_cols].dropna()
            df_clean = df.loc[df_clean.index].copy()

        # UMAP colored by top traits (uses PCA for feature selection)
        try:
            fig = create_umap_colored_by_top_traits(
                umap_results=umap_results,
                df=df_clean,
                trait_columns=trait_cols,
                trait_names=trait_cols,  # Use column names as display names
                pca_results=pca_results,
                n_traits=6,  # Show top 6 traits
                feature_selection=config.pca.feature_selection_strategy,
            )
            files.extend(
                self._save_figure(
                    fig,
                    "umap_top_traits",
                    output_dir,
                    formats,
                    dpi,
                    bbox_inches=config.static_viz.bbox_inches,
                    transparent=config.static_viz.transparent,
                )
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create UMAP top traits plot: {e}")

        return files

    def _create_trait_distributions(
        self,
        df: pd.DataFrame,
        trait_cols: list,
        config: Any,
        histograms_dir: Path,
        boxplots_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create trait distribution plots.

        Args:
            df: DataFrame with trait data.
            trait_cols: List of trait column names.
            config: Pipeline configuration.
            histograms_dir: Directory to save histogram plots (figures/trait_histograms/).
            boxplots_dir: Directory to save boxplot plots (figures/trait_boxplots/).
            formats: Output formats.
            dpi: DPI for figures.

        Returns:
            List of generated file paths.
        """
        files = []

        # Calculate adaptive batch sizes if enabled (Task 8.4)
        n_traits = len(trait_cols)
        histogram_batch_size = config.static_viz.histogram_batch_size
        boxplot_batch_size = config.static_viz.boxplot_batch_size

        if (
            config.adaptive_sizing.enabled
            and config.adaptive_sizing.adaptive_batch_size
            and n_traits > config.adaptive_sizing.batch_size_threshold
        ):
            # Increase batch size to reduce number of output files
            # Target: fewer than ~20 batch files per plot type
            max_batch = config.adaptive_sizing.max_batch_size
            histogram_batch_size = min(max_batch, max(histogram_batch_size, 36))
            boxplot_batch_size = min(max_batch, max(boxplot_batch_size, 36))
            logger.info(
                f"Adaptive batch sizing: {n_traits} traits, "
                f"using histogram_batch_size={histogram_batch_size}, "
                f"boxplot_batch_size={boxplot_batch_size}"
            )

        # Histograms (batched) - saved to figures/trait_histograms/
        fig = create_trait_histograms_batched(
            df, trait_cols, batch_size=histogram_batch_size
        )
        for i, subfig in enumerate(fig):
            files.extend(
                self._save_figure(
                    subfig,
                    f"trait_histograms_batch{i + 1}",
                    histograms_dir,
                    formats,
                    dpi,
                    bbox_inches=config.static_viz.bbox_inches,
                    transparent=config.static_viz.transparent,
                )
            )
            plt.close(subfig)
            # Periodically reclaim memory during large batch generation
            if (i + 1) % 10 == 0:
                gc.collect()

        # Boxplots by genotype (batched) - saved to figures/trait_boxplots/
        # Use hardcoded sanitized column name (data from QC pipeline already sanitized)
        genotype_col = "Genotype"
        if genotype_col in df.columns:
            # Build kwargs for boxplot function
            boxplot_kwargs = {
                "genotype_col": genotype_col,
                "batch_size": boxplot_batch_size,  # Uses adaptive size if enabled
            }
            # Use adaptive_sizing config for subplot dimensions if enabled (Task 6c.5)
            if config.adaptive_sizing.enabled:
                sizing = config.adaptive_sizing
                boxplot_kwargs["subplot_size"] = (
                    sizing.width_per_item,
                    sizing.height_per_item,
                )

            fig = create_trait_boxplots_by_genotype_batched(
                df,
                trait_cols,
                **boxplot_kwargs,
            )
            for i, subfig in enumerate(fig):
                files.extend(
                    self._save_figure(
                        subfig,
                        f"trait_boxplots_batch{i + 1}",
                        boxplots_dir,
                        formats,
                        dpi,
                        bbox_inches=config.static_viz.bbox_inches,
                        transparent=config.static_viz.transparent,
                    )
                )
                plt.close(subfig)
                # Periodically reclaim memory during large batch generation
                if (i + 1) % 10 == 0:
                    gc.collect()

        return files

    def _create_correlation_plot(
        self,
        df: pd.DataFrame,
        trait_cols: list,
        config: Any,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create correlation heatmap."""
        files = []

        fig = create_correlation_heatmap(df, trait_cols)
        files.extend(
            self._save_figure(
                fig,
                "trait_correlations",
                output_dir,
                formats,
                dpi,
                bbox_inches=config.static_viz.bbox_inches,
                transparent=config.static_viz.transparent,
            )
        )
        plt.close(fig)

        return files

    def _create_heritability_plots(
        self,
        heritability_results: dict,
        config: Any,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create heritability plots.

        Handles both single figure (small datasets) and paginated figures
        (large datasets with many traits).
        """
        files = []

        # Use configured threshold for visualization
        # Note: Shows the configured threshold regardless of whether filtering is enabled.
        # This ensures consistency with Step 8 plots and helps users understand what
        # threshold would be used if heritability filtering were enabled.
        threshold = config.heritability.threshold
        result = create_heritability_plot(heritability_results, threshold=threshold)

        # Handle both single figure and paginated list of figures
        if isinstance(result, list):
            # Multiple paginated figures
            for i, fig in enumerate(result, start=1):
                files.extend(
                    self._save_figure(
                        fig,
                        f"heritability_estimates_page{i:02d}",
                        output_dir,
                        formats,
                        dpi,
                        bbox_inches=config.static_viz.bbox_inches,
                        transparent=config.static_viz.transparent,
                    )
                )
                plt.close(fig)
        else:
            # Single figure (backward compatibility for small datasets)
            files.extend(
                self._save_figure(
                    result,
                    "heritability_estimates",
                    output_dir,
                    formats,
                    dpi,
                    bbox_inches=config.static_viz.bbox_inches,
                    transparent=config.static_viz.transparent,
                )
            )
            plt.close(result)

        return files

    def _create_phenotype_variation_plots(
        self,
        df: pd.DataFrame,
        heritability_results: dict,
        config: Any,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create phenotype variation plots for top heritable traits.

        Generates box plots with jittered points showing phenotypic variation
        across genotypes for the most heritable traits.
        """
        files = []

        # Get top N traits by heritability
        top_n = config.static_viz.phenotype_variation_top_n

        # Sort traits by H2 value
        traits_by_h2 = sorted(
            [
                (trait, data.get("H2", data.get("heritability", 0)))
                for trait, data in heritability_results.items()
                if trait in df.columns
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        if not traits_by_h2:
            logger.info("  No heritable traits found for phenotype variation plots")
            return files

        # Determine genotype column (sanitized name from QC pipeline)
        genotype_col = "Genotype"

        for trait, h2 in traits_by_h2:
            try:
                fig, _ = create_phenotype_variation_plot(
                    df,
                    trait=trait,
                    group_col=genotype_col,
                )
                # Sanitize trait name for filename
                safe_trait_name = trait.replace("/", "_").replace("\\", "_")
                files.extend(
                    self._save_figure(
                        fig,
                        f"phenotype_variation_{safe_trait_name}",
                        output_dir,
                        formats,
                        dpi,
                        bbox_inches=config.static_viz.bbox_inches,
                        transparent=config.static_viz.transparent,
                    )
                )
                plt.close(fig)
            except Exception as e:
                logger.warning(
                    f"Failed to create phenotype variation plot for {trait}: {e}"
                )
                continue

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

    def _create_regression_plots(
        self,
        df: pd.DataFrame,
        config: Any,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create regression plots for configured trait pairs.

        Generates scatter plots with linear regression lines for each
        pair of traits specified in config.static_viz.regression_trait_pairs.
        """
        files = []

        trait_pairs = config.static_viz.regression_trait_pairs

        for pair in trait_pairs:
            if len(pair) != 2:
                logger.warning(f"Invalid regression pair (expected 2 traits): {pair}")
                continue

            x_col, y_col = pair

            # Check columns exist
            if x_col not in df.columns or y_col not in df.columns:
                logger.warning(
                    f"Skipping regression {x_col} vs {y_col}: column(s) not in data"
                )
                continue

            try:
                fig = create_regression_plot(
                    df,
                    x_col=x_col,
                    y_col=y_col,
                    color_by="Genotype" if "Genotype" in df.columns else None,
                )
                # Sanitize trait names for filename
                safe_x = x_col.replace("/", "_").replace("\\", "_")
                safe_y = y_col.replace("/", "_").replace("\\", "_")
                files.extend(
                    self._save_figure(
                        fig,
                        f"regression_{safe_x}_vs_{safe_y}",
                        output_dir,
                        formats,
                        dpi,
                        bbox_inches=config.static_viz.bbox_inches,
                        transparent=config.static_viz.transparent,
                    )
                )
                plt.close(fig)
            except Exception as e:
                logger.warning(
                    f"Failed to create regression plot for {x_col} vs {y_col}: {e}"
                )
                continue

        return files

    def _create_genotype_image_grids(
        self,
        df: pd.DataFrame,
        pca_results: dict,
        image_paths: pd.Series,
        config: Any,
        output_dir: Path,
        formats: list,
        dpi: int,
    ) -> list[Path]:
        """Create genotype image grids for extreme genotypes identified by PCA.

        Task 10.20: Wire identify_extreme_genotypes_by_pc + create_genotype_image_grid
        into the pipeline.

        Args:
            df: DataFrame with trait data.
            pca_results: PCA analysis results from previous step.
            image_paths: Series mapping sample indices to image file paths.
            config: Pipeline configuration.
            output_dir: Directory to save figures.
            formats: List of output formats.
            dpi: Output DPI.

        Returns:
            List of generated file paths.
        """
        files = []
        genotype_col = config.columns.genotype
        barcode_col = config.columns.barcode

        try:
            # Identify extreme genotypes based on PCA
            extreme_df = identify_extreme_genotypes_by_pc(
                pca_results=pca_results,
                df=df,
                genotype_col=genotype_col,
                n_extreme=3,  # Top 3 extreme genotypes per PC
            )

            if extreme_df.empty:
                logger.info("    No extreme genotypes identified by PCA")
                return files

            # Convert image_paths to the image_links format expected by create_genotype_image_grid
            # image_links is Dict[barcode, Dict[image_type, Path]]
            # Use configured image_type (default: "features.png" for RhizoVision,
            # or "1.jpg"/"36.jpg" for cylinder scanners)
            image_type = config.static_viz.genotype_image_grid_image_type
            trait_cols = config.static_viz.genotype_image_grid_trait_cols

            image_links = {}

            # Detect format by checking if it's a dict with nested dict values
            # (from link_rhizovision_images_to_samples or link_cylinder_images_from_scan_path)
            is_nested_dict_format = (
                isinstance(image_paths, dict)
                and len(image_paths) > 0
                and isinstance(next(iter(image_paths.values())), dict)
            )

            if is_nested_dict_format:
                # Format: {barcode: {"features.png": Path, "1.jpg": Path, ...}}
                # Use directly, converting paths to Path objects
                for barcode, img_dict in image_paths.items():
                    image_links[barcode] = {
                        img_type: Path(path) if path else None
                        for img_type, path in img_dict.items()
                    }
            else:
                # Legacy format: pd.Series indexed by DataFrame row numbers
                for idx, path in image_paths.items():
                    if idx in df.index:
                        barcode = (
                            df.loc[idx, barcode_col]
                            if barcode_col in df.columns
                            else str(idx)
                        )
                        image_links[barcode] = {image_type: Path(path)}

            # Get unique extreme genotypes (column name matches genotype_col)
            unique_genotypes = extreme_df[genotype_col].unique()

            for genotype in unique_genotypes:
                try:
                    fig = create_genotype_image_grid(
                        df=df,
                        image_links=image_links,
                        genotype=genotype,
                        genotype_col=genotype_col,
                        barcode_col=barcode_col,
                        image_type=image_type,
                        trait_cols=trait_cols,
                    )
                    # Sanitize genotype name for filename
                    safe_genotype = (
                        str(genotype)
                        .replace("/", "_")
                        .replace("\\", "_")
                        .replace(" ", "_")
                    )
                    files.extend(
                        self._save_figure(
                            fig,
                            f"extreme_genotype_grid_{safe_genotype}",
                            output_dir,
                            formats,
                            dpi,
                            bbox_inches=config.static_viz.bbox_inches,
                            transparent=config.static_viz.transparent,
                        )
                    )
                    plt.close(fig)
                except Exception as e:
                    logger.warning(
                        f"Failed to create image grid for genotype {genotype}: {e}"
                    )
                    continue

            gc.collect()  # Clean up memory after generating grids

        except Exception as e:
            logger.warning(f"Failed to create genotype image grids: {e}")

        return files

    def _save_figure(
        self,
        fig,
        basename: str,
        output_dir: Path,
        formats: list,
        dpi: int,
        bbox_inches: str = "tight",
        transparent: bool = False,
    ) -> list[Path]:
        """Save figure in multiple formats."""
        files = []
        for fmt in formats:
            filepath = output_dir / f"{basename}.{fmt}"
            fig.savefig(
                filepath,
                dpi=dpi,
                bbox_inches=bbox_inches,
                transparent=transparent,
            )
            files.append(filepath)
        return files
