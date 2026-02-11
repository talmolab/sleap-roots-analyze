"""Step 10: Generate interactive visualizations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class GenerateInteractiveStep(BaseStep):
    """Generate interactive Plotly visualizations.

    Creates interactive HTML visualizations based on configuration:
    - Interactive PCA plots with hover data
    - Interactive UMAP plots with hover highlighting
    - Interactive scatter plots with genotype selection
    - Optional image tooltips on hover

    All figures are saved as standalone HTML files that can be
    opened in any web browser.

    Configuration:
        Control figure generation via config.interactive_viz:
        - enabled: Enable/disable step
        - create_pca_plots: Generate interactive PCA visualizations
        - create_umap_plots: Generate interactive UMAP visualizations
        - show_images_on_hover: Show image thumbnails on hover (requires image_paths)

    Outputs:
        - figures/interactive/*.html: Generated interactive plots
        - 10_interactive_figures_manifest.json: List of generated files

    Example:
        ```python
        config.interactive_viz.enabled = True
        config.interactive_viz.create_pca_plots = True
        config.interactive_viz.show_images_on_hover = True

        step = GenerateInteractiveStep()
        result = step.execute(data, config, run_dir, prev_result)

        # Open in browser: open figures/interactive/interactive_pca.html
        ```
    """

    def __init__(self):
        """Initialize GenerateInteractiveStep."""
        super().__init__(
            step_name="GenerateInteractive",
            description="Generate interactive Plotly visualizations",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute interactive figure generation.

        Args:
            data: DataFrame with trait data.
            config: Pipeline configuration with interactive_viz settings.
            run_dir: Directory to save outputs.
            prev_result: Result from previous step (contains PCA, UMAP, images).

        Returns:
            StepResult with DataFrame and list of generated HTML paths.
        """
        if not config.interactive_viz.enabled:
            logger.info("Interactive visualization disabled, skipping")
            return StepResult(
                data=data,
                metadata=prev_result.metadata if prev_result else {},
            )

        logger.info("Generating interactive visualizations...")
        df = data.copy()

        # Create output directory (figures/interactive/ per VIZ-OUTPUT-001)
        figures_dir = run_dir / "figures" / "interactive"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Get metadata from previous steps
        metadata = prev_result.metadata if prev_result else {}
        pca_results = metadata.get("pca_results")
        umap_results = metadata.get("umap_results")
        image_paths = metadata.get("image_paths")

        # Track generated files
        generated_files = []

        try:
            # Import interactive visualization functions
            # (deferred to avoid plotly dependency issues)
            from sleap_roots_analyze.interactive_visualization import (
                create_html_with_image_viewer,
                create_interactive_image_gallery,
                create_interactive_pca_plot,
                create_interactive_pca_with_images,
                create_interactive_scatter_with_images,
                create_interactive_umap_with_hover_highlight,
            )

            # 1. Interactive PCA plots
            if config.interactive_viz.create_pca_plots and pca_results:
                logger.info("  Creating interactive PCA plots...")
                generated_files.extend(
                    self._create_interactive_pca(
                        df,
                        pca_results,
                        image_paths,
                        config,
                        figures_dir,
                    )
                )

            # 2. Interactive UMAP plots
            if (
                config.interactive_viz.create_umap_plots
                and umap_results
                and "embedding" in umap_results
            ):
                logger.info("  Creating interactive UMAP plots...")
                generated_files.extend(
                    self._create_interactive_umap(
                        df,
                        umap_results,
                        image_paths,
                        config,
                        figures_dir,
                    )
                )

            # 3. Image-dependent interactive plots (Tasks 11.7-11.9)
            if image_paths is not None:
                generated_files.extend(
                    self._create_image_dependent_plots(
                        df,
                        pca_results,
                        image_paths,
                        config,
                        figures_dir,
                    )
                )
            else:
                # Log skipping message when image paths not available
                if (
                    config.interactive_viz.create_scatter_with_images
                    or config.interactive_viz.create_image_viewer
                    or config.interactive_viz.create_image_gallery
                ):
                    logger.info(
                        "  Skipping image-dependent interactive plots: image paths not available"
                    )

            logger.info(f"Generated {len(generated_files)} interactive figures")

            # Save manifest
            manifest = {
                "total_figures": len(generated_files),
                "files": [str(f.relative_to(run_dir)) for f in generated_files],
            }
            manifest_file = self.save_json(
                manifest, "10_interactive_figures_manifest.json", run_dir
            )

            # Update metadata
            new_metadata = {
                **metadata,
                "interactive_figures": generated_files,
                "interactive_figures_manifest": manifest,
            }

            return StepResult(
                data=df,
                metadata=new_metadata,
                files_generated=[manifest_file] + generated_files,
            )

        except ImportError as e:
            logger.warning(
                f"Interactive visualization requires plotly: {e}. Skipping..."
            )
            return StepResult(
                data=df,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Error generating interactive figures: {e}")
            raise

    def _create_interactive_pca(
        self,
        df: pd.DataFrame,
        pca_results: dict,
        image_paths: Optional[pd.Series],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Create interactive PCA plots."""
        from sleap_roots_analyze.interactive_visualization import (
            create_interactive_pca_plot,
            create_interactive_pca_with_images,
        )

        files = []
        genotype_col = config.columns.genotype
        barcode_col = config.columns.barcode

        # Interactive PCA scatter (basic)
        if "transformed_data" in pca_results:
            fig = create_interactive_pca_plot(
                pca_results,
                df,
                color_by=genotype_col,
            )
            filepath = output_dir / "interactive_pca.html"
            fig.write_html(filepath)
            files.append(filepath)

        # Interactive PCA with image hover (if images available)
        if (
            config.interactive_viz.show_images_on_hover
            and image_paths is not None
            and "transformed_data" in pca_results
        ):
            fig = create_interactive_pca_with_images(
                pca_results["transformed_data"],
                df,
                image_paths,
                genotype_col=genotype_col,
                id_col=barcode_col,
                title="Interactive PCA with Image Hover",
            )
            filepath = output_dir / "pca_with_images.html"
            fig.write_html(filepath)
            files.append(filepath)

        return files

    def _create_interactive_umap(
        self,
        df: pd.DataFrame,
        umap_results: dict,
        image_paths: Optional[pd.Series],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Create interactive UMAP plots."""
        from sleap_roots_analyze.interactive_visualization import (
            create_interactive_umap_with_hover_highlight,
        )

        files = []
        genotype_col = config.columns.genotype

        # Interactive UMAP with hover highlighting
        fig = create_interactive_umap_with_hover_highlight(
            umap_results,
            df,
            genotype_col=genotype_col,
            title="Interactive UMAP Visualization",
        )
        filepath = output_dir / "interactive_umap.html"
        fig.write_html(filepath)
        files.append(filepath)

        return files

    def _create_image_dependent_plots(
        self,
        df: pd.DataFrame,
        pca_results: Optional[dict],
        image_paths: pd.Series,
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Create image-dependent interactive plots (Tasks 11.7-11.9).

        These plots require image paths to be available in metadata.

        Args:
            df: DataFrame with trait data.
            pca_results: PCA results if available (for image viewer).
            image_paths: Series mapping sample indices to image file paths.
            config: Pipeline configuration.
            output_dir: Directory to save HTML files.

        Returns:
            List of generated HTML file paths.
        """
        from pathlib import Path as PathLib

        from sleap_roots_analyze.interactive_visualization import (
            create_html_with_image_viewer,
            create_interactive_image_gallery,
            create_interactive_scatter_with_images,
        )

        files = []
        genotype_col = config.columns.genotype
        barcode_col = config.columns.barcode

        # Convert image_paths to image_links Dict format
        # image_links: Dict[barcode, Dict[image_type, Path]]
        # Handle both formats:
        #   1. Dict[str, Dict[str, Path]] from link_rhizovision_images_to_samples()
        #   2. pd.Series indexed by DataFrame row numbers (legacy format)
        import pandas as pd

        image_links = {}

        # Detect format by checking if it's a dict (not Series) with nested dict values
        is_nested_dict_format = (
            isinstance(image_paths, dict)
            and len(image_paths) > 0
            and isinstance(next(iter(image_paths.values())), dict)
        )

        if is_nested_dict_format:
            # Format: {barcode: {"features.png": Path, ...}}
            # Use directly, converting paths to PathLib
            for barcode, img_dict in image_paths.items():
                image_links[barcode] = {
                    img_type: PathLib(path) if path else None
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
                    image_links[barcode] = {"features.png": PathLib(path)}

        # Get trait columns for gallery
        trait_cols = [
            col
            for col in df.columns
            if col not in [genotype_col, barcode_col, "Barcode", "geno", "rep"]
            and df[col].dtype in ["float64", "int64"]
        ]

        # Task 11.7: Scatter with images
        if config.interactive_viz.create_scatter_with_images:
            logger.info("  Creating scatter with images plot...")
            try:
                if len(trait_cols) >= 2:
                    fig = create_interactive_scatter_with_images(
                        df=df,
                        x_col=trait_cols[0],
                        y_col=trait_cols[1],
                        image_links=image_links,
                        color_by=genotype_col if genotype_col in df.columns else None,
                        id_col=barcode_col,
                        title="Interactive Scatter with Images",
                    )
                    filepath = output_dir / "scatter_with_images.html"
                    fig.write_html(filepath)
                    files.append(filepath)
            except Exception as e:
                logger.warning(f"Failed to create scatter with images: {e}")

        # Task 11.8: PCA image viewer (requires PCA results)
        if config.interactive_viz.create_image_viewer and pca_results:
            logger.info("  Creating PCA image viewer...")
            try:
                # Use PCA plot as base for image viewer
                if "transformed_data" in pca_results:
                    from sleap_roots_analyze.interactive_visualization import (
                        create_interactive_pca_plot,
                    )

                    pca_fig = create_interactive_pca_plot(
                        pca_results,
                        df,
                        color_by=genotype_col if genotype_col in df.columns else None,
                    )
                    filepath = output_dir / "pca_image_viewer.html"
                    create_html_with_image_viewer(
                        fig=pca_fig,
                        df=df,
                        image_links=image_links,
                        output_path=filepath,
                        id_col=barcode_col,
                    )
                    files.append(filepath)
            except Exception as e:
                logger.warning(f"Failed to create PCA image viewer: {e}")

        # Task 11.9: Image gallery
        if config.interactive_viz.create_image_gallery:
            logger.info("  Creating image gallery...")
            try:
                filepath = output_dir / "image_gallery.html"
                create_interactive_image_gallery(
                    df=df,
                    image_links=image_links,
                    trait_cols=(
                        trait_cols[:5] if trait_cols else []
                    ),  # Limit to 5 traits
                    output_path=filepath,
                    id_col=barcode_col,
                )
                files.append(filepath)
            except Exception as e:
                logger.warning(f"Failed to create image gallery: {e}")

        return files
