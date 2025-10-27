"""Step 12: Generate comprehensive pipeline summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class GenerateSummaryStep(BaseStep):
    """Generate comprehensive summary report of the visualization pipeline.

    This step:
    1. Collects results from all previous steps
    2. Generates markdown summary report
    3. Generates JSON summary for programmatic access
    4. Lists all generated outputs

    Input: DataFrame with trait data
    Output: DataFrame + summary files (markdown, JSON)
    """

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute the generate summary step.

        Args:
            data: DataFrame with trait data.
            config: Pipeline configuration.
            run_dir: Directory for this pipeline run.
            prev_result: Result from previous steps.

        Returns:
            StepResult with DataFrame and summary file paths.
        """
        logger.info("Generating pipeline summary...")

        metadata = prev_result.metadata

        # Prepare summary data
        summary_data = {
            "pipeline_name": config.pipeline_name,
            "version": config.version,
            "run_directory": str(run_dir),
            "data_file": config.data.csv_path,
            "n_samples": metadata.get("n_samples", len(data)),
            "n_traits_initial": metadata.get("n_traits", 0),
            "n_traits_final": len(metadata.get("trait_cols", [])),
            "configuration": self._extract_config_summary(config),
            "results": self._extract_results_summary(metadata),
        }

        # Generate markdown summary
        if "markdown" in config.summary.formats:
            md_path = run_dir / "SUMMARY.md"
            self._write_markdown_summary(summary_data, md_path, metadata)
            logger.info(f"Generated markdown summary: {md_path}")

        # Generate JSON summary
        if "json" in config.summary.formats:
            json_path = run_dir / "summary.json"
            self._write_json_summary(summary_data, json_path)
            logger.info(f"Generated JSON summary: {json_path}")

        # Generate HTML summary if requested
        if "html" in config.summary.formats:
            html_path = run_dir / "summary.html"
            self._write_html_summary(summary_data, html_path, metadata)
            logger.info(f"Generated HTML summary: {html_path}")

        return StepResult(
            data=data,
            metadata={**metadata, "summary_data": summary_data},
            message="Summary generation complete",
        )

    def _extract_config_summary(self, config) -> dict:
        """Extract key configuration settings for summary."""
        return {
            "pca": {
                "n_components": config.pca.n_components,
                "standardize": config.pca.standardize,
                "feature_selection_strategy": config.pca.feature_selection_strategy,
            },
            "umap_enabled": config.umap.enabled,
            "clustering_enabled": config.clustering.enabled,
            "statistics": {
                "calculate_anova": config.statistics.calculate_anova,
                "calculate_heritability": config.statistics.calculate_heritability,
            },
            "heritability_filtering": {
                "enabled": config.heritability.filter_enabled,
                "threshold": config.heritability.threshold,
            },
            "interesting_genotypes_enabled": config.interesting_genotypes.enabled,
            "adaptive_sizing_enabled": config.adaptive_sizing.enabled,
        }

    def _extract_results_summary(self, metadata: dict) -> dict:
        """Extract key results for summary."""
        results = {}

        if "n_pca_components" in metadata:
            results["pca"] = {
                "n_components": metadata["n_pca_components"],
                "explained_variance": f"{metadata.get('pca_explained_variance', 0):.1%}",
                "n_top_features": len(metadata.get("top_features", [])),
            }

        if "anova_results" in metadata:
            anova_df = metadata["anova_results"]
            results["anova"] = {
                "n_traits_tested": len(anova_df),
                "n_significant": int((anova_df["p_value"] < 0.05).sum()),
            }

        if "heritability_results" in metadata:
            h2_df = metadata["heritability_results"]
            results["heritability"] = {
                "mean_h2": float(h2_df["H2"].mean()),
                "median_h2": float(h2_df["H2"].median()),
                "min_h2": float(h2_df["H2"].min()),
                "max_h2": float(h2_df["H2"].max()),
            }

        if "n_traits_removed_by_heritability" in metadata:
            results["heritability_filtering"] = {
                "n_removed": metadata["n_traits_removed_by_heritability"],
                "threshold": metadata.get("heritability_threshold", 0),
            }

        return results

    def _write_markdown_summary(
        self, summary_data: dict, output_path: Path, metadata: dict
    ):
        """Write markdown summary report."""
        with open(output_path, "w") as f:
            f.write(f"# {summary_data['pipeline_name']} Summary\n\n")
            f.write(f"**Version:** {summary_data['version']}  \n")
            f.write(f"**Run Directory:** `{summary_data['run_directory']}`\n\n")

            # Data overview
            f.write("## Data Overview\n\n")
            f.write(f"- **Input File:** `{summary_data['data_file']}`\n")
            f.write(f"- **Samples:** {summary_data['n_samples']}\n")
            f.write(f"- **Traits (initial):** {summary_data['n_traits_initial']}\n")
            f.write(f"- **Traits (final):** {summary_data['n_traits_final']}\n")
            if "n_images_linked" in metadata:
                f.write(f"- **Images linked:** {metadata['n_images_linked']}\n")
            f.write("\n")

            # Configuration
            f.write("## Configuration\n\n")
            config = summary_data["configuration"]
            f.write(f"### PCA\n")
            f.write(f"- **Components:** {config['pca']['n_components']}\n")
            f.write(f"- **Standardize:** {config['pca']['standardize']}\n")
            f.write(
                f"- **Feature Selection:** {config['pca']['feature_selection_strategy']}\n\n"
            )

            f.write(f"### Analysis Features\n")
            f.write(
                f"- **UMAP:** {'Enabled' if config['umap_enabled'] else 'Disabled'}\n"
            )
            f.write(
                f"- **Clustering:** {'Enabled' if config['clustering_enabled'] else 'Disabled'}\n"
            )
            f.write(
                f"- **Interesting Genotypes:** {'Enabled' if config['interesting_genotypes_enabled'] else 'Disabled'}\n"
            )
            f.write(
                f"- **Adaptive Figure Sizing:** {'Enabled' if config['adaptive_sizing_enabled'] else 'Disabled'}\n\n"
            )

            # Results
            if summary_data["results"]:
                f.write("## Results\n\n")
                results = summary_data["results"]

                if "pca" in results:
                    f.write(f"### PCA\n")
                    f.write(
                        f"- **Components Kept:** {results['pca']['n_components']}\n"
                    )
                    f.write(
                        f"- **Variance Explained:** {results['pca']['explained_variance']}\n"
                    )
                    f.write(
                        f"- **Top Features Identified:** {results['pca']['n_top_features']}\n\n"
                    )

                if "anova" in results:
                    f.write(f"### ANOVA\n")
                    f.write(
                        f"- **Traits Tested:** {results['anova']['n_traits_tested']}\n"
                    )
                    f.write(
                        f"- **Significant (p<0.05):** {results['anova']['n_significant']}\n\n"
                    )

                if "heritability" in results:
                    f.write(f"### Heritability\n")
                    f.write(
                        f"- **Mean H²:** {results['heritability']['mean_h2']:.3f}\n"
                    )
                    f.write(
                        f"- **Median H²:** {results['heritability']['median_h2']:.3f}\n"
                    )
                    f.write(
                        f"- **Range:** {results['heritability']['min_h2']:.3f} - {results['heritability']['max_h2']:.3f}\n\n"
                    )

                if "heritability_filtering" in results:
                    f.write(f"### Heritability Filtering\n")
                    f.write(
                        f"- **Threshold:** H² >= {results['heritability_filtering']['threshold']}\n"
                    )
                    f.write(
                        f"- **Traits Removed:** {results['heritability_filtering']['n_removed']}\n\n"
                    )

            # Output files
            f.write("## Generated Outputs\n\n")
            f.write("See subdirectories for detailed outputs:\n")
            f.write("- `statistics/` - Statistical analysis results\n")
            f.write("- `pca/` - PCA analysis results\n")
            if config.get("umap_enabled"):
                f.write("- `umap/` - UMAP results\n")
            if config.get("clustering_enabled"):
                f.write("- `clustering/` - Clustering results\n")
            f.write("\n")

    def _write_json_summary(self, summary_data: dict, output_path: Path):
        """Write JSON summary."""
        # Convert any non-serializable objects
        json_data = self._make_json_serializable(summary_data)

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

    def _write_html_summary(
        self, summary_data: dict, output_path: Path, metadata: dict
    ):
        """Write HTML summary report."""
        # Simple HTML wrapper around markdown content
        # For Phase 2A, just convert markdown to HTML
        md_content = []
        md_path = output_path.parent / "SUMMARY.md"
        if md_path.exists():
            with open(md_path, "r") as f:
                md_content = f.readlines()

        with open(output_path, "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html>\n<head>\n")
            f.write(f"<title>{summary_data['pipeline_name']} Summary</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 40px; }\n")
            f.write("h1 { color: #333; }\n")
            f.write("h2 { color: #666; border-bottom: 1px solid #ddd; }\n")
            f.write("code { background-color: #f4f4f4; padding: 2px 4px; }\n")
            f.write("</style>\n")
            f.write("</head>\n<body>\n")
            f.write("<pre>\n")
            f.write("".join(md_content))
            f.write("</pre>\n")
            f.write("</body>\n</html>")

    def _make_json_serializable(self, obj):
        """Recursively convert objects to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return str(type(obj))  # Just note the type for complex objects
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
