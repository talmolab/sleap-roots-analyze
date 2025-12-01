"""Step 8: Perform statistical analysis (ANOVA and heritability)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.statistics import (
    calculate_heritability_estimates,
    calculate_trait_statistics,
    perform_anova_by_genotype,
)
from sleap_roots_analyze.visualization import create_heritability_plot


class StatisticalAnalysisStep(BaseStep):
    """Perform statistical analysis on cleaned data.

    This step calculates:
    - Basic trait statistics (mean, std, min, max, etc.)
    - ANOVA results for each trait
    - Broad-sense heritability (H²) for each trait

    Outputs:
        - 08_trait_statistics.json: Basic statistics for each trait
        - 08_anova_results.csv: ANOVA F-statistics and p-values
        - 08_heritability_results.csv: Heritability estimates and significance
        - 08_statistical_analysis_summary.json: Combined summary
    """

    def __init__(self):
        """Initialize StatisticalAnalysisStep."""
        super().__init__(
            step_name="StatisticalAnalysis",
            description="Perform statistical analysis (ANOVA and heritability)",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute statistical analysis.

        Args:
            data: DataFrame from previous step (RemoveOutliersStep).
            config: Pipeline configuration with:
                - columns.genotype: Genotype column name
                - columns.replicate: Replicate column name (optional)
            run_dir: Directory to save outputs.
            prev_result: Result from previous step.

        Returns:
            StepResult with original DataFrame and statistical results in metadata.
        """
        df = data.copy()

        # Get valid trait columns from previous step
        trait_cols = prev_result.metadata.get("valid_trait_names")
        if trait_cols is None:
            # Try to get from earlier step
            trait_cols = prev_result.metadata.get("trait_names")

        if trait_cols is None:
            raise ValueError(
                "Could not find trait column names in previous step metadata. "
                "Expected 'valid_trait_names' or 'trait_names'."
            )

        genotype_col = config.columns.genotype
        replicate_col = config.columns.replicate

        # 1. Calculate basic trait statistics
        trait_stats = calculate_trait_statistics(df=df, trait_cols=trait_cols)

        # 2. Perform ANOVA for each trait
        anova_results = perform_anova_by_genotype(
            df=df, trait_cols=trait_cols, genotype_col=genotype_col, alpha=0.05
        )

        # Convert ANOVA results to DataFrame for saving
        anova_records = []
        for trait, result in anova_results.items():
            if trait == "__calculation_metadata__":
                continue
            if "error" in result:
                anova_records.append(
                    {
                        "trait": trait,
                        "f_statistic": None,
                        "p_value": None,
                        "eta_squared": None,
                        "significant": None,
                        "n_groups": None,
                        "total_n": None,
                        "error": result["error"],
                    }
                )
            else:
                anova_records.append(
                    {
                        "trait": trait,
                        "f_statistic": result.get("f_statistic"),
                        "p_value": result.get("p_value"),
                        "eta_squared": result.get("eta_squared"),
                        "significant": result.get("significant"),
                        "n_groups": result.get("n_groups"),
                        "total_n": result.get("total_n"),
                        "error": None,
                    }
                )

        anova_df = pd.DataFrame(anova_records)

        # 3. Calculate heritability for each trait
        heritability_results = calculate_heritability_estimates(
            df=df,
            trait_cols=trait_cols,
            genotype_col=genotype_col,
            replicate_col=replicate_col,
            force_method=None,  # Use default mixed model approach
            remove_low_h2=False,  # Don't remove yet, that's Step 9
        )

        # Convert heritability results to DataFrame for saving
        heritability_records = []
        for trait, result in heritability_results.items():
            if trait == "__calculation_metadata__":
                continue
            if "error" in result:
                heritability_records.append(
                    {
                        "trait": trait,
                        "heritability": None,
                        "var_genetic": None,
                        "var_residual": None,
                        "mean_n_reps": None,
                        "n_genotypes": None,
                        "n_observations": None,
                        "model_type": None,
                        "error": result["error"],
                    }
                )
            else:
                heritability_records.append(
                    {
                        "trait": trait,
                        "heritability": result.get("heritability"),
                        "var_genetic": result.get("var_genetic"),
                        "var_residual": result.get("var_residual"),
                        "mean_n_reps": result.get("mean_n_reps"),
                        "n_genotypes": result.get("n_genotypes"),
                        "n_observations": result.get("n_observations"),
                        "model_type": result.get("model_type"),
                        "error": None,
                    }
                )

        heritability_df = pd.DataFrame(heritability_records)

        # 4. Create combined summary
        # Count valid ANOVA results
        valid_anova = [
            r
            for t, r in anova_results.items()
            if t != "__calculation_metadata__" and "error" not in r
        ]
        significant_p005 = sum(
            1
            for r in valid_anova
            if r.get("p_value") is not None and r["p_value"] < 0.05
        )
        significant_p001 = sum(
            1
            for r in valid_anova
            if r.get("p_value") is not None and r["p_value"] < 0.01
        )

        # Count valid heritability results
        valid_h2 = [
            r
            for t, r in heritability_results.items()
            if t != "__calculation_metadata__" and "error" not in r
        ]
        high_h2_60 = sum(
            1
            for r in valid_h2
            if r.get("heritability") is not None and r["heritability"] >= 0.60
        )
        mean_h2 = (
            sum(
                r["heritability"] for r in valid_h2 if r.get("heritability") is not None
            )
            / len(valid_h2)
            if valid_h2
            else None
        )

        summary = {
            "n_traits_analyzed": len(trait_cols),
            "n_samples": len(df),
            "n_genotypes": df[genotype_col].nunique(),
            "traits_analyzed": trait_cols,
            "anova_summary": {
                "traits_with_results": len(valid_anova),
                "significant_traits_p005": significant_p005,
                "significant_traits_p001": significant_p001,
            },
            "heritability_summary": {
                "traits_with_results": len(valid_h2),
                "high_heritability_traits_h60": high_h2_60,
                "mean_heritability": mean_h2,
            },
        }

        # Save outputs
        files = []
        files.append(self.save_json(trait_stats, "08_trait_statistics.json", run_dir))
        files.append(self.save_dataframe(anova_df, "08_anova_results.csv", run_dir))
        files.append(
            self.save_dataframe(heritability_df, "08_heritability_results.csv", run_dir)
        )
        files.append(
            self.save_json(summary, "08_statistical_analysis_summary.json", run_dir)
        )

        # Generate heritability plot
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        fig = create_heritability_plot(
            heritability_results=heritability_results,
            figsize=tuple(config.visualization.figsize),
            dpi=config.visualization.dpi,
        )
        heritability_plot_path = figures_dir / f"08_heritability_analysis.{config.visualization.figure_format}"
        fig.savefig(
            heritability_plot_path,
            dpi=config.visualization.dpi,
            bbox_inches=config.visualization.bbox_inches,
            facecolor=config.visualization.facecolor,
            edgecolor=config.visualization.edgecolor,
            transparent=config.visualization.transparent,
        )
        plt.close(fig)
        files.append(heritability_plot_path)

        # Create metadata
        metadata = {
            "trait_statistics": trait_stats,
            "anova_results": anova_results,
            "heritability_results": heritability_results,
            "summary": summary,
            "trait_names": trait_cols,  # Primary key (standardized)
            "valid_trait_names": trait_cols,  # For consistency
        }

        return StepResult(data=df, metadata=metadata, files_generated=files)
