"""Step 2: Calculate statistics (ANOVA, heritability, descriptive stats)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.statistics import (
    perform_anova_by_genotype,
    calculate_heritability_estimates,
)

logger = logging.getLogger(__name__)


class CalculateStatisticsStep(BaseStep):
    """Calculate descriptive statistics, ANOVA, and heritability.

    This step:
    1. Calculates descriptive statistics for all traits
    2. Runs ANOVA if configured
    3. Calculates broad-sense heritability if configured

    Input: DataFrame with trait data
    Output: DataFrame + metadata with statistics results
    """

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute the calculate statistics step.

        Args:
            data: DataFrame with trait data.
            config: Pipeline configuration.
            run_dir: Directory for this pipeline run.
            prev_result: Result from previous step (load_data_images).

        Returns:
            StepResult with DataFrame and statistics metadata.
        """
        trait_cols = prev_result.metadata["trait_cols"]
        genotype_col = prev_result.metadata["genotype_col"]
        replicate_col = prev_result.metadata["replicate_col"]

        logger.info(f"Calculating statistics for {len(trait_cols)} traits")

        # Calculate descriptive statistics
        descriptive_stats = data[trait_cols].describe()
        logger.info("Calculated descriptive statistics")

        metadata = {**prev_result.metadata, "descriptive_stats": descriptive_stats}

        # Calculate ANOVA if configured
        if config.statistics.calculate_anova:
            logger.info("Calculating ANOVA...")
            anova_dict = perform_anova_by_genotype(
                data,
                trait_cols=trait_cols,
                genotype_col=genotype_col,
                alpha=config.statistics.alpha,
            )
            # Convert dict to DataFrame for easier saving/access
            anova_results = pd.DataFrame(anova_dict).T
            metadata["anova_results"] = anova_results
            if "p_value" in anova_results.columns:
                n_significant = (
                    anova_results["p_value"] < config.statistics.alpha
                ).sum()
                logger.info(
                    f"ANOVA complete: {n_significant}/{len(trait_cols)} traits significant"
                )
            else:
                logger.info("ANOVA complete")

        # Calculate heritability if configured
        if config.statistics.calculate_heritability:
            if replicate_col is None:
                logger.warning(
                    "Cannot calculate heritability: replicate column not specified"
                )
            else:
                logger.info("Calculating broad-sense heritability...")
                h2_dict = calculate_heritability_estimates(
                    data,
                    trait_cols=trait_cols,
                    genotype_col=genotype_col,
                    replicate_col=replicate_col,
                )
                # Convert dict to DataFrame
                heritability_results = pd.DataFrame(h2_dict).T
                heritability_results["trait"] = heritability_results.index
                heritability_results = heritability_results.reset_index(drop=True)
                metadata["heritability_results"] = heritability_results
                if "H2" in heritability_results.columns:
                    mean_h2 = heritability_results["H2"].mean()
                    logger.info(f"Heritability complete: mean H² = {mean_h2:.3f}")
                else:
                    logger.info("Heritability complete")

        # Save statistics summary
        stats_dir = run_dir / "statistics"
        stats_dir.mkdir(exist_ok=True)

        # Save descriptive stats
        descriptive_stats.to_csv(stats_dir / "descriptive_stats.csv")

        if "anova_results" in metadata:
            metadata["anova_results"].to_csv(stats_dir / "anova_results.csv")

        if "heritability_results" in metadata:
            metadata["heritability_results"].to_csv(
                stats_dir / "heritability_results.csv"
            )

        logger.info(f"Saved statistics to: {stats_dir}")

        return StepResult(
            data=data,
            metadata=metadata,
        )
