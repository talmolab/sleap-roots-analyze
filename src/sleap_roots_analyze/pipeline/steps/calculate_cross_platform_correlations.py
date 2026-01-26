"""Step: Calculate cross-platform trait correlations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

from sleap_roots_analyze.cross_experiment_analysis import (
    calculate_genotype_means,
    calculate_correlations,
    calculate_correlation_ci,
    minimum_detectable_correlation,
    achieved_power,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


def _apply_fdr_correction_safe(
    p_values: pd.Series, alpha: float, method: str
) -> np.ndarray:
    """Apply FDR correction while handling NaN p-values.

    The statsmodels multipletests function returns all NaN if any input is NaN.
    This function filters out NaN values, applies correction to valid values,
    and merges results back preserving NaN for invalid correlations.

    Args:
        p_values: Series of p-values, may contain NaN
        alpha: Significance level for FDR correction
        method: FDR method ('fdr_bh', 'fdr_by', etc.)

    Returns:
        Array of adjusted p-values with NaN preserved for invalid inputs
    """
    # Initialize result array with NaN
    adjusted = np.full(len(p_values), np.nan)

    # Find valid (non-NaN) p-values
    valid_mask = ~p_values.isna()
    n_valid = valid_mask.sum()
    n_nan = (~valid_mask).sum()

    if n_nan > 0:
        logger.warning(
            f"Found {n_nan} NaN p-values out of {len(p_values)} total. "
            "These will be excluded from FDR correction and remain NaN."
        )

    if n_valid == 0:
        # All NaN - return all NaN
        logger.warning("All p-values are NaN. No FDR correction applied.")
        return adjusted

    if n_valid == 1:
        # Single valid p-value - no multiple testing correction needed
        # Adjusted p-value equals raw p-value
        adjusted[valid_mask] = p_values[valid_mask].values
        return adjusted

    # Apply FDR correction only to valid p-values
    _, p_adj, _, _ = multipletests(
        p_values[valid_mask].values,
        alpha=alpha,
        method=method,
    )

    # Place adjusted values back in original positions
    adjusted[valid_mask] = p_adj

    return adjusted


class CalculateCrossPlatformCorrelationsStep(BaseStep):
    """Calculate correlations between traits across two experimental platforms.

    This step:
    1. Retrieves aligned data from previous step
    2. Calculates genotype means for each trait in both experiments
    3. Computes pairwise correlations between all trait combinations
    4. Sorts correlations by absolute value
    5. Saves correlation results to CSV

    Outputs:
        - cross_platform_correlations.csv: All pairwise trait correlations
    """

    def __init__(self):
        """Initialize CalculateCrossPlatformCorrelationsStep."""
        super().__init__(
            step_name="CalculateCrossPlatformCorrelations",
            description="Calculate correlations between cross-platform traits",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the cross-platform correlation calculation step.

        Args:
            data: Dictionary containing exp1_df, exp2_df, and common_genotypes
            config: CrossPlatformConfig with correlation_method
            run_dir: Directory to save outputs
            prev_result: Previous step result with metadata about trait names

        Returns:
            StepResult with correlation DataFrame and metadata

        Raises:
            ValueError: If required data is missing
        """
        # Validate inputs
        if prev_result is None:
            raise ValueError("prev_result is required for correlation calculation")

        # Get data from previous step
        exp1_df = data["exp1_df"]
        exp2_df = data["exp2_df"]
        common_genotypes = data["common_genotypes"]

        # Get trait names from metadata
        exp1_traits = prev_result.metadata["exp1_trait_names"]
        exp2_traits = prev_result.metadata["exp2_trait_names"]

        # Calculate genotype means for each experiment
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

        # Filter to common genotypes only (genotype is the index)
        exp1_means = exp1_means[exp1_means.index.isin(common_genotypes)]
        exp2_means = exp2_means[exp2_means.index.isin(common_genotypes)]

        # Sort by genotype (index) to ensure alignment
        exp1_means = exp1_means.sort_index()
        exp2_means = exp2_means.sort_index()

        # Calculate correlations for all trait pairs
        # Store BOTH Pearson and Spearman for each pair (single source of truth)
        correlation_results = []
        n_filtered_low_n = 0  # Count pairs filtered due to insufficient genotypes

        for trait1 in exp1_traits:
            for trait2 in exp2_traits:
                # Extract trait values
                x = exp1_means[trait1].values
                y = exp2_means[trait2].values

                # Count valid genotypes (non-NaN in both traits)
                valid_mask = ~(np.isnan(x) | np.isnan(y))
                n_genotypes = valid_mask.sum()

                # Apply min_genotypes_for_correlation filter (hard filter)
                if n_genotypes < config.min_genotypes_for_correlation:
                    n_filtered_low_n += 1
                    logger.debug(
                        "Skipping %s vs %s: only %d genotypes (min: %d)",
                        trait1,
                        trait2,
                        n_genotypes,
                        config.min_genotypes_for_correlation,
                    )
                    continue

                # Calculate BOTH correlation methods
                spearman_r, spearman_p = calculate_correlations(x, y, method="spearman")
                pearson_r, pearson_p = calculate_correlations(x, y, method="pearson")

                # Calculate confidence intervals for both correlation methods
                spearman_ci_low, spearman_ci_high = calculate_correlation_ci(
                    spearman_r, n_genotypes, config.confidence_level
                )
                pearson_ci_low, pearson_ci_high = calculate_correlation_ci(
                    pearson_r, n_genotypes, config.confidence_level
                )

                # Calculate achieved power for the primary correlation
                if config.correlation_method == "spearman":
                    primary_r = spearman_r
                elif config.correlation_method == "pearson":
                    primary_r = pearson_r
                else:
                    # Kendall and any other unsupported methods are not valid for this step.
                    # Failing fast here avoids silently using Pearson when Kendall is configured.
                    raise ValueError(
                        f"Unsupported correlation_method for cross-platform correlations: "
                        f"{config.correlation_method!r}. "
                        "Supported methods are 'spearman' and 'pearson' for this pipeline step."
                    )
                power = achieved_power(
                    r=primary_r, n=n_genotypes, alpha=config.power_analysis_alpha
                )

                correlation_results.append(
                    {
                        "exp1_trait": trait1,
                        "exp2_trait": trait2,
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "spearman_r_ci_low": spearman_ci_low,
                        "spearman_r_ci_high": spearman_ci_high,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "pearson_r_ci_low": pearson_ci_low,
                        "pearson_r_ci_high": pearson_ci_high,
                        "n_genotypes": n_genotypes,
                        "achieved_power": power,
                    }
                )

        # Log filter summary
        if n_filtered_low_n > 0:
            total_pairs = len(exp1_traits) * len(exp2_traits)
            logger.info(
                f"Filtered {n_filtered_low_n}/{total_pairs} trait pairs with "
                f"n_genotypes < {config.min_genotypes_for_correlation}"
            )

        # Create DataFrame
        correlation_df = pd.DataFrame(correlation_results)

        # Handle empty results (all trait pairs filtered out)
        if len(correlation_df) == 0:
            # Create empty DataFrame with expected columns for downstream compatibility
            correlation_df = pd.DataFrame(
                columns=[
                    "exp1_trait",
                    "exp2_trait",
                    "spearman_r",
                    "spearman_p",
                    "spearman_r_ci_low",
                    "spearman_r_ci_high",
                    "pearson_r",
                    "pearson_p",
                    "pearson_r_ci_low",
                    "pearson_r_ci_high",
                    "n_genotypes",
                    "achieved_power",
                    "spearman_p_adjusted",
                    "pearson_p_adjusted",
                    "significant_fdr",
                ]
            )
            logger.warning(
                f"All {len(exp1_traits) * len(exp2_traits)} trait pairs were filtered out. "
                f"Consider lowering min_genotypes_for_correlation "
                f"(current: {config.min_genotypes_for_correlation})."
            )
        else:
            # Sort by absolute value of PRIMARY correlation (determined by config)
            # This determines ranking but both methods are stored
            primary_col = (
                "spearman_r" if config.correlation_method == "spearman" else "pearson_r"
            )
            correlation_df = correlation_df.assign(
                abs_correlation=correlation_df[primary_col].abs()
            )
            correlation_df = correlation_df.sort_values(
                "abs_correlation", ascending=False
            ).drop(columns=["abs_correlation"])
            correlation_df = correlation_df.reset_index(drop=True)

        # Apply multiple testing correction
        if config.fdr_correction_method != "none" and len(correlation_df) > 0:
            # Use safe FDR correction that handles NaN p-values
            # (multipletests returns all NaN if any input is NaN)
            correlation_df["spearman_p_adjusted"] = _apply_fdr_correction_safe(
                correlation_df["spearman_p"],
                alpha=config.significance_level,
                method=config.fdr_correction_method,
            )

            correlation_df["pearson_p_adjusted"] = _apply_fdr_correction_safe(
                correlation_df["pearson_p"],
                alpha=config.significance_level,
                method=config.fdr_correction_method,
            )

            # Add significance flag based on primary correlation method
            # NaN adjusted p-values result in False (not significant)
            primary_p_adj = (
                "spearman_p_adjusted"
                if config.correlation_method == "spearman"
                else "pearson_p_adjusted"
            )
            # Use fillna(False) to ensure NaN p-values are not marked significant
            correlation_df["significant_fdr"] = (
                correlation_df[primary_p_adj] < config.significance_level
            ).fillna(False)
        else:
            # No correction - adjusted equals raw
            correlation_df["spearman_p_adjusted"] = correlation_df["spearman_p"]
            correlation_df["pearson_p_adjusted"] = correlation_df["pearson_p"]
            # Significance based on raw p-values
            # NaN p-values result in False (not significant)
            primary_p = (
                "spearman_p" if config.correlation_method == "spearman" else "pearson_p"
            )
            correlation_df["significant_fdr"] = (
                correlation_df[primary_p] < config.significance_level
            ).fillna(False)

        # Save correlation results
        corr_output = run_dir / "cross_platform_correlations.csv"
        correlation_df.to_csv(corr_output, index=False)

        # Prepare result data (include exp1_df and exp2_df for downstream visualization)
        result_data = {
            "correlation_df": correlation_df,
            "exp1_df": exp1_df,
            "exp2_df": exp2_df,
        }

        # Calculate minimum detectable r for power analysis summary
        # Use modal n_genotypes from the results for the MDR calculation
        if len(correlation_df) > 0:
            modal_n = int(correlation_df["n_genotypes"].mode().iloc[0])
            mdr = minimum_detectable_correlation(
                n=modal_n,
                alpha=config.power_analysis_alpha,
                power=config.power_analysis_power,
            )
        else:
            modal_n = 0
            mdr = np.nan

        # Prepare metadata (pass through trait names and experiment names for visualization)
        metadata = {
            "total_correlations": len(correlation_df),
            "correlation_method": config.correlation_method,
            "fdr_correction_method": config.fdr_correction_method,
            "significance_level": config.significance_level,
            "confidence_level": config.confidence_level,
            "significant_correlations": (
                int(correlation_df["significant_fdr"].sum())
                if len(correlation_df) > 0
                else 0
            ),
            "exp1_traits": len(exp1_traits),
            "exp2_traits": len(exp2_traits),
            "exp1_trait_names": exp1_traits,
            "exp2_trait_names": exp2_traits,
            "exp1_name": prev_result.metadata.get("exp1_name", "Experiment 1"),
            "exp2_name": prev_result.metadata.get("exp2_name", "Experiment 2"),
            # Power analysis metadata
            "min_genotypes_for_correlation": config.min_genotypes_for_correlation,
            "n_correlations_filtered_low_n": n_filtered_low_n,
            "power_analysis_alpha": config.power_analysis_alpha,
            "power_analysis_power": config.power_analysis_power,
            "minimum_detectable_r": mdr,
            "modal_n_genotypes": modal_n,
        }

        return StepResult(
            data=result_data,
            metadata=metadata,
            files_generated=[str(corr_output)],
        )
