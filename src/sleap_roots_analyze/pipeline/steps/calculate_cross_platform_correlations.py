"""Step: Calculate cross-platform trait correlations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np

from sleap_roots_analyze.cross_experiment_analysis import (
    calculate_genotype_means,
    calculate_correlations,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


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
        correlation_results = []

        for trait1 in exp1_traits:
            for trait2 in exp2_traits:
                # Extract trait values
                x = exp1_means[trait1].values
                y = exp2_means[trait2].values

                # Calculate correlation
                r, p = calculate_correlations(
                    x, y, method=config.correlation_method
                )

                # Count valid genotypes (non-NaN in both traits)
                valid_mask = ~(np.isnan(x) | np.isnan(y))
                n_genotypes = valid_mask.sum()

                correlation_results.append(
                    {
                        "exp1_trait": trait1,
                        "exp2_trait": trait2,
                        "correlation": r,
                        "p_value": p,
                        "n_genotypes": n_genotypes,
                    }
                )

        # Create DataFrame
        correlation_df = pd.DataFrame(correlation_results)

        # Sort by absolute correlation (descending)
        correlation_df = correlation_df.assign(
            abs_correlation=correlation_df["correlation"].abs()
        )
        correlation_df = correlation_df.sort_values(
            "abs_correlation", ascending=False
        ).drop(columns=["abs_correlation"])
        correlation_df = correlation_df.reset_index(drop=True)

        # Save correlation results
        corr_output = run_dir / "cross_platform_correlations.csv"
        correlation_df.to_csv(corr_output, index=False)

        # Prepare result data (include exp1_df and exp2_df for downstream visualization)
        result_data = {
            "correlation_df": correlation_df,
            "exp1_df": exp1_df,
            "exp2_df": exp2_df,
        }

        # Prepare metadata (pass through trait names and experiment names for visualization)
        metadata = {
            "total_correlations": len(correlation_df),
            "correlation_method": config.correlation_method,
            "exp1_traits": len(exp1_traits),
            "exp2_traits": len(exp2_traits),
            "exp1_trait_names": exp1_traits,
            "exp2_trait_names": exp2_traits,
            "exp1_name": prev_result.metadata.get("exp1_name", "Experiment 1"),
            "exp2_name": prev_result.metadata.get("exp2_name", "Experiment 2"),
        }

        return StepResult(
            data=result_data,
            metadata=metadata,
            files_generated=[str(corr_output)],
        )
