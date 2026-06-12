"""Step: Load and align cross-platform experimental data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.cross_experiment_analysis import load_and_align_experiments
from sleap_roots_analyze.data_cleanup import get_trait_columns
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.validation import validate_cross_platform_experiment


class LoadCrossPlatformDataStep(BaseStep):
    """Load and align data from two experimental platforms.

    This step:
    1. Loads data from two experiments
    2. Identifies common genotypes
    3. Filters by minimum samples per genotype
    4. Identifies trait columns in each experiment
    5. Saves alignment summary and loaded data

    Outputs:
        - cross_platform_exp1_loaded.csv: Experiment 1 data
        - cross_platform_exp2_loaded.csv: Experiment 2 data
        - cross_platform_alignment_summary.csv: Summary of common genotypes and sample counts
    """

    def __init__(self):
        """Initialize LoadCrossPlatformDataStep."""
        super().__init__(
            step_name="LoadCrossPlatformData",
            description="Load and align cross-platform experimental data",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the cross-platform data loading step.

        Args:
            data: Unused (data is loaded from paths in config)
            config: CrossPlatformConfig with:
                - exp1_data_path: Path to experiment 1 CSV
                - exp1_name: Name of experiment 1
                - exp1_genotype_col: Genotype column name in exp1
                - exp2_data_path: Path to experiment 2 CSV
                - exp2_name: Name of experiment 2
                - exp2_genotype_col: Genotype column name in exp2
                - min_samples_per_genotype: Minimum samples required per genotype
                - exp1_exclude_cols: Columns to exclude from exp1 trait analysis
                - exp2_exclude_cols: Columns to exclude from exp2 trait analysis
            run_dir: Directory to save outputs
            prev_result: Previous step result (unused)

        Returns:
            StepResult with loaded and aligned DataFrames and metadata

        Raises:
            ValueError: If no common genotypes are found
        """
        # Load and align experiments
        exp1_df, exp2_df, common_genotypes = load_and_align_experiments(
            exp1_path=config.exp1_data_path,
            exp2_path=config.exp2_data_path,
            genotype_col1=config.exp1_genotype_col,
            genotype_col2=config.exp2_genotype_col,
        )

        # Input-contract validation at the entry boundary (issue #154). Each aligned
        # experiment frame already carries canonical genotype/replicate columns; the
        # side-check runs on a discarded copy and never alters exp1_df/exp2_df.
        # Degrades to a logged no-op when contracts is absent or validate_input=="off".
        validate_cross_platform_experiment(exp1_df, mode=config.validate_input)
        validate_cross_platform_experiment(exp2_df, mode=config.validate_input)

        # Get trait columns for each experiment
        # Note: column names are standardized to "genotype" and "replicate"
        # Set barcode_col to None since it may not exist in the loaded data
        # Pass config exclude_cols via additional_exclude param to filter metadata
        exp1_traits = get_trait_columns(
            exp1_df,
            barcode_col=None,
            genotype_col="genotype",
            replicate_col="replicate",
            additional_exclude=config.exp1_exclude_cols,
        )
        exp2_traits = get_trait_columns(
            exp2_df,
            barcode_col=None,
            genotype_col="genotype",
            replicate_col="replicate",
            additional_exclude=config.exp2_exclude_cols,
        )

        # Filter genotypes by minimum sample count
        # Note: load_and_align_experiments standardizes column names to "genotype"
        valid_genotypes = []
        for geno in common_genotypes:
            exp1_count = (exp1_df["genotype"] == geno).sum()
            exp2_count = (exp2_df["genotype"] == geno).sum()

            if (
                exp1_count >= config.min_samples_per_genotype
                and exp2_count >= config.min_samples_per_genotype
            ):
                valid_genotypes.append(geno)

        if len(valid_genotypes) == 0:
            raise ValueError(
                f"No common genotypes found with at least "
                f"{config.min_samples_per_genotype} samples in both experiments"
            )

        # Create alignment summary
        alignment_summary = []
        for geno in valid_genotypes:
            exp1_count = (exp1_df["genotype"] == geno).sum()
            exp2_count = (exp2_df["genotype"] == geno).sum()
            alignment_summary.append(
                {
                    "genotype": geno,
                    "exp1_samples": exp1_count,
                    "exp2_samples": exp2_count,
                }
            )

        alignment_df = pd.DataFrame(alignment_summary)

        # Save outputs
        exp1_output = run_dir / "cross_platform_exp1_loaded.csv"
        exp2_output = run_dir / "cross_platform_exp2_loaded.csv"
        alignment_output = run_dir / "cross_platform_alignment_summary.csv"

        exp1_df.to_csv(exp1_output, index=False)
        exp2_df.to_csv(exp2_output, index=False)
        alignment_df.to_csv(alignment_output, index=False)

        # Prepare result data
        result_data = {
            "exp1_df": exp1_df,
            "exp2_df": exp2_df,
            "common_genotypes": valid_genotypes,
        }

        # Prepare metadata
        metadata = {
            "exp1_name": config.exp1_name,
            "exp2_name": config.exp2_name,
            "exp1_samples": len(exp1_df),
            "exp2_samples": len(exp2_df),
            "exp1_traits": len(exp1_traits),
            "exp2_traits": len(exp2_traits),
            "exp1_trait_names": exp1_traits,
            "exp2_trait_names": exp2_traits,
            "common_genotypes": len(valid_genotypes),
            "min_samples_per_genotype": config.min_samples_per_genotype,
        }

        return StepResult(
            data=result_data,
            metadata=metadata,
            files_generated=[
                str(exp1_output),
                str(exp2_output),
                str(alignment_output),
            ],
        )
