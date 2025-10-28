"""Step 6: Heritability analysis and filtering."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class HeritabilityAnalysisStep(BaseStep):
    """Analyze heritability and optionally filter low heritability traits.

    This step:
    1. Uses heritability results from calculate_statistics step
    2. Optionally filters traits below heritability threshold
    3. Updates trait list for downstream steps

    Input: DataFrame with trait data
    Output: DataFrame (potentially filtered) + updated metadata
    """

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute the heritability analysis step.

        Args:
            data: DataFrame with trait data.
            config: Pipeline configuration.
            run_dir: Directory for this pipeline run.
            prev_result: Result from previous step (calculate_statistics).

        Returns:
            StepResult with potentially filtered DataFrame and metadata.
        """
        # Check if filtering is enabled
        if not config.heritability.filter_enabled:
            logger.info("Heritability filtering disabled, passing through")
            return StepResult(
                data=data,
                metadata=prev_result.metadata,
            )

        # Check if heritability was calculated
        if "heritability_results" not in prev_result.metadata:
            logger.warning(
                "Heritability filtering enabled but heritability was not calculated. "
                "Skipping filtering."
            )
            return StepResult(
                data=data,
                metadata=prev_result.metadata,
            )

        heritability_df = prev_result.metadata["heritability_results"]
        threshold = config.heritability.threshold
        trait_cols = prev_result.metadata["trait_cols"]

        logger.info(f"Filtering traits with H² >= {threshold}")

        # Filter traits by heritability
        high_h2_traits = heritability_df[heritability_df["H2"] >= threshold][
            "trait"
        ].tolist()

        # Keep only traits that are both in trait_cols and have high heritability
        filtered_trait_cols = [t for t in trait_cols if t in high_h2_traits]

        n_removed = len(trait_cols) - len(filtered_trait_cols)
        logger.info(
            f"Filtered {n_removed} low heritability traits, {len(filtered_trait_cols)} remain"
        )

        # Update metadata with filtered trait list
        metadata = {
            **prev_result.metadata,
            "trait_cols": filtered_trait_cols,
            "n_traits": len(filtered_trait_cols),
            "heritability_threshold": threshold,
            "n_traits_removed_by_heritability": n_removed,
        }

        # Save filtering summary
        filter_summary_path = run_dir / "heritability_filtering_summary.txt"
        with open(filter_summary_path, "w") as f:
            f.write(f"Heritability Filtering Summary\n")
            f.write(f"{'=' * 40}\n\n")
            f.write(f"Threshold: H² >= {threshold}\n")
            f.write(f"Traits before filtering: {len(trait_cols)}\n")
            f.write(f"Traits after filtering: {len(filtered_trait_cols)}\n")
            f.write(f"Traits removed: {n_removed}\n\n")
            f.write(f"High heritability traits (n={len(filtered_trait_cols)}):\n")
            for trait in filtered_trait_cols:
                h2 = heritability_df[heritability_df["trait"] == trait]["H2"].values[0]
                f.write(f"  - {trait}: H² = {h2:.3f}\n")

        logger.info(f"Saved filtering summary to: {filter_summary_path}")

        return StepResult(
            data=data,
            metadata=metadata,
        )
