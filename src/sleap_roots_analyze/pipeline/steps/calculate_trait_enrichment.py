"""Step: Trait-level binomial enrichment for cross-platform correlations.

This is a config-gated downstream node in the cross-platform DAG. It tests
whether the number of nominally significant (``p < significance_level``)
trait correlations for this platform pair deviates from chance, using an exact
binomial test (no FDR). It runs only when ``config.enrichment_enabled`` is set;
otherwise it is a pass-through.

Per-pair enrichment is a clean DAG step (the pipeline is pairwise). Enrichment
pooled across all platform pairs (the "Combined" result) is an all-platforms
synthesis concern handled by the public
:func:`sleap_roots_analyze.trait_correlation_enrichment` function, not this step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from sleap_roots_analyze.pc_correlations.enrichment import (
    calculate_enrichment_test,
    count_significant,
    results_to_dataframe,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class CalculateTraitEnrichmentStep(BaseStep):
    """Binomial enrichment test on this platform pair's correlation results.

    The input correlation table is already representative-only when
    ``trait_reduction_method="clustering"`` upstream (one trait per cluster), so
    counting its rows directly gives the representative-trait enrichment; with
    ``trait_reduction_method="none"`` it is the full-trait enrichment. Either way
    the step counts the rows of the produced correlation table — it does not need
    to re-filter representatives.
    """

    def __init__(self):
        """Initialize CalculateTraitEnrichmentStep."""
        super().__init__(
            step_name="CalculateTraitEnrichment",
            description="Binomial enrichment test on nominal correlation significance",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the trait-enrichment step.

        Args:
            data: Dict from the correlation step (contains ``correlation_df``).
            config: ``CrossPlatformConfig`` (``enrichment_enabled``,
                ``enrichment_p_value_column``, ``significance_level``,
                ``confidence_level``, ``correlation_method``).
            run_dir: Directory to save outputs.
            prev_result: The correlation step's result (metadata: experiment
                names, etc.).

        Returns:
            StepResult passing ``data`` through unchanged (so visualization
            still receives the correlation table), with enrichment metadata and
            — when enabled — a written ``trait_enrichment.csv``.
        """
        prev_metadata = prev_result.metadata if prev_result is not None else {}

        if not getattr(config, "enrichment_enabled", False):
            logger.info("Trait enrichment disabled, skipping")
            return StepResult(
                data=data,
                metadata={**prev_metadata, "enrichment_enabled": False},
                files_generated=[],
            )

        correlation_df = data["correlation_df"]
        p_col = config.enrichment_p_value_column
        exp1_name = prev_metadata.get("exp1_name", "Experiment 1")
        exp2_name = prev_metadata.get("exp2_name", "Experiment 2")
        pair_label = f"{exp1_name} vs {exp2_name}"

        n_significant, n_tests = count_significant(
            correlation_df, p_value_column=p_col, alpha=config.significance_level
        )
        result = calculate_enrichment_test(
            n_significant=n_significant,
            n_tests=n_tests,
            platform_pair=pair_label,
            alpha=config.significance_level,
            confidence_level=config.confidence_level,
        )

        output_file = run_dir / "trait_enrichment.csv"
        results_to_dataframe([result]).to_csv(output_file, index=False)
        logger.info(
            "Trait enrichment (%s): %d/%d significant, fold=%.2f (%s)",
            pair_label,
            result.n_significant,
            result.n_tests,
            result.fold_enrichment,
            result.interpretation,
        )

        return StepResult(
            data=data,
            metadata={
                **prev_metadata,
                "enrichment_enabled": True,
                "enrichment_p_value_column": p_col,
                "enrichment_n_tests": result.n_tests,
                "enrichment_n_significant": result.n_significant,
                "enrichment_fold": result.fold_enrichment,
                "enrichment_interpretation": result.interpretation,
            },
            files_generated=[str(output_file)],
        )
