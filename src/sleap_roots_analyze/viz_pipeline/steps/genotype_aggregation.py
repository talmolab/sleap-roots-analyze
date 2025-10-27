"""Step 8: Genotype aggregation (stub - to be implemented in Phase 2B)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class GenotypeAggregationStep(BaseStep):
    """Aggregate data by genotype for comparisons (Phase 2B)."""

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute genotype aggregation (stub)."""
        # TODO: Implement in Phase 2B
        logger.warning("Genotype aggregation not yet implemented (Phase 2B)")
        return StepResult(
            data=data,
            metadata=prev_result.metadata,
            message="Genotype aggregation not implemented (Phase 2B)",
        )
