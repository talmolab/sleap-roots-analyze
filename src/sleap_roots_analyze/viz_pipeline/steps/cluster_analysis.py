"""Step 5: Clustering analysis (stub - to be implemented in Phase 2C)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class ClusterAnalysisStep(BaseStep):
    """Perform clustering analysis (Phase 2C)."""

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute clustering analysis (stub)."""
        if not config.clustering.enabled:
            logger.info("Clustering analysis disabled, skipping")
            return StepResult(
                data=data,
                metadata=prev_result.metadata,
                message="Clustering disabled",
            )

        # TODO: Implement in Phase 2C
        logger.warning("Clustering analysis not yet implemented (Phase 2C)")
        return StepResult(
            data=data,
            metadata=prev_result.metadata,
            message="Clustering not implemented (Phase 2C)",
        )
