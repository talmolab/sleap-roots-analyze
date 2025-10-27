"""Step 4: UMAP analysis (stub - to be implemented in Phase 2C)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class UMAPAnalysisStep(BaseStep):
    """Perform UMAP dimensionality reduction (Phase 2C)."""

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute UMAP analysis (stub)."""
        if not config.umap.enabled:
            logger.info("UMAP analysis disabled, skipping")
            return StepResult(
                data=data,
                metadata=prev_result.metadata,
                message="UMAP disabled",
            )

        # TODO: Implement in Phase 2C
        logger.warning("UMAP analysis not yet implemented (Phase 2C)")
        return StepResult(
            data=data,
            metadata=prev_result.metadata,
            message="UMAP not implemented (Phase 2C)",
        )
