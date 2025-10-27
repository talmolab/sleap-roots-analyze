"""Step 9: Generate static figures (stub - to be implemented in Phase 2B)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class GenerateStaticFiguresStep(BaseStep):
    """Generate publication-quality static figures (Phase 2B)."""

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute static figure generation (stub)."""
        if not config.static_viz.enabled:
            logger.info("Static visualization disabled, skipping")
            return StepResult(
                data=data,
                metadata=prev_result.metadata,
                message="Static viz disabled",
            )

        # TODO: Implement in Phase 2B
        logger.warning("Static figure generation not yet implemented (Phase 2B)")
        return StepResult(
            data=data,
            metadata=prev_result.metadata,
            message="Static figures not implemented (Phase 2B)",
        )
