"""Step 11: Generate dashboards (stub - to be implemented in Phase 2C)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class GenerateDashboardsStep(BaseStep):
    """Generate summary dashboards (Phase 2C)."""

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute dashboard generation (stub)."""
        if not config.dashboard.enabled:
            logger.info("Dashboard generation disabled, skipping")
            return StepResult(
                data=data,
                metadata=prev_result.metadata,
            )

        # TODO: Implement in Phase 2C
        logger.warning("Dashboard generation not yet implemented (Phase 2C)")
        return StepResult(
            data=data,
            metadata=prev_result.metadata,
        )
