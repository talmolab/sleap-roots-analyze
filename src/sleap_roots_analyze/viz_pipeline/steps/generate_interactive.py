"""Step 10: Generate interactive visualizations (stub - to be implemented in Phase 2B)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class GenerateInteractiveStep(BaseStep):
    """Generate interactive visualizations with image hover (Phase 2B)."""

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute interactive visualization generation (stub)."""
        if not config.interactive_viz.enabled:
            logger.info("Interactive visualization disabled, skipping")
            return StepResult(
                data=data,
                metadata=prev_result.metadata,
                message="Interactive viz disabled",
            )

        # TODO: Implement in Phase 2B
        logger.warning(
            "Interactive visualization generation not yet implemented (Phase 2B)"
        )
        return StepResult(
            data=data,
            metadata=prev_result.metadata,
            message="Interactive viz not implemented (Phase 2B)",
        )
