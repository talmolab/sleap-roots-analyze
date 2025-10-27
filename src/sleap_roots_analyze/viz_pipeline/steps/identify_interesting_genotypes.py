"""Step 7: Identify interesting genotypes (stub - to be implemented in Phase 2C)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class IdentifyInterestingGenotypesStep(BaseStep):
    """Identify extreme and heritable genotypes (Phase 2C)."""

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute interesting genotypes identification (stub)."""
        if not config.interesting_genotypes.enabled:
            logger.info("Interesting genotypes identification disabled, skipping")
            return StepResult(
                data=data,
                metadata=prev_result.metadata,
                message="Interesting genotypes disabled",
            )

        # TODO: Implement in Phase 2C
        logger.warning(
            "Interesting genotypes identification not yet implemented (Phase 2C)"
        )
        return StepResult(
            data=data,
            metadata=prev_result.metadata,
            message="Interesting genotypes not implemented (Phase 2C)",
        )
