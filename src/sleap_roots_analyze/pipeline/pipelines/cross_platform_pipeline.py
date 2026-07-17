"""Cross-Platform Pipeline orchestrator for comparing traits across experiments.

This module provides the CrossPlatformPipeline class that orchestrates the
cross-platform analysis steps in a NetworkX-based DAG for reproducible, automated
cross-experiment analysis.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List

from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig
from sleap_roots_analyze.pipeline.pipelines.base_pipeline import BasePipeline
from sleap_roots_analyze.pipeline.task import Task
from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
    LoadCrossPlatformDataStep,
)
from sleap_roots_analyze.pipeline.steps.reduce_trait_redundancy import (
    ReduceTraitRedundancyStep,
)
from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
    CalculateCrossPlatformCorrelationsStep,
)
from sleap_roots_analyze.pipeline.steps.calculate_trait_enrichment import (
    CalculateTraitEnrichmentStep,
)
from sleap_roots_analyze.pipeline.steps.visualize_cross_platform import (
    VisualizeCrossPlatformStep,
)
from sleap_roots_analyze.pipeline.steps.predict_cross_platform import (
    PredictCrossPlatformStep,
)


class CrossPlatformPipeline(BasePipeline):
    """Cross-platform analysis pipeline for comparing traits across experiments.

    This pipeline performs:
    1. Load and align data from two experiments
    2. Reduce trait redundancy (optional, config-gated)
    3. Calculate correlations between all trait pairs
    4. Trait-level binomial enrichment (optional, config-gated)
    5. Generate visualizations of correlations

    The pipeline is designed to compare traits across different experimental
    platforms or conditions to identify which traits capture similar biological
    variation patterns.
    """

    def __init__(
        self,
        config: CrossPlatformConfig,
        output_dir: Path | str = "./cross_platform_runs",
        log_filename: str | None = None,
    ):
        """Initialize CrossPlatformPipeline.

        Args:
            config: CrossPlatformConfig with experiment paths and analysis parameters
            output_dir: Base directory for output (default: "./cross_platform_runs")
            log_filename: Custom log filename for run directory. If None, uses
                "pipeline.log".
        """
        # Sanitize experiment names for safe folder naming
        exp1_safe = self._sanitize_folder_name(config.exp1_name)
        exp2_safe = self._sanitize_folder_name(config.exp2_name)

        super().__init__(
            config=config,
            output_dir=Path(output_dir),
            pipeline_name=f"cross_platform_{exp1_safe}_vs_{exp2_safe}",
            log_filename=log_filename,
        )

    @staticmethod
    def _sanitize_folder_name(name: str) -> str:
        """Sanitize a name for safe use in folder paths.

        Replaces spaces with underscores and removes special characters that
        may cause issues on Windows, macOS, or Linux filesystems.

        Args:
            name: The name to sanitize

        Returns:
            Sanitized name safe for folder paths
        """
        # Replace spaces with underscores
        sanitized = name.replace(" ", "_")
        # Remove parentheses and their contents (e.g., "(QC'd)" -> "")
        sanitized = re.sub(r"\s*\([^)]*\)", "", sanitized)
        # Remove any remaining special characters (keep alphanumeric, underscore, hyphen)
        sanitized = re.sub(r"[^\w\-]", "", sanitized)
        # Remove any trailing underscores
        sanitized = sanitized.rstrip("_")
        return sanitized

    def create_tasks(self) -> List[Task]:
        """Create the cross-platform analysis task graph with 5 steps.

        Returns:
            List of Tasks defining the pipeline DAG:
                - Step 1: LoadCrossPlatformData
                - Step 2: ReduceTraitRedundancy (optional, based on config)
                - Step 3: CalculateCrossPlatformCorrelations
                - Step 4: CalculateTraitEnrichment (optional, based on config)
                - Step 5: VisualizeCrossPlatform
        """
        tasks = []

        # Step 1: Load and align cross-platform data
        tasks.append(
            Task(
                func=self._run_load_cross_platform_data,
                name="01_load_cross_platform_data",
                depends_on=[],
                description=f"Load and align {self.config.exp1_name} vs {self.config.exp2_name}",
            )
        )

        # Step 2: Reduce trait redundancy (optional - always runs but may be no-op)
        tasks.append(
            Task(
                func=self._run_reduce_trait_redundancy,
                name="02_reduce_trait_redundancy",
                depends_on=["01_load_cross_platform_data"],
                description=f"Reduce trait redundancy ({self.config.trait_reduction_method})",
            )
        )

        # Step 3: Calculate correlations between all trait pairs
        tasks.append(
            Task(
                func=self._run_calculate_correlations,
                name="03_calculate_correlations",
                depends_on=["02_reduce_trait_redundancy"],
                description=f"Calculate correlations using {self.config.correlation_method}",
            )
        )

        # Step 4: Trait-level enrichment (optional - always runs but may be no-op)
        tasks.append(
            Task(
                func=self._run_calculate_trait_enrichment,
                name="04_calculate_trait_enrichment",
                depends_on=["03_calculate_correlations"],
                description=(
                    "Binomial enrichment on nominal significance "
                    f"(enabled={self.config.enrichment_enabled})"
                ),
            )
        )

        # Step 5: Visualize correlations
        tasks.append(
            Task(
                func=self._run_visualize_cross_platform,
                name="05_visualize_cross_platform",
                depends_on=["04_calculate_trait_enrichment"],
                description="Generate correlation visualizations",
            )
        )

        # Step 6: Predict cross-platform genotype values (optional, config-gated;
        # entirely absent -- not merely skipped -- when disabled, per Decision 1).
        if self.config.prediction.enabled:
            tasks.append(
                Task(
                    func=self._run_predict_cross_platform,
                    name="06_predict_cross_platform",
                    depends_on=[
                        "01_load_cross_platform_data",
                        "05_visualize_cross_platform",
                    ],
                    description="Predict cross-platform genotype values via LOGO-CV",
                )
            )

        return tasks

    def _run_load_cross_platform_data(self, config, run_dir, logger, **kwargs):
        """Execute Step 1: Load Cross-Platform Data."""
        logger.info("Step 1/5: Loading and aligning cross-platform data...")
        step = LoadCrossPlatformDataStep()
        result = step.execute(
            data=None, config=config, run_dir=run_dir, prev_result=None
        )
        return result

    def _run_reduce_trait_redundancy(self, config, run_dir, logger, **kwargs):
        """Execute Step 2: Reduce Trait Redundancy."""
        logger.info(
            "Step 2/5: Reducing trait redundancy (method=%s)...",
            config.trait_reduction_method,
        )
        prev_task_result = kwargs.get("01_load_cross_platform_data")
        prev_step_result = prev_task_result.data
        step = ReduceTraitRedundancyStep()
        result = step.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_calculate_correlations(self, config, run_dir, logger, **kwargs):
        """Execute Step 3: Calculate Cross-Platform Correlations."""
        logger.info("Step 3/5: Calculating correlations between all trait pairs...")
        prev_task_result = kwargs.get("02_reduce_trait_redundancy")
        prev_step_result = prev_task_result.data
        step = CalculateCrossPlatformCorrelationsStep()
        result = step.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_calculate_trait_enrichment(self, config, run_dir, logger, **kwargs):
        """Execute Step 4: Trait-Level Enrichment (config-gated; pass-through)."""
        logger.info(
            "Step 4/5: Trait enrichment (enabled=%s)...", config.enrichment_enabled
        )
        prev_task_result = kwargs.get("03_calculate_correlations")
        prev_step_result = prev_task_result.data
        step = CalculateTraitEnrichmentStep()
        result = step.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_visualize_cross_platform(self, config, run_dir, logger, **kwargs):
        """Execute Step 5: Visualize Cross-Platform Correlations."""
        logger.info("Step 5/5: Generating correlation visualizations...")
        prev_task_result = kwargs.get("04_calculate_trait_enrichment")
        prev_step_result = prev_task_result.data
        step = VisualizeCrossPlatformStep()
        result = step.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_predict_cross_platform(self, config, run_dir, logger, **kwargs):
        """Execute Step 6: Predict Cross-Platform Genotype Values (optional).

        Reads data from task 1's result only. ``kwargs["05_visualize_cross_platform"]``
        is depended-upon solely to guarantee DAG ordering (steps 1-5 complete
        before prediction runs) -- its ``data`` is never read here (Decision 15).
        """
        logger.info("Step 6/6: Predicting cross-platform genotype values...")
        prev_task_result = kwargs["01_load_cross_platform_data"]
        prev_step_result = prev_task_result.data
        step = PredictCrossPlatformStep()
        result = step.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result
