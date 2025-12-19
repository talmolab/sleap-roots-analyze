"""Cross-Platform Pipeline orchestrator for comparing traits across experiments.

This module provides the CrossPlatformPipeline class that orchestrates the 3 cross-platform
analysis steps in a NetworkX-based DAG for reproducible, automated cross-experiment analysis.
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
from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
    CalculateCrossPlatformCorrelationsStep,
)
from sleap_roots_analyze.pipeline.steps.visualize_cross_platform import (
    VisualizeCrossPlatformStep,
)


class CrossPlatformPipeline(BasePipeline):
    """Cross-platform analysis pipeline for comparing traits across experiments.

    This pipeline performs:
    1. Load and align data from two experiments
    2. Calculate correlations between all trait pairs
    3. Generate visualizations of correlations

    The pipeline is designed to compare traits across different experimental
    platforms or conditions to identify which traits capture similar biological
    variation patterns.
    """

    def __init__(
        self,
        config: CrossPlatformConfig,
        output_dir: Path | str = "./cross_platform_runs",
    ):
        """Initialize CrossPlatformPipeline.

        Args:
            config: CrossPlatformConfig with experiment paths and analysis parameters
            output_dir: Base directory for output (default: "./cross_platform_runs")
        """
        # Sanitize experiment names for safe folder naming
        exp1_safe = self._sanitize_folder_name(config.exp1_name)
        exp2_safe = self._sanitize_folder_name(config.exp2_name)

        super().__init__(
            config=config,
            output_dir=Path(output_dir),
            pipeline_name=f"cross_platform_{exp1_safe}_vs_{exp2_safe}",
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
        """Create the cross-platform analysis task graph with 3 steps.

        Returns:
            List of Tasks defining the pipeline DAG:
                - Step 1: LoadCrossPlatformData
                - Step 2: CalculateCrossPlatformCorrelations
                - Step 3: VisualizeCrossPlatform
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

        # Step 2: Calculate correlations between all trait pairs
        tasks.append(
            Task(
                func=self._run_calculate_correlations,
                name="02_calculate_correlations",
                depends_on=["01_load_cross_platform_data"],
                description=f"Calculate correlations using {self.config.correlation_method}",
            )
        )

        # Step 3: Visualize correlations
        tasks.append(
            Task(
                func=self._run_visualize_cross_platform,
                name="03_visualize_cross_platform",
                depends_on=["02_calculate_correlations"],
                description="Generate correlation visualizations",
            )
        )

        return tasks

    def _run_load_cross_platform_data(self, config, run_dir, logger, **kwargs):
        """Execute Step 1: Load Cross-Platform Data."""
        logger.info("Step 1/3: Loading and aligning cross-platform data...")
        step = LoadCrossPlatformDataStep()
        result = step.execute(
            data=None, config=config, run_dir=run_dir, prev_result=None
        )
        return result

    def _run_calculate_correlations(self, config, run_dir, logger, **kwargs):
        """Execute Step 2: Calculate Cross-Platform Correlations."""
        logger.info("Step 2/3: Calculating correlations between all trait pairs...")
        prev_task_result = kwargs.get("01_load_cross_platform_data")
        prev_step_result = prev_task_result.data
        step = CalculateCrossPlatformCorrelationsStep()
        result = step.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_visualize_cross_platform(self, config, run_dir, logger, **kwargs):
        """Execute Step 3: Visualize Cross-Platform Correlations."""
        logger.info("Step 3/3: Generating correlation visualizations...")
        prev_task_result = kwargs.get("02_calculate_correlations")
        prev_step_result = prev_task_result.data
        step = VisualizeCrossPlatformStep()
        result = step.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result
