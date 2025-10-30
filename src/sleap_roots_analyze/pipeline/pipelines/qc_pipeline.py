"""QC Pipeline orchestrator for trait quality control workflow.

This module provides the main QCPipeline class that orchestrates all 10 QC steps
in a NetworkX-based DAG for reproducible, automated quality control of trait data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from sleap_roots_analyze.pipeline.config import QCPipelineConfig, validate_qc_config
from sleap_roots_analyze.pipeline.pipelines.base_pipeline import BasePipeline
from sleap_roots_analyze.pipeline.task import Task
from sleap_roots_analyze.pipeline.steps.load_data import LoadDataStep
from sleap_roots_analyze.pipeline.steps.cleanup_traits import CleanupTraitsStep
from sleap_roots_analyze.pipeline.steps.validate_clean import ValidateCleanStep
from sleap_roots_analyze.pipeline.steps.exploratory_analysis import (
    ExploratoryAnalysisStep,
)
from sleap_roots_analyze.pipeline.steps.detect_outliers import DetectOutliersStep
from sleap_roots_analyze.pipeline.steps.visualize_outliers import VisualizeOutliersStep
from sleap_roots_analyze.pipeline.steps.remove_outliers import RemoveOutliersStep
from sleap_roots_analyze.pipeline.steps.statistical_analysis import (
    StatisticalAnalysisStep,
)
from sleap_roots_analyze.pipeline.steps.filter_heritability import (
    FilterHeritabilityStep,
)
from sleap_roots_analyze.pipeline.steps.generate_summary import GenerateSummaryStep


class QCPipeline(BasePipeline):
    """Quality Control pipeline for trait data.

    This pipeline implements a 10-step quality control workflow:
    1. LoadData - Load and validate CSV data
    2. CleanupTraits - Remove problematic traits (high zeros/NaNs)
    3. ValidateClean - Validate cleaned data
    4. ExploratoryAnalysis - Generate EDA visualizations
    5. DetectOutliers - Run configured outlier detection methods
    6. VisualizeOutliers - Visualize outlier detection results
    7. RemoveOutliers - Remove outliers using configured strategy
    8. StatisticalAnalysis - Calculate ANOVA and heritability
    9. FilterHeritability - Filter low heritability traits (optional)
    10. GenerateSummary - Generate comprehensive pipeline summary

    The pipeline uses NetworkX-based DAG execution for automatic step ordering
    and dependency management.

    Args:
        config: QCPipelineConfig object with all configuration settings.
        output_dir: Directory for pipeline outputs. A timestamped subdirectory
            will be created for this run.
        validate: Whether to validate configuration before execution (default: True).

    Example:
        >>> from sleap_roots_analyze.pipeline import QCPipeline, load_config
        >>> config = load_config("qc_mahalanobis.yaml")
        >>> config.data.csv_path = "my_data.csv"
        >>> pipeline = QCPipeline(config, output_dir="./runs")
        >>> results = pipeline.run()
        >>> # Output in: runs/qc_pipeline_YYYYMMDD_HHMMSS/
    """

    def __init__(
        self,
        config: QCPipelineConfig,
        output_dir: str | Path = "./qc_runs",
        validate: bool = True,
    ):
        """Initialize the QC pipeline.

        Args:
            config: QCPipelineConfig object.
            output_dir: Directory for pipeline outputs.
            validate: Whether to validate config before execution.

        Raises:
            ValueError: If config validation fails.
        """
        # Validate configuration if requested
        if validate:
            validate_qc_config(config)

        # Initialize base pipeline
        super().__init__(
            config=config,
            output_dir=output_dir,
            pipeline_name=config.pipeline_name,
            version=config.version,
        )

        # Store config for use in create_tasks
        self.config: PipelineConfig = config

        # Initialize all step instances
        self.step_1_load_data = LoadDataStep()
        self.step_2_cleanup_traits = CleanupTraitsStep()
        self.step_3_validate_clean = ValidateCleanStep()
        self.step_4_exploratory_analysis = ExploratoryAnalysisStep()
        self.step_5_detect_outliers = DetectOutliersStep()
        self.step_6_visualize_outliers = VisualizeOutliersStep()
        self.step_7_remove_outliers = RemoveOutliersStep()
        self.step_8_statistical_analysis = StatisticalAnalysisStep()
        self.step_9_filter_heritability = FilterHeritabilityStep()
        self.step_10_generate_summary = GenerateSummaryStep()

    def create_tasks(self) -> List[Task]:
        """Create the 10-step QC pipeline task graph.

        This method creates a linear chain of 10 tasks with proper dependencies.
        The NetworkX DAG executor will ensure they execute in the correct order.

        Returns:
            List of Task objects representing the QC pipeline steps.
        """
        tasks = []

        # Step 1: Load Data (no dependencies - first step)
        tasks.append(
            Task(
                func=self._run_load_data,
                name="01_load_data",
                depends_on=[],
                description="Load and validate trait data from CSV",
            )
        )

        # Step 2: Cleanup Traits (depends on load_data)
        tasks.append(
            Task(
                func=self._run_cleanup_traits,
                name="02_cleanup_traits",
                depends_on=["01_load_data"],
                description="Remove problematic traits (high zeros/NaNs)",
            )
        )

        # Step 3: Validate Clean (depends on cleanup_traits)
        tasks.append(
            Task(
                func=self._run_validate_clean,
                name="03_validate_clean",
                depends_on=["02_cleanup_traits"],
                description="Validate cleaned data",
            )
        )

        # Step 4: Exploratory Analysis (depends on validate_clean)
        tasks.append(
            Task(
                func=self._run_exploratory_analysis,
                name="04_exploratory_analysis",
                depends_on=["03_validate_clean"],
                description="Generate exploratory data analysis visualizations",
            )
        )

        # Step 5: Detect Outliers (depends on exploratory_analysis)
        tasks.append(
            Task(
                func=self._run_detect_outliers,
                name="05_detect_outliers",
                depends_on=["04_exploratory_analysis"],
                description="Run configured outlier detection methods",
            )
        )

        # Step 6: Visualize Outliers (depends on detect_outliers)
        tasks.append(
            Task(
                func=self._run_visualize_outliers,
                name="06_visualize_outliers",
                depends_on=["05_detect_outliers"],
                description="Visualize outlier detection results",
            )
        )

        # Step 7: Remove Outliers (depends on visualize_outliers)
        tasks.append(
            Task(
                func=self._run_remove_outliers,
                name="07_remove_outliers",
                depends_on=["06_visualize_outliers"],
                description="Remove outliers using configured strategy",
            )
        )

        # Step 8: Statistical Analysis (depends on remove_outliers)
        tasks.append(
            Task(
                func=self._run_statistical_analysis,
                name="08_statistical_analysis",
                depends_on=["07_remove_outliers"],
                description="Calculate ANOVA and heritability statistics",
            )
        )

        # Step 9: Filter Heritability (depends on statistical_analysis)
        tasks.append(
            Task(
                func=self._run_filter_heritability,
                name="09_filter_heritability",
                depends_on=["08_statistical_analysis"],
                description="Filter low heritability traits (if enabled)",
            )
        )

        # Step 10: Generate Summary (depends on filter_heritability)
        tasks.append(
            Task(
                func=self._run_generate_summary,
                name="10_generate_summary",
                depends_on=["09_filter_heritability"],
                description="Generate comprehensive pipeline summary",
            )
        )

        return tasks

    # Task wrapper methods - these adapt steps to the Task interface
    # Each method receives dependency results as kwargs with the dependency task name

    def _run_load_data(self, config, run_dir, logger):
        """Execute Step 1: Load Data."""
        logger.info("Step 1/10: Loading data...")
        result = self.step_1_load_data.execute(
            data=None, config=config, run_dir=run_dir, prev_result=None
        )
        return result

    def _run_cleanup_traits(self, config, run_dir, logger, **kwargs):
        """Execute Step 2: Cleanup Traits."""
        logger.info("Step 2/10: Cleaning up traits...")
        prev_task_result = kwargs.get("01_load_data")
        # TaskResult.data contains the StepResult
        prev_step_result = prev_task_result.data
        result = self.step_2_cleanup_traits.execute(
            data=prev_step_result.data,  # DataFrame
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,  # StepResult
        )
        return result

    def _run_validate_clean(self, config, run_dir, logger, **kwargs):
        """Execute Step 3: Validate Clean."""
        logger.info("Step 3/10: Validating cleaned data...")
        prev_task_result = kwargs.get("02_cleanup_traits")
        # TaskResult.data contains the StepResult
        prev_step_result = prev_task_result.data
        result = self.step_3_validate_clean.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_exploratory_analysis(self, config, run_dir, logger, **kwargs):
        """Execute Step 4: Exploratory Analysis."""
        logger.info("Step 4/10: Generating exploratory analysis...")
        prev_task_result = kwargs.get("03_validate_clean")
        # TaskResult.data contains the StepResult
        prev_step_result = prev_task_result.data
        result = self.step_4_exploratory_analysis.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_detect_outliers(self, config, run_dir, logger, **kwargs):
        """Execute Step 5: Detect Outliers."""
        logger.info("Step 5/10: Detecting outliers...")
        prev_task_result = kwargs.get("04_exploratory_analysis")
        prev_step_result = prev_task_result.data
        result = self.step_5_detect_outliers.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_visualize_outliers(self, config, run_dir, logger, **kwargs):
        """Execute Step 6: Visualize Outliers."""
        logger.info("Step 6/10: Visualizing outliers...")
        prev_task_result = kwargs.get("05_detect_outliers")
        # TaskResult.data contains the StepResult
        prev_step_result = prev_task_result.data
        result = self.step_6_visualize_outliers.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_remove_outliers(self, config, run_dir, logger, **kwargs):
        """Execute Step 7: Remove Outliers."""
        logger.info("Step 7/10: Removing outliers...")
        prev_task_result = kwargs.get("06_visualize_outliers")
        # TaskResult.data contains the StepResult
        prev_step_result = prev_task_result.data
        result = self.step_7_remove_outliers.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_statistical_analysis(self, config, run_dir, logger, **kwargs):
        """Execute Step 8: Statistical Analysis."""
        logger.info("Step 8/10: Running statistical analysis...")
        prev_task_result = kwargs.get("07_remove_outliers")
        # TaskResult.data contains the StepResult
        prev_step_result = prev_task_result.data
        result = self.step_8_statistical_analysis.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_filter_heritability(self, config, run_dir, logger, **kwargs):
        """Execute Step 9: Filter Heritability."""
        logger.info("Step 9/10: Filtering by heritability...")
        prev_task_result = kwargs.get("08_statistical_analysis")
        # TaskResult.data contains the StepResult
        prev_step_result = prev_task_result.data
        result = self.step_9_filter_heritability.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_generate_summary(self, config, run_dir, logger, **kwargs):
        """Execute Step 10: Generate Summary."""
        logger.info("Step 10/10: Generating pipeline summary...")
        prev_task_result = kwargs.get("09_filter_heritability")
        # TaskResult.data contains the StepResult
        prev_step_result = prev_task_result.data
        result = self.step_10_generate_summary.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result
