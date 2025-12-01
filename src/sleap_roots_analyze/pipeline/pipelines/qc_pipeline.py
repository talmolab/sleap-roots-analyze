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
from sleap_roots_analyze.pipeline.steps.load_root_core_data import LoadRootCoreDataStep
from sleap_roots_analyze.pipeline.steps.transform_depth_data import (
    TransformDepthDataStep,
)
from sleap_roots_analyze.pipeline.steps.qc_core_level import QCCoreLevelStep
from sleap_roots_analyze.pipeline.steps.aggregate_cores import AggregateCoresStep
from sleap_roots_analyze.pipeline.steps.reshape_for_trait_qc import (
    ReshapeForTraitQCStep,
)
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

    This pipeline implements a quality control workflow with optional root core
    processing (Steps 0a-0e) followed by standard trait-level QC (Steps 1-10).

    **Root Core Processing (Optional - if config.root_core is set):**
    0a. LoadRootCoreData - Load biomass/counting data for all sources
    0b. TransformDepthData - Calculate depth_cm and pivot to long format
    0c. QCCoreLevel - Detect and optionally remove outlier cores (optional)
    0d. AggregateCores - Aggregate 3 cores to biological replicate level
    0e. ReshapeForTraitQC - Pivot to wide format with prefixed columns

    **Standard Trait-Level QC:**
    1. LoadData - Load and validate CSV data (or use root core output)
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
        self.config: QCPipelineConfig = config

        # Initialize root core processing step instances (optional)
        self.step_0a_load_root_core = LoadRootCoreDataStep()
        self.step_0b_transform_depth = TransformDepthDataStep()
        self.step_0c_qc_core_level = QCCoreLevelStep()
        self.step_0d_aggregate_cores = AggregateCoresStep()
        self.step_0e_reshape_for_qc = ReshapeForTraitQCStep()

        # Initialize all standard QC step instances
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
        """Create the QC pipeline task graph (with optional root core steps).

        If config.root_core is set, creates Steps 0a-0e before standard QC steps.
        The NetworkX DAG executor will ensure they execute in the correct order.

        Returns:
            List of Task objects representing the QC pipeline steps.
        """
        tasks = []

        # Check if root core processing is configured
        if self.config.root_core is not None:
            # Step 0a: Load Root Core Data (first step if root core enabled)
            tasks.append(
                Task(
                    func=self._run_load_root_core_data,
                    name="00a_load_root_core_data",
                    depends_on=[],
                    description="Load root core biomass/counting data from all sources",
                )
            )

            # Step 0b: Transform Depth Data
            tasks.append(
                Task(
                    func=self._run_transform_depth_data,
                    name="00b_transform_depth_data",
                    depends_on=["00a_load_root_core_data"],
                    description="Calculate depth_cm and pivot to long format",
                )
            )

            # Step 0c: QC Core Level (optional outlier detection)
            tasks.append(
                Task(
                    func=self._run_qc_core_level,
                    name="00c_qc_core_level",
                    depends_on=["00b_transform_depth_data"],
                    description="Detect and optionally remove outlier cores",
                )
            )

            # Step 0d: Aggregate Cores
            tasks.append(
                Task(
                    func=self._run_aggregate_cores,
                    name="00d_aggregate_cores",
                    depends_on=["00c_qc_core_level"],
                    description="Aggregate 3 cores to biological replicate level",
                )
            )

            # Step 0e: Reshape For Trait QC
            tasks.append(
                Task(
                    func=self._run_reshape_for_trait_qc,
                    name="00e_reshape_for_trait_qc",
                    depends_on=["00d_aggregate_cores"],
                    description="Pivot to wide format with prefixed columns",
                )
            )

            # Step 1 now depends on Step 0e if root core enabled
            first_step_dependency = ["00e_reshape_for_trait_qc"]
        else:
            # No root core processing - Step 1 has no dependencies
            first_step_dependency = []

        # Step 1: Load Data
        tasks.append(
            Task(
                func=self._run_load_data,
                name="01_load_data",
                depends_on=first_step_dependency,
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

    # Root Core Processing Wrapper Methods (Steps 0a-0e)

    def _run_load_root_core_data(self, config, run_dir, logger):
        """Execute Step 0a: Load Root Core Data."""
        logger.info("Step 0a: Loading root core data...")
        result = self.step_0a_load_root_core.execute(
            data=None, config=config, run_dir=run_dir, prev_result=None
        )
        return result

    def _run_transform_depth_data(self, config, run_dir, logger, **kwargs):
        """Execute Step 0b: Transform Depth Data."""
        logger.info("Step 0b: Transforming depth data...")
        prev_task_result = kwargs.get("00a_load_root_core_data")
        prev_step_result = prev_task_result.data
        result = self.step_0b_transform_depth.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_qc_core_level(self, config, run_dir, logger, **kwargs):
        """Execute Step 0c: QC Core Level."""
        logger.info("Step 0c: Running core-level QC...")
        prev_task_result = kwargs.get("00b_transform_depth_data")
        prev_step_result = prev_task_result.data
        result = self.step_0c_qc_core_level.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_aggregate_cores(self, config, run_dir, logger, **kwargs):
        """Execute Step 0d: Aggregate Cores."""
        logger.info("Step 0d: Aggregating cores to replicate level...")
        prev_task_result = kwargs.get("00c_qc_core_level")
        prev_step_result = prev_task_result.data
        result = self.step_0d_aggregate_cores.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_reshape_for_trait_qc(self, config, run_dir, logger, **kwargs):
        """Execute Step 0e: Reshape For Trait QC."""
        logger.info("Step 0e: Reshaping data for trait-level QC...")
        prev_task_result = kwargs.get("00d_aggregate_cores")
        prev_step_result = prev_task_result.data
        result = self.step_0e_reshape_for_qc.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    # Standard QC Pipeline Wrapper Methods (Steps 1-10)

    def _run_load_data(self, config, run_dir, logger, **kwargs):
        """Execute Step 1: Load Data (or use root core output)."""
        # Check if root core processing provided data
        if "00e_reshape_for_trait_qc" in kwargs:
            logger.info("Step 1/10: Using root core processed data...")
            prev_task_result = kwargs.get("00e_reshape_for_trait_qc")
            prev_step_result = prev_task_result.data
            # Pass through the reshaped root core data
            result = self.step_1_load_data.execute(
                data=prev_step_result.data,
                config=config,
                run_dir=run_dir,
                prev_result=prev_step_result,
            )
        else:
            logger.info("Step 1/10: Loading data from CSV...")
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
