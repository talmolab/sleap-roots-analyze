"""Visualization Pipeline orchestrator for comprehensive trait visualization.

This module provides the main VizPipeline class that orchestrates all 12
visualization steps in a NetworkX-based DAG for reproducible, automated
generation of publication-quality plots and interactive visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from sleap_roots_analyze.pipeline.pipelines.base_pipeline import BasePipeline
from sleap_roots_analyze.pipeline.task import Task
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.config import (
    VizPipelineConfig,
    validate_viz_config,
)
from sleap_roots_analyze.pipeline.steps import (
    LoadDataAndImagesStep,
    StatisticalAnalysisStep,
    PCAAnalysisStep,
    UMAPAnalysisStep,
    ClusterAnalysisStep,
    FilterHeritabilityStep,
    IdentifyInterestingGenotypesStep,
    GenotypeAggregationStep,
    GenerateStaticFiguresStep,
    GenerateInteractiveStep,
    GenerateDashboardsStep,
    GenerateSummaryVizStep,
)


class VizPipeline(BasePipeline):
    """Visualization pipeline for comprehensive trait visualization.

    This pipeline implements a 12-step visualization workflow:
    1. LoadDataAndImages - Load trait data and link images
    2. CalculateStatistics - Calculate ANOVA, heritability, descriptive stats
    3. PCAAnalysis - Perform PCA and identify top features
    4. UMAPAnalysis - Perform UMAP dimensionality reduction (optional)
    5. ClusterAnalysis - Perform clustering analysis (optional)
    6. HeritabilityAnalysis - Calculate and filter by heritability
    7. IdentifyInterestingGenotypes - Find extreme/heritable genotypes
    8. GenotypeAggregation - Aggregate data by genotype
    9. GenerateStaticFigures - Create publication-quality plots
    10. GenerateInteractive - Create interactive visualizations
    11. GenerateDashboards - Create summary dashboards (optional)
    12. GenerateSummary - Generate comprehensive summary report

    The pipeline uses NetworkX-based DAG execution for automatic step ordering
    and dependency management, with adaptive figure sizing.

    Args:
        config: VizPipelineConfig object with all configuration settings.
        output_dir: Directory for pipeline outputs. A timestamped subdirectory
            will be created for this run.
        validate: Whether to validate configuration before execution (default: True).

    Example:
        >>> from sleap_roots_analyze.pipeline import VizPipeline, load_viz_config
        >>> from omegaconf import OmegaConf
        >>> # Load and configure
        >>> omega_conf = OmegaConf.load("viz_standard.yaml")
        >>> OmegaConf.set_struct(omega_conf, False)
        >>> omega_conf.data.csv_path = "my_data.csv"
        >>> OmegaConf.set_struct(omega_conf, True)
        >>> # Convert to structured config
        >>> from sleap_roots_analyze.pipeline import VizPipelineConfig
        >>> structured = OmegaConf.structured(VizPipelineConfig)
        >>> merged = OmegaConf.merge(structured, omega_conf)
        >>> config = OmegaConf.to_object(merged)
        >>> # Run pipeline
        >>> pipeline = VizPipeline(config, output_dir="./viz_runs")
        >>> results = pipeline.run()
        >>> # Output in: viz_runs/viz_pipeline_YYYYMMDD_HHMMSS/
    """

    def __init__(
        self,
        config: VizPipelineConfig,
        output_dir: str | Path = "./viz_runs",
        validate: bool = True,
        log_filename: str | None = None,
    ):
        """Initialize the visualization pipeline.

        Args:
            config: VizPipelineConfig object.
            output_dir: Directory for pipeline outputs.
            validate: Whether to validate config before execution.
            log_filename: Custom log filename for run directory. If None, uses
                "pipeline.log".

        Raises:
            ValueError: If config validation fails.
        """
        # Validate configuration if requested
        if validate:
            validate_viz_config(config)

        # Initialize base pipeline
        super().__init__(
            config=config,
            output_dir=output_dir,
            pipeline_name=config.pipeline_name,
            version=config.version,
            log_filename=log_filename,
        )

        # Store config for use in create_tasks
        self.config: VizPipelineConfig = config

        # Initialize all step instances
        self.step_1_load_data_images = LoadDataAndImagesStep()
        self.step_2_calculate_statistics = StatisticalAnalysisStep()
        self.step_3_pca_analysis = PCAAnalysisStep()
        self.step_4_umap_analysis = UMAPAnalysisStep()
        self.step_5_cluster_analysis = ClusterAnalysisStep()
        self.step_6_heritability_analysis = FilterHeritabilityStep()
        self.step_7_identify_interesting_genotypes = IdentifyInterestingGenotypesStep()
        self.step_8_genotype_aggregation = GenotypeAggregationStep()
        self.step_9_generate_static_figures = GenerateStaticFiguresStep()
        self.step_10_generate_interactive = GenerateInteractiveStep()
        self.step_11_generate_dashboards = GenerateDashboardsStep()
        self.step_12_generate_summary = GenerateSummaryVizStep()

    def create_tasks(self) -> List[Task]:
        """Create the 12-step visualization pipeline task graph.

        This method creates a linear chain of 12 tasks with proper dependencies.
        The NetworkX DAG executor will ensure they execute in the correct order.

        Returns:
            List of Task objects representing the visualization pipeline steps.
        """
        tasks = []

        # Step 1: Load Data and Images (no dependencies - first step)
        tasks.append(
            Task(
                func=self._run_load_data_images,
                name="01_load_data_images",
                depends_on=[],
                description="Load trait data and link images",
            )
        )

        # Step 2: Calculate Statistics (depends on load_data_images)
        tasks.append(
            Task(
                func=self._run_calculate_statistics,
                name="02_calculate_statistics",
                depends_on=["01_load_data_images"],
                description="Calculate ANOVA, heritability, and descriptive statistics",
            )
        )

        # Step 3: PCA Analysis (depends on calculate_statistics)
        tasks.append(
            Task(
                func=self._run_pca_analysis,
                name="03_pca_analysis",
                depends_on=["02_calculate_statistics"],
                description="Perform PCA and identify top features",
            )
        )

        # Step 4: UMAP Analysis (depends on pca_analysis, optional)
        tasks.append(
            Task(
                func=self._run_umap_analysis,
                name="04_umap_analysis",
                depends_on=["03_pca_analysis"],
                description="Perform UMAP dimensionality reduction (optional)",
            )
        )

        # Step 5: Cluster Analysis (depends on pca_analysis, optional)
        tasks.append(
            Task(
                func=self._run_cluster_analysis,
                name="05_cluster_analysis",
                depends_on=["03_pca_analysis"],
                description="Perform clustering analysis (optional)",
            )
        )

        # Step 6: Heritability Analysis (depends on calculate_statistics)
        tasks.append(
            Task(
                func=self._run_heritability_analysis,
                name="06_heritability_analysis",
                depends_on=["02_calculate_statistics"],
                description="Analyze heritability and filter traits (optional)",
            )
        )

        # Step 7: Identify Interesting Genotypes
        # Depends on: pca_analysis, calculate_statistics
        tasks.append(
            Task(
                func=self._run_identify_interesting_genotypes,
                name="07_identify_interesting_genotypes",
                depends_on=["03_pca_analysis", "02_calculate_statistics"],
                description="Identify extreme and heritable genotypes",
            )
        )

        # Step 8: Genotype Aggregation
        # Depends on all analysis steps
        tasks.append(
            Task(
                func=self._run_genotype_aggregation,
                name="08_genotype_aggregation",
                depends_on=[
                    "02_calculate_statistics",
                    "03_pca_analysis",
                    "06_heritability_analysis",
                ],
                description="Aggregate data by genotype for comparisons",
            )
        )

        # Step 9: Generate Static Figures
        # Depends on all analysis steps
        tasks.append(
            Task(
                func=self._run_generate_static_figures,
                name="09_generate_static_figures",
                depends_on=[
                    "03_pca_analysis",
                    "04_umap_analysis",
                    "05_cluster_analysis",
                    "06_heritability_analysis",
                    "07_identify_interesting_genotypes",
                    "08_genotype_aggregation",
                ],
                description="Generate publication-quality static figures",
            )
        )

        # Step 10: Generate Interactive Visualizations
        # Depends on all analysis steps
        tasks.append(
            Task(
                func=self._run_generate_interactive,
                name="10_generate_interactive",
                depends_on=[
                    "01_load_data_images",
                    "03_pca_analysis",
                    "04_umap_analysis",
                    "05_cluster_analysis",
                ],
                description="Generate interactive visualizations with image hover",
            )
        )

        # Step 11: Generate Dashboards (optional)
        # Depends on all visualization generation
        tasks.append(
            Task(
                func=self._run_generate_dashboards,
                name="11_generate_dashboards",
                depends_on=[
                    "09_generate_static_figures",
                    "10_generate_interactive",
                ],
                description="Generate summary dashboards (optional)",
            )
        )

        # Step 12: Generate Summary Report
        # Depends on all previous steps
        tasks.append(
            Task(
                func=self._run_generate_summary,
                name="12_generate_summary",
                depends_on=[
                    "02_calculate_statistics",
                    "03_pca_analysis",
                    "06_heritability_analysis",
                    "07_identify_interesting_genotypes",
                    "09_generate_static_figures",
                    "10_generate_interactive",
                    "11_generate_dashboards",
                ],
                description="Generate comprehensive pipeline summary",
            )
        )

        return tasks

    # Task wrapper methods - these adapt steps to the Task interface
    # Each method receives dependency results as kwargs with the dependency task name

    def _run_load_data_images(self, config, run_dir, logger):
        """Execute Step 1: Load Data and Images."""
        logger.info("Step 1/12: Loading data and linking images...")
        result = self.step_1_load_data_images.execute(
            data=None, config=config, run_dir=run_dir, prev_result=None
        )
        return result

    def _run_calculate_statistics(self, config, run_dir, logger, **kwargs):
        """Execute Step 2: Calculate Statistics."""
        logger.info("Step 2/12: Calculating statistics...")
        prev_task_result = kwargs.get("01_load_data_images")
        prev_step_result = prev_task_result.data
        result = self.step_2_calculate_statistics.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_pca_analysis(self, config, run_dir, logger, **kwargs):
        """Execute Step 3: PCA Analysis."""
        logger.info("Step 3/12: Performing PCA analysis...")
        prev_task_result = kwargs.get("02_calculate_statistics")
        prev_step_result = prev_task_result.data
        result = self.step_3_pca_analysis.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_umap_analysis(self, config, run_dir, logger, **kwargs):
        """Execute Step 4: UMAP Analysis."""
        logger.info("Step 4/12: Performing UMAP analysis...")
        prev_task_result = kwargs.get("03_pca_analysis")
        prev_step_result = prev_task_result.data
        result = self.step_4_umap_analysis.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_cluster_analysis(self, config, run_dir, logger, **kwargs):
        """Execute Step 5: Cluster Analysis."""
        logger.info("Step 5/12: Performing cluster analysis...")
        prev_task_result = kwargs.get("03_pca_analysis")
        prev_step_result = prev_task_result.data
        result = self.step_5_cluster_analysis.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_heritability_analysis(self, config, run_dir, logger, **kwargs):
        """Execute Step 6: Heritability Analysis."""
        logger.info("Step 6/12: Analyzing heritability...")
        prev_task_result = kwargs.get("02_calculate_statistics")
        prev_step_result = prev_task_result.data
        result = self.step_6_heritability_analysis.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_identify_interesting_genotypes(self, config, run_dir, logger, **kwargs):
        """Execute Step 7: Identify Interesting Genotypes."""
        logger.info("Step 7/12: Identifying interesting genotypes...")
        # Get both PCA and statistics results
        pca_result = kwargs.get("03_pca_analysis").data
        stats_result = kwargs.get("02_calculate_statistics").data

        # Merge PCA and stats metadata for step (needs heritability & ANOVA when implemented)
        combined_result = StepResult(
            data=pca_result.data,
            metadata={
                **pca_result.metadata,
                "heritability_results": stats_result.metadata.get(
                    "heritability_results"
                ),
                "anova_results": stats_result.metadata.get("anova_results"),
                "trait_statistics": stats_result.metadata.get("trait_statistics"),
            },
            files_generated=pca_result.files_generated,
        )

        result = self.step_7_identify_interesting_genotypes.execute(
            data=combined_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=combined_result,
        )
        return result

    def _run_genotype_aggregation(self, config, run_dir, logger, **kwargs):
        """Execute Step 8: Genotype Aggregation."""
        logger.info("Step 8/12: Aggregating genotype data...")
        prev_task_result = kwargs.get("06_heritability_analysis")
        prev_step_result = prev_task_result.data
        result = self.step_8_genotype_aggregation.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_generate_static_figures(self, config, run_dir, logger, **kwargs):
        """Execute Step 9: Generate Static Figures."""
        logger.info("Step 9/12: Generating static figures...")
        # Primary input from genotype aggregation
        prev_task_result = kwargs.get("08_genotype_aggregation")
        prev_step_result = prev_task_result.data

        # CRITICAL FIX: Merge PCA and UMAP results into metadata
        # The static figures step needs PCA/UMAP results, but they're on a different
        # branch of the DAG (PCA → interesting_genotypes vs statistics → heritability → aggregation)
        # So we explicitly grab PCA and UMAP results from kwargs and merge them in
        combined_metadata = {**prev_step_result.metadata}

        # Merge PCA results
        pca_task_result = kwargs.get("03_pca_analysis")
        if pca_task_result:
            pca_step_result = pca_task_result.data
            combined_metadata.update(
                {
                    "pca_results": pca_step_result.metadata.get("pca_results"),
                    "top_features": pca_step_result.metadata.get("top_features"),
                    "n_pca_components": pca_step_result.metadata.get(
                        "n_pca_components"
                    ),
                    "pca_explained_variance": pca_step_result.metadata.get(
                        "pca_explained_variance"
                    ),
                    # Issue #80: trait_names/original_trait_names must come
                    # from the PCA branch too, not just pca_results/top_features
                    # above — otherwise this step keeps the pre-PCA trait list
                    # relayed from the heritability/aggregation branch.
                    "trait_names": pca_step_result.metadata.get("trait_names"),
                    "original_trait_names": pca_step_result.metadata.get(
                        "original_trait_names"
                    ),
                }
            )

        # Merge UMAP results
        umap_task_result = kwargs.get("04_umap_analysis")
        if umap_task_result:
            umap_step_result = umap_task_result.data
            umap_results = umap_step_result.metadata.get("umap_results")
            if umap_results:
                combined_metadata["umap_results"] = umap_results

        # Create combined result with merged metadata
        combined_result = StepResult(
            data=prev_step_result.data,
            metadata=combined_metadata,
            files_generated=prev_step_result.files_generated,
        )

        result = self.step_9_generate_static_figures.execute(
            data=combined_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=combined_result,
        )
        return result

    def _run_generate_interactive(self, config, run_dir, logger, **kwargs):
        """Execute Step 10: Generate Interactive Visualizations."""
        logger.info("Step 10/12: Generating interactive visualizations...")
        # Get results from UMAP analysis (has most complete metadata including
        # PCA results, image_paths, and umap_results)
        umap_task_result = kwargs.get("04_umap_analysis")
        if umap_task_result and umap_task_result.data:
            prev_step_result = umap_task_result.data
        else:
            # Fallback to PCA if UMAP didn't run
            pca_task_result = kwargs.get("03_pca_analysis")
            prev_step_result = pca_task_result.data
        result = self.step_10_generate_interactive.execute(
            data=prev_step_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=prev_step_result,
        )
        return result

    def _run_generate_dashboards(self, config, run_dir, logger, **kwargs):
        """Execute Step 11: Generate Dashboards."""
        logger.info("Step 11/12: Generating dashboards...")
        # Combine static and interactive results
        static_result = kwargs.get("09_generate_static_figures").data
        result = self.step_11_generate_dashboards.execute(
            data=static_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=static_result,
        )
        return result

    def _run_generate_summary(self, config, run_dir, logger, **kwargs):
        """Execute Step 12: Generate Summary Report."""
        logger.info("Step 12/12: Generating summary report...")
        # Collect all important results for summary
        stats_result = kwargs.get("02_calculate_statistics").data
        result = self.step_12_generate_summary.execute(
            data=stats_result.data,
            config=config,
            run_dir=run_dir,
            prev_result=stats_result,
        )
        return result
