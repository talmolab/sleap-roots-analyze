"""Pipeline steps - all reusable step implementations.

This module contains all step implementations used by both QC and Viz pipelines.
Steps are organized alphabetically and exported for easy access.
"""

from __future__ import annotations

# Root Core Processing Steps (Steps 0a-0f)
from sleap_roots_analyze.pipeline.steps.aggregate_cores import AggregateCoresStep
from sleap_roots_analyze.pipeline.steps.load_root_core_data import LoadRootCoreDataStep
from sleap_roots_analyze.pipeline.steps.qc_core_level import QCCoreLevelStep
from sleap_roots_analyze.pipeline.steps.reshape_for_trait_qc import (
    ReshapeForTraitQCStep,
)
from sleap_roots_analyze.pipeline.steps.transform_depth_data import (
    TransformDepthDataStep,
)
from sleap_roots_analyze.pipeline.steps.visualize_depth_profiles import (
    VisualizeDepthProfilesStep,
)

# QC Pipeline Steps
from sleap_roots_analyze.pipeline.steps.cleanup_traits import CleanupTraitsStep
from sleap_roots_analyze.pipeline.steps.detect_outliers import DetectOutliersStep
from sleap_roots_analyze.pipeline.steps.exploratory_analysis import (
    ExploratoryAnalysisStep,
)
from sleap_roots_analyze.pipeline.steps.filter_heritability import (
    FilterHeritabilityStep,
)
from sleap_roots_analyze.pipeline.steps.generate_summary import GenerateSummaryStep
from sleap_roots_analyze.pipeline.steps.load_data import LoadDataStep
from sleap_roots_analyze.pipeline.steps.remove_outliers import RemoveOutliersStep
from sleap_roots_analyze.pipeline.steps.statistical_analysis import (
    StatisticalAnalysisStep,
)
from sleap_roots_analyze.pipeline.steps.validate_clean import ValidateCleanStep
from sleap_roots_analyze.pipeline.steps.visualize_outliers import VisualizeOutliersStep

# Viz Pipeline Steps
from sleap_roots_analyze.pipeline.steps.cluster_analysis import ClusterAnalysisStep
from sleap_roots_analyze.pipeline.steps.generate_dashboards import (
    GenerateDashboardsStep,
)
from sleap_roots_analyze.pipeline.steps.generate_interactive import (
    GenerateInteractiveStep,
)
from sleap_roots_analyze.pipeline.steps.generate_static_figures import (
    GenerateStaticFiguresStep,
)
from sleap_roots_analyze.pipeline.steps.generate_summary_viz import (
    GenerateSummaryStep as GenerateSummaryVizStep,
)
from sleap_roots_analyze.pipeline.steps.genotype_aggregation import (
    GenotypeAggregationStep,
)
from sleap_roots_analyze.pipeline.steps.identify_interesting_genotypes import (
    IdentifyInterestingGenotypesStep,
)
from sleap_roots_analyze.pipeline.steps.load_data_images import LoadDataAndImagesStep
from sleap_roots_analyze.pipeline.steps.pca_analysis import PCAAnalysisStep
from sleap_roots_analyze.pipeline.steps.umap_analysis import UMAPAnalysisStep

# Cross-Platform Analysis Steps
from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
    CalculateCrossPlatformCorrelationsStep,
)
from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
    LoadCrossPlatformDataStep,
)
from sleap_roots_analyze.pipeline.steps.visualize_cross_platform import (
    VisualizeCrossPlatformStep,
)

__all__ = [
    # Root Core Steps
    "AggregateCoresStep",
    "LoadRootCoreDataStep",
    "QCCoreLevelStep",
    "ReshapeForTraitQCStep",
    "TransformDepthDataStep",
    "VisualizeDepthProfilesStep",
    # QC Steps
    "CleanupTraitsStep",
    "DetectOutliersStep",
    "ExploratoryAnalysisStep",
    "FilterHeritabilityStep",
    "GenerateSummaryStep",
    "LoadDataStep",
    "RemoveOutliersStep",
    "StatisticalAnalysisStep",
    "ValidateCleanStep",
    "VisualizeOutliersStep",
    # Viz Steps
    "ClusterAnalysisStep",
    "GenerateDashboardsStep",
    "GenerateInteractiveStep",
    "GenerateStaticFiguresStep",
    "GenerateSummaryVizStep",
    "GenotypeAggregationStep",
    "IdentifyInterestingGenotypesStep",
    "LoadDataAndImagesStep",
    "PCAAnalysisStep",
    "UMAPAnalysisStep",
    # Cross-Platform Steps
    "CalculateCrossPlatformCorrelationsStep",
    "LoadCrossPlatformDataStep",
    "VisualizeCrossPlatformStep",
]
