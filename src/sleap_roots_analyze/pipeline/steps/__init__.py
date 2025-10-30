"""Pipeline steps - all reusable step implementations.

This module contains all step implementations used by both QC and Viz pipelines.
Steps are organized alphabetically and exported for easy access.
"""

from __future__ import annotations

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

__all__ = [
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
]
