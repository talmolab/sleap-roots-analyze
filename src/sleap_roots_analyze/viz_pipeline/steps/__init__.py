"""Visualization pipeline steps.

This module contains all step implementations for the visualization pipeline.
"""

from __future__ import annotations

from sleap_roots_analyze.viz_pipeline.steps.calculate_statistics import (
    CalculateStatisticsStep,
)
from sleap_roots_analyze.viz_pipeline.steps.cluster_analysis import ClusterAnalysisStep
from sleap_roots_analyze.viz_pipeline.steps.generate_dashboards import (
    GenerateDashboardsStep,
)
from sleap_roots_analyze.viz_pipeline.steps.generate_interactive import (
    GenerateInteractiveStep,
)
from sleap_roots_analyze.viz_pipeline.steps.generate_static_figures import (
    GenerateStaticFiguresStep,
)
from sleap_roots_analyze.viz_pipeline.steps.generate_summary import GenerateSummaryStep
from sleap_roots_analyze.viz_pipeline.steps.genotype_aggregation import (
    GenotypeAggregationStep,
)
from sleap_roots_analyze.viz_pipeline.steps.heritability_analysis import (
    HeritabilityAnalysisStep,
)
from sleap_roots_analyze.viz_pipeline.steps.identify_interesting_genotypes import (
    IdentifyInterestingGenotypesStep,
)
from sleap_roots_analyze.viz_pipeline.steps.load_data_images import (
    LoadDataAndImagesStep,
)
from sleap_roots_analyze.viz_pipeline.steps.pca_analysis import PCAAnalysisStep
from sleap_roots_analyze.viz_pipeline.steps.umap_analysis import UMAPAnalysisStep

__all__ = [
    "CalculateStatisticsStep",
    "ClusterAnalysisStep",
    "GenerateDashboardsStep",
    "GenerateInteractiveStep",
    "GenerateStaticFiguresStep",
    "GenerateSummaryStep",
    "GenotypeAggregationStep",
    "HeritabilityAnalysisStep",
    "IdentifyInterestingGenotypesStep",
    "LoadDataAndImagesStep",
    "PCAAnalysisStep",
    "UMAPAnalysisStep",
]
