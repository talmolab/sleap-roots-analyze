"""Pipeline configuration - components, pipeline configs, and utilities.

This module provides all configuration classes and utilities for both QC and
visualization pipelines.
"""

from sleap_roots_analyze.pipeline.config.components import (
    AdaptiveSizingConfig,
    CleanupConfig,
    ClusteringConfig,
    ColumnConfig,
    CoreQCConfig,
    CrossPlatformConfig,
    DashboardConfig,
    DataConfig,
    GMMOutlierConfig,
    HeritabilityConfig,
    HierarchicalOutlierConfig,
    InteractiveVisualizationConfig,
    InterestingGenotypesConfig,
    IsolationForestConfig,
    KMeansOutlierConfig,
    LoggingConfig,
    MahalanobisConfig,
    MergeTraitsConfig,
    OutlierDetectionConfig,
    OutlierRemovalConfig,
    PCAConfig,
    PCAOutlierConfig,
    RootCoreConfig,
    RootCoreSourceConfig,
    StaticVisualizationConfig,
    StatisticsConfig,
    SummaryConfig,
    UMAPConfig,
    VisualizationConfig,
)
from sleap_roots_analyze.pipeline.config.qc_config import QCPipelineConfig
from sleap_roots_analyze.pipeline.config.utils import (
    get_default_qc_config,
    get_default_viz_config,
    load_cross_platform_config,
    load_qc_config,
    load_viz_config,
    merge_qc_configs,
    save_qc_config,
    save_viz_config,
    validate_qc_config,
    validate_viz_config,
)
from sleap_roots_analyze.pipeline.config.viz_config import VizPipelineConfig

__all__ = [
    # Reusable config components
    "AdaptiveSizingConfig",
    "CleanupConfig",
    "ClusteringConfig",
    "ColumnConfig",
    "CoreQCConfig",
    "CrossPlatformConfig",
    "DashboardConfig",
    "DataConfig",
    "GMMOutlierConfig",
    "HeritabilityConfig",
    "HierarchicalOutlierConfig",
    "InteractiveVisualizationConfig",
    "InterestingGenotypesConfig",
    "IsolationForestConfig",
    "KMeansOutlierConfig",
    "LoggingConfig",
    "MahalanobisConfig",
    "MergeTraitsConfig",
    "OutlierDetectionConfig",
    "OutlierRemovalConfig",
    "PCAConfig",
    "PCAOutlierConfig",
    "RootCoreConfig",
    "RootCoreSourceConfig",
    "StaticVisualizationConfig",
    "StatisticsConfig",
    "SummaryConfig",
    "UMAPConfig",
    "VisualizationConfig",
    # Pipeline configs
    "QCPipelineConfig",
    "VizPipelineConfig",
    # QC config utilities
    "load_qc_config",
    "save_qc_config",
    "get_default_qc_config",
    "validate_qc_config",
    "merge_qc_configs",
    # Viz config utilities
    "load_viz_config",
    "save_viz_config",
    "get_default_viz_config",
    "validate_viz_config",
    # Cross-platform config utilities
    "load_cross_platform_config",
]
