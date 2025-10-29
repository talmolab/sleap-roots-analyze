"""QC Pipeline configuration.

This module defines the QC pipeline configuration that composes reusable
configuration components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING

from sleap_roots_analyze.pipeline.config.components import (
    CleanupConfig,
    ClusteringConfig,
    ColumnConfig,
    DataConfig,
    HeritabilityConfig,
    LoggingConfig,
    OutlierDetectionConfig,
    OutlierRemovalConfig,
    PCAConfig,
    VisualizationConfig,
)


@dataclass
class QCPipelineConfig:
    """QC Pipeline configuration - composes reusable components.

    Attributes:
        pipeline_name: Name of the pipeline.
        version: Pipeline version.
        enable_parallel: Whether to enable parallel task execution.
        columns: Column name configuration.
        data: Data loading and processing configuration.
        cleanup: Data cleanup configuration.
        outlier_detection: Outlier detection configuration.
        outlier_removal: Outlier removal configuration.
        heritability: Heritability filtering configuration.
        pca: PCA analysis configuration.
        clustering: Clustering analysis configuration.
        visualization: Visualization generation configuration.
        logging: Logging configuration.
    """

    pipeline_name: str = MISSING
    version: str = "1.0"
    enable_parallel: bool = False

    # Compose reusable components
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    outlier_detection: OutlierDetectionConfig = field(
        default_factory=OutlierDetectionConfig
    )
    outlier_removal: OutlierRemovalConfig = field(default_factory=OutlierRemovalConfig)
    heritability: HeritabilityConfig = field(default_factory=HeritabilityConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
