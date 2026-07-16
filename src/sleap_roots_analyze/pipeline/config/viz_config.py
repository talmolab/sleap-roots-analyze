"""Visualization Pipeline configuration.

This module defines the visualization pipeline configuration that composes
reusable configuration components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING

from sleap_roots_analyze.pipeline.config.components import (
    AdaptiveSizingConfig,
    ClusteringConfig,
    ColumnConfig,
    DashboardConfig,
    DataConfig,
    HeritabilityConfig,
    InteractiveVisualizationConfig,
    InterestingGenotypesConfig,
    LoggingConfig,
    PCAConfig,
    StaticVisualizationConfig,
    StatisticsConfig,
    SummaryConfig,
    UMAPConfig,
)


@dataclass
class VizPipelineConfig:
    """Visualization Pipeline configuration - composes reusable components.

    Attributes:
        pipeline_name: Name of the pipeline.
        version: Pipeline version.
        enable_parallel: Whether to enable parallel task execution.
        columns: Column name configuration.
        data: Data loading configuration.
        statistics: Statistical analysis configuration.
        pca: PCA analysis configuration.
        umap: UMAP analysis configuration.
        clustering: Clustering analysis configuration.
        heritability: Heritability filtering configuration.
        interesting_genotypes: Interesting genotypes identification configuration.
        adaptive_sizing: Adaptive figure sizing configuration.
        static_viz: Static visualization configuration.
        interactive_viz: Interactive visualization configuration.
        dashboard: Dashboard configuration.
        summary: Summary report configuration.
        logging: Logging configuration.
    """

    pipeline_name: str = MISSING
    version: str = "1.0"
    enable_parallel: bool = False

    # Compose reusable components
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    data: DataConfig = field(default_factory=DataConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    heritability: HeritabilityConfig = field(default_factory=HeritabilityConfig)
    interesting_genotypes: InterestingGenotypesConfig = field(
        default_factory=InterestingGenotypesConfig
    )
    adaptive_sizing: AdaptiveSizingConfig = field(default_factory=AdaptiveSizingConfig)
    static_viz: StaticVisualizationConfig = field(
        default_factory=StaticVisualizationConfig
    )
    interactive_viz: InteractiveVisualizationConfig = field(
        default_factory=InteractiveVisualizationConfig
    )
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        """Auto-exclude fixed_effects columns from the trait_cols scan (#114, 7.3).

        `LoadDataAndImagesStep` fixes `trait_cols` once, at the very start of
        the pipeline, using only `data.additional_exclude_cols` plus a
        hardcoded metadata-substring list -- it has no knowledge of
        `statistics.fixed_effects` on its own. A fixed_effects name outside
        that hardcoded list (e.g. "block") would otherwise be silently
        treated as a phenotypic trait everywhere upstream of the statistics
        step. Unioning it into `additional_exclude_cols` here, at
        config-construction time, closes that gap without any step-ordering
        change. Deduplicates against names already present (e.g. a
        fixed_effects name that also happens to match the hardcoded
        substring list, or that a user already listed explicitly).
        """
        if self.statistics.fixed_effects:
            self.data.additional_exclude_cols = list(
                dict.fromkeys(
                    (self.data.additional_exclude_cols or [])
                    + self.statistics.fixed_effects
                )
            )
