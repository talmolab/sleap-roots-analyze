"""Configuration management for visualization pipeline.

This module provides structured configuration classes for the visualization pipeline
using OmegaConf dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import MISSING, OmegaConf


@dataclass
class ColumnConfig:
    """Configuration for column names in the trait data.

    Attributes:
        barcode: Name of the barcode/plant ID column.
        genotype: Name of the genotype column.
        replicate: Name of the replicate column (None if not present).
        image_path: Name of the image path column (optional, for image linking).
    """

    barcode: str = "Barcode"
    genotype: str = "geno"
    replicate: Optional[str] = "rep"
    image_path: Optional[str] = "image_path"


@dataclass
class DataConfig:
    """Configuration for data loading.

    Attributes:
        csv_path: Path to trait CSV file.
        image_dir: Directory containing images (optional).
        traits_to_include: List of trait names to include. If None, includes all.
        traits_to_exclude: List of trait names to exclude.
    """

    csv_path: str = MISSING
    image_dir: Optional[str] = None
    traits_to_include: Optional[List[str]] = None
    traits_to_exclude: List[str] = field(default_factory=list)


@dataclass
class StatisticsConfig:
    """Configuration for statistical calculations.

    Attributes:
        calculate_anova: Whether to calculate ANOVA.
        calculate_heritability: Whether to calculate heritability.
        alpha: Significance level for tests.
    """

    calculate_anova: bool = True
    calculate_heritability: bool = True
    alpha: float = 0.05


@dataclass
class PCAConfig:
    """Configuration for PCA analysis.

    Attributes:
        n_components: Number of components (or variance ratio if < 1).
        standardize: Whether to standardize data before PCA.
        feature_selection_strategy: Strategy for selecting top features.
        n_top_features: Number of top features to select per component.
    """

    n_components: float = 0.95
    standardize: bool = True
    feature_selection_strategy: str = "top_variance"
    n_top_features: int = 10


@dataclass
class UMAPConfig:
    """Configuration for UMAP analysis.

    Attributes:
        enabled: Whether to perform UMAP analysis.
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance for UMAP.
        metric: Distance metric for UMAP.
        random_state: Random state for reproducibility.
    """

    enabled: bool = False
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42


@dataclass
class ClusteringConfig:
    """Configuration for clustering analysis.

    Attributes:
        enabled: Whether to perform clustering.
        methods: List of clustering methods to use (kmeans, gmm, hierarchical).
        n_clusters: Number of clusters (None = auto-optimize).
        auto_optimize: Whether to automatically optimize number of clusters.
        min_clusters: Minimum clusters for auto-optimization.
        max_clusters: Maximum clusters for auto-optimization.
    """

    enabled: bool = False
    methods: List[str] = field(default_factory=lambda: ["kmeans"])
    n_clusters: Optional[int] = None
    auto_optimize: bool = True
    min_clusters: int = 2
    max_clusters: int = 10


@dataclass
class HeritabilityConfig:
    """Configuration for heritability filtering.

    Attributes:
        filter_enabled: Whether to filter traits by heritability.
        threshold: Minimum heritability (H²) threshold for filtering.
    """

    filter_enabled: bool = False
    threshold: float = 0.60


@dataclass
class InterestingGenotypesConfig:
    """Configuration for identifying interesting genotypes.

    Attributes:
        enabled: Whether to identify interesting genotypes.
        methods: List of methods (pc_extreme, trait_extreme, heritable_extreme).
        pc_threshold: Z-score threshold for PC-based extremes.
        trait_percentile: Percentile threshold for trait-based extremes.
        min_heritability: Minimum heritability for heritable extremes.
        max_genotypes: Maximum number of genotypes to identify per method.
        generate_image_grids: Whether to generate image grids.
        images_per_genotype: Number of images to show per genotype.
    """

    enabled: bool = True
    methods: List[str] = field(
        default_factory=lambda: ["pc_extreme", "trait_extreme", "heritable_extreme"]
    )
    pc_threshold: float = 2.0
    trait_percentile: float = 95.0
    min_heritability: float = 0.60
    max_genotypes: int = 10
    generate_image_grids: bool = True
    images_per_genotype: int = 9


@dataclass
class AdaptiveSizingConfig:
    """Configuration for adaptive figure sizing.

    Attributes:
        enabled: Whether to use adaptive sizing.
        base_width: Base width for single-column figures.
        base_height: Base height for single-row figures.
        width_per_item: Width increment per additional column/item.
        height_per_item: Height increment per additional row/item.
        min_width: Minimum figure width.
        max_width: Maximum figure width.
        min_height: Minimum figure height.
        max_height: Maximum figure height.
    """

    enabled: bool = True
    base_width: float = 8.0
    base_height: float = 6.0
    width_per_item: float = 2.0
    height_per_item: float = 2.0
    min_width: float = 6.0
    max_width: float = 20.0
    min_height: float = 4.0
    max_height: float = 16.0


@dataclass
class StaticVisualizationConfig:
    """Configuration for static visualization generation.

    Attributes:
        enabled: Whether to generate static plots.
        formats: List of output formats (png, pdf, svg).
        dpi: DPI for raster formats.
        create_pca_plots: Whether to create PCA plots.
        create_umap_plots: Whether to create UMAP plots.
        create_cluster_plots: Whether to create clustering plots.
        create_trait_distributions: Whether to create trait histograms.
        create_trait_correlations: Whether to create correlation plots.
        create_heritability_plots: Whether to create heritability plots.
        create_genotype_comparisons: Whether to create genotype comparison plots.
    """

    enabled: bool = True
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 300
    create_pca_plots: bool = True
    create_umap_plots: bool = False
    create_cluster_plots: bool = False
    create_trait_distributions: bool = True
    create_trait_correlations: bool = True
    create_heritability_plots: bool = True
    create_genotype_comparisons: bool = True


@dataclass
class InteractiveVisualizationConfig:
    """Configuration for interactive visualization generation.

    Attributes:
        enabled: Whether to generate interactive plots.
        create_pca_plots: Whether to create interactive PCA plots.
        create_umap_plots: Whether to create interactive UMAP plots.
        create_cluster_plots: Whether to create interactive clustering plots.
        show_images_on_hover: Whether to show images on hover (requires image_dir).
    """

    enabled: bool = True
    create_pca_plots: bool = True
    create_umap_plots: bool = False
    create_cluster_plots: bool = False
    show_images_on_hover: bool = True


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation.

    Attributes:
        enabled: Whether to generate dashboards.
        create_summary_dashboard: Whether to create summary dashboard.
        create_trait_dashboard: Whether to create per-trait dashboards.
    """

    enabled: bool = False
    create_summary_dashboard: bool = True
    create_trait_dashboard: bool = False


@dataclass
class SummaryConfig:
    """Configuration for summary report generation.

    Attributes:
        enabled: Whether to generate summary report.
        formats: List of output formats (markdown, html, json).
    """

    enabled: bool = True
    formats: List[str] = field(default_factory=lambda: ["markdown", "json"])


@dataclass
class LoggingConfig:
    """Configuration for logging.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_to_file: Whether to log to a file.
        log_file: Path to log file (if log_to_file is True).
    """

    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "viz_pipeline.log"


@dataclass
class VizPipelineConfig:
    """Top-level visualization pipeline configuration.

    Attributes:
        pipeline_name: Name of the pipeline.
        version: Pipeline version.
        enable_parallel: Whether to enable parallel task execution (future).
        columns: Column name configuration.
        data: Data loading configuration.
        statistics: Statistics calculation configuration.
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


def load_viz_config(config_path: str | Path) -> VizPipelineConfig:
    """Load visualization pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        VizPipelineConfig object with loaded configuration.

    Example:
        >>> config = load_viz_config("viz_standard.yaml")
    """
    config_path = Path(config_path)
    omega_conf = OmegaConf.load(config_path)
    # Merge with structured config to get proper VizPipelineConfig object
    structured = OmegaConf.structured(VizPipelineConfig)
    merged = OmegaConf.merge(structured, omega_conf)
    return OmegaConf.to_object(merged)


def save_viz_config(config: VizPipelineConfig, config_path: str | Path) -> None:
    """Save visualization pipeline configuration to a YAML file.

    Args:
        config: VizPipelineConfig object to save.
        config_path: Path to save the YAML file.

    Example:
        >>> config = VizPipelineConfig(pipeline_name="viz_pipeline")
        >>> save_viz_config(config, "viz_config.yaml")
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to OmegaConf and save
    omega_conf = OmegaConf.structured(config)
    OmegaConf.save(omega_conf, config_path)


def get_default_viz_config(pipeline_name: str = "viz_pipeline") -> VizPipelineConfig:
    """Get a default visualization pipeline configuration.

    Args:
        pipeline_name: Name for the pipeline.

    Returns:
        VizPipelineConfig with default values.

    Example:
        >>> config = get_default_viz_config("viz_pipeline")
    """
    return VizPipelineConfig(pipeline_name=pipeline_name)


def validate_viz_config(config: VizPipelineConfig) -> None:
    """Validate a visualization pipeline configuration.

    Args:
        config: VizPipelineConfig object to validate.

    Raises:
        ValueError: If configuration is invalid.

    Example:
        >>> config = VizPipelineConfig(pipeline_name="test")
        >>> validate_viz_config(config)  # Raises if invalid
    """
    # Check required fields
    if config.pipeline_name == MISSING:
        raise ValueError("pipeline_name is required")

    # Validate data config
    if config.data.csv_path == MISSING:
        raise ValueError("data.csv_path is required")

    # Validate PCA config
    if config.pca.n_components <= 0:
        raise ValueError("pca.n_components must be positive")

    valid_pca_strategies = [
        "extreme",
        "top_absolute",
        "top_contribution",
        "top_variance",
    ]
    if config.pca.feature_selection_strategy not in valid_pca_strategies:
        raise ValueError(
            f"pca.feature_selection_strategy must be one of {valid_pca_strategies}"
        )

    # Validate clustering config if enabled
    if config.clustering.enabled:
        valid_cluster_methods = ["kmeans", "gmm", "hierarchical"]
        for method in config.clustering.methods:
            if method not in valid_cluster_methods:
                raise ValueError(
                    f"clustering.methods contains invalid method '{method}'. "
                    f"Valid methods: {valid_cluster_methods}"
                )

        if (
            not config.clustering.auto_optimize
            and config.clustering.n_clusters is not None
        ):
            if config.clustering.n_clusters < 2:
                raise ValueError("clustering.n_clusters must be at least 2")

    # Validate interesting genotypes config if enabled
    if config.interesting_genotypes.enabled:
        valid_methods = ["pc_extreme", "trait_extreme", "heritable_extreme"]
        for method in config.interesting_genotypes.methods:
            if method not in valid_methods:
                raise ValueError(
                    f"interesting_genotypes.methods contains invalid method '{method}'. "
                    f"Valid methods: {valid_methods}"
                )

    # Validate static viz formats
    if config.static_viz.enabled:
        valid_formats = ["png", "pdf", "svg"]
        for fmt in config.static_viz.formats:
            if fmt not in valid_formats:
                raise ValueError(
                    f"static_viz.formats contains invalid format '{fmt}'. "
                    f"Valid formats: {valid_formats}"
                )

    # Validate summary formats
    if config.summary.enabled:
        valid_formats = ["markdown", "html", "json"]
        for fmt in config.summary.formats:
            if fmt not in valid_formats:
                raise ValueError(
                    f"summary.formats contains invalid format '{fmt}'. "
                    f"Valid formats: {valid_formats}"
                )

    # Validate logging config
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.logging.level not in valid_log_levels:
        raise ValueError(f"logging.level must be one of {valid_log_levels}")
