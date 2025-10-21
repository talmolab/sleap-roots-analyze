"""Configuration management using OmegaConf.

This module provides structured configuration classes for pipeline setup and
execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import MISSING, OmegaConf


@dataclass
class DataConfig:
    """Configuration for data loading and processing.

    Attributes:
        input_path: Path to input data file.
        output_dir: Directory for output files.
        traits_to_include: List of trait names to include. If None, includes all.
        traits_to_exclude: List of trait names to exclude.
        min_heritability: Minimum heritability threshold for trait filtering.
    """

    input_path: str = MISSING
    output_dir: str = "./outputs"
    traits_to_include: Optional[List[str]] = None
    traits_to_exclude: List[str] = field(default_factory=list)
    min_heritability: float = 0.0


@dataclass
class OutlierDetectionConfig:
    """Configuration for outlier detection.

    Attributes:
        method: Outlier detection method (mahalanobis, zscore, iqr).
        threshold: Threshold for outlier detection (method-specific).
        use_pca: Whether to use PCA before outlier detection.
        n_components: Number of PCA components (or variance ratio if < 1).
        robust_covariance: Whether to use robust covariance estimation.
    """

    method: str = "mahalanobis"
    threshold: float = 0.01
    use_pca: bool = True
    n_components: float = 0.95
    robust_covariance: bool = False


@dataclass
class PCAConfig:
    """Configuration for PCA analysis.

    Attributes:
        n_components: Number of components (or variance ratio if < 1).
        standardize: Whether to standardize data before PCA.
        feature_selection_strategy: Strategy for selecting top features
            (extreme, top_absolute, top_contribution, top_variance).
        n_top_features: Number of top features to select per component.
    """

    n_components: float = 0.95
    standardize: bool = True
    feature_selection_strategy: str = "top_variance"
    n_top_features: int = 10


@dataclass
class ClusteringConfig:
    """Configuration for clustering analysis.

    Attributes:
        method: Clustering method (kmeans, gmm, hierarchical).
        n_clusters: Number of clusters.
        auto_optimize: Whether to automatically optimize number of clusters.
        min_clusters: Minimum clusters for auto-optimization.
        max_clusters: Maximum clusters for auto-optimization.
    """

    method: str = "kmeans"
    n_clusters: int = 3
    auto_optimize: bool = False
    min_clusters: int = 2
    max_clusters: int = 10


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation.

    Attributes:
        create_pca_plots: Whether to create PCA plots.
        create_umap_plots: Whether to create UMAP plots.
        create_cluster_plots: Whether to create clustering plots.
        create_outlier_plots: Whether to create outlier detection plots.
        interactive: Whether to create interactive plots (Plotly).
        dpi: DPI for static plots.
        figsize: Figure size for static plots (width, height).
    """

    create_pca_plots: bool = True
    create_umap_plots: bool = False
    create_cluster_plots: bool = False
    create_outlier_plots: bool = True
    interactive: bool = False
    dpi: int = 300
    figsize: tuple[int, int] = (10, 8)


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
    log_file: str = "pipeline.log"


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration.

    Attributes:
        pipeline_name: Name of the pipeline.
        version: Pipeline version.
        enable_parallel: Whether to enable parallel task execution (future).
        data: Data configuration.
        outlier_detection: Outlier detection configuration.
        pca: PCA configuration.
        clustering: Clustering configuration.
        visualization: Visualization configuration.
        logging: Logging configuration.
    """

    pipeline_name: str = MISSING
    version: str = "1.0"
    enable_parallel: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    outlier_detection: OutlierDetectionConfig = field(
        default_factory=OutlierDetectionConfig
    )
    pca: PCAConfig = field(default_factory=PCAConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        PipelineConfig object with loaded configuration.

    Example:
        >>> config = load_config("config.yaml")
    """
    config_path = Path(config_path)
    omega_conf = OmegaConf.load(config_path)
    # Merge with structured config to get proper PipelineConfig object
    structured = OmegaConf.structured(PipelineConfig)
    merged = OmegaConf.merge(structured, omega_conf)
    return OmegaConf.to_object(merged)


def save_config(config: PipelineConfig, config_path: str | Path) -> None:
    """Save configuration to a YAML file.

    Args:
        config: PipelineConfig object to save.
        config_path: Path to save the YAML file.

    Example:
        >>> config = PipelineConfig(pipeline_name="qc_pipeline")
        >>> save_config(config, "config.yaml")
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to OmegaConf and save
    omega_conf = OmegaConf.structured(config)
    OmegaConf.save(omega_conf, config_path)


def get_default_config(pipeline_name: str = "pipeline") -> PipelineConfig:
    """Get a default configuration.

    Args:
        pipeline_name: Name for the pipeline.

    Returns:
        PipelineConfig with default values.

    Example:
        >>> config = get_default_config("qc_pipeline")
    """
    return PipelineConfig(pipeline_name=pipeline_name)


def merge_configs(
    base_config: PipelineConfig,
    override_dict: Dict[str, Any],
) -> PipelineConfig:
    """Merge a base configuration with overrides.

    Args:
        base_config: Base PipelineConfig object.
        override_dict: Dictionary of configuration overrides.

    Returns:
        Merged PipelineConfig object.

    Example:
        >>> base = get_default_config()
        >>> overrides = {"data": {"input_path": "data.csv"}}
        >>> config = merge_configs(base, overrides)
    """
    # Convert base config to OmegaConf
    base_omega = OmegaConf.structured(base_config)

    # Create OmegaConf from override dict
    override_omega = OmegaConf.create(override_dict)

    # Merge
    merged = OmegaConf.merge(base_omega, override_omega)

    # Convert back to PipelineConfig
    return OmegaConf.to_object(merged)


def validate_config(config: PipelineConfig) -> None:
    """Validate a pipeline configuration.

    Args:
        config: PipelineConfig object to validate.

    Raises:
        ValueError: If configuration is invalid.

    Example:
        >>> config = PipelineConfig(pipeline_name="test")
        >>> validate_config(config)  # Raises if invalid
    """
    # Check required fields
    if config.pipeline_name == MISSING:
        raise ValueError("pipeline_name is required")

    # Validate data config
    if config.data.input_path == MISSING:
        raise ValueError("data.input_path is required")

    # Validate outlier detection config
    valid_outlier_methods = ["mahalanobis", "zscore", "iqr"]
    if config.outlier_detection.method not in valid_outlier_methods:
        raise ValueError(
            f"outlier_detection.method must be one of {valid_outlier_methods}"
        )

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

    # Validate clustering config
    valid_cluster_methods = ["kmeans", "gmm", "hierarchical"]
    if config.clustering.method not in valid_cluster_methods:
        raise ValueError(f"clustering.method must be one of {valid_cluster_methods}")

    if config.clustering.n_clusters < 2:
        raise ValueError("clustering.n_clusters must be at least 2")

    # Validate logging config
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.logging.level not in valid_log_levels:
        raise ValueError(f"logging.level must be one of {valid_log_levels}")
