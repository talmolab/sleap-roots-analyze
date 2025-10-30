"""Configuration utilities for all pipelines.

This module provides helper functions for loading, saving, and validating
pipeline configurations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from omegaconf import MISSING, OmegaConf

from sleap_roots_analyze.pipeline.config.qc_config import QCPipelineConfig
from sleap_roots_analyze.pipeline.config.viz_config import VizPipelineConfig


# ============================================================================
# QC Pipeline Configuration Utilities
# ============================================================================


def load_qc_config(config_path: str | Path) -> QCPipelineConfig:
    """Load QC pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        QCPipelineConfig object with loaded configuration.

    Example:
        >>> config = load_qc_config("qc_standard.yaml")
    """
    config_path = Path(config_path)
    omega_conf = OmegaConf.load(config_path)
    structured = OmegaConf.structured(QCPipelineConfig)
    merged = OmegaConf.merge(structured, omega_conf)
    return OmegaConf.to_object(merged)


def save_qc_config(config: QCPipelineConfig, config_path: str | Path) -> None:
    """Save QC pipeline configuration to a YAML file.

    Args:
        config: QCPipelineConfig object to save.
        config_path: Path to save the YAML file.

    Example:
        >>> config = QCPipelineConfig(pipeline_name="qc_pipeline")
        >>> save_qc_config(config, "qc_config.yaml")
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    omega_conf = OmegaConf.structured(config)
    OmegaConf.save(omega_conf, config_path)


def get_default_qc_config(pipeline_name: str = "qc_pipeline") -> QCPipelineConfig:
    """Get a default QC pipeline configuration.

    Args:
        pipeline_name: Name for the pipeline.

    Returns:
        QCPipelineConfig with default values.

    Example:
        >>> config = get_default_qc_config("qc_pipeline")
    """
    return QCPipelineConfig(pipeline_name=pipeline_name)


def validate_qc_config(config: QCPipelineConfig) -> None:
    """Validate a QC pipeline configuration.

    Args:
        config: QCPipelineConfig object to validate.

    Raises:
        ValueError: If configuration is invalid.

    Example:
        >>> config = QCPipelineConfig(pipeline_name="test")
        >>> validate_qc_config(config)
    """
    # Check required fields
    if config.pipeline_name == MISSING:
        raise ValueError("pipeline_name is required")

    # Validate data config
    if config.data.csv_path == MISSING:
        raise ValueError("data.csv_path is required")

    # Validate outlier detection config
    valid_traditional_methods = ["pca", "isolation_forest", "mahalanobis"]
    valid_clustering_methods = ["kmeans", "gmm", "hierarchical"]

    for method in config.outlier_detection.traditional_methods:
        if method not in valid_traditional_methods:
            raise ValueError(
                f"outlier_detection.traditional_methods contains invalid method '{method}'. "
                f"Valid methods: {valid_traditional_methods}"
            )

    for method in config.outlier_detection.clustering_methods:
        if method not in valid_clustering_methods:
            raise ValueError(
                f"outlier_detection.clustering_methods contains invalid method '{method}'. "
                f"Valid methods: {valid_clustering_methods}"
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
    if config.clustering.enabled:
        for method in config.clustering.methods:
            if method not in valid_cluster_methods:
                raise ValueError(
                    f"clustering.methods contains invalid method '{method}'. "
                    f"Valid methods: {valid_cluster_methods}"
                )

        if (
            config.clustering.n_clusters is not None
            and config.clustering.n_clusters < 2
        ):
            raise ValueError("clustering.n_clusters must be at least 2")

    # Validate outlier removal config
    valid_removal_strategies = ["single", "consensus", "subset"]
    if config.outlier_removal.strategy not in valid_removal_strategies:
        raise ValueError(
            f"outlier_removal.strategy must be one of {valid_removal_strategies}"
        )

    # For "single" strategy, validate the method is configured
    if config.outlier_removal.strategy == "single":
        all_configured_methods = (
            config.outlier_detection.traditional_methods
            + config.outlier_detection.clustering_methods
        )
        if config.outlier_removal.method not in all_configured_methods:
            if not all_configured_methods:
                raise ValueError(
                    f"outlier_removal.method '{config.outlier_removal.method}' "
                    f"specified, but no outlier detection methods are configured. "
                    f"Add methods to outlier_detection.traditional_methods or clustering_methods."
                )
            else:
                raise ValueError(
                    f"outlier_removal.method '{config.outlier_removal.method}' "
                    f"not in configured detection methods: {all_configured_methods}"
                )

    # For "subset" strategy, validate min_methods is reasonable
    if config.outlier_removal.strategy == "subset":
        all_configured_methods = (
            config.outlier_detection.traditional_methods
            + config.outlier_detection.clustering_methods
        )
        if config.outlier_removal.min_methods < 1:
            raise ValueError("outlier_removal.min_methods must be at least 1")
        if config.outlier_removal.min_methods > len(all_configured_methods):
            raise ValueError(
                f"outlier_removal.min_methods ({config.outlier_removal.min_methods}) "
                f"cannot exceed number of configured methods ({len(all_configured_methods)})"
            )

    # Validate logging config
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.logging.level not in valid_log_levels:
        raise ValueError(f"logging.level must be one of {valid_log_levels}")


def merge_qc_configs(
    base_config: QCPipelineConfig,
    override_dict: Dict[str, Any],
) -> QCPipelineConfig:
    """Merge a base QC configuration with overrides.

    Args:
        base_config: Base QCPipelineConfig object.
        override_dict: Dictionary of configuration overrides.

    Returns:
        Merged QCPipelineConfig object.

    Example:
        >>> base = get_default_qc_config()
        >>> overrides = {"data": {"csv_path": "data.csv"}}
        >>> config = merge_qc_configs(base, overrides)
    """
    base_omega = OmegaConf.structured(base_config)
    override_omega = OmegaConf.create(override_dict)
    merged = OmegaConf.merge(base_omega, override_omega)
    return OmegaConf.to_object(merged)


# ============================================================================
# Visualization Pipeline Configuration Utilities
# ============================================================================


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
        >>> validate_viz_config(config)
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
