"""Configuration utilities for all pipelines.

This module provides helper functions for loading, saving, and validating
pipeline configurations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from omegaconf import MISSING, OmegaConf

from sleap_roots_analyze.pipeline.config.qc_config import QCPipelineConfig
from sleap_roots_analyze.pipeline.config.viz_config import VizPipelineConfig
from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig


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


def validate_explicit_config(config: QCPipelineConfig) -> None:
    """Validate that critical parameters are explicitly configured.

    This validation ensures users explicitly set critical parameters that affect
    analysis results, preventing silent failures from unintended defaults.

    Args:
        config: QCPipelineConfig object to validate.

    Raises:
        ValueError: If required parameters are missing or use default values.

    Warns:
        UserWarning: If important-but-optional parameters are not set.

    Example:
        >>> config = load_qc_config("my_config.yaml")
        >>> validate_explicit_config(config)  # Errors if required params missing
    """
    import warnings

    errors = []
    warnings_list = []

    # REQUIRED: Cleanup thresholds (affect data entry)
    if config.cleanup.max_nan_fraction is None:
        errors.append(
            "cleanup.max_nan_fraction must be explicitly set\n"
            "  Recommended: 0.25 (removes samples with >25% missing data)\n"
            "  Range: 0.0-1.0 (lower = stricter)"
        )
    if config.cleanup.max_zeros_per_trait is None:
        errors.append(
            "cleanup.max_zeros_per_trait must be explicitly set\n"
            "  Recommended: 0.5 (removes traits with >50% zeros)\n"
            "  Range: 0.0-1.0 (lower = stricter)"
        )

    # REQUIRED: Column mappings (dataset-specific)
    if config.columns.genotype is None:
        errors.append(
            "columns.genotype must be explicitly set (dataset-specific)\n"
            "  Examples: 'geno', 'genotype', 'accession', 'salk_geno'"
        )
    # OPTIONAL: columns.replicate may be None (or omitted). Its values are never
    # used in any computation — heritability fits value ~ 1 + (1|genotype) and
    # weights by rows-per-genotype, not by replicate. Datasets with no replicate
    # factor (e.g. cylinder data) leave it unset. See issue #142.

    # REQUIRED: PCA n_components (affects dimensionality)
    if config.pca.n_components is None:
        errors.append(
            "pca.n_components must be explicitly set\n"
            "  Recommended: 0.95 (retain components explaining 95% variance)\n"
            "  Range: 0.90-0.99 (higher = more components retained)"
        )

    # REQUIRED: Heritability threshold (if filtering enabled)
    if config.heritability.enabled and config.heritability.threshold is None:
        errors.append(
            "heritability.threshold must be explicitly set when filtering enabled\n"
            "  Typical range: 0.3-0.6\n"
            "  Higher = more stringent (fewer traits retained)"
        )

    # REQUIRED: Root core aggregation method (if root_core processing enabled)
    if config.root_core is not None:
        for i, source in enumerate(config.root_core.sources):
            if source.aggregation_method is None:
                errors.append(
                    f"root_core.sources[{i}].aggregation_method must be explicitly set\n"
                    f"  Recommended: 'median' (robust to outliers and typos)\n"
                    f"  Alternative: 'mean' (if no outliers expected)"
                )

    # REQUIRED: Outlier removal strategy (if outlier detection configured)
    has_outlier_detection = (
        len(config.outlier_detection.traditional_methods) > 0
        or len(config.outlier_detection.clustering_methods) > 0
    )
    if has_outlier_detection and config.outlier_removal.strategy is None:
        errors.append(
            "outlier_removal.strategy must be set when outlier detection enabled\n"
            "  Options:\n"
            "    - 'remove': Delete outlier samples from dataset\n"
            "    - 'flag': Add outlier_* columns but keep samples\n"
            "    - 'none': Detect but don't remove or flag"
        )

    # IMPORTANT BUT OPTIONAL: Outlier detection
    if not has_outlier_detection:
        warnings_list.append(
            "No outlier detection methods configured (traditional_methods and clustering_methods are empty).\n"
            "  This is valid if you only want data cleanup (NaN/zero removal).\n"
            "  Consider adding outlier detection for robust QC:\n"
            "    - traditional_methods: ['mahalanobis_pca', 'isolation_forest']\n"
            "    - clustering_methods: ['dbscan']\n"
            "  See configs/templates/ for examples."
        )

    # Raise errors if any required parameters missing
    if errors:
        error_msg = (
            "Configuration Validation Failed\n"
            "================================================================\n"
            "Critical parameters must be explicitly set to avoid silent failures.\n\n"
        )
        error_msg += "\n\n".join(errors)
        error_msg += (
            "\n\n================================================================\n"
            "See configuration templates in configs/templates/:\n"
            "   - qc_cleanup_only_template.yaml (data cleanup only)\n"
            "   - qc_full_pipeline_template.yaml (cleanup + outlier detection)\n"
        )
        raise ValueError(error_msg)

    # Issue warnings for important-but-optional parameters
    for warning_msg in warnings_list:
        warnings.warn(warning_msg, UserWarning, stacklevel=2)


def validate_qc_config(config: QCPipelineConfig, check_files: bool = True) -> None:
    """Validate a QC pipeline configuration.

    This function performs both explicit configuration validation (checking that
    critical parameters are set) and structural validation (checking that values
    are valid).

    Args:
        config: QCPipelineConfig object to validate.
        check_files: If True, verify data files exist and validate group_by column (default: True).

    Raises:
        ValueError: If configuration is invalid.

    Warns:
        UserWarning: If important-but-optional parameters are not set.

    Example:
        >>> config = QCPipelineConfig(pipeline_name="test")
        >>> validate_qc_config(config)
    """
    # First, validate explicit configuration (required parameters set)
    validate_explicit_config(config)

    # Then validate structural correctness
    # Check required fields
    if config.pipeline_name == MISSING:
        raise ValueError("pipeline_name is required")

    # Validate data config
    if config.data.csv_path == MISSING:
        raise ValueError("data.csv_path is required")

    # Validate input-contract validation mode (issue #144)
    valid_validate_input_modes = ["off", "warn", "strict"]
    if config.data.validate_input not in valid_validate_input_modes:
        raise ValueError(
            f"data.validate_input must be one of off | warn | strict, "
            f"got '{config.data.validate_input}'"
        )

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

    # Validate outlier removal config (only if outlier detection is configured)
    has_outlier_detection = (
        len(config.outlier_detection.traditional_methods) > 0
        or len(config.outlier_detection.clustering_methods) > 0
    )

    if has_outlier_detection:
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

    # Validate group_by column exists in data (if group_by specified and check_files enabled)
    if check_files and config.data.group_by is not None:
        from pathlib import Path

        if config.data.csv_path and Path(config.data.csv_path).exists():
            try:
                df = pd.read_csv(config.data.csv_path, nrows=0)  # Read only header
                if config.data.group_by not in df.columns:
                    available = list(df.columns)
                    raise ValueError(
                        f"group_by column '{config.data.group_by}' not found in CSV. "
                        f"Available columns: {available}"
                    )
            except pd.errors.EmptyDataError:
                # Empty CSV will fail at runtime anyway, skip validation
                pass


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

    Returns:
        None. Writes the configuration to ``config_path`` as a side effect,
        creating parent directories as needed.

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

    Returns:
        None. Returns normally when the configuration is valid; otherwise raises.

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

    # Validate heritability configuration consistency
    if config.heritability.enabled and not config.statistics.calculate_heritability:
        raise ValueError(
            "Invalid configuration: heritability.enabled=True but "
            "statistics.calculate_heritability=False.\n"
            "Cannot filter traits by heritability without calculating heritability "
            "first.\n"
            "Either:\n"
            "  - Set statistics.calculate_heritability=True, OR\n"
            "  - Set heritability.enabled=False"
        )

    # Warn if UMAP is enabled but not yet implemented (Phase 2C stub)
    if config.umap.enabled:
        import warnings

        warnings.warn(
            "UMAP analysis is not yet implemented (Phase 2C stub). "
            "The umap.enabled setting will be ignored and UMAP steps will be skipped.",
            UserWarning,
            stacklevel=2,
        )


# ============================================================================
# Cross-Platform Pipeline Configuration Utilities
# ============================================================================


def load_cross_platform_config(config_path: str | Path) -> CrossPlatformConfig:
    """Load cross-platform analysis configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        CrossPlatformConfig object with loaded configuration.

    Example:
        >>> config = load_cross_platform_config("cross_platform_analysis.yaml")
    """
    config_path = Path(config_path)
    omega_conf = OmegaConf.load(config_path)
    structured = OmegaConf.structured(CrossPlatformConfig)
    merged = OmegaConf.merge(structured, omega_conf)
    return OmegaConf.to_object(merged)
