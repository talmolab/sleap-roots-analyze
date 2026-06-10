"""Tests for pipeline configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import MISSING

from sleap_roots_analyze.pipeline.config import (
    ClusteringConfig,
    DataConfig,
    LoggingConfig,
    OutlierDetectionConfig,
    PCAConfig,
    QCPipelineConfig,
    VisualizationConfig,
    get_default_qc_config,
    load_qc_config,
    merge_qc_configs,
    save_qc_config,
    validate_qc_config,
)


def test_data_config_creation():
    """Test creating a DataConfig."""
    config = DataConfig(
        csv_path="data.csv",
        image_dir="./images",
        additional_exclude_cols=["QC_flag"],
    )

    assert config.csv_path == "data.csv"
    assert config.image_dir == "./images"
    assert config.additional_exclude_cols == ["QC_flag"]
    assert config.traits_to_include is None
    assert config.traits_to_exclude == []


def test_data_config_group_by_defaults_to_none():
    """Test that DataConfig.group_by defaults to None."""
    config = DataConfig(csv_path="data.csv")

    assert config.group_by is None


def test_data_config_group_by_can_be_set():
    """Test that DataConfig.group_by can be set to a column name."""
    config = DataConfig(csv_path="data.csv", group_by="plant_age_days")

    assert config.group_by == "plant_age_days"


def test_outlier_detection_config_defaults():
    """Test OutlierDetectionConfig defaults."""
    config = OutlierDetectionConfig()

    assert config.traditional_methods == []
    assert config.clustering_methods == []
    # Check nested configs have defaults
    assert config.pca.explained_variance == 0.95
    assert config.isolation_forest.contamination == 0.1
    assert config.mahalanobis.variance_threshold == 0.95
    assert config.mahalanobis.use_chi_squared is True


def test_pca_config_defaults():
    """Test PCAConfig defaults."""
    config = PCAConfig()

    assert config.n_components == 0.95
    assert config.standardize is True
    assert config.feature_selection_strategy == "top_variance"
    assert config.n_top_features == 10


def test_clustering_config_creation():
    """Test creating a ClusteringConfig."""
    config = ClusteringConfig(
        enabled=True,
        methods=["gmm"],
        n_clusters=5,
        auto_optimize=True,
        min_clusters=2,
        max_clusters=8,
    )

    assert config.methods == ["gmm"]
    assert config.n_clusters == 5
    assert config.auto_optimize is True
    assert config.min_clusters == 2
    assert config.max_clusters == 8


def test_visualization_config_defaults():
    """Test VisualizationConfig defaults."""
    config = VisualizationConfig()

    assert config.create_pca_plots is True
    assert config.create_umap_plots is False
    assert config.create_cluster_plots is False
    assert config.create_outlier_plots is True
    assert config.interactive is False
    assert config.dpi == 300
    assert config.figsize == (10, 8)


def test_logging_config_defaults():
    """Test LoggingConfig defaults."""
    config = LoggingConfig()

    assert config.level == "INFO"
    assert config.log_to_file is True
    assert config.log_file == "pipeline.log"


def test_pipeline_config_creation():
    """Test creating a QCPipelineConfig."""
    config = QCPipelineConfig(
        pipeline_name="test_pipeline",
        version="2.0",
    )

    assert config.pipeline_name == "test_pipeline"
    assert config.version == "2.0"
    assert config.enable_parallel is False
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.outlier_detection, OutlierDetectionConfig)
    assert isinstance(config.pca, PCAConfig)


def test_get_default_config():
    """Test getting a default configuration."""
    config = get_default_qc_config("my_pipeline")

    assert config.pipeline_name == "my_pipeline"
    assert config.version == "1.0"
    assert isinstance(config.data, DataConfig)


def test_save_and_load_config(tmp_path):
    """Test saving and loading configuration."""
    # Create a config
    original = QCPipelineConfig(
        pipeline_name="test",
        version="1.5",
    )
    original.data.csv_path = "test_data.csv"
    original.data.additional_exclude_cols = ["QC_flag"]
    original.pca.n_components = 0.9

    # Save
    config_path = tmp_path / "config.yaml"
    save_qc_config(original, config_path)

    assert config_path.exists()

    # Load
    loaded = load_qc_config(config_path)

    assert loaded.pipeline_name == "test"
    assert loaded.version == "1.5"
    assert loaded.data.csv_path == "test_data.csv"
    assert loaded.data.additional_exclude_cols == ["QC_flag"]
    assert loaded.pca.n_components == 0.9


def test_save_config_creates_directory(tmp_path):
    """Test that save_qc_config creates parent directories."""
    config = get_default_qc_config("test")
    config.data.csv_path = "data.csv"

    config_path = tmp_path / "nested" / "dir" / "config.yaml"
    save_qc_config(config, config_path)

    assert config_path.exists()


def test_merge_configs():
    """Test merging configurations."""
    base = get_default_qc_config("base")
    base.data.csv_path = "base.csv"
    base.pca.n_components = 0.95

    overrides = {
        "data": {"csv_path": "override.csv", "additional_exclude_cols": ["QC_flag"]},
        "pca": {"n_components": 0.8},
    }

    merged = merge_qc_configs(base, overrides)

    # Overridden values
    assert merged.data.csv_path == "override.csv"
    assert merged.data.additional_exclude_cols == ["QC_flag"]
    assert merged.pca.n_components == 0.8

    # Non-overridden values should be preserved
    assert merged.pipeline_name == "base"


def test_merge_configs_nested():
    """Test merging with nested overrides."""
    base = get_default_qc_config("test")
    base.data.csv_path = "base.csv"

    overrides = {
        "outlier_detection": {
            "traditional_methods": ["pca", "mahalanobis"],
            "pca": {"explained_variance": 0.90},
        }
    }

    merged = merge_qc_configs(base, overrides)

    assert merged.outlier_detection.traditional_methods == ["pca", "mahalanobis"]
    assert merged.outlier_detection.pca.explained_variance == 0.90
    # Other outlier_detection fields should be preserved
    assert merged.outlier_detection.clustering_methods == []


def test_validate_config_valid():
    """Test validating a valid configuration."""
    config = QCPipelineConfig(
        pipeline_name="test",
    )
    config.data.csv_path = "data.csv"
    # Need to configure at least one outlier detection method
    # since outlier_removal.method defaults to "mahalanobis"
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    # Should not raise
    validate_qc_config(config)


def test_validate_config_replicate_optional():
    """columns.replicate=None passes validation (issue #142)."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    config.outlier_detection.traditional_methods = ["mahalanobis"]
    config.columns.replicate = None

    # Should not raise — replicate values are never used in any computation.
    validate_qc_config(config, check_files=False)


def test_validate_config_genotype_still_required():
    """Relaxing replicate must not relax genotype, which remains required."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    config.outlier_detection.traditional_methods = ["mahalanobis"]
    config.columns.genotype = None

    with pytest.raises(ValueError, match="columns.genotype must be explicitly set"):
        validate_qc_config(config, check_files=False)


def test_validate_config_missing_pipeline_name():
    """Test validation fails for missing pipeline_name."""
    config = QCPipelineConfig(
        pipeline_name=MISSING,
    )
    config.data.csv_path = "data.csv"

    with pytest.raises(ValueError, match="pipeline_name is required"):
        validate_qc_config(config)


def test_validate_config_missing_csv_path():
    """Test validation fails for missing csv_path."""
    config = QCPipelineConfig(
        pipeline_name="test",
    )
    # data.csv_path is MISSING by default

    with pytest.raises(ValueError, match="data.csv_path is required"):
        validate_qc_config(config)


def test_validate_config_invalid_traditional_outlier_method():
    """Test validation fails for invalid traditional outlier detection method."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    config.outlier_detection.traditional_methods = ["invalid"]

    with pytest.raises(ValueError, match="traditional_methods contains invalid method"):
        validate_qc_config(config)


def test_validate_config_valid_outlier_methods():
    """Test validation passes for all valid outlier methods."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"

    # Test traditional methods
    config.outlier_detection.traditional_methods = [
        "pca",
        "isolation_forest",
        "mahalanobis",
    ]
    validate_qc_config(config)  # Should not raise

    # Test clustering methods
    config.outlier_detection.traditional_methods = []
    config.outlier_detection.clustering_methods = ["kmeans", "gmm", "hierarchical"]
    config.outlier_removal.method = "kmeans"  # Update to match clustering methods
    validate_qc_config(config)  # Should not raise


def test_validate_config_invalid_pca_components():
    """Test validation fails for invalid PCA components."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    config.pca.n_components = -1

    with pytest.raises(ValueError, match="pca.n_components must be positive"):
        validate_qc_config(config)


def test_validate_config_invalid_pca_strategy():
    """Test validation fails for invalid PCA feature selection strategy."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    config.pca.feature_selection_strategy = "invalid"

    with pytest.raises(ValueError, match="pca.feature_selection_strategy"):
        validate_qc_config(config)


def test_validate_config_valid_pca_strategies():
    """Test validation passes for all valid PCA strategies."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    # Need to configure at least one outlier detection method
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    for strategy in [
        "extreme",
        "top_absolute",
        "top_contribution",
        "top_variance",
    ]:
        config.pca.feature_selection_strategy = strategy
        validate_qc_config(config)  # Should not raise


def test_validate_config_invalid_clustering_method():
    """Test validation fails for invalid clustering method."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    config.clustering.enabled = True
    config.clustering.methods = ["invalid"]

    with pytest.raises(ValueError, match="clustering.methods"):
        validate_qc_config(config)


def test_validate_config_valid_clustering_methods():
    """Test validation passes for all valid clustering methods."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    # Need to configure at least one outlier detection method
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    for method in ["kmeans", "gmm", "hierarchical"]:
        config.clustering.method = method
        validate_qc_config(config)  # Should not raise


def test_validate_config_invalid_n_clusters():
    """Test validation fails for invalid number of clusters."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    config.clustering.enabled = True
    config.clustering.methods = ["kmeans"]
    config.clustering.n_clusters = 1

    with pytest.raises(ValueError, match="clustering.n_clusters"):
        validate_qc_config(config)


def test_validate_config_invalid_logging_level():
    """Test validation fails for invalid logging level."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    # Need to configure at least one outlier detection method
    config.outlier_detection.traditional_methods = ["mahalanobis"]
    config.logging.level = "INVALID"

    with pytest.raises(ValueError, match="logging.level"):
        validate_qc_config(config)


def test_validate_config_valid_logging_levels():
    """Test validation passes for all valid logging levels."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    # Need to configure at least one outlier detection method
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        config.logging.level = level
        validate_qc_config(config)  # Should not raise


def test_validate_config_with_group_by():
    """Test validation passes when group_by is set."""
    config = QCPipelineConfig(pipeline_name="test")
    config.data.csv_path = "data.csv"
    config.data.group_by = "plant_age_days"
    config.outlier_detection.traditional_methods = ["mahalanobis"]

    # Should not raise - group_by column existence validated at data load time
    validate_qc_config(config)
