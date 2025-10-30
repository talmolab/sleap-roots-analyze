"""Tests for visualization pipeline configuration."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from pathlib import Path

from sleap_roots_analyze.pipeline.config import (
    AdaptiveSizingConfig,
    ClusteringConfig,
    ColumnConfig,
    DataConfig,
    HeritabilityConfig,
    InterestingGenotypesConfig,
    PCAConfig,
    StatisticsConfig,
    UMAPConfig,
    VizPipelineConfig,
    load_viz_config,
    save_viz_config,
    validate_viz_config,
)


class TestColumnConfig:
    """Tests for ColumnConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ColumnConfig()
        assert config.barcode == "Barcode"
        assert config.genotype == "geno"
        assert config.replicate == "rep"
        assert config.image_path == "image_path"

    def test_custom_values(self):
        """Test custom column names."""
        config = ColumnConfig(
            barcode="ID",
            genotype="variety",
            replicate="block",
            image_path="img_file",
        )
        assert config.barcode == "ID"
        assert config.genotype == "variety"
        assert config.replicate == "block"
        assert config.image_path == "img_file"


class TestDataConfig:
    """Tests for DataConfig."""

    def test_csv_path_required(self):
        """Test that csv_path is required (MISSING by default)."""
        from omegaconf import MISSING

        config = DataConfig()
        assert config.csv_path == MISSING

    def test_optional_fields(self):
        """Test optional fields have correct defaults."""
        config = DataConfig()
        assert config.image_dir is None
        assert config.traits_to_include is None
        assert config.traits_to_exclude == []

    def test_custom_values(self):
        """Test setting custom values."""
        config = DataConfig(
            csv_path="data.csv",
            image_dir="./images",
            traits_to_include=["trait1", "trait2"],
            traits_to_exclude=["bad_trait"],
        )
        assert config.csv_path == "data.csv"
        assert config.image_dir == "./images"
        assert config.traits_to_include == ["trait1", "trait2"]
        assert config.traits_to_exclude == ["bad_trait"]


class TestPCAConfig:
    """Tests for PCAConfig."""

    def test_defaults(self):
        """Test default PCA configuration."""
        config = PCAConfig()
        assert config.n_components == 0.95
        assert config.standardize is True
        assert config.feature_selection_strategy == "top_variance"
        assert config.n_top_features == 10

    def test_custom_values(self):
        """Test custom PCA configuration."""
        config = PCAConfig(
            n_components=5,
            standardize=False,
            feature_selection_strategy="extreme",
            n_top_features=20,
        )
        assert config.n_components == 5
        assert config.standardize is False
        assert config.feature_selection_strategy == "extreme"
        assert config.n_top_features == 20


class TestAdaptiveSizingConfig:
    """Tests for AdaptiveSizingConfig."""

    def test_defaults(self):
        """Test default adaptive sizing configuration."""
        config = AdaptiveSizingConfig()
        assert config.enabled is True
        assert config.base_width == 8.0
        assert config.base_height == 6.0
        assert config.min_width == 6.0
        assert config.max_width == 20.0

    def test_disabled(self):
        """Test with sizing disabled."""
        config = AdaptiveSizingConfig(enabled=False)
        assert config.enabled is False


class TestVizPipelineConfig:
    """Tests for VizPipelineConfig."""

    def test_pipeline_name_required(self):
        """Test that pipeline_name is required."""
        from omegaconf import MISSING

        config = VizPipelineConfig()
        assert config.pipeline_name == MISSING

    def test_nested_configs_have_defaults(self):
        """Test that nested configs are created with defaults."""
        config = VizPipelineConfig(pipeline_name="test")
        assert isinstance(config.columns, ColumnConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.pca, PCAConfig)
        assert isinstance(config.adaptive_sizing, AdaptiveSizingConfig)

    def test_full_config_creation(self):
        """Test creating a complete configuration."""
        config = VizPipelineConfig(pipeline_name="test_pipeline", version="2.0")
        config.data.csv_path = "test.csv"
        config.pca.n_components = 10
        config.umap.enabled = True

        assert config.pipeline_name == "test_pipeline"
        assert config.version == "2.0"
        assert config.data.csv_path == "test.csv"
        assert config.pca.n_components == 10
        assert config.umap.enabled is True


class TestValidateVizConfig:
    """Tests for validate_viz_config function."""

    def test_valid_minimal_config(self, viz_config_minimal):
        """Test validation passes for valid minimal config."""
        # Should not raise
        validate_viz_config(viz_config_minimal)

    def test_missing_pipeline_name_raises(self):
        """Test that missing pipeline_name raises error."""
        config = VizPipelineConfig()
        config.data.csv_path = "test.csv"
        with pytest.raises(ValueError, match="pipeline_name is required"):
            validate_viz_config(config)

    def test_missing_csv_path_raises(self):
        """Test that missing csv_path raises error."""
        config = VizPipelineConfig(pipeline_name="test")
        with pytest.raises(ValueError, match="csv_path is required"):
            validate_viz_config(config)

    def test_invalid_pca_n_components_raises(self):
        """Test that invalid n_components raises error."""
        config = VizPipelineConfig(pipeline_name="test")
        config.data.csv_path = "test.csv"
        config.pca.n_components = 0  # Invalid
        with pytest.raises(ValueError, match="n_components must be positive"):
            validate_viz_config(config)

    def test_invalid_pca_strategy_raises(self):
        """Test that invalid feature selection strategy raises error."""
        config = VizPipelineConfig(pipeline_name="test")
        config.data.csv_path = "test.csv"
        config.pca.feature_selection_strategy = "invalid_strategy"
        with pytest.raises(
            ValueError, match="feature_selection_strategy must be one of"
        ):
            validate_viz_config(config)

    def test_invalid_clustering_method_raises(self):
        """Test that invalid clustering method raises error."""
        config = VizPipelineConfig(pipeline_name="test")
        config.data.csv_path = "test.csv"
        config.clustering.enabled = True
        config.clustering.methods = ["invalid_method"]
        with pytest.raises(ValueError, match="contains invalid method"):
            validate_viz_config(config)

    def test_invalid_clustering_n_clusters_raises(self):
        """Test that invalid n_clusters raises error."""
        config = VizPipelineConfig(pipeline_name="test")
        config.data.csv_path = "test.csv"
        config.clustering.enabled = True
        config.clustering.auto_optimize = False
        config.clustering.n_clusters = 1  # Too few
        with pytest.raises(ValueError, match="n_clusters must be at least 2"):
            validate_viz_config(config)

    def test_invalid_genotype_method_raises(self):
        """Test that invalid genotype identification method raises error."""
        config = VizPipelineConfig(pipeline_name="test")
        config.data.csv_path = "test.csv"
        config.interesting_genotypes.enabled = True
        config.interesting_genotypes.methods = ["invalid_method"]
        with pytest.raises(ValueError, match="contains invalid method"):
            validate_viz_config(config)

    def test_invalid_static_viz_format_raises(self):
        """Test that invalid static viz format raises error."""
        config = VizPipelineConfig(pipeline_name="test")
        config.data.csv_path = "test.csv"
        config.static_viz.formats = ["invalid_format"]
        with pytest.raises(ValueError, match="contains invalid format"):
            validate_viz_config(config)

    def test_invalid_summary_format_raises(self):
        """Test that invalid summary format raises error."""
        config = VizPipelineConfig(pipeline_name="test")
        config.data.csv_path = "test.csv"
        config.summary.formats = ["invalid_format"]
        with pytest.raises(ValueError, match="contains invalid format"):
            validate_viz_config(config)


class TestLoadVizConfig:
    """Tests for load_viz_config function."""

    def test_load_viz_standard_preset(self):
        """Test loading viz_standard.yaml preset."""
        config_path = Path("configs/viz_standard.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        # Load as OmegaConf first
        omega_conf = OmegaConf.load(config_path)
        assert omega_conf.pipeline_name == "viz_standard"
        assert omega_conf.pca.n_components == 0.95

    def test_load_viz_minimal_preset(self):
        """Test loading viz_minimal.yaml preset."""
        config_path = Path("configs/viz_minimal.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        omega_conf = OmegaConf.load(config_path)
        assert omega_conf.pipeline_name == "viz_minimal"
        assert omega_conf.statistics.calculate_heritability is False

    def test_load_viz_comprehensive_preset(self):
        """Test loading viz_comprehensive.yaml preset."""
        config_path = Path("configs/viz_comprehensive.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        omega_conf = OmegaConf.load(config_path)
        assert omega_conf.pipeline_name == "viz_comprehensive"
        assert omega_conf.umap.enabled is True
        assert omega_conf.clustering.enabled is True

    def test_load_viz_publication_preset(self):
        """Test loading viz_publication.yaml preset."""
        config_path = Path("configs/viz_publication.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        omega_conf = OmegaConf.load(config_path)
        assert omega_conf.pipeline_name == "viz_publication"
        assert omega_conf.static_viz.dpi == 600
        assert "svg" in omega_conf.static_viz.formats


class TestSaveVizConfig:
    """Tests for save_viz_config function."""

    def test_save_and_load_config(self, viz_config_minimal, tmp_path):
        """Test saving and loading configuration."""
        config_path = tmp_path / "test_config.yaml"

        # Save
        save_viz_config(viz_config_minimal, config_path)
        assert config_path.exists()

        # Load back
        loaded_omega = OmegaConf.load(config_path)
        assert loaded_omega.pipeline_name == "test_viz_minimal"
        assert loaded_omega.data.csv_path == "dummy.csv"

    def test_save_creates_parent_directories(self, viz_config_minimal, tmp_path):
        """Test that save_viz_config creates parent directories."""
        nested_path = tmp_path / "nested" / "dir" / "config.yaml"

        save_viz_config(viz_config_minimal, nested_path)
        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_roundtrip_preserves_values(self, viz_config_with_stats, tmp_path):
        """Test that save/load preserves all values."""
        config_path = tmp_path / "roundtrip.yaml"

        # Save
        save_viz_config(viz_config_with_stats, config_path)

        # Load back
        loaded_omega = OmegaConf.load(config_path)

        # Check key values preserved
        assert loaded_omega.pipeline_name == "test_viz_stats"
        assert loaded_omega.statistics.calculate_anova is True
        assert loaded_omega.statistics.calculate_heritability is True
        assert loaded_omega.umap.enabled is False


class TestConfigIntegration:
    """Integration tests for config system."""

    def test_config_modification_workflow(self, tmp_path):
        """Test typical workflow of loading preset and modifying."""
        # Start with minimal config
        config = VizPipelineConfig(pipeline_name="custom_viz")
        config.data.csv_path = "my_data.csv"
        config.data.image_dir = "./images"

        # Enable some features
        config.statistics.calculate_anova = True
        config.statistics.calculate_heritability = True
        config.pca.n_components = 0.90
        config.interesting_genotypes.enabled = True

        # Should be valid
        validate_viz_config(config)

        # Save and load back
        config_path = tmp_path / "custom.yaml"
        save_viz_config(config, config_path)

        loaded = OmegaConf.load(config_path)
        assert loaded.pipeline_name == "custom_viz"
        assert loaded.data.csv_path == "my_data.csv"
        assert loaded.pca.n_components == 0.90

    def test_preset_override_pattern(self, tmp_path):
        """Test pattern of loading preset and overriding specific values."""
        # Create base config
        base_config = VizPipelineConfig(pipeline_name="base")
        base_config.data.csv_path = "base.csv"
        base_config.pca.n_components = 0.95

        # Save base
        base_path = tmp_path / "base.yaml"
        save_viz_config(base_config, base_path)

        # Load and override
        loaded_omega = OmegaConf.load(base_path)
        loaded_omega.data.csv_path = "override.csv"
        loaded_omega.pca.n_components = 10

        # Convert to structured config
        structured = OmegaConf.structured(VizPipelineConfig)
        merged = OmegaConf.merge(structured, loaded_omega)
        final_config = OmegaConf.to_object(merged)

        assert final_config.data.csv_path == "override.csv"
        assert final_config.pca.n_components == 10
