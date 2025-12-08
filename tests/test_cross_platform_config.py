"""Tests for cross-platform analysis configuration."""

from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError


def test_cross_platform_config_valid(cross_platform_config_dict):
    """Test valid CrossPlatformConfig creation with all required fields."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    config = CrossPlatformConfig(**cross_platform_config_dict)

    assert config.exp1_data_path == "exp1_data.csv"
    assert config.exp1_name == "Cylinder"
    assert config.exp1_genotype_col == "Geno"
    assert config.exp2_data_path == "exp2_data.csv"
    assert config.exp2_name == "Turface"
    assert config.exp2_genotype_col == "geno"
    assert config.correlation_method == "spearman"
    assert config.min_samples_per_genotype == 3
    assert config.significance_level == 0.05
    assert config.top_n_correlations == 20
    assert config.top_n_joint_plots == 6
    assert config.top_n_boxplots == 6
    assert config.figsize_summary == (14, 12)
    assert config.figsize_joint == (10, 10)
    assert config.figsize_boxplot == (14, 6)


def test_cross_platform_config_defaults():
    """Test CrossPlatformConfig with only required fields uses defaults."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    config = CrossPlatformConfig(
        exp1_data_path="exp1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="geno1",
        exp2_data_path="exp2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno2",
    )

    # Check defaults
    assert config.correlation_method == "spearman"
    assert config.min_samples_per_genotype == 3
    assert config.significance_level == 0.05
    assert config.top_n_correlations == 20
    assert config.top_n_joint_plots == 6
    assert config.top_n_boxplots == 6
    assert config.figsize_summary == (14, 12)
    assert config.figsize_joint == (10, 10)
    assert config.figsize_boxplot == (14, 6)


def test_cross_platform_config_missing_exp1_path():
    """Test that missing exp1_data_path raises TypeError."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    with pytest.raises(TypeError, match="missing.*required.*exp1_data_path"):
        CrossPlatformConfig(
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
        )


def test_cross_platform_config_missing_exp1_name():
    """Test that missing exp1_name raises TypeError."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    with pytest.raises(TypeError, match="missing.*required.*exp1_name"):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
        )


def test_cross_platform_config_missing_exp1_genotype_col():
    """Test that missing exp1_genotype_col raises TypeError."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    with pytest.raises(TypeError, match="missing.*required.*exp1_genotype_col"):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
        )


def test_cross_platform_config_missing_exp2_path():
    """Test that missing exp2_data_path raises TypeError."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    with pytest.raises(TypeError, match="missing.*required.*exp2_data_path"):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
        )


def test_cross_platform_config_missing_exp2_name():
    """Test that missing exp2_name raises TypeError."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    with pytest.raises(TypeError, match="missing.*required.*exp2_name"):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_genotype_col="geno2",
        )


def test_cross_platform_config_missing_exp2_genotype_col():
    """Test that missing exp2_genotype_col raises TypeError."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    with pytest.raises(TypeError, match="missing.*required.*exp2_genotype_col"):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
        )


def test_cross_platform_config_invalid_correlation_method():
    """Test that invalid correlation_method raises ValueError."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    with pytest.raises(ValueError, match="correlation_method.*must be one of"):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
            correlation_method="invalid_method",
        )


def test_cross_platform_config_pearson_method():
    """Test CrossPlatformConfig with pearson correlation method."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    config = CrossPlatformConfig(
        exp1_data_path="exp1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="geno1",
        exp2_data_path="exp2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno2",
        correlation_method="pearson",
    )

    assert config.correlation_method == "pearson"


def test_cross_platform_config_kendall_method():
    """Test CrossPlatformConfig with kendall correlation method."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    config = CrossPlatformConfig(
        exp1_data_path="exp1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="geno1",
        exp2_data_path="exp2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno2",
        correlation_method="kendall",
    )

    assert config.correlation_method == "kendall"


def test_cross_platform_config_custom_parameters(cross_platform_config_dict):
    """Test CrossPlatformConfig with custom parameter values."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    cross_platform_config_dict["min_samples_per_genotype"] = 5
    cross_platform_config_dict["significance_level"] = 0.01
    cross_platform_config_dict["top_n_correlations"] = 50
    cross_platform_config_dict["top_n_joint_plots"] = 10
    cross_platform_config_dict["top_n_boxplots"] = 8

    config = CrossPlatformConfig(**cross_platform_config_dict)

    assert config.min_samples_per_genotype == 5
    assert config.significance_level == 0.01
    assert config.top_n_correlations == 50
    assert config.top_n_joint_plots == 10
    assert config.top_n_boxplots == 8


def test_cross_platform_config_frozen():
    """Test that CrossPlatformConfig is frozen (immutable)."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    config = CrossPlatformConfig(
        exp1_data_path="exp1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="geno1",
        exp2_data_path="exp2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno2",
    )

    with pytest.raises(FrozenInstanceError):
        config.correlation_method = "pearson"