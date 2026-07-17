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


def test_cross_platform_config_validate_input_defaults_to_warn():
    """validate_input defaults to 'warn' on CrossPlatformConfig (issue #154)."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    config = CrossPlatformConfig(
        exp1_data_path="exp1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="geno1",
        exp2_data_path="exp2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno2",
    )
    assert config.validate_input == "warn"


def test_cross_platform_config_invalid_validate_input():
    """Invalid validate_input raises ValueError (issue #154)."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    with pytest.raises(ValueError, match=r"validate_input.*off \| warn \| strict"):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
            validate_input="lenient",
        )


def test_cross_platform_config_valid_validate_input_modes():
    """Each valid validate_input mode is accepted (issue #154)."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    for mode in ["off", "warn", "strict"]:
        config = CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
            validate_input=mode,
        )
        assert config.validate_input == mode


def test_load_cross_platform_config_yaml_round_trips_validate_input(tmp_path):
    """The YAML load path honors validate_input and runs __post_init__ (issue #154)."""
    from sleap_roots_analyze.pipeline.config.utils import load_cross_platform_config

    yaml_path = tmp_path / "xp.yaml"
    yaml_path.write_text(
        "exp1_data_path: e1.csv\n"
        "exp1_name: Exp1\n"
        "exp1_genotype_col: geno1\n"
        "exp2_data_path: e2.csv\n"
        "exp2_name: Exp2\n"
        "exp2_genotype_col: geno2\n"
        "validate_input: strict\n"
    )
    config = load_cross_platform_config(yaml_path)
    assert config.validate_input == "strict"


def test_load_cross_platform_config_yaml_rejects_invalid_validate_input(tmp_path):
    """An invalid validate_input in YAML is rejected at load (via __post_init__) (#154)."""
    from sleap_roots_analyze.pipeline.config.utils import load_cross_platform_config

    yaml_path = tmp_path / "xp_bad.yaml"
    yaml_path.write_text(
        "exp1_data_path: e1.csv\n"
        "exp1_name: Exp1\n"
        "exp1_genotype_col: geno1\n"
        "exp2_data_path: e2.csv\n"
        "exp2_name: Exp2\n"
        "exp2_genotype_col: geno2\n"
        "validate_input: lenient\n"
    )
    with pytest.raises(ValueError, match=r"validate_input.*off \| warn \| strict"):
        load_cross_platform_config(yaml_path)


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


# =============================================================================
# trait_reduction_target validation tests
# =============================================================================


def test_config_requires_target_when_clustering_enabled():
    """Test that trait_reduction_target is required when clustering is enabled."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    with pytest.raises(
        ValueError,
        match="trait_reduction_target must be specified when trait_reduction_method is 'clustering'",
    ):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
            trait_reduction_method="clustering",
            # trait_reduction_target not specified - should fail
        )


def test_config_accepts_valid_reduction_targets():
    """Test that all valid trait_reduction_target values are accepted."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    for target in ["exp1", "exp2", "both"]:
        config = CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
            trait_reduction_method="clustering",
            trait_reduction_target=target,
        )
        assert config.trait_reduction_target == target


def test_config_allows_no_target_when_clustering_disabled():
    """Test that trait_reduction_target is not required when clustering is disabled."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    # Should succeed without trait_reduction_target when method is "none"
    config = CrossPlatformConfig(
        exp1_data_path="exp1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="geno1",
        exp2_data_path="exp2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno2",
        trait_reduction_method="none",
        # No trait_reduction_target - should succeed
    )
    assert config.trait_reduction_method == "none"
    assert config.trait_reduction_target is None


def test_config_rejects_invalid_reduction_target():
    """Test that invalid trait_reduction_target values are rejected."""
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    with pytest.raises(
        ValueError, match="trait_reduction_target must be one of.*got 'invalid'"
    ):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
            trait_reduction_method="clustering",
            trait_reduction_target="invalid",
        )


# =============================================================================
# PredictionConfig tests (Tier 3.5, add-prediction-pipeline-step, #196)
# tasks.md Section 2 -- PredictionConfig standalone (no CrossPlatformConfig
# nesting assertions here; those live solely in Section 3, task 3.1, per
# design.md's round-1 reconciliation).
# =============================================================================


def test_prediction_config_defaults_to_disabled():
    """PredictionConfig() defaults to enabled=False (tasks.md 2.1)."""
    from sleap_roots_analyze.pipeline.config.components import PredictionConfig

    config = PredictionConfig()
    assert config.enabled is False


def test_prediction_config_validation_skipped_when_disabled():
    """No validation runs at all when enabled=False (Decision 4, tasks.md 2.2)."""
    from sleap_roots_analyze.pipeline.config.components import PredictionConfig

    # Every field below is individually invalid; none of it should raise
    # because enabled=False short-circuits __post_init__ entirely.
    config = PredictionConfig(
        enabled=False,
        predictor_source="not_a_real_value",
        source_blup_path="/does/not/exist",
    )
    assert config.predictor_source == "not_a_real_value"


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("predictor_source", "not_a_real_value"),
        ("reduction_method", "not_a_real_value"),
        ("representative_selection_metric", "heritability"),
        ("representative_selection_metric", "not_a_real_value"),
    ],
)
def test_prediction_config_rejects_invalid_enum_fields(field_name, invalid_value):
    """Invalid enum field values raise ValueError naming the field (tasks.md 2.3)."""
    from sleap_roots_analyze.pipeline.config.components import PredictionConfig

    kwargs = {
        "enabled": True,
        "predictor_source": "genotype_means",
        field_name: invalid_value,
    }
    with pytest.raises(ValueError, match=field_name):
        PredictionConfig(**kwargs)


def test_prediction_config_rejects_invalid_comparison_methods_entry():
    """An invalid entry inside comparison_methods raises ValueError (tasks.md 2.3)."""
    from sleap_roots_analyze.pipeline.config.components import PredictionConfig

    with pytest.raises(ValueError, match="comparison_methods"):
        PredictionConfig(
            enabled=True,
            predictor_source="genotype_means",
            reduction_method="pls_latent",
            comparison_methods=["not_a_real_value"],
        )


def test_prediction_config_rejects_duplicate_method_in_comparison_methods():
    """comparison_methods duplicating reduction_method raises ValueError (tasks.md 2.4)."""
    from sleap_roots_analyze.pipeline.config.components import PredictionConfig

    with pytest.raises(ValueError, match="comparison_methods"):
        PredictionConfig(
            enabled=True,
            predictor_source="genotype_means",
            reduction_method="pls_latent",
            comparison_methods=["pls_latent"],
        )


def test_prediction_config_rejects_duplicate_entries_within_comparison_methods():
    """A method listed twice within comparison_methods itself raises (tasks.md 2.4a)."""
    from sleap_roots_analyze.pipeline.config.components import PredictionConfig

    with pytest.raises(ValueError, match="comparison_methods"):
        PredictionConfig(
            enabled=True,
            predictor_source="genotype_means",
            reduction_method="pls_latent",
            comparison_methods=["representatives", "representatives"],
        )


def test_prediction_config_blup_preflight_check_missing_path(tmp_path):
    """A nonexistent source_blup_path/target_blup_path raises at construction (tasks.md 2.5)."""
    from sleap_roots_analyze.pipeline.config.components import PredictionConfig

    existing = tmp_path / "exists.csv"
    existing.write_text("Genotype,trait_a\nG01,1.0\n")

    with pytest.raises(ValueError, match="source_blup_path"):
        PredictionConfig(
            enabled=True,
            predictor_source="blup",
            source_blup_path=str(tmp_path / "does_not_exist.csv"),
            target_blup_path=str(existing),
        )

    with pytest.raises(ValueError, match="target_blup_path"):
        PredictionConfig(
            enabled=True,
            predictor_source="blup",
            source_blup_path=str(existing),
            target_blup_path=str(tmp_path / "does_not_exist.csv"),
        )


def test_prediction_config_genotype_means_does_not_require_blup_paths():
    """predictor_source=genotype_means needs no BLUP paths (Decision 2, tasks.md 2.6)."""
    from sleap_roots_analyze.pipeline.config.components import PredictionConfig

    config = PredictionConfig(
        enabled=True,
        predictor_source="genotype_means",
        source_blup_path=None,
        target_blup_path=None,
    )
    assert config.source_blup_path is None
    assert config.target_blup_path is None


# =============================================================================
# CrossPlatformConfig <-> PredictionConfig wiring tests (tasks.md Section 3)
# =============================================================================


def test_cross_platform_config_gains_prediction_field():
    """CrossPlatformConfig().prediction is a default PredictionConfig (tasks.md 3.1)."""
    from sleap_roots_analyze.pipeline.config.components import (
        CrossPlatformConfig,
        PredictionConfig,
    )

    config = CrossPlatformConfig(
        exp1_data_path="exp1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="geno1",
        exp2_data_path="exp2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno2",
    )
    assert isinstance(config.prediction, PredictionConfig)
    assert config.prediction == PredictionConfig()


def test_cross_platform_config_validates_platform_pairs_direction_against_exp_names():
    """A platform_pairs entry not matching exp1_name/exp2_name raises (tasks.md 3.2)."""
    from sleap_roots_analyze.pipeline.config.components import (
        CrossPlatformConfig,
        PredictionConfig,
    )

    with pytest.raises(ValueError, match="platform_pairs"):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
            prediction=PredictionConfig(
                enabled=True,
                predictor_source="genotype_means",
                platform_pairs=[{"source": "not_exp1_or_exp2", "target": "also_not"}],
            ),
        )


@pytest.mark.parametrize(
    "source_name,target_name",
    [("Exp1", "Exp2"), ("Exp2", "Exp1")],
)
def test_cross_platform_config_accepts_valid_platform_pairs_direction(
    source_name, target_name
):
    """platform_pairs direction is accepted in either order (tasks.md 3.3)."""
    from sleap_roots_analyze.pipeline.config.components import (
        CrossPlatformConfig,
        PredictionConfig,
    )

    config = CrossPlatformConfig(
        exp1_data_path="exp1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="geno1",
        exp2_data_path="exp2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno2",
        prediction=PredictionConfig(
            enabled=True,
            predictor_source="genotype_means",
            platform_pairs=[{"source": source_name, "target": target_name}],
        ),
    )
    assert config.prediction.platform_pairs == [
        {"source": source_name, "target": target_name}
    ]


def test_cross_platform_config_rejects_empty_platform_pairs_when_enabled():
    """prediction.enabled=True with the default empty platform_pairs raises (tasks.md 3.3a)."""
    from sleap_roots_analyze.pipeline.config.components import (
        CrossPlatformConfig,
        PredictionConfig,
    )

    with pytest.raises(ValueError, match="platform_pairs"):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
            prediction=PredictionConfig(
                enabled=True,
                predictor_source="genotype_means",
                platform_pairs=[],
            ),
        )


def test_cross_platform_config_rejects_multiple_platform_pairs_entries():
    """prediction.enabled=True with 2 platform_pairs entries raises (tasks.md 3.3b)."""
    from sleap_roots_analyze.pipeline.config.components import (
        CrossPlatformConfig,
        PredictionConfig,
    )

    with pytest.raises(ValueError, match="platform_pairs"):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
            prediction=PredictionConfig(
                enabled=True,
                predictor_source="genotype_means",
                platform_pairs=[
                    {"source": "Exp1", "target": "Exp2"},
                    {"source": "Exp2", "target": "Exp1"},
                ],
            ),
        )


def test_cross_platform_config_rejects_non_dict_platform_pairs_entry():
    """A single platform_pairs entry that isn't a dict raises ValueError, not AttributeError.

    Found during code review: `platform_pairs: ["Cylinder"]` (a plausible
    YAML-authoring mistake -- a bare string instead of a
    {source, target} mapping) passed the cardinality check (exactly one
    entry) and then crashed with an unhelpful `AttributeError` from
    `pair.get("source")`, rather than the clean `ValueError` every other
    validation failure in this config raises.
    """
    from sleap_roots_analyze.pipeline.config.components import (
        CrossPlatformConfig,
        PredictionConfig,
    )

    with pytest.raises(ValueError, match="platform_pairs"):
        CrossPlatformConfig(
            exp1_data_path="exp1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="geno1",
            exp2_data_path="exp2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno2",
            prediction=PredictionConfig(
                enabled=True,
                predictor_source="genotype_means",
                platform_pairs=["Exp1"],
            ),
        )
