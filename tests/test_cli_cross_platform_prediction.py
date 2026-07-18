"""CLI dry-run tests for the prediction step (Tier 3.5, #196, tasks.md Section 7)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from sleap_roots_analyze.cli import cli

HARNESS_DIR = Path(__file__).parent / "fixtures" / "harness" / "cross_platform"


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


def test_cli_cross_platform_dry_run_lists_prediction_step_when_enabled(runner):
    """--dry-run lists a 6th step when prediction.enabled=True (tasks.md 7.1)."""
    config_path = HARNESS_DIR / "cross_platform_prediction_wiring.yaml"

    result = runner.invoke(cli, ["cross-platform", str(config_path), "--dry-run"])

    assert result.exit_code == 0
    assert "PredictCrossPlatform" in result.output


def test_cli_cross_platform_dry_run_omits_prediction_step_when_disabled(runner):
    """--dry-run has exactly the existing 5 steps when disabled (tasks.md 7.2)."""
    config_path = HARNESS_DIR / "cross_platform_prediction_wiring_baseline.yaml"

    result = runner.invoke(cli, ["cross-platform", str(config_path), "--dry-run"])

    assert result.exit_code == 0
    assert "PredictCrossPlatform" not in result.output
    for step_name in (
        "LoadCrossPlatformData",
        "ReduceTraitRedundancy",
        "CalculateCrossPlatformCorrelations",
        "CalculateTraitEnrichment",
        "VisualizeCrossPlatform",
    ):
        assert step_name in result.output


# =============================================================================
# Tier 4 (add-prediction-permutation-and-figure, #200), tasks.md 8.4/8.5 --
# --dry-run listing for VisualizePredictionStep (7th step).
# =============================================================================


def test_cli_cross_platform_dry_run_lists_visualize_prediction_step_when_enabled(
    runner,
):
    """--dry-run lists a 7th step when prediction.visualize=True (tasks.md 8.4)."""
    config_path = HARNESS_DIR / "cross_platform_prediction_wiring_visualize.yaml"

    result = runner.invoke(cli, ["cross-platform", str(config_path), "--dry-run"])

    assert result.exit_code == 0
    assert "VisualizePrediction" in result.output


def test_cli_cross_platform_dry_run_omits_it_when_disabled(runner):
    """--dry-run has exactly the existing 6 steps when visualize=False (tasks.md 8.4)."""
    config_path = HARNESS_DIR / "cross_platform_prediction_wiring.yaml"

    result = runner.invoke(cli, ["cross-platform", str(config_path), "--dry-run"])

    assert result.exit_code == 0
    assert "VisualizePrediction" not in result.output
    assert "PredictCrossPlatform" in result.output
