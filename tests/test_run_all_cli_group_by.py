"""Tests for CLI --group-by flag interaction with run-all pipeline runner.

These tests verify that the CLI --group-by flag correctly triggers
viz fan-out, even when the config file has no group_by or a different one.

Bug: Viz fan-out detection only checked config.data.group_by, not the
effective group_by (CLI > config). This meant CLI-based grouping didn't
trigger viz fan-out.

Fix: Track effective group_by (CLI flag takes precedence) and use it
for viz fan-out detection.
"""

import logging
from pathlib import Path

import pytest
import yaml

from sleap_roots_analyze.pipeline.config.utils import (
    get_default_qc_config,
    get_default_viz_config,
    save_qc_config,
    save_viz_config,
)
from sleap_roots_analyze.pipeline_runner import PipelineRunner


@pytest.fixture
def test_dataset_with_groups(tmp_path):
    """Create test dataset with multiple plant_age_days groups."""
    csv_path = tmp_path / "data.csv"
    rows = ["barcode,genotype,replicate,plant_age_days,site,trait1,trait2"]

    # plant_age_days=0 (10 samples)
    for i in range(10):
        rows.append(f"p{i},A,{i % 3},0,site1,{1.0 + i * 0.1},{2.0 + i * 0.1}")

    # plant_age_days=3 (10 samples)
    for i in range(10, 20):
        rows.append(f"p{i},B,{i % 3},3,site1,{3.0 + i * 0.1},{4.0 + i * 0.1}")

    # plant_age_days=5 (10 samples)
    for i in range(20, 30):
        rows.append(f"p{i},C,{i % 3},5,site2,{5.0 + i * 0.1},{6.0 + i * 0.1}")

    csv_path.write_text("\n".join(rows))
    return csv_path


@pytest.fixture
def qc_config_no_group_by(tmp_path, test_dataset_with_groups):
    """Create QC config WITHOUT group_by field set."""
    config_path = tmp_path / "qc_config.yaml"

    config = get_default_qc_config()
    config.pipeline_name = "test_qc"
    config.data.csv_path = str(test_dataset_with_groups)
    config.data.group_by = None  # NO group_by in config
    config.columns.barcode = "barcode"
    config.columns.genotype = "genotype"
    config.columns.replicate = "replicate"
    config.cleanup.min_samples_per_trait = 5

    save_qc_config(config, config_path)
    return config_path


@pytest.fixture
def qc_config_with_site_group_by(tmp_path, test_dataset_with_groups):
    """Create QC config with group_by="site"."""
    config_path = tmp_path / "qc_config_site.yaml"

    config = get_default_qc_config()
    config.pipeline_name = "test_qc_site"
    config.data.csv_path = str(test_dataset_with_groups)
    config.data.group_by = "site"  # Config groups by site
    config.columns.barcode = "barcode"
    config.columns.genotype = "genotype"
    config.columns.replicate = "replicate"
    config.cleanup.min_samples_per_trait = 5

    save_qc_config(config, config_path)
    return config_path


@pytest.fixture
def viz_config(tmp_path):
    """Create basic viz config."""
    config_path = tmp_path / "viz_config.yaml"

    config = get_default_viz_config()
    config.pipeline_name = "test_viz"
    config.data.csv_path = "placeholder.csv"  # Will be updated by runner
    config.columns.barcode = "Barcode"
    config.columns.genotype = "Genotype"
    config.columns.replicate = "Replicate"
    config.data.image_dir = None

    save_viz_config(config, config_path)
    return config_path


@pytest.fixture
def run_manifest(tmp_path, qc_config_no_group_by, viz_config):
    """Create run manifest."""
    manifest_path = tmp_path / "manifest.yaml"

    manifest = {
        "run_name": "Test Run",
        "description": "Test CLI group-by flag",
        "qc_configs": ["qc_config.yaml"],
        "viz_configs": ["viz_config.yaml"],
        "qc_mapping": {
            "viz_config.yaml": "qc_config.yaml"
        },
    }

    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f)

    return manifest_path


class TestRunAllCLIGroupBy:
    """Test that CLI --group-by flag correctly triggers viz fan-out."""

    def test_cli_group_by_triggers_viz_fanout_when_config_has_no_group_by(
        self, tmp_path, run_manifest
    ):
        """run-all --group-by should trigger viz fan-out even when config has no group_by.

        This is the CRITICAL bug: CLI grouping should work even if config
        doesn't specify group_by.
        """
        # Run pipeline runner with CLI --group-by flag
        runner = PipelineRunner(
            manifest_path=run_manifest,
            output_dir=tmp_path / "output",
            group_by="plant_age_days",  # CLI flag (config has no group_by)
        )
        runner.run_all()

        # CRITICAL: Verify viz ran for EACH group (not just once)
        viz_results = runner.run_results["viz"]

        # Should have 3 viz results (one per plant_age_days group)
        # Result keys should be "viz_config.yaml:plant_age_days_0", etc.
        expected_groups = ["plant_age_days_0", "plant_age_days_3", "plant_age_days_5"]

        for group in expected_groups:
            result_key = f"viz_config.yaml:{group}"
            assert result_key in viz_results, (
                f"Viz fan-out failed for group {group}. "
                f"CLI --group-by flag should trigger fan-out even when "
                f"config has no group_by.\n"
                f"Found viz result keys: {list(viz_results.keys())}"
            )

        # Should NOT have a non-grouped viz result
        assert "viz_config.yaml" not in viz_results, (
            "Should not have non-grouped viz result when using CLI --group-by"
        )

    def test_cli_group_by_overrides_config_group_by_for_viz_fanout(
        self, tmp_path, qc_config_with_site_group_by, viz_config
    ):
        """CLI --group-by should override config group_by for viz fan-out detection.

        If config says group_by="site" but CLI says --group-by plant_age_days,
        viz should fan out by plant_age_days (not site).
        """
        # Create manifest with site-grouped QC config
        manifest_path = tmp_path / "manifest_site.yaml"
        manifest = {
            "run_name": "Test Override",
            "qc_configs": ["qc_config_site.yaml"],
            "viz_configs": ["viz_config.yaml"],
            "qc_mapping": {"viz_config.yaml": "qc_config_site.yaml"},
        }
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f)

        # Run with CLI --group-by plant_age_days (overrides config's "site")
        runner = PipelineRunner(
            manifest_path=manifest_path,
            output_dir=tmp_path / "output",
            group_by="plant_age_days",  # CLI overrides config
        )
        runner.run_all()

        # Viz fan-out should detect "plant_age_days" groups (from CLI)
        # NOT "site" groups (from config)
        viz_results = runner.run_results["viz"]

        # Should have plant_age_days groups
        assert any(
            "plant_age_days_" in key for key in viz_results.keys()
        ), (
            "Viz should fan out by plant_age_days (CLI flag), "
            f"not site (config). Found: {list(viz_results.keys())}"
        )

        # Should NOT have site groups
        assert not any(
            "site" in key and "plant_age_days" not in key
            for key in viz_results.keys()
        ), (
            "Should not have site-based groups when CLI overrides with "
            "plant_age_days"
        )

    def test_effective_group_by_logged_correctly(
        self, tmp_path, run_manifest, caplog
    ):
        """Runner should log CLI/config/effective group_by for transparency."""
        with caplog.at_level(logging.INFO):
            runner = PipelineRunner(
                manifest_path=run_manifest,
                output_dir=tmp_path / "output",
                group_by="plant_age_days",  # CLI flag
            )
            runner.run_all()

        # Should log effective group_by during QC execution
        log_messages = [record.message for record in caplog.records]

        # Look for log showing CLI/config/effective group_by
        found_effective_log = any(
            "effective=" in msg and "plant_age_days" in msg
            for msg in log_messages
        )

        assert found_effective_log, (
            "Should log effective group_by (CLI > config) for transparency.\n"
            f"Log messages: {log_messages}"
        )

    def test_viz_fanout_creates_per_group_directories(
        self, tmp_path, run_manifest
    ):
        """Viz fan-out should create separate output directories per group."""
        runner = PipelineRunner(
            manifest_path=run_manifest,
            output_dir=tmp_path / "output",
            group_by="plant_age_days",
        )
        runner.run_all()

        # Verify that viz ran for all 3 groups by checking run_results
        viz_results = runner.run_results["viz"]

        # Should have 3 viz results (one per group)
        expected_groups = ["plant_age_days_0", "plant_age_days_3", "plant_age_days_5"]

        for group in expected_groups:
            result_key = f"viz_config.yaml:{group}"
            assert result_key in viz_results, (
                f"Viz should have run for group {group}"
            )

            # Verify the viz run was successful
            result = viz_results[result_key]
            assert result["success"], (
                f"Viz for group {group} should have completed successfully"
            )

        # Also verify viz output directory exists
        viz_base = runner.run_dir / "viz"
        assert viz_base.exists(), "Viz output directory should exist"
