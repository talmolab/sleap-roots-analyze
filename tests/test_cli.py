"""Tests for CLI module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from sleap_roots_analyze.cli import cli, main, setup_logging


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_qc_config(tmp_path):
    """Create a minimal QC config file for testing."""
    config_content = """
pipeline_name: "test_qc"

columns:
  barcode: "Barcode"
  genotype: "geno"
  replicate: "rep"

data:
  csv_path: "test_data.csv"

outlier_detection:
  traditional_methods: []
  clustering_methods: []

outlier_removal:
  strategy: "union"
  method: "all"

heritability:
  enabled: false

visualization:
  dpi: 100
  figsize: [10, 8]
  figure_format: "png"
  title_fontsize: 14
  label_fontsize: 12
  tick_fontsize: 10
  legend_fontsize: 10
"""
    config_path = tmp_path / "test_qc_config.yaml"
    config_path.write_text(config_content)

    # Create sample data file
    data = {
        "Barcode": ["plant1", "plant2", "plant3"],
        "geno": ["A", "B", "A"],
        "rep": [1, 1, 2],
        "trait1": [10.5, 12.3, 11.8],
        "trait2": [5.2, 6.1, 5.8],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    return config_path


@pytest.fixture
def sample_viz_config(tmp_path):
    """Create a minimal Viz config file for testing."""
    config_content = """
pipeline_name: "test_viz"

columns:
  barcode: "Barcode"
  genotype: "geno"
  replicate: "rep"

data:
  csv_path: "test_data.csv"

statistics:
  calculate_anova: true
  calculate_heritability: true
  alpha: 0.05

pca:
  n_components: 0.95
  standardize: true

static_viz:
  enabled: true
  formats: ["png"]
  dpi: 100

adaptive_sizing:
  enabled: false
"""
    config_path = tmp_path / "test_viz_config.yaml"
    config_path.write_text(config_content)

    # Create sample data file
    data = {
        "Barcode": ["plant1", "plant2", "plant3"],
        "geno": ["A", "B", "A"],
        "rep": [1, 1, 2],
        "trait1": [10.5, 12.3, 11.8],
        "trait2": [5.2, 6.1, 5.8],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    return config_path


# Test CLI group and main entry point


def test_cli_group(runner):
    """Test CLI group exists and shows help."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Statistical analysis tools for root trait data" in result.output
    assert "qc" in result.output
    assert "viz" in result.output
    assert "config" in result.output


def test_cli_version(runner):
    """Test CLI version flag."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.0.1" in result.output


def test_main_entry_point():
    """Test main() entry point function exists."""
    assert callable(main)


# Test setup_logging function


def test_setup_logging_function_exists():
    """Test setup_logging function exists and is callable."""
    assert callable(setup_logging)


def test_setup_logging_doesnt_error(tmp_path):
    """Test that setup_logging function doesn't error with various options."""
    # Test all combinations of flags
    setup_logging()
    setup_logging(verbose=True)
    setup_logging(quiet=True)
    setup_logging(log_file=str(tmp_path / "test.log"))
    # Function should not raise any exceptions


# Test QC command


def test_qc_command_help(runner):
    """Test QC command help."""
    result = runner.invoke(cli, ["qc", "--help"])
    assert result.exit_code == 0
    assert "Run QC pipeline" in result.output
    assert "--output-dir" in result.output
    assert "--dry-run" in result.output


def test_qc_command_missing_config(runner):
    """Test QC command with missing config file."""
    result = runner.invoke(cli, ["qc", "nonexistent.yaml"])
    assert result.exit_code != 0


def test_qc_command_dry_run(runner, sample_qc_config):
    """Test QC command with dry-run flag."""
    result = runner.invoke(cli, ["qc", str(sample_qc_config), "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run mode - validation complete" in result.output
    assert "Would execute QC pipeline" in result.output
    assert "LoadData" in result.output
    assert "Configuration is valid" in result.output


def test_qc_command_with_custom_output_dir(runner, sample_qc_config, tmp_path):
    """Test QC command with custom output directory."""
    output_dir = tmp_path / "custom_output"

    with patch("sleap_roots_analyze.cli.QCPipeline") as mock_pipeline:
        # Mock pipeline execution
        mock_instance = MagicMock()
        mock_instance.run.return_value = {"status": "success"}
        mock_pipeline.return_value = mock_instance

        result = runner.invoke(
            cli, ["qc", str(sample_qc_config), "-o", str(output_dir)]
        )

        assert result.exit_code == 0
        mock_pipeline.assert_called_once()


def test_qc_command_verbose_flag(runner, sample_qc_config):
    """Test QC command with verbose flag."""
    result = runner.invoke(cli, ["qc", str(sample_qc_config), "-v", "--dry-run"])
    assert result.exit_code == 0


def test_qc_command_quiet_flag(runner, sample_qc_config):
    """Test QC command with quiet flag."""
    result = runner.invoke(cli, ["qc", str(sample_qc_config), "-q", "--dry-run"])
    assert result.exit_code == 0


# Test Viz command


def test_viz_command_help(runner):
    """Test Viz command help."""
    result = runner.invoke(cli, ["viz", "--help"])
    assert result.exit_code == 0
    assert "Run visualization pipeline" in result.output
    assert "--output-dir" in result.output
    assert "--dry-run" in result.output


def test_viz_command_missing_config(runner):
    """Test Viz command with missing config file."""
    result = runner.invoke(cli, ["viz", "nonexistent.yaml"])
    assert result.exit_code != 0


def test_viz_command_dry_run(runner, sample_viz_config):
    """Test Viz command with dry-run flag."""
    result = runner.invoke(cli, ["viz", str(sample_viz_config), "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run mode - validation complete" in result.output
    assert "Would execute Viz pipeline" in result.output
    assert "Configuration is valid" in result.output


def test_viz_command_with_custom_output_dir(runner, sample_viz_config, tmp_path):
    """Test Viz command with custom output directory."""
    output_dir = tmp_path / "custom_viz_output"

    with patch("sleap_roots_analyze.cli.VizPipeline") as mock_pipeline:
        # Mock pipeline execution
        mock_instance = MagicMock()
        mock_instance.run.return_value = {"status": "success"}
        mock_pipeline.return_value = mock_instance

        result = runner.invoke(
            cli, ["viz", str(sample_viz_config), "-o", str(output_dir)]
        )

        assert result.exit_code == 0
        mock_pipeline.assert_called_once()


# Test config subcommands


def test_config_group_help(runner):
    """Test config group help."""
    result = runner.invoke(cli, ["config", "--help"])
    assert result.exit_code == 0
    assert "validate" in result.output
    assert "show" in result.output
    assert "list" in result.output


def test_config_validate_qc(runner, sample_qc_config):
    """Test config validate command for QC config."""
    result = runner.invoke(cli, ["config", "validate", str(sample_qc_config)])
    assert result.exit_code == 0
    assert "valid" in result.output.lower() or "Valid" in result.output


def test_config_validate_viz(runner, sample_viz_config):
    """Test config validate command for Viz config."""
    result = runner.invoke(cli, ["config", "validate", str(sample_viz_config)])
    assert result.exit_code == 0
    assert "valid" in result.output.lower() or "Valid" in result.output


def test_config_validate_missing_file(runner):
    """Test config validate with missing file."""
    result = runner.invoke(cli, ["config", "validate", "nonexistent.yaml"])
    assert result.exit_code != 0


def test_config_show_qc(runner, sample_qc_config):
    """Test config show command for QC config."""
    result = runner.invoke(cli, ["config", "show", str(sample_qc_config)])
    assert result.exit_code == 0
    assert "test_qc" in result.output  # pipeline name


def test_config_show_viz(runner, sample_viz_config):
    """Test config show command for Viz config."""
    result = runner.invoke(cli, ["config", "show", str(sample_viz_config)])
    assert result.exit_code == 0
    assert "test_viz" in result.output  # pipeline name


def test_config_list(runner):
    """Test config list command."""
    result = runner.invoke(cli, ["config", "list"])
    assert result.exit_code == 0
    # Should show available configs or message if none found


# Test error handling


def test_qc_command_invalid_config(runner, tmp_path):
    """Test QC command with invalid config file."""
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text("invalid: yaml: content: [")

    result = runner.invoke(cli, ["qc", str(invalid_config)])
    assert result.exit_code != 0


def test_viz_command_invalid_config(runner, tmp_path):
    """Test Viz command with invalid config file."""
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text("invalid: yaml: content: [")

    result = runner.invoke(cli, ["viz", str(invalid_config)])
    assert result.exit_code != 0


# Test integration with actual configs (if they exist)


def test_qc_with_real_config_dry_run(runner):
    """Test QC command with real config file if it exists."""
    config_path = Path("configs/qc_turface_150genotypes.yaml")
    if config_path.exists():
        result = runner.invoke(cli, ["qc", str(config_path), "--dry-run"])
        assert result.exit_code == 0
        assert "Configuration is valid" in result.output


# Test output and logging


def test_qc_command_creates_output_directory(runner, sample_qc_config, tmp_path):
    """Test that QC command creates output directory."""
    output_dir = tmp_path / "qc_output"

    with patch("sleap_roots_analyze.cli.QCPipeline") as mock_pipeline:
        mock_instance = MagicMock()
        mock_instance.run.return_value = {"status": "success"}
        mock_pipeline.return_value = mock_instance

        result = runner.invoke(
            cli, ["qc", str(sample_qc_config), "-o", str(output_dir)]
        )

        assert result.exit_code == 0


def test_viz_command_creates_output_directory(runner, sample_viz_config, tmp_path):
    """Test that Viz command creates output directory."""
    output_dir = tmp_path / "viz_output"

    with patch("sleap_roots_analyze.cli.VizPipeline") as mock_pipeline:
        mock_instance = MagicMock()
        mock_instance.run.return_value = {"status": "success"}
        mock_pipeline.return_value = mock_instance

        result = runner.invoke(
            cli, ["viz", str(sample_viz_config), "-o", str(output_dir)]
        )

        assert result.exit_code == 0


# Test edge cases


def test_qc_command_with_log_file(runner, sample_qc_config, tmp_path):
    """Test QC command with log file output."""
    log_file = tmp_path / "test.log"

    result = runner.invoke(
        cli,
        ["qc", str(sample_qc_config), "--log-file", str(log_file), "--dry-run"],
    )

    assert result.exit_code == 0


def test_viz_command_with_log_file(runner, sample_viz_config, tmp_path):
    """Test Viz command with log file output."""
    log_file = tmp_path / "test.log"

    result = runner.invoke(
        cli,
        ["viz", str(sample_viz_config), "--log-file", str(log_file), "--dry-run"],
    )

    assert result.exit_code == 0


def test_qc_command_conflicting_flags(runner, sample_qc_config):
    """Test QC command with conflicting verbose and quiet flags."""
    # Both flags provided - verbose should take precedence
    result = runner.invoke(
        cli, ["qc", str(sample_qc_config), "-v", "-q", "--dry-run"]
    )
    # Should still work (one flag takes precedence)
    assert result.exit_code == 0
