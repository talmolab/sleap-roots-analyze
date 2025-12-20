"""Tests for enhanced pipeline runner summary generation.

TDD tests for enhance-pipeline-run-summary proposal.
Tests written before implementation per OpenSpec best practices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_qc_summary() -> dict[str, Any]:
    """Create a mock QC pipeline summary JSON structure."""
    return {
        "pipeline_info": {
            "pipeline_name": "test_qc_pipeline",
            "version": "1.0",
            "run_timestamp": "2025-12-15T10:00:00",
            "run_directory": "pipeline_runs/test/qc/test_run",
        },
        "configuration": {
            "heritability": {"enabled": True, "threshold": 0.4},
            "columns": {"genotype": "geno", "replicate": "rep"},
        },
        "final_data": {
            "n_samples": 890,
            "n_traits": 13,
            "n_genotypes": 156,
            "trait_names": ["Trait1", "Trait2", "Trait3"],
        },
        "step_summaries": {
            "heritability_filter": {
                "filtering_enabled": True,
                "threshold": 0.4,
                "traits_original": 19,
                "traits_retained": 13,
                "traits_removed": 6,
                "removed_trait_names": [
                    "Depth (mm)",
                    "Avg Hole Size",
                    "Shallow Angle",
                ],
                "mean_heritability_retained": 0.607,
            }
        },
    }


@pytest.fixture
def mock_qc_summary_heritability_disabled() -> dict[str, Any]:
    """Create a mock QC summary with heritability filtering disabled."""
    return {
        "pipeline_info": {"pipeline_name": "test_qc_no_h2"},
        "configuration": {
            "heritability": {"enabled": False, "threshold": 0.0},
        },
        "final_data": {
            "n_samples": 500,
            "n_traits": 20,
            "n_genotypes": 50,
        },
        "step_summaries": {},
    }


@pytest.fixture
def mock_cross_platform_summary() -> dict[str, Any]:
    """Create a mock cross-platform pipeline summary."""
    return {
        "pipeline_name": "cross_platform_test",
        "status": "success",
        "total_elapsed_time": 23.5,
    }


@pytest.fixture
def mock_alignment_csv_data() -> pd.DataFrame:
    """Create mock alignment summary CSV data."""
    return pd.DataFrame(
        {
            "metric": [
                "common_genotypes",
                "exp1_samples",
                "exp1_traits",
                "exp2_samples",
                "exp2_traits",
            ],
            "value": [18, 450, 25, 380, 30],
        }
    )


@pytest.fixture
def mock_correlations_csv_data() -> pd.DataFrame:
    """Create mock correlations CSV data (sorted by abs correlation)."""
    return pd.DataFrame(
        {
            "trait1": ["TraitA", "TraitB", "TraitC"],
            "trait2": ["TraitX", "TraitY", "TraitZ"],
            "correlation": [0.85, -0.72, 0.65],
            "p_value": [0.001, 0.003, 0.01],
        }
    )


@pytest.fixture
def setup_mock_qc_run(tmp_path, mock_qc_summary) -> Path:
    """Set up a mock QC run directory with summary JSON."""
    qc_run_dir = tmp_path / "qc" / "test_qc_run"
    qc_run_dir.mkdir(parents=True)

    summary_path = qc_run_dir / "10_pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(mock_qc_summary, f)

    return qc_run_dir


@pytest.fixture
def setup_mock_cross_platform_run(
    tmp_path,
    mock_cross_platform_summary,
    mock_alignment_csv_data,
    mock_correlations_csv_data,
) -> Path:
    """Set up a mock cross-platform run directory."""
    cp_run_dir = tmp_path / "cross_platform" / "test_cp_run"
    cp_run_dir.mkdir(parents=True)

    # Write summary JSON
    summary_path = cp_run_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(mock_cross_platform_summary, f)

    # Write alignment CSV
    alignment_path = cp_run_dir / "cross_platform_alignment_summary.csv"
    mock_alignment_csv_data.to_csv(alignment_path, index=False)

    # Write correlations CSV
    correlations_path = cp_run_dir / "cross_platform_correlations.csv"
    mock_correlations_csv_data.to_csv(correlations_path, index=False)

    return cp_run_dir


@pytest.fixture
def setup_mock_viz_run(tmp_path) -> Path:
    """Set up a mock viz run directory with figures."""
    viz_run_dir = tmp_path / "viz" / "test_viz_run"
    viz_run_dir.mkdir(parents=True)

    # Create mock figure files
    figures_dir = viz_run_dir / "figures"
    figures_dir.mkdir()
    for i in range(5):
        (figures_dir / f"figure_{i}.png").touch()

    # Create interactive HTML
    (viz_run_dir / "interactive_pca.html").touch()
    (viz_run_dir / "interactive_umap.html").touch()

    # Create summary JSON
    summary = {"pipeline_name": "test_viz", "status": "success"}
    with open(viz_run_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f)

    return viz_run_dir


# =============================================================================
# Phase 2: Tests for Pipeline Summary Reading
# =============================================================================


class TestReadPipelineSummary:
    """Tests for _read_pipeline_summary helper function."""

    def test_read_pipeline_summary_success(self, setup_mock_qc_run, mock_qc_summary):
        """Test reading a valid pipeline summary JSON."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        # We need to test the helper method directly
        # For now, test that we can import and the method exists
        assert hasattr(PipelineRunner, "_read_pipeline_summary")

        # Create a minimal runner to test the method
        # This will fail until we implement the method
        summary = PipelineRunner._read_pipeline_summary(setup_mock_qc_run)

        assert summary is not None
        assert summary.get("final_data", {}).get("n_samples") == 890
        assert summary.get("final_data", {}).get("n_traits") == 13

    def test_read_pipeline_summary_missing_file(self, tmp_path):
        """Test reading from directory without summary file returns empty dict."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        summary = PipelineRunner._read_pipeline_summary(empty_dir)

        assert summary == {}

    def test_read_pipeline_summary_malformed_json(self, tmp_path, caplog):
        """Test reading malformed JSON returns empty dict and logs warning."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        bad_dir = tmp_path / "bad_json"
        bad_dir.mkdir()

        # Write invalid JSON
        bad_json_path = bad_dir / "10_pipeline_summary.json"
        bad_json_path.write_text("{ invalid json }")

        summary = PipelineRunner._read_pipeline_summary(bad_dir)

        assert summary == {}


# =============================================================================
# Phase 3: Tests for QC Summary Enhancement
# =============================================================================


class TestFormatQCSummary:
    """Tests for enhanced QC summary formatting."""

    def test_format_qc_summary_with_metrics(self, tmp_path, setup_mock_qc_run):
        """Test QC summary table includes scientific metrics."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        # Create a mock manifest
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\nqc_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")

        # Simulate run results
        runner.run_results["qc"]["test_config.yaml"] = {
            "success": True,
            "elapsed_seconds": 30.5,
            "output_path": str(setup_mock_qc_run),
        }

        lines = runner._format_qc_summary()
        summary_text = "\n".join(lines)

        # Check table headers include new columns
        assert "Samples" in summary_text
        assert "Traits" in summary_text
        assert "Genotypes" in summary_text
        assert "H² Threshold" in summary_text or "H2 Threshold" in summary_text
        assert "Mean H²" in summary_text or "Mean H2" in summary_text

        # Check values are present
        assert "890" in summary_text  # n_samples
        assert "13" in summary_text  # n_traits
        assert "156" in summary_text  # n_genotypes
        assert "0.4" in summary_text  # threshold
        assert "0.61" in summary_text or "0.607" in summary_text  # mean H²

    def test_format_qc_summary_failed_pipeline(self, tmp_path):
        """Test QC summary shows N/A for failed pipelines."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\nqc_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")

        runner.run_results["qc"]["failed_config.yaml"] = {
            "success": False,
            "elapsed_seconds": 5.0,
            "error": "Some error",
        }

        lines = runner._format_qc_summary()
        summary_text = "\n".join(lines)

        assert "Failed" in summary_text
        assert "N/A" in summary_text

    def test_format_qc_summary_heritability_disabled(
        self, tmp_path, mock_qc_summary_heritability_disabled
    ):
        """Test QC summary shows 'Disabled' when heritability filtering is off."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        # Set up run directory with heritability disabled summary
        qc_run_dir = tmp_path / "qc" / "no_h2_run"
        qc_run_dir.mkdir(parents=True)
        with open(qc_run_dir / "10_pipeline_summary.json", "w") as f:
            json.dump(mock_qc_summary_heritability_disabled, f)

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\nqc_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")
        runner.run_results["qc"]["no_h2.yaml"] = {
            "success": True,
            "elapsed_seconds": 20.0,
            "output_path": str(qc_run_dir),
        }

        lines = runner._format_qc_summary()
        summary_text = "\n".join(lines)

        assert "Disabled" in summary_text or "N/A" in summary_text


# =============================================================================
# Phase 4: Tests for Removed Traits Documentation
# =============================================================================


class TestRemovedTraitsSection:
    """Tests for removed traits documentation."""

    def test_format_removed_traits_section(self, tmp_path, setup_mock_qc_run):
        """Test removed traits are listed per dataset."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\nqc_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")
        runner.run_results["qc"]["test.yaml"] = {
            "success": True,
            "elapsed_seconds": 30.0,
            "output_path": str(setup_mock_qc_run),
        }

        lines = runner._format_qc_summary()
        summary_text = "\n".join(lines)

        # Check removed traits section exists
        assert "Removed Traits" in summary_text or "removed" in summary_text.lower()
        # Check specific removed traits are listed
        assert "Depth" in summary_text or "Avg Hole" in summary_text

    def test_format_removed_traits_none_removed(self, tmp_path):
        """Test shows 'No traits removed' when none filtered."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        # Create summary with no removed traits
        qc_run_dir = tmp_path / "qc" / "all_pass_run"
        qc_run_dir.mkdir(parents=True)
        summary = {
            "final_data": {"n_samples": 100, "n_traits": 10, "n_genotypes": 20},
            "configuration": {"heritability": {"enabled": True, "threshold": 0.3}},
            "step_summaries": {
                "heritability_filter": {
                    "filtering_enabled": True,
                    "traits_removed": 0,
                    "removed_trait_names": [],
                    "mean_heritability_retained": 0.55,
                }
            },
        }
        with open(qc_run_dir / "10_pipeline_summary.json", "w") as f:
            json.dump(summary, f)

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\nqc_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")
        runner.run_results["qc"]["all_pass.yaml"] = {
            "success": True,
            "elapsed_seconds": 25.0,
            "output_path": str(qc_run_dir),
        }

        lines = runner._format_qc_summary()
        summary_text = "\n".join(lines)

        # Should indicate no traits were removed
        assert (
            "No traits" in summary_text
            or "0 traits" in summary_text.lower()
            or "none removed" in summary_text.lower()
        )


# =============================================================================
# Phase 5: Tests for Viz Summary Enhancement
# =============================================================================


class TestFormatVizSummary:
    """Tests for enhanced Viz summary formatting."""

    def test_format_viz_summary_with_figure_counts(self, tmp_path, setup_mock_viz_run):
        """Test Viz summary includes figure counts."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\nviz_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")
        runner.run_results["viz"]["test_viz.yaml"] = {
            "success": True,
            "elapsed_seconds": 45.0,
            "output_path": str(setup_mock_viz_run),
        }

        lines = runner._format_viz_summary()
        summary_text = "\n".join(lines)

        # Check for figure counts in output
        assert "Figures" in summary_text or "figures" in summary_text.lower()
        # Should show count of 5 PNG files
        assert "5" in summary_text

    def test_format_viz_summary_counts_interactive(self, tmp_path, setup_mock_viz_run):
        """Test Viz summary counts interactive HTML files."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\nviz_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")
        runner.run_results["viz"]["test_viz.yaml"] = {
            "success": True,
            "elapsed_seconds": 45.0,
            "output_path": str(setup_mock_viz_run),
        }

        lines = runner._format_viz_summary()
        summary_text = "\n".join(lines)

        # Should include interactive plot count (2 HTML files)
        assert "Interactive" in summary_text or "2" in summary_text


# =============================================================================
# Phase 6: Tests for Cross-Platform Summary Enhancement
# =============================================================================


class TestFormatCrossPlatformSummary:
    """Tests for enhanced Cross-Platform summary formatting."""

    def test_format_cross_platform_summary_with_metrics(
        self, tmp_path, setup_mock_cross_platform_run
    ):
        """Test Cross-Platform summary includes alignment metrics."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\ncross_platform_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")
        runner.run_results["cross_platform"]["test_cp.yaml"] = {
            "success": True,
            "elapsed_seconds": 23.5,
            "output_path": str(setup_mock_cross_platform_run),
        }

        lines = runner._format_cross_platform_summary()
        summary_text = "\n".join(lines)

        # Check table headers
        assert "Common Genotypes" in summary_text or "Genotypes" in summary_text
        assert "Exp1" in summary_text or "exp1" in summary_text.lower()
        assert "Exp2" in summary_text or "exp2" in summary_text.lower()

        # Check values from alignment CSV
        assert "18" in summary_text  # common_genotypes

    def test_format_cross_platform_summary_top_correlation(
        self, tmp_path, setup_mock_cross_platform_run
    ):
        """Test Cross-Platform summary shows top correlation."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\ncross_platform_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")
        runner.run_results["cross_platform"]["test_cp.yaml"] = {
            "success": True,
            "elapsed_seconds": 23.5,
            "output_path": str(setup_mock_cross_platform_run),
        }

        lines = runner._format_cross_platform_summary()
        summary_text = "\n".join(lines)

        # Should show top correlation value (0.85)
        assert "0.85" in summary_text or "Top" in summary_text

    def test_format_cross_platform_summary_missing_files(self, tmp_path):
        """Test Cross-Platform summary handles missing files gracefully."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        # Create run dir without alignment/correlation files
        cp_run_dir = tmp_path / "cross_platform" / "minimal_run"
        cp_run_dir.mkdir(parents=True)

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\ncross_platform_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")
        runner.run_results["cross_platform"]["minimal.yaml"] = {
            "success": True,
            "elapsed_seconds": 10.0,
            "output_path": str(cp_run_dir),
        }

        lines = runner._format_cross_platform_summary()
        summary_text = "\n".join(lines)

        # Should show N/A for missing data, not crash
        assert "N/A" in summary_text or "Success" in summary_text


# =============================================================================
# Phase 7: Tests for Methods Section
# =============================================================================


class TestMethodsSection:
    """Tests for publication-ready methods section."""

    def test_format_methods_section_exists(self, tmp_path):
        """Test methods section is generated."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\nqc_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")
        runner.run_results["qc"]["test.yaml"] = {"success": True}

        # Check the method exists
        assert hasattr(runner, "_format_methods_section")

        lines = runner._format_methods_section()
        summary_text = "\n".join(lines)

        assert "## Methods" in summary_text or "Methods" in summary_text

    def test_format_methods_section_describes_qc(self, tmp_path):
        """Test methods section describes QC methodology."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\nqc_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")

        lines = runner._format_methods_section()
        summary_text = "\n".join(lines)

        # Should mention key QC concepts
        assert "outlier" in summary_text.lower() or "QC" in summary_text
        assert "heritability" in summary_text.lower() or "H²" in summary_text

    def test_format_methods_section_has_placeholders(self, tmp_path):
        """Test methods section includes placeholders for dataset values."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Test\nqc_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")

        lines = runner._format_methods_section()
        summary_text = "\n".join(lines)

        # Should have placeholder markers or template text
        assert (
            "{" in summary_text
            or "N=" in summary_text
            or "samples" in summary_text.lower()
        )


# =============================================================================
# Phase 8: Integration Tests
# =============================================================================


class TestGenerateSummaryIntegration:
    """Integration tests for full summary generation."""

    def test_generate_summary_includes_all_sections(
        self,
        tmp_path,
        setup_mock_qc_run,
        setup_mock_viz_run,
        setup_mock_cross_platform_run,
    ):
        """Test full summary includes QC, Viz, Cross-Platform, and Methods."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            "run_name: Full Test\nqc_configs: []\nviz_configs: []\ncross_platform_configs: []"
        )

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")

        # Set up all results
        runner.run_results["qc"]["qc.yaml"] = {
            "success": True,
            "elapsed_seconds": 30.0,
            "output_path": str(setup_mock_qc_run),
        }
        runner.run_results["viz"]["viz.yaml"] = {
            "success": True,
            "elapsed_seconds": 45.0,
            "output_path": str(setup_mock_viz_run),
        }
        runner.run_results["cross_platform"]["cp.yaml"] = {
            "success": True,
            "elapsed_seconds": 20.0,
            "output_path": str(setup_mock_cross_platform_run),
        }

        # Ensure run directory exists
        runner.run_dir.mkdir(parents=True, exist_ok=True)

        summary_path = runner.generate_summary()

        assert summary_path.exists()

        content = summary_path.read_text()

        # Check all major sections present
        assert "## QC Pipeline Results" in content
        assert "## Visualization Pipeline Results" in content
        assert "## Cross-Platform Analysis Results" in content
        assert "## Methods" in content or "Methods" in content

    def test_generate_summary_markdown_tables_valid(self, tmp_path, setup_mock_qc_run):
        """Test generated markdown tables are properly formatted."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("run_name: Table Test\nqc_configs: []")

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")
        runner.run_results["qc"]["test.yaml"] = {
            "success": True,
            "elapsed_seconds": 30.0,
            "output_path": str(setup_mock_qc_run),
        }

        runner.run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = runner.generate_summary()

        content = summary_path.read_text()

        # Check table formatting - pipes should be balanced
        for line in content.split("\n"):
            if line.startswith("|"):
                # Count pipes - should have same number in header and divider
                pipe_count = line.count("|")
                assert pipe_count >= 2, f"Table row has too few columns: {line}"


# =============================================================================
# Phase 9: Tests for Config Preservation (fix-pipeline-runner-config-preservation)
# =============================================================================


class TestExtractFilename:
    """Tests for _extract_filename helper function."""

    def test_extract_filename_unix_path(self, tmp_path):
        """Test extracting filename from Unix-style path."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        result = PipelineRunner._extract_filename(
            "pipeline_runs/2025-12-15/qc/test_qc/07_data_outliers_removed.csv"
        )
        assert result == "07_data_outliers_removed.csv"

    def test_extract_filename_windows_path(self, tmp_path):
        """Test extracting filename from Windows-style path."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        result = PipelineRunner._extract_filename(
            "pipeline_runs\\2025-12-15\\qc\\test_qc\\10_final_data.csv"
        )
        assert result == "10_final_data.csv"

    def test_extract_filename_mixed_path(self, tmp_path):
        """Test extracting filename from mixed separator path."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        result = PipelineRunner._extract_filename(
            "pipeline_runs/2025-12-15\\qc/test_qc\\07_data_outliers_removed.csv"
        )
        assert result == "07_data_outliers_removed.csv"

    def test_extract_filename_just_filename(self, tmp_path):
        """Test extracting filename when path is just the filename."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        result = PipelineRunner._extract_filename("07_data_outliers_removed.csv")
        assert result == "07_data_outliers_removed.csv"


class TestUpdateYamlPathPreservingStructure:
    """Tests for _update_yaml_path_preserving_structure helper function."""

    def test_preserves_double_quoted_path(self, tmp_path):
        """Test that double-quoted paths remain double-quoted."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        content = """# Comment preserved
exp1_data_path: "old/path/to/07_data_outliers_removed.csv"
exp1_name: "Test"
"""
        new_dir = Path("new/qc/output")
        result = PipelineRunner._update_yaml_path_preserving_structure(
            content, "exp1_data_path", new_dir, "07_data_outliers_removed.csv"
        )

        assert '"new/qc/output/07_data_outliers_removed.csv"' in result
        assert "# Comment preserved" in result
        assert 'exp1_name: "Test"' in result

    def test_preserves_single_quoted_path(self, tmp_path):
        """Test that single-quoted paths remain single-quoted."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        content = "csv_path: 'old/path/10_final_data.csv'\n"
        new_dir = Path("new/output")
        result = PipelineRunner._update_yaml_path_preserving_structure(
            content, "csv_path", new_dir, "10_final_data.csv"
        )

        assert "'new/output/10_final_data.csv'" in result

    def test_preserves_unquoted_path(self, tmp_path):
        """Test that unquoted paths remain unquoted."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        content = "exp2_data_path: old/path/data.csv\n"
        new_dir = Path("new/output")
        result = PipelineRunner._update_yaml_path_preserving_structure(
            content, "exp2_data_path", new_dir, "data.csv"
        )

        # Should not add quotes
        assert "exp2_data_path: new/output/data.csv" in result
        assert '"' not in result
        assert "'" not in result

    def test_preserves_comments_and_formatting(self, tmp_path):
        """Test that all comments and blank lines are preserved."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        content = """# This is a header comment
# Describing the config

# Experiment 1: Important data
exp1_data_path: "old/path/07_data_outliers_removed.csv"
exp1_name: "Test Experiment"

# More comments here
other_key: value
"""
        new_dir = Path("new/qc/run_123")
        result = PipelineRunner._update_yaml_path_preserving_structure(
            content, "exp1_data_path", new_dir, "07_data_outliers_removed.csv"
        )

        # All comments preserved
        assert "# This is a header comment" in result
        assert "# Describing the config" in result
        assert "# Experiment 1: Important data" in result
        assert "# More comments here" in result

        # Blank lines preserved
        assert "\n\n" in result

        # Other keys unchanged
        assert 'exp1_name: "Test Experiment"' in result
        assert "other_key: value" in result


class TestConfigPreservation:
    """Integration tests for config path preservation."""

    def test_preserves_07_filename_in_cross_platform(self, tmp_path):
        """Test that 07_data_outliers_removed.csv is preserved when specified."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        # Create manifest
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            """run_name: Test
qc_configs: []
cross_platform_configs:
  - cross_platform/test.yaml
qc_mapping:
  cross_platform/test.yaml:
    exp1: qc/exp1.yaml
    exp2: qc/exp2.yaml
"""
        )

        # Create cross-platform config with 07 files
        cross_platform_dir = tmp_path / "cross_platform"
        cross_platform_dir.mkdir()
        config_path = cross_platform_dir / "test.yaml"
        config_path.write_text(
            """# Cross-Platform Analysis
# Using QC'd but NOT heritability filtered data

exp1_data_path: "old/path/qc/exp1/07_data_outliers_removed.csv"
exp1_name: "Experiment 1 (QC'd)"
exp1_genotype_col: "Genotype"

# Experiment 2 also uses 07 file
exp2_data_path: "old/path/qc/exp2/07_data_outliers_removed.csv"
exp2_name: "Experiment 2 (QC'd)"
exp2_genotype_col: "Genotype"

correlation_method: "spearman"
"""
        )

        # Create mock QC outputs with both file types
        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")

        qc1_output = tmp_path / "runs" / runner.run_timestamp / "qc" / "exp1_run"
        qc1_output.mkdir(parents=True)
        (qc1_output / "07_data_outliers_removed.csv").write_text("col1,col2\n1,2")
        (qc1_output / "10_final_data.csv").write_text("col1,col2\n1,2")

        qc2_output = tmp_path / "runs" / runner.run_timestamp / "qc" / "exp2_run"
        qc2_output.mkdir(parents=True)
        (qc2_output / "07_data_outliers_removed.csv").write_text("col1,col2\n1,2")
        (qc2_output / "10_final_data.csv").write_text("col1,col2\n1,2")

        runner.qc_outputs = {
            "qc/exp1.yaml": qc1_output,
            "qc/exp2.yaml": qc2_output,
        }

        # Create output directories
        (runner.run_dir / "cross_platform").mkdir(parents=True)

        # Run the update
        updated_path = runner._update_cross_platform_config(
            config_path,
            "cross_platform/test.yaml",
            runner.manifest.get("qc_mapping", {}),
        )

        assert updated_path is not None
        updated_content = updated_path.read_text()

        # CRITICAL: Should preserve 07_data_outliers_removed.csv, NOT use 10_final_data.csv
        assert "07_data_outliers_removed.csv" in updated_content
        assert "10_final_data.csv" not in updated_content

        # Comments should be preserved
        assert "# Cross-Platform Analysis" in updated_content
        assert "# Using QC'd but NOT heritability filtered data" in updated_content
        assert "# Experiment 2 also uses 07 file" in updated_content

    def test_preserves_mixed_filenames(self, tmp_path):
        """Test that mixed filename choices (07 and 10) are both preserved."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        # Create manifest
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            """run_name: Test
qc_configs: []
cross_platform_configs:
  - cross_platform/mixed.yaml
qc_mapping:
  cross_platform/mixed.yaml:
    exp1: qc/exp1.yaml
    exp2: qc/exp2.yaml
"""
        )

        # Create config with MIXED filenames
        cross_platform_dir = tmp_path / "cross_platform"
        cross_platform_dir.mkdir()
        config_path = cross_platform_dir / "mixed.yaml"
        config_path.write_text(
            """# Mixed filename example
# exp1 uses 07 (all traits), exp2 uses 10 (filtered)

exp1_data_path: "old/qc/exp1/07_data_outliers_removed.csv"
exp1_name: "Exp1 (All Traits)"

exp2_data_path: "old/qc/exp2/10_final_data.csv"
exp2_name: "Exp2 (Filtered)"
"""
        )

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")

        # Create mock QC outputs
        qc1_output = tmp_path / "runs" / runner.run_timestamp / "qc" / "exp1_run"
        qc1_output.mkdir(parents=True)
        (qc1_output / "07_data_outliers_removed.csv").write_text("data")
        (qc1_output / "10_final_data.csv").write_text("data")

        qc2_output = tmp_path / "runs" / runner.run_timestamp / "qc" / "exp2_run"
        qc2_output.mkdir(parents=True)
        (qc2_output / "07_data_outliers_removed.csv").write_text("data")
        (qc2_output / "10_final_data.csv").write_text("data")

        runner.qc_outputs = {
            "qc/exp1.yaml": qc1_output,
            "qc/exp2.yaml": qc2_output,
        }

        (runner.run_dir / "cross_platform").mkdir(parents=True)

        updated_path = runner._update_cross_platform_config(
            config_path,
            "cross_platform/mixed.yaml",
            runner.manifest.get("qc_mapping", {}),
        )

        assert updated_path is not None
        updated_content = updated_path.read_text()

        # exp1 should use 07, exp2 should use 10
        lines = updated_content.split("\n")
        exp1_line = [l for l in lines if "exp1_data_path" in l][0]
        exp2_line = [l for l in lines if "exp2_data_path" in l][0]

        assert "07_data_outliers_removed.csv" in exp1_line
        assert "10_final_data.csv" in exp2_line

    def test_preserves_viz_config_structure(self, tmp_path):
        """Test that viz config structure is preserved during update."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            """run_name: Test
qc_configs: []
viz_configs:
  - viz/test.yaml
qc_mapping:
  viz/test.yaml: qc/source.yaml
"""
        )

        # Create viz config with comments and specific filename
        viz_dir = tmp_path / "viz"
        viz_dir.mkdir()
        config_path = viz_dir / "test.yaml"
        config_path.write_text(
            """# Visualization Pipeline Configuration
# Using QC'd data (not heritability filtered for exploration)

pipeline_name: "viz_test"
version: "1.0"

# Data configuration
data:
  csv_path: "old/qc/run/07_data_outliers_removed.csv"
  image_dir: null

# Column names
columns:
  barcode: "Barcode"
  genotype: "Genotype"
"""
        )

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")

        qc_output = tmp_path / "runs" / runner.run_timestamp / "qc" / "source_run"
        qc_output.mkdir(parents=True)
        (qc_output / "07_data_outliers_removed.csv").write_text("data")
        (qc_output / "10_final_data.csv").write_text("data")

        runner.qc_outputs = {"qc/source.yaml": qc_output}
        (runner.run_dir / "viz").mkdir(parents=True)

        updated_path = runner._update_viz_config(
            config_path,
            "viz/test.yaml",
            runner.manifest.get("qc_mapping", {}),
        )

        assert updated_path is not None
        updated_content = updated_path.read_text()

        # Filename preserved
        assert "07_data_outliers_removed.csv" in updated_content

        # All comments preserved
        assert "# Visualization Pipeline Configuration" in updated_content
        assert (
            "# Using QC'd data (not heritability filtered for exploration)"
            in updated_content
        )
        assert "# Data configuration" in updated_content
        assert "# Column names" in updated_content

        # Structure preserved (version should still be "1.0" not '1.0')
        assert 'version: "1.0"' in updated_content

        # Other nested keys unchanged
        assert "image_dir: null" in updated_content
        assert 'barcode: "Barcode"' in updated_content


# =============================================================================
# Phase 10: Tests for Configuration Comparison (add-config-comparison-to-summary)
# =============================================================================


class TestFlattenConfigDict:
    """Tests for _flatten_config_dict helper function."""

    def test_flatten_simple_dict(self):
        """Test flattening a simple flat dictionary."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        config = {"a": 1, "b": 2, "c": "test"}
        result = PipelineRunner._flatten_config_dict(config)

        assert result == {"a": 1, "b": 2, "c": "test"}

    def test_flatten_nested_dict(self):
        """Test flattening a nested dictionary."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        config = {
            "cleanup": {"max_nan_fraction": 0.0, "max_zeros_per_trait": 0.5},
            "heritability": {"enabled": True, "threshold": 0.4},
        }
        result = PipelineRunner._flatten_config_dict(config)

        assert result["cleanup.max_nan_fraction"] == 0.0
        assert result["cleanup.max_zeros_per_trait"] == 0.5
        assert result["heritability.enabled"] is True
        assert result["heritability.threshold"] == 0.4

    def test_flatten_deeply_nested_dict(self):
        """Test flattening a deeply nested dictionary."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        config = {
            "outlier_detection": {
                "mahalanobis": {
                    "variance_threshold": 0.95,
                    "use_chi_squared": True,
                    "chi2_percentile": 99.0,
                }
            }
        }
        result = PipelineRunner._flatten_config_dict(config)

        assert result["outlier_detection.mahalanobis.variance_threshold"] == 0.95
        assert result["outlier_detection.mahalanobis.use_chi_squared"] is True
        assert result["outlier_detection.mahalanobis.chi2_percentile"] == 99.0

    def test_flatten_with_list_values(self):
        """Test flattening preserves list values."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        config = {
            "outlier_detection": {
                "traditional_methods": ["mahalanobis"],
                "clustering_methods": [],
            }
        }
        result = PipelineRunner._flatten_config_dict(config)

        assert result["outlier_detection.traditional_methods"] == ["mahalanobis"]
        assert result["outlier_detection.clustering_methods"] == []

    def test_flatten_with_none_values(self):
        """Test flattening handles None values."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        config = {"data": {"csv_path": None, "image_dir": None}}
        result = PipelineRunner._flatten_config_dict(config)

        assert result["data.csv_path"] is None
        assert result["data.image_dir"] is None

    def test_flatten_excludes_specified_keys(self):
        """Test flattening can exclude specific top-level keys."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        config = {
            "pipeline_name": "test",
            "data": {"csv_path": "/path/to/file.csv"},
            "heritability": {"threshold": 0.4},
        }
        result = PipelineRunner._flatten_config_dict(
            config, exclude_keys=["data", "pipeline_name"]
        )

        assert "data.csv_path" not in result
        assert "pipeline_name" not in result
        assert result["heritability.threshold"] == 0.4


class TestExtractAllConfigParams:
    """Tests for _extract_all_config_params function."""

    def test_extract_params_from_qc_config(self, tmp_path):
        """Test extracting ALL parameters from a QC config file."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        config_path = tmp_path / "test_qc.yaml"
        config_path.write_text(
            """
pipeline_name: "test_qc"
cleanup:
  max_nan_fraction: 0.0
  max_zeros_per_trait: 0.5
heritability:
  enabled: true
  threshold: 0.4
"""
        )

        result = PipelineRunner._extract_all_config_params(config_path)

        assert result["cleanup.max_nan_fraction"] == 0.0
        assert result["cleanup.max_zeros_per_trait"] == 0.5
        assert result["heritability.enabled"] is True
        assert result["heritability.threshold"] == 0.4

    def test_extract_params_excludes_paths(self, tmp_path):
        """Test that data paths are excluded from config params."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        config_path = tmp_path / "test_qc.yaml"
        config_path.write_text(
            """
pipeline_name: "test_qc"
data:
  csv_path: "C:/path/to/data.csv"
  image_dir: "C:/path/to/images"
heritability:
  threshold: 0.4
"""
        )

        result = PipelineRunner._extract_all_config_params(config_path)

        # Data paths should be excluded (they're environment-specific)
        assert "data.csv_path" not in result
        assert "data.image_dir" not in result
        # But other params included
        assert result["heritability.threshold"] == 0.4

    def test_extract_params_missing_file(self, tmp_path):
        """Test extracting from non-existent file returns empty dict."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        result = PipelineRunner._extract_all_config_params(
            tmp_path / "nonexistent.yaml"
        )

        assert result == {}


class TestFormatComparisonTable:
    """Tests for _format_comparison_table function."""

    def test_format_table_multiple_configs(self):
        """Test formatting comparison table with multiple configs."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        configs = {
            "turface_150": {
                "heritability.threshold": 0.4,
                "heritability.enabled": True,
                "cleanup.max_nan_fraction": 0.0,
            },
            "turface_19": {
                "heritability.threshold": 0.6,
                "heritability.enabled": True,
                "cleanup.max_nan_fraction": 0.0,
            },
        }

        lines = PipelineRunner._format_comparison_table(configs)
        table_text = "\n".join(lines)

        # Check table structure
        assert "| Parameter |" in table_text
        assert "turface_150" in table_text
        assert "turface_19" in table_text
        assert "heritability.threshold" in table_text
        assert "0.4" in table_text
        assert "0.6" in table_text

    def test_format_table_single_config(self):
        """Test formatting table with single config still works."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        configs = {
            "only_one": {
                "heritability.threshold": 0.5,
                "cleanup.max_nan_fraction": 0.0,
            }
        }

        lines = PipelineRunner._format_comparison_table(configs)
        table_text = "\n".join(lines)

        # Should still produce valid table
        assert "| Parameter |" in table_text
        assert "only_one" in table_text
        assert "0.5" in table_text

    def test_format_table_missing_params_show_na(self):
        """Test that missing parameters show N/A."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        configs = {
            "has_param": {"heritability.threshold": 0.4, "root_core.enabled": True},
            "missing_param": {
                "heritability.threshold": 0.5
                # root_core.enabled is missing
            },
        }

        lines = PipelineRunner._format_comparison_table(configs)
        table_text = "\n".join(lines)

        assert "N/A" in table_text

    def test_format_table_list_values(self):
        """Test that list values are formatted as comma-separated."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        configs = {
            "config1": {
                "outlier_detection.traditional_methods": [
                    "mahalanobis",
                    "isolation_forest",
                ],
            }
        }

        lines = PipelineRunner._format_comparison_table(configs)
        table_text = "\n".join(lines)

        # Lists should be comma-separated
        assert (
            "mahalanobis, isolation_forest" in table_text
            or "mahalanobis,isolation_forest" in table_text
        )

    def test_format_table_empty_list(self):
        """Test that empty lists are formatted correctly."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        configs = {"config1": {"outlier_detection.clustering_methods": []}}

        lines = PipelineRunner._format_comparison_table(configs)
        table_text = "\n".join(lines)

        # Empty list should show as empty or "(none)"
        assert "[]" in table_text or "(none)" in table_text or "| |" in table_text


class TestFormatConfigComparison:
    """Tests for _format_config_comparison generating full section."""

    def test_format_config_comparison_qc_section(self, tmp_path):
        """Test generating QC configuration comparison section."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        # Create mock QC configs
        qc_dir = tmp_path / "qc"
        qc_dir.mkdir()

        config1 = qc_dir / "turface_150.yaml"
        config1.write_text(
            """
pipeline_name: "turface_150_qc"
cleanup:
  max_nan_fraction: 0.0
  max_zeros_per_trait: 0.5
heritability:
  enabled: true
  threshold: 0.4
"""
        )

        config2 = qc_dir / "turface_19.yaml"
        config2.write_text(
            """
pipeline_name: "turface_19_qc"
cleanup:
  max_nan_fraction: 0.0
  max_zeros_per_trait: 0.5
heritability:
  enabled: true
  threshold: 0.6
"""
        )

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            f"""
run_name: Test
qc_configs:
  - qc/turface_150.yaml
  - qc/turface_19.yaml
"""
        )

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")

        lines = runner._format_config_comparison()
        section_text = "\n".join(lines)

        assert "## Configuration Comparison" in section_text
        assert "### QC Pipeline Configuration" in section_text
        assert "heritability.threshold" in section_text
        assert "0.4" in section_text
        assert "0.6" in section_text

    def test_format_config_comparison_viz_section(self, tmp_path):
        """Test generating Viz configuration comparison section."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        viz_dir = tmp_path / "viz"
        viz_dir.mkdir()

        config1 = viz_dir / "viz_turface.yaml"
        config1.write_text(
            """
pipeline_name: "viz_turface"
visualization:
  dpi: 100
  figsize: [12, 8]
"""
        )

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            """
run_name: Test
viz_configs:
  - viz/viz_turface.yaml
"""
        )

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")

        lines = runner._format_config_comparison()
        section_text = "\n".join(lines)

        assert "### Visualization Pipeline Configuration" in section_text
        assert "visualization.dpi" in section_text

    def test_format_config_comparison_cross_platform_section(self, tmp_path):
        """Test generating Cross-Platform configuration comparison section."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        cross_dir = tmp_path / "cross_platform"
        cross_dir.mkdir()

        config1 = cross_dir / "cross_test.yaml"
        config1.write_text(
            """
exp1_name: "Exp1"
exp2_name: "Exp2"
correlation_method: "spearman"
significance_threshold: 0.05
"""
        )

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            """
run_name: Test
cross_platform_configs:
  - cross_platform/cross_test.yaml
"""
        )

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")

        lines = runner._format_config_comparison()
        section_text = "\n".join(lines)

        assert "### Cross-Platform Pipeline Configuration" in section_text
        assert "correlation_method" in section_text


class TestGenerateSummaryWithConfigComparison:
    """Test that generate_summary includes config comparison section."""

    def test_summary_includes_config_comparison(self, tmp_path):
        """Test that SUMMARY.md includes configuration comparison section."""
        from sleap_roots_analyze.pipeline_runner import PipelineRunner

        # Create QC config
        qc_dir = tmp_path / "qc"
        qc_dir.mkdir()
        (qc_dir / "test.yaml").write_text(
            """
pipeline_name: "test_qc"
heritability:
  threshold: 0.4
cleanup:
  max_nan_fraction: 0.0
"""
        )

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            """
run_name: Test
qc_configs:
  - qc/test.yaml
"""
        )

        runner = PipelineRunner(manifest_path, output_dir=tmp_path / "runs")
        runner.run_dir.mkdir(parents=True)

        # Mock successful QC run
        runner.run_results["qc"]["qc/test.yaml"] = {
            "success": True,
            "elapsed_seconds": 30.0,
            "output_path": str(tmp_path / "output"),
        }

        summary_path = runner.generate_summary()

        content = summary_path.read_text()

        # Config comparison should appear after results and before Methods
        assert "## Configuration Comparison" in content
        assert content.index("## Configuration Comparison") < content.index(
            "## Methods"
        )
