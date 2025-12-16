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
