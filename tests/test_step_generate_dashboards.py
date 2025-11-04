"""Tests for GenerateDashboardsStep (Step 11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import GenerateDashboardsStep
from tests.fixtures_visualization import verify_html_structure


class TestGenerateDashboardsBasic:
    """Test basic functionality of GenerateDashboardsStep."""

    def test_step_initialization(self):
        """Test that step initializes with correct name and description."""
        step = GenerateDashboardsStep()

        assert step.step_name == "GenerateDashboards"
        assert "dashboard" in step.description.lower()

    def test_basic_execution(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
        tmp_path,
    ):
        """Test basic step execution with minimal config."""
        step = GenerateDashboardsStep()

        result = step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=prev_result_with_generated_figures.metadata["static_figures"][
                0
            ].parent.parent,  # Get run_dir from fixture
            prev_result=prev_result_with_generated_figures,
        )

        # Check result structure
        assert isinstance(result, StepResult)
        assert isinstance(result.data, type(sample_trait_data))
        assert result.data.equals(sample_trait_data)

        # Check metadata exists
        assert "trait_names" in result.metadata
        assert "dashboard_path" in result.metadata

        # Check files were generated
        assert len(result.files_generated) > 0

    def test_step_disabled(
        self,
        dashboard_config_disabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that step skips execution when disabled."""
        step = GenerateDashboardsStep()

        result = step.execute(
            data=sample_trait_data,
            config=dashboard_config_disabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check result structure
        assert isinstance(result, StepResult)
        assert result.data.equals(sample_trait_data)

        # Check no dashboard was generated
        assert "dashboard_path" not in result.metadata or not result.metadata.get(
            "dashboard_path"
        )
        assert len(result.files_generated) == 0

        # Check dashboard file doesn't exist
        dashboard_file = tmp_path / "dashboard.html"
        assert not dashboard_file.exists()

    def test_dashboard_file_creation(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
        tmp_path,
    ):
        """Test that dashboard.html file is created."""
        step = GenerateDashboardsStep()

        step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=prev_result_with_generated_figures.metadata["static_figures"][
                0
            ].parent.parent,
            prev_result=prev_result_with_generated_figures,
        )

        # Check dashboard file exists
        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent
        dashboard_file = run_dir / "dashboard.html"
        assert dashboard_file.exists()


class TestGenerateDashboardsHTMLStructure:
    """Test HTML structure and content."""

    def test_html_validity(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
    ):
        """Test that generated HTML has valid structure."""
        step = GenerateDashboardsStep()

        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent

        step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=run_dir,
            prev_result=prev_result_with_generated_figures,
        )

        # Check HTML structure
        dashboard_file = run_dir / "dashboard.html"
        assert verify_html_structure(dashboard_file)

    def test_html_contains_summary_stats(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
    ):
        """Test that dashboard contains summary statistics."""
        step = GenerateDashboardsStep()

        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent

        step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=run_dir,
            prev_result=prev_result_with_generated_figures,
        )

        # Check dashboard contains summary info
        dashboard_file = run_dir / "dashboard.html"
        content = dashboard_file.read_text(encoding="utf-8")

        # Should contain sample count
        assert str(len(sample_trait_data)) in content

        # Should contain figure count info
        total_figs = len(
            prev_result_with_generated_figures.metadata["static_figures"]
        ) + len(prev_result_with_generated_figures.metadata["interactive_figures"])
        assert str(total_figs) in content or "figure" in content.lower()

    def test_html_contains_css_styling(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
    ):
        """Test that dashboard contains CSS styling."""
        step = GenerateDashboardsStep()

        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent

        step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=run_dir,
            prev_result=prev_result_with_generated_figures,
        )

        # Check CSS is embedded
        dashboard_file = run_dir / "dashboard.html"
        content = dashboard_file.read_text(encoding="utf-8")

        assert "<style>" in content or "stylesheet" in content.lower()

    def test_html_contains_figure_links(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
    ):
        """Test that dashboard contains links to figures."""
        step = GenerateDashboardsStep()

        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent

        result = step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=run_dir,
            prev_result=prev_result_with_generated_figures,
        )

        # Check dashboard has links to figures
        dashboard_file = run_dir / "dashboard.html"
        content = dashboard_file.read_text(encoding="utf-8")

        # Should have references to figure directories
        assert "static_figures" in content or "interactive_figures" in content


class TestGenerateDashboardsWithStaticFigures:
    """Test dashboard generation with static figures."""

    def test_dashboard_includes_static_figures(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
    ):
        """Test that dashboard includes static figure references."""
        step = GenerateDashboardsStep()

        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent

        result = step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=run_dir,
            prev_result=prev_result_with_generated_figures,
        )

        # Check dashboard references static figures
        dashboard_file = run_dir / "dashboard.html"
        content = dashboard_file.read_text(encoding="utf-8")

        # Should contain references to at least one static figure
        static_files = prev_result_with_generated_figures.metadata["static_figures"]
        assert any(f.name in content for f in static_files)

    def test_dashboard_with_no_static_figures(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test dashboard generation when no static figures exist."""
        step = GenerateDashboardsStep()

        result = step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Should still create dashboard
        dashboard_file = tmp_path / "dashboard.html"
        assert dashboard_file.exists()

        # Should handle empty state gracefully
        content = dashboard_file.read_text(encoding="utf-8")
        assert verify_html_structure(dashboard_file)


class TestGenerateDashboardsWithInteractiveFigures:
    """Test dashboard generation with interactive figures."""

    def test_dashboard_includes_interactive_figures(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
    ):
        """Test that dashboard includes interactive figure references."""
        step = GenerateDashboardsStep()

        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent

        result = step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=run_dir,
            prev_result=prev_result_with_generated_figures,
        )

        # Check dashboard references interactive figures
        dashboard_file = run_dir / "dashboard.html"
        content = dashboard_file.read_text(encoding="utf-8")

        # Should contain references to at least one interactive figure
        interactive_files = prev_result_with_generated_figures.metadata[
            "interactive_figures"
        ]
        assert any(f.name in content for f in interactive_files)


class TestGenerateDashboardsFileList:
    """Test file list generation in dashboard."""

    def test_dashboard_contains_file_list(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
    ):
        """Test that dashboard contains a file list section."""
        step = GenerateDashboardsStep()

        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent

        result = step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=run_dir,
            prev_result=prev_result_with_generated_figures,
        )

        # Check dashboard has file list
        dashboard_file = run_dir / "dashboard.html"
        content = dashboard_file.read_text(encoding="utf-8")

        # Should have some indicator of files (could be a list, table, etc.)
        assert "file" in content.lower() or ".html" in content or ".png" in content


class TestGenerateDashboardsManifest:
    """Test manifest generation."""

    def test_manifest_created(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
    ):
        """Test that manifest JSON file is created."""
        step = GenerateDashboardsStep()

        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent

        result = step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=run_dir,
            prev_result=prev_result_with_generated_figures,
        )

        # Check manifest file exists
        manifest_file = run_dir / "11_dashboard_manifest.json"
        assert manifest_file.exists()

    def test_manifest_content_accuracy(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
    ):
        """Test that manifest contains accurate information."""
        step = GenerateDashboardsStep()

        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent

        result = step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=run_dir,
            prev_result=prev_result_with_generated_figures,
        )

        # Check manifest metadata
        manifest = result.metadata["dashboard_manifest"]
        assert "dashboard_path" in manifest
        assert "total_static_figures" in manifest
        assert "total_interactive_figures" in manifest
        assert "run_directory" in manifest


class TestGenerateDashboardsMetadata:
    """Test metadata handling and propagation."""

    def test_metadata_propagation(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
    ):
        """Test that metadata from previous step is preserved."""
        step = GenerateDashboardsStep()

        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent

        result = step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=run_dir,
            prev_result=prev_result_with_generated_figures,
        )

        # Check original metadata is preserved
        assert "trait_names" in result.metadata
        assert (
            result.metadata["trait_names"]
            == prev_result_with_generated_figures.metadata["trait_names"]
        )

        # Check new metadata is added
        assert "dashboard_path" in result.metadata
        assert "dashboard_manifest" in result.metadata

    def test_files_generated_list(
        self,
        dashboard_config_enabled,
        sample_trait_data,
        prev_result_with_generated_figures,
    ):
        """Test that files_generated list is populated correctly."""
        step = GenerateDashboardsStep()

        run_dir = prev_result_with_generated_figures.metadata["static_figures"][
            0
        ].parent.parent

        result = step.execute(
            data=sample_trait_data,
            config=dashboard_config_enabled,
            run_dir=run_dir,
            prev_result=prev_result_with_generated_figures,
        )

        # Check files_generated is not empty
        assert len(result.files_generated) > 0

        # Check all files in list exist
        for file_path in result.files_generated:
            assert (
                file_path.exists()
            ), f"File in files_generated doesn't exist: {file_path}"
