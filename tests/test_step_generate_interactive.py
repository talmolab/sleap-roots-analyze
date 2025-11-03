"""Tests for GenerateInteractiveStep (Step 10)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import GenerateInteractiveStep
from tests.fixtures_visualization import verify_html_structure


class TestGenerateInteractiveBasic:
    """Test basic functionality of GenerateInteractiveStep."""

    def test_step_initialization(self):
        """Test that step initializes with correct name and description."""
        step = GenerateInteractiveStep()

        assert step.step_name == "GenerateInteractive"
        assert "interactive" in step.description.lower()

    def test_basic_execution(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """Test basic step execution with minimal config."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Check result structure
        assert isinstance(result, StepResult)
        assert isinstance(result.data, type(sample_trait_data))
        assert result.data.equals(sample_trait_data)

        # Check metadata exists
        assert "trait_names" in result.metadata
        assert "interactive_figures" in result.metadata
        assert "interactive_figures_manifest" in result.metadata

        # Check files were generated
        assert len(result.files_generated) > 0

    def test_step_disabled(
        self,
        interactive_viz_config_disabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that step skips execution when disabled."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_disabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check result structure
        assert isinstance(result, StepResult)
        assert result.data.equals(sample_trait_data)

        # Check no figures were generated
        assert "interactive_figures" not in result.metadata or not result.metadata.get(
            "interactive_figures"
        )
        assert len(result.files_generated) == 0

        # Check output directory wasn't created
        interactive_dir = tmp_path / "interactive_figures"
        if interactive_dir.exists():
            # If directory exists, it should be empty
            assert len(list(interactive_dir.iterdir())) == 0

    def test_output_directory_creation(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """Test that output directory is created."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Check directory was created
        interactive_dir = tmp_path / "interactive_figures"
        assert interactive_dir.exists()
        assert interactive_dir.is_dir()

        # Check files exist in directory
        files_in_dir = list(interactive_dir.iterdir())
        assert len(files_in_dir) > 0


class TestGenerateInteractivePCAPlots:
    """Test interactive PCA plot generation."""

    def test_interactive_pca_generated(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that interactive PCA plot is generated when PCA results available."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check interactive PCA plot exists
        interactive_dir = tmp_path / "interactive_figures"
        pca_html = interactive_dir / "interactive_pca.html"
        assert pca_html.exists(), "Missing interactive_pca.html"

        # Verify it's a valid HTML file
        assert verify_html_structure(pca_html)

    def test_interactive_pca_skipped_without_results(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that interactive PCA is skipped when no PCA results available."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check no interactive PCA plot exists
        interactive_dir = tmp_path / "interactive_figures"
        if interactive_dir.exists():
            pca_html = interactive_dir / "interactive_pca.html"
            assert not pca_html.exists()

    def test_interactive_pca_disabled_in_config(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that interactive PCA respects config setting."""
        # Disable interactive PCA plots
        interactive_viz_config_enabled.interactive_viz.create_pca_plots = False

        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check no interactive PCA plot exists
        interactive_dir = tmp_path / "interactive_figures"
        if interactive_dir.exists():
            pca_html = interactive_dir / "interactive_pca.html"
            assert not pca_html.exists()


class TestGenerateInteractiveUMAPPlots:
    """Test interactive UMAP plot generation."""

    def test_interactive_umap_generated(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """Test that interactive UMAP plot is generated when UMAP results available."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Check interactive UMAP plot exists
        interactive_dir = tmp_path / "interactive_figures"
        umap_html = interactive_dir / "interactive_umap.html"
        assert umap_html.exists(), "Missing interactive_umap.html"

        # Verify it's a valid HTML file
        assert verify_html_structure(umap_html)

    def test_interactive_umap_skipped_without_results(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that interactive UMAP is skipped when no UMAP results available."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check no interactive UMAP plot exists
        interactive_dir = tmp_path / "interactive_figures"
        if interactive_dir.exists():
            umap_html = interactive_dir / "interactive_umap.html"
            assert not umap_html.exists()

    def test_interactive_umap_disabled_in_config(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """Test that interactive UMAP respects config setting."""
        # Disable interactive UMAP plots
        interactive_viz_config_enabled.interactive_viz.create_umap_plots = False

        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Check no interactive UMAP plot exists
        interactive_dir = tmp_path / "interactive_figures"
        if interactive_dir.exists():
            umap_html = interactive_dir / "interactive_umap.html"
            assert not umap_html.exists()


class TestGenerateInteractiveDependencyHandling:
    """Test plotly dependency handling."""

    @patch.dict("sys.modules", {"sleap_roots_analyze.interactive_visualization": None})
    def test_missing_plotly_dependency(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test graceful handling when plotly is not available."""
        # Mock module unavailable - this will cause ImportError when trying to import from it

        step = GenerateInteractiveStep()

        # Should not raise error, just skip
        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check result is returned without figures
        assert isinstance(result, StepResult)
        assert result.data.equals(sample_trait_data)

        # No figures should be generated
        interactive_dir = tmp_path / "interactive_figures"
        if interactive_dir.exists():
            assert len(list(interactive_dir.glob("*.html"))) == 0


class TestGenerateInteractiveManifest:
    """Test manifest generation and accuracy."""

    def test_manifest_created(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """Test that manifest JSON file is created."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Check manifest file exists
        manifest_file = tmp_path / "10_interactive_figures_manifest.json"
        assert manifest_file.exists()

    def test_manifest_content_accuracy(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """Test that manifest contains accurate information."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Check manifest metadata
        manifest = result.metadata["interactive_figures_manifest"]
        assert "total_figures" in manifest
        assert "files" in manifest

        # Verify file count matches
        assert manifest["total_figures"] == len(manifest["files"])
        assert len(result.metadata["interactive_figures"]) == manifest["total_figures"]

    def test_manifest_file_paths_exist(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """Test that all files listed in manifest actually exist."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Check all listed files exist
        manifest = result.metadata["interactive_figures_manifest"]
        for file_path in manifest["files"]:
            full_path = tmp_path / file_path
            assert (
                full_path.exists()
            ), f"File listed in manifest doesn't exist: {file_path}"


class TestGenerateInteractiveHTMLValidity:
    """Test HTML file structure and validity."""

    def test_html_contains_plotly_script(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that generated HTML contains plotly script tags."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check PCA HTML contains plotly references
        interactive_dir = tmp_path / "interactive_figures"
        pca_html = interactive_dir / "interactive_pca.html"

        if pca_html.exists():
            content = pca_html.read_text(encoding="utf-8")
            # Should contain plotly references (either CDN link or embedded script)
            assert "plotly" in content.lower() or "plotly.js" in content.lower()

    def test_html_is_standalone(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that HTML files are standalone (can be opened independently)."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check HTML has complete structure
        interactive_dir = tmp_path / "interactive_figures"
        pca_html = interactive_dir / "interactive_pca.html"

        if pca_html.exists():
            content = pca_html.read_text(encoding="utf-8")
            # Plotly HTML files should have basic HTML structure
            # Note: Plotly may not include DOCTYPE declaration
            assert "<html" in content.lower()
            assert "<head" in content.lower()
            assert "<body" in content.lower()


class TestGenerateInteractiveMetadata:
    """Test metadata handling and propagation."""

    def test_metadata_propagation(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that metadata from previous step is preserved."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check original metadata is preserved
        assert "trait_names" in result.metadata
        assert (
            result.metadata["trait_names"]
            == prev_result_with_pca.metadata["trait_names"]
        )

        # Check new metadata is added
        assert "interactive_figures" in result.metadata
        assert "interactive_figures_manifest" in result.metadata

    def test_files_generated_list(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """Test that files_generated list is populated correctly."""
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Check files_generated is not empty
        assert len(result.files_generated) > 0

        # Check all files in list exist
        for file_path in result.files_generated:
            assert (
                file_path.exists()
            ), f"File in files_generated doesn't exist: {file_path}"
