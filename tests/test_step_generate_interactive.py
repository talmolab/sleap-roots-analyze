"""Tests for GenerateInteractiveStep (Step 10)."""

from __future__ import annotations

from unittest.mock import patch

from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import GenerateInteractiveStep
from tests.fixtures_visualization import verify_html_structure


class TestInteractiveFigureOrganization:
    """Test that interactive figures are organized into figures/interactive/ subdirectory.

    Per VIZ-OUTPUT-001: Interactive figures saved to figures/interactive/
    """

    def test_no_interactive_figures_directory_created(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that NO interactive_figures/ directory is created (legacy structure).

        Per VIZ-LEGACY-001: interactive_figures/ directory at run root is removed.
        """
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Verify interactive_figures/ does NOT exist at run root
        legacy_dir = tmp_path / "interactive_figures"
        assert not legacy_dir.exists(), (
            "interactive_figures/ directory should NOT be created. "
            "Interactive figures should go to figures/interactive/"
        )

    def test_interactive_figures_saved_to_figures_interactive_subdirectory(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that interactive figures are saved to figures/interactive/ subdirectory.

        Per VIZ-OUTPUT-001 Scenario: Interactive figures saved to figures/interactive/
        """
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check interactive figures exist in figures/interactive/
        interactive_dir = tmp_path / "figures" / "interactive"
        assert interactive_dir.exists(), "figures/interactive/ directory should be created"

        # Should have at least one HTML file
        html_files = list(interactive_dir.glob("*.html"))
        assert len(html_files) > 0, "Should have HTML files in figures/interactive/"


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
        interactive_dir = tmp_path / "figures" / "interactive"
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
        interactive_dir = tmp_path / "figures" / "interactive"
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
        interactive_dir = tmp_path / "figures" / "interactive"
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
        interactive_dir = tmp_path / "figures" / "interactive"
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
        interactive_dir = tmp_path / "figures" / "interactive"
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
        interactive_dir = tmp_path / "figures" / "interactive"
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
        interactive_dir = tmp_path / "figures" / "interactive"
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
        interactive_dir = tmp_path / "figures" / "interactive"
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
        # Mock module unavailable - this will cause ImportError
        # when trying to import from it

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
        interactive_dir = tmp_path / "figures" / "interactive"
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
        interactive_dir = tmp_path / "figures" / "interactive"
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
        interactive_dir = tmp_path / "figures" / "interactive"
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


class TestInteractiveImageDependentPlots:
    """Tests for Section 11: Interactive plots that depend on image paths."""

    def test_config_accepts_image_dependent_fields(
        self,
        interactive_viz_config_enabled,
    ):
        """Test that config accepts image-dependent plot fields."""
        config = interactive_viz_config_enabled.interactive_viz

        # Check new fields exist with correct defaults
        assert hasattr(config, "create_scatter_with_images")
        assert hasattr(config, "create_image_viewer")
        assert hasattr(config, "create_image_gallery")

        assert config.create_scatter_with_images is True
        assert config.create_image_viewer is True
        assert config.create_image_gallery is True

    def test_image_dependent_plots_skipped_without_image_paths(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that image-dependent plots are skipped when image paths not available."""
        step = GenerateInteractiveStep()

        # Enable all image-dependent plots
        interactive_viz_config_enabled.interactive_viz.create_scatter_with_images = True
        interactive_viz_config_enabled.interactive_viz.create_image_viewer = True
        interactive_viz_config_enabled.interactive_viz.create_image_gallery = True

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,  # Has PCA but no image paths
        )

        # Check image-dependent plots don't exist (no image paths available)
        interactive_dir = tmp_path / "figures" / "interactive"
        if interactive_dir.exists():
            scatter_with_images = interactive_dir / "scatter_with_images.html"
            pca_image_viewer = interactive_dir / "pca_image_viewer.html"
            image_gallery = interactive_dir / "image_gallery.html"

            # These should not exist because no image paths are in metadata
            assert not scatter_with_images.exists(), (
                "scatter_with_images.html should not exist without image paths"
            )
            assert not pca_image_viewer.exists(), (
                "pca_image_viewer.html should not exist without image paths"
            )
            assert not image_gallery.exists(), (
                "image_gallery.html should not exist without image paths"
            )

    def test_image_dependent_plots_disabled_in_config(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that image-dependent plots respect config disable settings."""
        step = GenerateInteractiveStep()

        # Disable all image-dependent plots
        interactive_viz_config_enabled.interactive_viz.create_scatter_with_images = False
        interactive_viz_config_enabled.interactive_viz.create_image_viewer = False
        interactive_viz_config_enabled.interactive_viz.create_image_gallery = False

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check result is valid
        assert isinstance(result, StepResult)
        assert result.data.equals(sample_trait_data)

    def test_scatter_with_images_generated_when_image_paths_available(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """Task 11.3: This test MUST FAIL until task 11.7 is implemented.

        The step should call create_interactive_scatter_with_images()
        when image_paths metadata is present and config is enabled.
        """
        import numpy as np
        import pandas as pd

        step = GenerateInteractiveStep()

        # Enable scatter with images
        interactive_viz_config_enabled.interactive_viz.create_scatter_with_images = True

        n_samples = len(sample_trait_data)
        n_components = 3

        # Create mock PCA results
        mock_pca = {
            "transformed_data": np.random.randn(n_samples, n_components),
            "cumulative_variance_ratio": np.array([0.50, 0.75, 0.95]),
            "explained_variance_ratio": np.array([0.50, 0.25, 0.20]),
        }

        # Create mock image paths (one per sample)
        image_paths = pd.Series(
            [f"/mock/path/image_{i}.png" for i in range(n_samples)],
            index=sample_trait_data.index,
        )

        # Create prev_result with both PCA results AND image paths
        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "trait_names": ["trait1", "trait2", "trait3"],
                "pca_results": mock_pca,
                "image_paths": image_paths,  # Key addition - image paths available
            },
        )

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Check that scatter_with_images.html was generated
        interactive_dir = tmp_path / "figures" / "interactive"
        scatter_file = interactive_dir / "scatter_with_images.html"

        assert scatter_file.exists(), (
            "Should generate scatter_with_images.html when image paths are available. "
            "Task 11.7: create_interactive_scatter_with_images() must be wired into "
            "generate_interactive.py, guarded by image path availability."
        )

    def test_pca_image_viewer_generated_when_image_paths_available(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """Task 11.4: This test MUST FAIL until task 11.8 is implemented.

        The step should call create_html_with_image_viewer() using PCA plot as base
        when image_paths metadata is present and PCA plot exists.
        """
        import numpy as np
        import pandas as pd

        step = GenerateInteractiveStep()

        # Enable image viewer
        interactive_viz_config_enabled.interactive_viz.create_image_viewer = True

        n_samples = len(sample_trait_data)
        n_components = 3

        # Create mock PCA results
        mock_pca = {
            "transformed_data": np.random.randn(n_samples, n_components),
            "cumulative_variance_ratio": np.array([0.50, 0.75, 0.95]),
            "explained_variance_ratio": np.array([0.50, 0.25, 0.20]),
        }

        # Create mock image paths (one per sample)
        image_paths = pd.Series(
            [f"/mock/path/image_{i}.png" for i in range(n_samples)],
            index=sample_trait_data.index,
        )

        # Create prev_result with both PCA results AND image paths
        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "trait_names": ["trait1", "trait2", "trait3"],
                "pca_results": mock_pca,
                "image_paths": image_paths,  # Key addition - image paths available
            },
        )

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Check that pca_image_viewer.html was generated
        interactive_dir = tmp_path / "figures" / "interactive"
        viewer_file = interactive_dir / "pca_image_viewer.html"

        assert viewer_file.exists(), (
            "Should generate pca_image_viewer.html when image paths and PCA plot are available. "
            "Task 11.8: create_html_with_image_viewer() must be wired into "
            "generate_interactive.py, using PCA plot as base figure."
        )

    def test_image_gallery_generated_when_image_paths_available(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """Task 11.5: This test MUST FAIL until task 11.9 is implemented.

        The step should call create_interactive_image_gallery()
        when image_paths metadata is present and config is enabled.
        """
        import numpy as np
        import pandas as pd

        step = GenerateInteractiveStep()

        # Enable image gallery
        interactive_viz_config_enabled.interactive_viz.create_image_gallery = True

        n_samples = len(sample_trait_data)

        # Create mock image paths (one per sample)
        image_paths = pd.Series(
            [f"/mock/path/image_{i}.png" for i in range(n_samples)],
            index=sample_trait_data.index,
        )

        # Create prev_result with image paths (PCA not required for gallery)
        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "trait_names": ["trait1", "trait2", "trait3"],
                "image_paths": image_paths,  # Key addition - image paths available
            },
        )

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Check that image_gallery.html was generated
        interactive_dir = tmp_path / "figures" / "interactive"
        gallery_file = interactive_dir / "image_gallery.html"

        assert gallery_file.exists(), (
            "Should generate image_gallery.html when image paths are available. "
            "Task 11.9: create_interactive_image_gallery() must be wired into "
            "generate_interactive.py, guarded by image path availability."
        )

    def test_image_dependent_plots_skipped_with_log_message(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
        caplog,
    ):
        """Task 11.6: Test that image-dependent plots are skipped WITH LOG MESSAGE.

        This test ensures that when image paths are not available:
        1. No image-dependent HTML files are generated
        2. A log message explains WHY the plots were skipped

        This test MUST FAIL until tasks 11.7-11.9 are implemented with skip logging.
        """
        import logging

        step = GenerateInteractiveStep()

        # Enable all image-dependent plots
        interactive_viz_config_enabled.interactive_viz.create_scatter_with_images = True
        interactive_viz_config_enabled.interactive_viz.create_image_viewer = True
        interactive_viz_config_enabled.interactive_viz.create_image_gallery = True

        with caplog.at_level(logging.INFO):
            result = step.execute(
                data=sample_trait_data,
                config=interactive_viz_config_enabled,
                run_dir=tmp_path,
                prev_result=prev_result_with_pca,  # Has PCA but NO image paths
            )

        # Check image-dependent plots don't exist
        interactive_dir = tmp_path / "figures" / "interactive"
        if interactive_dir.exists():
            assert not (interactive_dir / "scatter_with_images.html").exists()
            assert not (interactive_dir / "pca_image_viewer.html").exists()
            assert not (interactive_dir / "image_gallery.html").exists()

        # CRITICAL: Check that a skip log message was emitted
        # This ensures the code has actual skip logic, not just missing implementation
        log_messages = caplog.text.lower()
        assert "image" in log_messages and "skip" in log_messages, (
            "Should log a message explaining that image-dependent plots were skipped "
            "due to missing image paths. Tasks 11.7-11.9 must include skip logging. "
            f"Actual log: {caplog.text}"
        )
