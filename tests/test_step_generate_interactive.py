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
        assert (
            interactive_dir.exists()
        ), "figures/interactive/ directory should be created"

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
            assert (
                not scatter_with_images.exists()
            ), "scatter_with_images.html should not exist without image paths"
            assert (
                not pca_image_viewer.exists()
            ), "pca_image_viewer.html should not exist without image paths"
            assert (
                not image_gallery.exists()
            ), "image_gallery.html should not exist without image paths"

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
        interactive_viz_config_enabled.interactive_viz.create_scatter_with_images = (
            False
        )
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

    def test_pca_with_images_generates_when_show_images_on_hover_enabled(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """TDD Test: PCA with images plot generates without error.

        Bug: _create_interactive_pca() passes wrong arguments to
        create_interactive_pca_with_images():
        - Passes pca_results["transformed_data"] instead of full pca_results dict
        - Passes genotype_col= instead of color_by=

        This causes: TypeError: create_interactive_pca_with_images()
        got an unexpected keyword argument 'genotype_col'

        Fix: Use correct parameter names and pass full pca_results dict.
        """
        import numpy as np
        from pathlib import Path

        step = GenerateInteractiveStep()

        # Enable show_images_on_hover which triggers create_interactive_pca_with_images
        interactive_viz_config_enabled.interactive_viz.show_images_on_hover = True

        # Get barcodes for image paths dict
        barcodes = sample_trait_data["Barcode"].tolist()
        n_samples = len(sample_trait_data)
        n_components = 3

        # Create mock PCA results (full dict required, not just transformed_data)
        mock_pca = {
            "transformed_data": np.random.randn(n_samples, n_components),
            "cumulative_variance_ratio": np.array([0.50, 0.75, 0.95]),
            "explained_variance_ratio": np.array([0.50, 0.25, 0.20]),
            "loadings": np.random.randn(3, n_components),  # Needed for loadings display
            "eigenvalues": np.array([2.5, 1.5, 0.5]),
        }

        # Create mock image paths in DICT format (matching barcodes)
        image_paths_dict = {
            barcode: {"features.png": Path(f"/mock/path/{barcode}_features.png")}
            for barcode in barcodes
        }

        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "trait_names": ["trait1", "trait2", "trait3"],
                "pca_results": mock_pca,
                "image_paths": image_paths_dict,
            },
        )

        # This should NOT raise TypeError about 'genotype_col'
        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Verify pca_with_images.html was generated
        interactive_dir = tmp_path / "figures" / "interactive"
        pca_images_file = interactive_dir / "pca_with_images.html"

        assert pca_images_file.exists(), (
            "Should generate pca_with_images.html when show_images_on_hover is True "
            "and image_paths are available. Bug: _create_interactive_pca passes "
            "genotype_col= instead of color_by= to create_interactive_pca_with_images()"
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

    def test_scatter_with_images_handles_dict_format_from_link_function(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """TDD Test: _create_image_dependent_plots handles dict format.

        The link_rhizovision_images_to_samples() function returns:
            Dict[str, Dict[str, Optional[Path]]]
            i.e., {barcode: {"features.png": Path, "seg.png": Path}}

        The current code incorrectly assumes image_paths is a pd.Series
        indexed by DataFrame row numbers. This test verifies the fix.
        """
        import numpy as np
        from pathlib import Path

        step = GenerateInteractiveStep()

        # Enable scatter with images
        interactive_viz_config_enabled.interactive_viz.create_scatter_with_images = True

        # Get existing barcode values from sample_trait_data (they are like "sample_0", "sample_1")
        # The config uses barcode="Barcode" (capital B) which matches sample_trait_data column
        barcodes = sample_trait_data["Barcode"].tolist()

        n_samples = len(sample_trait_data)
        n_components = 3

        # Create mock PCA results
        mock_pca = {
            "transformed_data": np.random.randn(n_samples, n_components),
            "cumulative_variance_ratio": np.array([0.50, 0.75, 0.95]),
            "explained_variance_ratio": np.array([0.50, 0.25, 0.20]),
        }

        # Create mock image paths in DICT format (as returned by link_rhizovision_images_to_samples)
        # Keys must match the EXISTING Barcode column values in the data
        # Format: {barcode: {img_type: Path}}
        image_paths_dict = {
            barcode: {"features.png": Path(f"/mock/path/{barcode}_features.png")}
            for barcode in barcodes
        }

        # Create prev_result with dict-format image paths
        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "trait_names": ["trait1", "trait2", "trait3"],
                "pca_results": mock_pca,
                "image_paths": image_paths_dict,  # Dict format, NOT pd.Series
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
            "Should generate scatter_with_images.html when image paths are provided "
            "in dict format {barcode: {img_type: Path}} from link_rhizovision_images_to_samples(). "
            "Fix: _create_image_dependent_plots must handle both dict and Series formats."
        )

        # CRITICAL: Verify that the HTML actually contains barcode references
        # This ensures image_links was properly populated from the dict format
        html_content = scatter_file.read_text(encoding="utf-8", errors="ignore")

        # Check that at least some barcodes appear in the HTML (hover data or customdata)
        # Barcodes are like "sample_0", "sample_1", etc.
        barcode_found = any(barcode in html_content for barcode in barcodes[:3])
        assert barcode_found, (
            "scatter_with_images.html should contain barcode references from image_links. "
            "This indicates the dict format {barcode: {img_type: Path}} was properly converted. "
            "Bug: _create_image_dependent_plots checks 'if idx in df.index' but df.index is "
            "integers (0,1,2...) while idx is a barcode string. This always fails."
        )

    def test_image_gallery_handles_dict_format_from_link_function(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """TDD Test: Image gallery works with dict format from link function."""
        import numpy as np
        from pathlib import Path

        step = GenerateInteractiveStep()

        # Enable image gallery
        interactive_viz_config_enabled.interactive_viz.create_image_gallery = True

        # Get existing barcode values from sample_trait_data
        barcodes = sample_trait_data["Barcode"].tolist()

        n_samples = len(sample_trait_data)
        n_components = 3

        # Create mock PCA results
        mock_pca = {
            "transformed_data": np.random.randn(n_samples, n_components),
            "cumulative_variance_ratio": np.array([0.50, 0.75, 0.95]),
            "explained_variance_ratio": np.array([0.50, 0.25, 0.20]),
        }

        # Create mock image paths in DICT format (keys match Barcode column values)
        image_paths_dict = {
            barcode: {"features.png": Path(f"/mock/path/{barcode}_features.png")}
            for barcode in barcodes
        }

        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "trait_names": ["trait1", "trait2", "trait3"],
                "pca_results": mock_pca,
                "image_paths": image_paths_dict,
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
            "Should generate image_gallery.html when image paths are provided "
            "in dict format from link_rhizovision_images_to_samples()."
        )


class TestImagePathsMetadataFlowIntegration:
    """Integration tests for image_paths metadata flow through Viz pipeline.

    These tests verify that image_paths set by LoadDataAndImagesStep
    flows through StatisticalAnalysisStep and PCAAnalysisStep to
    GenerateInteractiveStep, enabling image-dependent plots.

    The bug fixed: StatisticalAnalysisStep was not preserving previous
    step's metadata (missing **prev_result.metadata), causing image_paths
    to be lost and all image-dependent interactive plots to be skipped.
    """

    def test_image_paths_flows_through_statistical_to_pca_to_interactive(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """Integration test: image_paths flows from load -> stats -> pca -> interactive.

        This simulates the real Viz pipeline flow:
        1. LoadDataAndImagesStep sets image_paths in metadata
        2. StatisticalAnalysisStep MUST preserve image_paths
        3. PCAAnalysisStep preserves image_paths
        4. GenerateInteractiveStep reads image_paths and creates image-dependent plots

        Without the fix, step 2 would drop image_paths and step 4 would skip all
        image-dependent plots (scatter_with_images, image_gallery, etc.).
        """
        import numpy as np
        from pathlib import Path

        from sleap_roots_analyze.pipeline.steps import (
            StatisticalAnalysisStep,
            PCAAnalysisStep,
            GenerateInteractiveStep,
        )

        # Get existing barcode values
        barcodes = sample_trait_data["Barcode"].tolist()

        # Step 1: Simulate LoadDataAndImagesStep output (with image_paths)
        mock_image_paths = {
            barcode: {"features.png": Path(f"/mock/path/{barcode}_features.png")}
            for barcode in barcodes
        }

        load_result = StepResult(
            data=sample_trait_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2", "trait3"],
                "trait_names": ["trait1", "trait2", "trait3"],
                "n_samples": len(sample_trait_data),
                "image_paths": mock_image_paths,  # This must flow through
            },
        )

        # Step 2: Run StatisticalAnalysisStep
        stats_step = StatisticalAnalysisStep()
        stats_result = stats_step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=load_result,
        )

        # Verify image_paths is preserved after StatisticalAnalysisStep
        assert "image_paths" in stats_result.metadata, (
            "StatisticalAnalysisStep must preserve image_paths from previous step. "
            "This is the critical fix - without **prev_result.metadata, image_paths is lost."
        )

        # Step 3: Run PCAAnalysisStep
        pca_step = PCAAnalysisStep()
        pca_result = pca_step.execute(
            data=stats_result.data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=stats_result,
        )

        # Verify image_paths is still preserved after PCAAnalysisStep
        assert (
            "image_paths" in pca_result.metadata
        ), "PCAAnalysisStep must preserve image_paths."
        assert pca_result.metadata["image_paths"] == mock_image_paths

        # Step 4: Run GenerateInteractiveStep
        interactive_viz_config_enabled.interactive_viz.create_scatter_with_images = True
        interactive_viz_config_enabled.interactive_viz.create_image_gallery = True

        interactive_step = GenerateInteractiveStep()
        interactive_result = interactive_step.execute(
            data=pca_result.data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=pca_result,
        )

        # Verify image-dependent plots were generated
        interactive_dir = tmp_path / "figures" / "interactive"
        scatter_file = interactive_dir / "scatter_with_images.html"
        gallery_file = interactive_dir / "image_gallery.html"

        assert scatter_file.exists(), (
            "scatter_with_images.html should be generated when image_paths "
            "flows through the pipeline. This proves the metadata propagation fix works."
        )
        assert gallery_file.exists(), (
            "image_gallery.html should be generated when image_paths "
            "flows through the pipeline."
        )


class TestInteractiveUMAPVisualizationQuality:
    """Test interactive UMAP visualization quality requirements.

    These tests verify that interactive UMAP matches interactive PCA quality:
    - Points colored by genotype (not all grey)
    - Barcode shown on hover
    """

    def test_interactive_umap_colored_by_genotype(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """TDD Test: Interactive UMAP should color points by genotype.

        The current implementation uses create_interactive_umap_with_hover_highlight()
        which shows all points in grey. It should instead use a function that
        colors points by genotype, like the interactive PCA does.

        This test checks that the HTML output contains multiple colors (not just grey).
        """
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Check interactive UMAP exists
        interactive_dir = tmp_path / "figures" / "interactive"
        umap_html = interactive_dir / "interactive_umap.html"
        assert umap_html.exists(), "interactive_umap.html should exist"

        # Read HTML content
        html_content = umap_html.read_text(encoding="utf-8", errors="ignore")

        # The HTML should NOT contain "lightgrey" as the dominant color
        # (create_interactive_umap_with_hover_highlight uses lightgrey for all points)
        # Instead, it should contain color scale references for genotypes

        # Check for genotype coloring indicators:
        # 1. Multiple genotype names should appear (geno_0, geno_1, etc.)
        genotypes = sample_trait_data["Genotype"].unique()
        genotype_found = any(g in html_content for g in genotypes)
        assert genotype_found, (
            "Interactive UMAP should display genotype names in the visualization. "
            "Current implementation shows all points in grey."
        )

        # 2. Should NOT use the "legendonly" visibility pattern for colored traces
        # (create_interactive_umap_with_hover_highlight hides colored traces by default)
        # The new implementation should show colored traces directly
        assert '"visible":"legendonly"' not in html_content or html_content.count(
            '"visible":"legendonly"'
        ) < len(genotypes), (
            "Interactive UMAP should NOT hide genotype traces by default. "
            "Points should be colored by genotype and visible immediately."
        )

    def test_interactive_umap_shows_barcode_on_hover(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """TDD Test: Interactive UMAP should show Barcode on hover.

        The current implementation shows genotype name on hover, but should
        also show the Barcode for sample identification.

        This test checks that barcode values appear in hover data.
        """
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Check interactive UMAP exists
        interactive_dir = tmp_path / "figures" / "interactive"
        umap_html = interactive_dir / "interactive_umap.html"
        assert umap_html.exists(), "interactive_umap.html should exist"

        # Read HTML content
        html_content = umap_html.read_text(encoding="utf-8", errors="ignore")

        # Check that barcode values appear in the HTML
        # Barcodes are like "sample_0", "sample_1", etc.
        barcodes = sample_trait_data["Barcode"].tolist()
        barcode_found = any(b in html_content for b in barcodes[:5])

        assert barcode_found, (
            "Interactive UMAP should include Barcode values for hover display. "
            "This allows users to identify individual samples. "
            f"Expected barcodes like: {barcodes[:3]}"
        )

    def test_interactive_umap_matches_pca_style(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """TDD Test: Interactive UMAP should have similar style to interactive PCA.

        Both plots should:
        - Color points by genotype
        - Show sample info on hover (including Barcode)
        - Use the same color palette

        This test verifies structural similarity between the two HTML outputs.
        """
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Both plots should exist
        interactive_dir = tmp_path / "figures" / "interactive"
        pca_html = interactive_dir / "interactive_pca.html"
        umap_html = interactive_dir / "interactive_umap.html"

        assert pca_html.exists(), "interactive_pca.html should exist"
        assert umap_html.exists(), "interactive_umap.html should exist"

        pca_content = pca_html.read_text(encoding="utf-8", errors="ignore")
        umap_content = umap_html.read_text(encoding="utf-8", errors="ignore")

        # Both should use scatter plot type
        assert "scatter" in pca_content.lower()
        assert "scatter" in umap_content.lower()

        # Both should reference the genotype column
        genotype_col = interactive_viz_config_enabled.columns.genotype
        assert genotype_col in pca_content or "Genotype" in pca_content
        assert (
            genotype_col in umap_content or "Genotype" in umap_content
        ), f"UMAP should reference genotype column '{genotype_col}' like PCA does"


class TestInteractiveUMAPMetadataAndAlignment:
    """Integration tests for UMAP metadata preservation and data alignment.

    These tests verify that:
    - Metadata flows correctly through the interactive step
    - Data is properly aligned with UMAP embedding using clean_indices
    """

    def test_umap_metadata_preserved_through_interactive_step(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """Integration test: Metadata is preserved through GenerateInteractiveStep.

        Verifies:
        - umap_results flows from UMAPAnalysisStep to GenerateInteractiveStep
        - image_paths preserved in output metadata
        - trait_names preserved in output metadata
        """
        step = GenerateInteractiveStep()

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Verify umap_results preserved
        assert (
            "umap_results" in result.metadata
        ), "umap_results should be preserved in output metadata"
        assert (
            "embedding" in result.metadata["umap_results"]
        ), "umap_results should contain embedding"

        # Verify trait_names preserved
        assert (
            "trait_names" in result.metadata
        ), "trait_names should be preserved in output metadata"
        assert (
            result.metadata["trait_names"]
            == prev_result_with_all_viz_data.metadata["trait_names"]
        ), "trait_names should be unchanged"

    def test_interactive_umap_uses_clean_indices_for_alignment(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """Integration test: DataFrame is aligned with UMAP embedding using clean_indices.

        Verifies that when UMAP drops rows (due to NaN), the DataFrame is properly
        aligned so that Barcode values in hover data match the correct samples.
        """
        import numpy as np

        step = GenerateInteractiveStep()

        # Create UMAP results with only a subset of samples (simulating NaN removal)
        n_samples = len(sample_trait_data)
        subset_indices = list(range(0, n_samples, 2))  # Every other sample
        n_subset = len(subset_indices)

        mock_umap = {
            "embedding": np.random.randn(n_subset, 2),
            "n_neighbors": 15,
            "min_dist": 0.1,
            "clean_indices": subset_indices,  # Key: only these rows were used
        }

        mock_pca = {
            "transformed_data": np.random.randn(n_samples, 3),
            "explained_variance_ratio": np.array([0.5, 0.3, 0.2]),
            "cumulative_variance_ratio": np.array([0.5, 0.8, 1.0]),
        }

        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "trait_names": ["trait1", "trait2", "trait3"],
                "umap_results": mock_umap,
                "pca_results": mock_pca,
            },
        )

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Check UMAP plot was generated
        interactive_dir = tmp_path / "figures" / "interactive"
        umap_html = interactive_dir / "interactive_umap.html"
        assert umap_html.exists(), "interactive_umap.html should exist"

        # Read HTML and verify correct barcodes are present
        html_content = umap_html.read_text(encoding="utf-8", errors="ignore")

        # Only barcodes from subset_indices should appear
        # (since those are the samples in the embedding)
        expected_barcodes = [
            sample_trait_data.iloc[i]["Barcode"] for i in subset_indices[:3]
        ]
        barcode_found = any(b in html_content for b in expected_barcodes)

        assert barcode_found, (
            f"UMAP should contain barcodes from clean_indices samples. "
            f"Expected barcodes like: {expected_barcodes}"
        )

    def test_interactive_umap_full_dataframe_when_no_clean_indices(
        self,
        interactive_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """Test: Full DataFrame used when UMAP results have no clean_indices.

        When clean_indices is not provided, all samples should be used.
        """
        import numpy as np

        step = GenerateInteractiveStep()

        n_samples = len(sample_trait_data)

        # UMAP results WITHOUT clean_indices
        mock_umap = {
            "embedding": np.random.randn(n_samples, 2),
            "n_neighbors": 15,
            "min_dist": 0.1,
            # No clean_indices key
        }

        mock_pca = {
            "transformed_data": np.random.randn(n_samples, 3),
            "explained_variance_ratio": np.array([0.5, 0.3, 0.2]),
            "cumulative_variance_ratio": np.array([0.5, 0.8, 1.0]),
        }

        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "trait_names": ["trait1", "trait2", "trait3"],
                "umap_results": mock_umap,
                "pca_results": mock_pca,
            },
        )

        result = step.execute(
            data=sample_trait_data,
            config=interactive_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Check UMAP plot was generated
        interactive_dir = tmp_path / "figures" / "interactive"
        umap_html = interactive_dir / "interactive_umap.html"
        assert umap_html.exists(), "interactive_umap.html should exist"

        # All barcodes should be present since full DataFrame is used
        html_content = umap_html.read_text(encoding="utf-8", errors="ignore")
        all_barcodes = sample_trait_data["Barcode"].tolist()
        barcode_found = any(b in html_content for b in all_barcodes[:5])

        assert (
            barcode_found
        ), "UMAP should contain barcodes when no clean_indices provided"
