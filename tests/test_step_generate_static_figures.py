"""Tests for GenerateStaticFiguresStep (Step 9)."""

from __future__ import annotations

import matplotlib

from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import GenerateStaticFiguresStep
from tests.fixtures_visualization import (
    count_files_by_extension,
    setup_matplotlib_backend,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestGenerateStaticFiguresBasic:
    """Test basic functionality of GenerateStaticFiguresStep."""

    def test_step_initialization(self):
        """Test that step initializes with correct name and description."""
        step = GenerateStaticFiguresStep()

        assert step.step_name == "GenerateStaticFigures"
        assert "static figures" in step.description.lower()

    def test_basic_execution(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test basic step execution with minimal config."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check result structure
        assert isinstance(result, StepResult)
        assert isinstance(result.data, type(sample_trait_data))
        assert result.data.equals(sample_trait_data)

        # Check metadata exists
        assert "trait_names" in result.metadata
        assert "static_figures" in result.metadata
        assert "static_figures_manifest" in result.metadata

        # Check files were generated
        assert len(result.files_generated) > 0

    def test_step_disabled(
        self,
        static_viz_config_disabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that step skips execution when disabled."""
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_disabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check result structure
        assert isinstance(result, StepResult)
        assert result.data.equals(sample_trait_data)

        # Check no figures were generated
        assert "static_figures" not in result.metadata or not result.metadata.get(
            "static_figures"
        )
        assert len(result.files_generated) == 0

        # Check output directory wasn't created
        static_dir = tmp_path / "static_figures"
        if static_dir.exists():
            # If directory exists, it should be empty
            assert len(list(static_dir.iterdir())) == 0

    def test_output_directory_creation(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that output directory is created."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check directory was created
        static_dir = tmp_path / "static_figures"
        assert static_dir.exists()
        assert static_dir.is_dir()

        # Check files exist in directory
        files_in_dir = list(static_dir.iterdir())
        assert len(files_in_dir) > 0


class TestGenerateStaticFiguresPCAPlots:
    """Test PCA plot generation."""

    def test_pca_plots_generated(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that PCA plots are generated when PCA results available."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check PCA plots exist
        static_dir = tmp_path / "static_figures"
        pca_files = [
            "pca_scree_plot.png",
            "pca_biplot.png",
            "pca_feature_variance.png",
            "pca_feature_loadings.png",
            "pca_pc_boxplots.png",
        ]

        for pca_file in pca_files:
            assert (static_dir / pca_file).exists(), f"Missing {pca_file}"

    def test_pca_plots_skipped_without_results(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that PCA plots are skipped when no PCA results available."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check no PCA-specific plots exist
        static_dir = tmp_path / "static_figures"
        pca_files = list(static_dir.glob("pca_*.png"))
        assert len(pca_files) == 0

    def test_pca_plots_disabled_in_config(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that PCA plots respect config setting."""
        setup_matplotlib_backend()

        # Disable PCA plots
        static_viz_config_enabled.static_viz.create_pca_plots = False

        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check no PCA plots exist
        static_dir = tmp_path / "static_figures"
        pca_files = list(static_dir.glob("pca_*.png"))
        assert len(pca_files) == 0


class TestGenerateStaticFiguresTraitDistributions:
    """Test trait distribution plot generation."""

    def test_trait_distributions_generated(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that trait distribution plots are generated."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check distribution plots exist
        static_dir = tmp_path / "static_figures"
        histogram_files = list(static_dir.glob("trait_histograms_*.png"))
        assert len(histogram_files) > 0

    def test_trait_boxplots_by_genotype(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that boxplots by genotype are generated."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check boxplot files exist
        static_dir = tmp_path / "static_figures"
        boxplot_files = list(static_dir.glob("trait_boxplots_by_genotype_*.png"))
        assert len(boxplot_files) > 0

    def test_distributions_disabled_in_config(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that trait distributions respect config setting."""
        setup_matplotlib_backend()

        # Disable trait distributions
        static_viz_config_enabled.static_viz.create_trait_distributions = False

        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check no distribution plots exist
        static_dir = tmp_path / "static_figures"
        histogram_files = list(static_dir.glob("trait_histograms_*.png"))
        boxplot_files = list(static_dir.glob("trait_boxplots_*.png"))
        assert len(histogram_files) == 0
        assert len(boxplot_files) == 0


class TestGenerateStaticFiguresCorrelations:
    """Test correlation plot generation."""

    def test_correlation_heatmap_generated(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that correlation heatmap is generated."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check correlation heatmap exists
        static_dir = tmp_path / "static_figures"
        assert (static_dir / "trait_correlations.png").exists()

    def test_correlations_disabled_in_config(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that correlations respect config setting."""
        setup_matplotlib_backend()

        # Disable correlations
        static_viz_config_enabled.static_viz.create_trait_correlations = False

        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check no correlation plot exists
        static_dir = tmp_path / "static_figures"
        assert not (static_dir / "trait_correlations.png").exists()


class TestGenerateStaticFiguresHeritability:
    """Test heritability plot generation."""

    def test_heritability_plots_generated(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_heritability,
        tmp_path,
    ):
        """Test that heritability plots are generated when results available."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_heritability,
        )

        # Check heritability plot exists
        static_dir = tmp_path / "static_figures"
        assert (static_dir / "heritability_estimates.png").exists()

    def test_heritability_plots_skipped_without_results(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that heritability plots are skipped when no results available."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check no heritability plot exists
        static_dir = tmp_path / "static_figures"
        assert not (static_dir / "heritability_estimates.png").exists()

    def test_heritability_plots_disabled_in_config(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_heritability,
        tmp_path,
    ):
        """Test that heritability plots respect config setting."""
        setup_matplotlib_backend()

        # Disable heritability plots
        static_viz_config_enabled.static_viz.create_heritability_plots = False

        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_heritability,
        )

        # Check no heritability plot exists
        static_dir = tmp_path / "static_figures"
        assert not (static_dir / "heritability_estimates.png").exists()


class TestGenerateStaticFiguresFormats:
    """Test multiple output format support."""

    def test_multiple_formats_generated(
        self,
        static_viz_config_multiformat,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that figures are saved in multiple formats."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_multiformat,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check multiple format files exist
        static_dir = tmp_path / "static_figures"

        # Count files by extension
        png_count = count_files_by_extension(static_dir, "png")
        pdf_count = count_files_by_extension(static_dir, "pdf")
        svg_count = count_files_by_extension(static_dir, "svg")

        assert png_count > 0, "No PNG files generated"
        assert pdf_count > 0, "No PDF files generated"
        assert svg_count > 0, "No SVG files generated"

        # All formats should have same count (same figures, different formats)
        assert png_count == pdf_count == svg_count


class TestGenerateStaticFiguresManifest:
    """Test manifest generation and accuracy."""

    def test_manifest_created(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that manifest JSON file is created."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check manifest file exists
        manifest_file = tmp_path / "09_static_figures_manifest.json"
        assert manifest_file.exists()

    def test_manifest_content_accuracy(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that manifest contains accurate information."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check manifest metadata
        manifest = result.metadata["static_figures_manifest"]
        assert "total_figures" in manifest
        assert "formats" in manifest
        assert "dpi" in manifest
        assert "files" in manifest

        # Verify file count matches
        assert manifest["total_figures"] == len(manifest["files"])
        assert len(result.metadata["static_figures"]) == manifest["total_figures"]

    def test_manifest_file_paths_exist(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that all files listed in manifest actually exist."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check all listed files exist
        manifest = result.metadata["static_figures_manifest"]
        for file_path in manifest["files"]:
            full_path = tmp_path / file_path
            assert (
                full_path.exists()
            ), f"File listed in manifest doesn't exist: {file_path}"


class TestGenerateStaticFiguresMetadata:
    """Test metadata handling and propagation."""

    def test_metadata_propagation(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that metadata from previous step is preserved."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check original metadata is preserved
        assert "trait_names" in result.metadata
        assert (
            result.metadata["trait_names"]
            == prev_result_minimal.metadata["trait_names"]
        )

        # Check new metadata is added
        assert "static_figures" in result.metadata
        assert "static_figures_manifest" in result.metadata

    def test_files_generated_list(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that files_generated list is populated correctly."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check files_generated is not empty
        assert len(result.files_generated) > 0

        # Check all files in list exist
        for file_path in result.files_generated:
            assert (
                file_path.exists()
            ), f"File in files_generated doesn't exist: {file_path}"


class TestGenerateStaticFiguresGenotypeHighlighting:
    """Test genotype highlighting parameter passing."""

    def test_passes_genotypes_to_color_to_pca_biplot(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
        monkeypatch,
    ):
        """Test that genotypes_to_color is passed to create_pca_biplot."""
        from unittest.mock import Mock
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        # Configure genotype highlighting
        genotypes_to_color = ["GH_7401", "GH_7391", "GH_7361"]
        static_viz_config_enabled.static_viz.genotypes_to_color = genotypes_to_color

        # Mock the visualization functions at the module where they're used
        mock_biplot = Mock(return_value=Mock())  # Returns a mock figure
        monkeypatch.setattr(generate_static_figures, "create_pca_biplot", mock_biplot)

        # Mock other visualization functions to avoid errors
        mock_scree = Mock(return_value=Mock())
        mock_heatmap = Mock(return_value=(Mock(), Mock()))
        mock_boxplot = Mock(return_value=Mock())
        monkeypatch.setattr(
            generate_static_figures, "create_pca_scree_plot", mock_scree
        )
        monkeypatch.setattr(
            generate_static_figures, "create_feature_contribution_heatmap", mock_heatmap
        )
        monkeypatch.setattr(
            generate_static_figures, "create_pc_genotype_boxplots", mock_boxplot
        )

        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Verify create_pca_biplot was called with genotypes_to_color
        assert mock_biplot.called
        call_kwargs = mock_biplot.call_args.kwargs
        assert (
            "genotypes_to_color" in call_kwargs
        ), "genotypes_to_color not passed to create_pca_biplot"
        assert call_kwargs["genotypes_to_color"] == genotypes_to_color

    def test_passes_highlight_genotypes_to_pca_biplot(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
        monkeypatch,
    ):
        """Test that highlight_genotypes is passed to create_pca_biplot."""
        from unittest.mock import Mock
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        # Configure genotype highlighting
        highlight_genotypes = ["GH_7401"]
        static_viz_config_enabled.static_viz.highlight_genotypes = highlight_genotypes

        # Mock functions
        mock_biplot = Mock(return_value=Mock())
        mock_scree = Mock(return_value=Mock())
        mock_heatmap = Mock(return_value=(Mock(), Mock()))
        mock_boxplot = Mock(return_value=Mock())
        monkeypatch.setattr(generate_static_figures, "create_pca_biplot", mock_biplot)
        monkeypatch.setattr(
            generate_static_figures, "create_pca_scree_plot", mock_scree
        )
        monkeypatch.setattr(
            generate_static_figures, "create_feature_contribution_heatmap", mock_heatmap
        )
        monkeypatch.setattr(
            generate_static_figures, "create_pc_genotype_boxplots", mock_boxplot
        )

        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Verify create_pca_biplot was called with highlight_genotypes
        assert mock_biplot.called
        call_kwargs = mock_biplot.call_args.kwargs
        assert (
            "highlight_genotypes" in call_kwargs
        ), "highlight_genotypes not passed to create_pca_biplot"
        assert call_kwargs["highlight_genotypes"] == highlight_genotypes

    def test_passes_highlight_genotypes_to_pc_boxplots(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
        monkeypatch,
    ):
        """Test that highlight_genotypes is passed to create_pc_genotype_boxplots."""
        from unittest.mock import Mock
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        # Configure genotype highlighting
        highlight_genotypes = ["GH_7401", "GH_7391"]
        static_viz_config_enabled.static_viz.highlight_genotypes = highlight_genotypes

        # Mock functions
        mock_biplot = Mock(return_value=Mock())
        mock_scree = Mock(return_value=Mock())
        mock_heatmap = Mock(return_value=(Mock(), Mock()))
        mock_boxplot = Mock(return_value=Mock())
        monkeypatch.setattr(generate_static_figures, "create_pca_biplot", mock_biplot)
        monkeypatch.setattr(
            generate_static_figures, "create_pca_scree_plot", mock_scree
        )
        monkeypatch.setattr(
            generate_static_figures, "create_feature_contribution_heatmap", mock_heatmap
        )
        monkeypatch.setattr(
            generate_static_figures, "create_pc_genotype_boxplots", mock_boxplot
        )

        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Verify create_pc_genotype_boxplots was called with highlight_genotypes (if conditions met)
        if mock_boxplot.called:
            call_kwargs = mock_boxplot.call_args.kwargs
            assert (
                "highlight_genotypes" in call_kwargs
            ), "highlight_genotypes not passed to create_pc_genotype_boxplots"
            assert call_kwargs["highlight_genotypes"] == highlight_genotypes

    def test_both_parameters_passed_together(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
        monkeypatch,
    ):
        """Test that both highlighting parameters work together."""
        from unittest.mock import Mock
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        # Configure both highlighting parameters
        genotypes_to_color = ["GH_7401", "GH_7391", "GH_7361"]
        highlight_genotypes = ["GH_7401"]
        static_viz_config_enabled.static_viz.genotypes_to_color = genotypes_to_color
        static_viz_config_enabled.static_viz.highlight_genotypes = highlight_genotypes

        # Mock functions
        mock_biplot = Mock(return_value=Mock())
        mock_scree = Mock(return_value=Mock())
        mock_heatmap = Mock(return_value=(Mock(), Mock()))
        mock_boxplot = Mock(return_value=Mock())
        monkeypatch.setattr(generate_static_figures, "create_pca_biplot", mock_biplot)
        monkeypatch.setattr(
            generate_static_figures, "create_pca_scree_plot", mock_scree
        )
        monkeypatch.setattr(
            generate_static_figures, "create_feature_contribution_heatmap", mock_heatmap
        )
        monkeypatch.setattr(
            generate_static_figures, "create_pc_genotype_boxplots", mock_boxplot
        )

        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Verify both parameters passed to PCA biplot
        biplot_kwargs = mock_biplot.call_args.kwargs
        assert biplot_kwargs["genotypes_to_color"] == genotypes_to_color
        assert biplot_kwargs["highlight_genotypes"] == highlight_genotypes

        # Verify highlight_genotypes passed to boxplots (if conditions met)
        if mock_boxplot.called:
            boxplot_kwargs = mock_boxplot.call_args.kwargs
            assert boxplot_kwargs["highlight_genotypes"] == highlight_genotypes

    def test_none_values_passed_when_not_configured(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
        monkeypatch,
    ):
        """Test that None is passed when highlighting not configured (backward compat)."""
        from unittest.mock import Mock
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        # Ensure highlighting is None (default)
        assert static_viz_config_enabled.static_viz.genotypes_to_color is None
        assert static_viz_config_enabled.static_viz.highlight_genotypes is None

        # Mock functions
        mock_biplot = Mock(return_value=Mock())
        mock_scree = Mock(return_value=Mock())
        mock_heatmap = Mock(return_value=(Mock(), Mock()))
        mock_boxplot = Mock(return_value=Mock())
        monkeypatch.setattr(generate_static_figures, "create_pca_biplot", mock_biplot)
        monkeypatch.setattr(
            generate_static_figures, "create_pca_scree_plot", mock_scree
        )
        monkeypatch.setattr(
            generate_static_figures, "create_feature_contribution_heatmap", mock_heatmap
        )
        monkeypatch.setattr(
            generate_static_figures, "create_pc_genotype_boxplots", mock_boxplot
        )

        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Verify None values passed (backward compatibility)
        biplot_kwargs = mock_biplot.call_args.kwargs
        assert biplot_kwargs["genotypes_to_color"] is None
        assert biplot_kwargs["highlight_genotypes"] is None

        if mock_boxplot.called:
            boxplot_kwargs = mock_boxplot.call_args.kwargs
            assert boxplot_kwargs["highlight_genotypes"] is None
