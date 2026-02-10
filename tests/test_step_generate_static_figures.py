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


class TestMemoryManagement:
    """Tests for memory management during figure generation."""

    def test_plt_close_called_after_batch_figure_save(
        self, sample_trait_data, static_viz_config_enabled, prev_result_minimal, tmp_path, monkeypatch
    ):
        """Verify plt.close() is called after saving each batch figure."""
        import matplotlib.pyplot as plt
        from unittest.mock import patch, MagicMock
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        # Track plt.close calls
        close_calls = []
        original_close = plt.close

        def tracking_close(fig=None):
            close_calls.append(fig)
            original_close(fig)

        with patch.object(plt, "close", side_effect=tracking_close):
            # Create minimal config for trait distributions only
            config = static_viz_config_enabled
            config.static_viz.create_pca_plots = False
            config.static_viz.create_heritability_plots = False
            config.static_viz.create_trait_correlations = False
            config.static_viz.create_genotype_comparisons = False

            result = step.execute(
                data=sample_trait_data,
                config=config,
                run_dir=tmp_path,
                prev_result=prev_result_minimal,
            )

        # Should have called plt.close at least once for histogram batches
        assert len(close_calls) > 0, "plt.close should be called after saving figures"

    def test_gc_collect_called_periodically_in_batch_generation(
        self, monkeypatch
    ):
        """Verify gc.collect() is called periodically during batch generation."""
        import gc
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        # Check that gc module is imported in generate_static_figures
        assert hasattr(generate_static_figures, "gc"), "gc module should be imported"

        # Verify the code has gc.collect() calls by inspecting the module
        import inspect
        source = inspect.getsource(generate_static_figures.GenerateStaticFiguresStep)
        assert "gc.collect()" in source, "gc.collect() should be called in batch generation loops"

    def test_no_figure_handle_accumulation(self):
        """Test that generating many figures in sequence doesn't accumulate handles."""
        import matplotlib.pyplot as plt
        from sleap_roots_analyze.visualization import create_trait_histograms
        import numpy as np
        import pandas as pd

        setup_matplotlib_backend()

        # Get initial figure count
        initial_figs = len(plt.get_fignums())

        # Generate 20 figures in sequence (simulating batch generation)
        for i in range(20):
            # Create simple test data
            data = pd.DataFrame({
                "trait1": np.random.randn(50),
                "trait2": np.random.randn(50),
            })
            fig = create_trait_histograms(data, ["trait1", "trait2"])
            plt.close(fig)

        # Check that figure count hasn't grown significantly
        final_figs = len(plt.get_fignums())
        accumulated = final_figs - initial_figs

        assert accumulated <= 2, (
            f"Figure handles accumulated: {accumulated}. "
            f"Expected <= 2 (initial: {initial_figs}, final: {final_figs})"
        )

    def test_batch_generation_memory_bounds(
        self, sample_trait_data, static_viz_config_enabled, prev_result_minimal, tmp_path
    ):
        """Test that batch generation keeps figure count within reasonable bounds."""
        import matplotlib.pyplot as plt

        setup_matplotlib_backend()

        step = GenerateStaticFiguresStep()

        # Record figure count before
        initial_figs = len(plt.get_fignums())

        # Run with minimal config
        config = static_viz_config_enabled
        config.static_viz.create_pca_plots = False
        config.static_viz.create_heritability_plots = False
        config.static_viz.create_trait_correlations = False

        result = step.execute(
            data=sample_trait_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Record figure count after
        final_figs = len(plt.get_fignums())

        # Should not accumulate many open figures
        # Allow some tolerance for figures that might not be closed immediately
        max_allowed_accumulation = 5
        accumulated = final_figs - initial_figs

        assert accumulated <= max_allowed_accumulation, (
            f"Too many figures accumulated during batch generation: {accumulated}. "
            f"Expected <= {max_allowed_accumulation}"
        )


class TestBatchFileReduction:
    """Tests for Section 8: Batch File Reduction."""

    def test_config_accepts_save_pdf_option(self, static_viz_config_enabled):
        """Test that StaticVisualizationConfig accepts save_pdf option."""
        assert hasattr(static_viz_config_enabled.static_viz, "save_pdf")
        # Default should be True (generate PDF alongside PNG)
        assert static_viz_config_enabled.static_viz.save_pdf is True

    def test_no_pdf_files_when_save_pdf_disabled(
        self, static_viz_config_enabled, sample_trait_data, prev_result_minimal, tmp_path
    ):
        """Test that no PDF files are generated when save_pdf is False, even if pdf in formats."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        # Disable PDF generation but include pdf in formats
        # The pipeline should respect save_pdf=False and skip PDF generation
        static_viz_config_enabled.static_viz.save_pdf = False
        static_viz_config_enabled.static_viz.formats = ["png", "pdf"]  # PDF is in formats

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check no PDF files exist (save_pdf=False should override formats list)
        static_dir = tmp_path / "static_figures"
        pdf_files = list(static_dir.glob("*.pdf"))
        assert len(pdf_files) == 0, (
            f"Should not generate PDF files when save_pdf=False, found: {pdf_files}"
        )

    def test_pdf_files_generated_when_save_pdf_enabled(
        self, static_viz_config_enabled, sample_trait_data, prev_result_minimal, tmp_path
    ):
        """Test that PDF files are generated when save_pdf is True."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        # Enable PDF generation
        static_viz_config_enabled.static_viz.save_pdf = True
        static_viz_config_enabled.static_viz.formats = ["png", "pdf"]

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check PDF files exist
        static_dir = tmp_path / "static_figures"
        pdf_files = list(static_dir.glob("*.pdf"))
        assert len(pdf_files) > 0, (
            "Should generate PDF files when save_pdf=True and pdf in formats"
        )


class TestMissingPlotsWiring:
    """Tests for Section 10: Wiring missing notebook plots into pipeline.

    These tests verify that plots from notebooks are properly wired
    into the pipeline step execution.
    """

    # --- 10a: PCA Feature Contribution Bar Chart ---

    def test_pca_feature_contributions_generated_when_pca_results_exist(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that PCA feature contributions bar chart is generated when PCA results exist."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check feature contributions plot exists
        static_dir = tmp_path / "static_figures"
        assert (static_dir / "pca_feature_contributions.png").exists(), (
            "Missing pca_feature_contributions.png when PCA results exist"
        )

    def test_pca_feature_contributions_not_generated_without_pca_results(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that PCA feature contributions is NOT generated when PCA results missing."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check feature contributions plot does NOT exist
        static_dir = tmp_path / "static_figures"
        assert not (static_dir / "pca_feature_contributions.png").exists(), (
            "pca_feature_contributions.png should not exist without PCA results"
        )

    # --- 10a2: PCA Feature Contribution Config ---

    def test_config_accepts_feature_contribution_fields(
        self,
        static_viz_config_enabled,
    ):
        """Test that config accepts feature contribution plot fields.

        Task 10.3a-b: StaticVisualizationConfig should have variance_threshold and top_n.
        """
        assert hasattr(
            static_viz_config_enabled.static_viz, "feature_contribution_variance_threshold"
        )
        assert hasattr(static_viz_config_enabled.static_viz, "feature_contribution_top_n")
        # Check defaults
        assert static_viz_config_enabled.static_viz.feature_contribution_variance_threshold is None
        assert static_viz_config_enabled.static_viz.feature_contribution_top_n == 20

    def test_passes_variance_threshold_from_pca_config_when_none(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
        monkeypatch,
    ):
        """Test that variance_threshold is passed from pca.n_components when static_viz is None.

        Task 10.3d: When feature_contribution_variance_threshold is None, inherit from pca.n_components.
        """
        from unittest.mock import Mock
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        # Set up config
        static_viz_config_enabled.static_viz.feature_contribution_variance_threshold = None
        static_viz_config_enabled.pca.n_components = 0.80  # Variance threshold

        # Mock the function to capture call args
        mock_contrib = Mock(return_value=Mock())
        monkeypatch.setattr(
            generate_static_figures, "create_feature_contribution_plot", mock_contrib
        )

        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Verify function was called with variance_threshold from pca config
        assert mock_contrib.called
        call_kwargs = mock_contrib.call_args.kwargs
        assert "variance_threshold" in call_kwargs
        assert call_kwargs["variance_threshold"] == 0.80

    def test_passes_explicit_variance_threshold_from_config(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
        monkeypatch,
    ):
        """Test that explicit variance_threshold from static_viz config is used.

        Task 10.3d: When feature_contribution_variance_threshold is set, use that value.
        """
        from unittest.mock import Mock
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        # Set explicit variance threshold different from pca.n_components
        static_viz_config_enabled.static_viz.feature_contribution_variance_threshold = 0.90
        static_viz_config_enabled.pca.n_components = 0.80

        # Mock the function to capture call args
        mock_contrib = Mock(return_value=Mock())
        monkeypatch.setattr(
            generate_static_figures, "create_feature_contribution_plot", mock_contrib
        )

        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Verify function was called with explicit variance_threshold
        assert mock_contrib.called
        call_kwargs = mock_contrib.call_args.kwargs
        assert "variance_threshold" in call_kwargs
        assert call_kwargs["variance_threshold"] == 0.90

    def test_passes_top_n_from_config(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
        monkeypatch,
    ):
        """Test that top_n is passed from config.

        Task 10.3e: top_n should come from feature_contribution_top_n config.
        """
        from unittest.mock import Mock
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        # Set custom top_n
        static_viz_config_enabled.static_viz.feature_contribution_top_n = 15

        # Mock the function to capture call args
        mock_contrib = Mock(return_value=Mock())
        monkeypatch.setattr(
            generate_static_figures, "create_feature_contribution_plot", mock_contrib
        )

        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Verify function was called with top_n from config
        assert mock_contrib.called
        call_kwargs = mock_contrib.call_args.kwargs
        assert "top_n" in call_kwargs
        assert call_kwargs["top_n"] == 15

    # --- 6b: PCA PC Boxplots Variance Threshold ---

    def test_pc_boxplots_uses_variance_threshold_from_pca_config(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
        monkeypatch,
    ):
        """Test that PC boxplots use variance_threshold from pca.n_components when <1.

        Task 6b.2: Same logic as feature contribution plot.
        """
        from unittest.mock import Mock
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        # Set pca.n_components to variance threshold (<1)
        static_viz_config_enabled.pca.n_components = 0.80

        # Mock the function to capture call args
        mock_boxplots = Mock(return_value=Mock())
        monkeypatch.setattr(
            generate_static_figures, "create_pc_genotype_boxplots", mock_boxplots
        )

        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Verify function was called with variance_threshold from pca config
        assert mock_boxplots.called
        call_kwargs = mock_boxplots.call_args.kwargs
        assert "variance_threshold" in call_kwargs or "n_components" in call_kwargs
        # If pca.n_components < 1, it should be passed as variance_threshold
        if "variance_threshold" in call_kwargs:
            assert call_kwargs["variance_threshold"] == 0.80
        else:
            # Or n_components should be None (to trigger variance threshold mode)
            assert call_kwargs.get("n_components") is None

    # --- 10b: Phenotype Variation Plots ---

    def test_config_accepts_phenotype_variation_fields(
        self,
        static_viz_config_enabled,
    ):
        """Test that config accepts phenotype variation plot fields."""
        # Check that fields exist and have correct defaults
        assert hasattr(static_viz_config_enabled.static_viz, "create_phenotype_variation_plots")
        assert hasattr(static_viz_config_enabled.static_viz, "phenotype_variation_top_n")
        assert static_viz_config_enabled.static_viz.create_phenotype_variation_plots is True
        assert static_viz_config_enabled.static_viz.phenotype_variation_top_n == 10

    def test_phenotype_variation_plots_generated_when_heritability_exists(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_heritability,
        tmp_path,
    ):
        """Test that phenotype variation plots are generated when heritability results exist."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        # Limit to just 2 plots for faster test
        static_viz_config_enabled.static_viz.phenotype_variation_top_n = 2

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_heritability,
        )

        # Check phenotype variation plots exist
        static_dir = tmp_path / "static_figures"
        variation_files = list(static_dir.glob("phenotype_variation_*.png"))
        assert len(variation_files) >= 1, (
            "Should generate at least 1 phenotype variation plot when heritability exists"
        )

    def test_phenotype_variation_plots_skipped_without_heritability(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that phenotype variation plots are skipped when heritability missing."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check no phenotype variation plots exist
        static_dir = tmp_path / "static_figures"
        variation_files = list(static_dir.glob("phenotype_variation_*.png"))
        assert len(variation_files) == 0, (
            "Should not generate phenotype variation plots without heritability results"
        )

    def test_phenotype_variation_plots_disabled_in_config(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_heritability,
        tmp_path,
    ):
        """Test that phenotype variation plots respect config disable setting."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        # Disable the plots
        static_viz_config_enabled.static_viz.create_phenotype_variation_plots = False

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_heritability,
        )

        # Check no phenotype variation plots exist
        static_dir = tmp_path / "static_figures"
        variation_files = list(static_dir.glob("phenotype_variation_*.png"))
        assert len(variation_files) == 0, (
            "Should not generate phenotype variation plots when disabled in config"
        )

    # --- 10c: Regression Plots ---

    def test_config_accepts_regression_trait_pairs(
        self,
        static_viz_config_enabled,
    ):
        """Test that config accepts regression_trait_pairs field."""
        assert hasattr(static_viz_config_enabled.static_viz, "regression_trait_pairs")
        # Default should be empty list
        assert static_viz_config_enabled.static_viz.regression_trait_pairs == []

    def test_regression_plots_generated_for_configured_pairs(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that regression plots are generated for each configured pair."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        # Configure trait pairs for regression
        static_viz_config_enabled.static_viz.regression_trait_pairs = [
            ["trait1", "trait2"],
            ["trait3", "trait4"],
        ]

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check regression plots exist
        static_dir = tmp_path / "static_figures"
        regression_files = list(static_dir.glob("regression_*.png"))
        assert len(regression_files) == 2, (
            f"Expected 2 regression plots, got {len(regression_files)}"
        )

    def test_no_regression_plots_when_pairs_empty(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that no regression plots are generated when regression_trait_pairs is empty."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        # Ensure pairs are empty (default)
        static_viz_config_enabled.static_viz.regression_trait_pairs = []

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check no regression plots exist
        static_dir = tmp_path / "static_figures"
        regression_files = list(static_dir.glob("regression_*.png"))
        assert len(regression_files) == 0, (
            "Should not generate regression plots when pairs list is empty"
        )

    # --- 10d: Genotype Image Grids ---

    def test_config_accepts_genotype_image_grids(
        self,
        static_viz_config_enabled,
    ):
        """Test that config accepts create_genotype_image_grids field."""
        assert hasattr(static_viz_config_enabled.static_viz, "create_genotype_image_grids")
        # Default should be True
        assert static_viz_config_enabled.static_viz.create_genotype_image_grids is True

    def test_genotype_image_grids_skipped_without_image_paths(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that genotype image grids are skipped when image paths not available."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        # Ensure create_genotype_image_grids is enabled
        static_viz_config_enabled.static_viz.create_genotype_image_grids = True

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,  # Has PCA but no image paths
        )

        # Check no image grid files exist (skipped due to missing image paths)
        static_dir = tmp_path / "static_figures"
        grid_files = list(static_dir.glob("genotype_grid_*.png"))
        assert len(grid_files) == 0, (
            "Should skip genotype image grids when image paths not available"
        )

    def test_genotype_image_grids_disabled_in_config(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that genotype image grids respect config disable setting."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        # Disable genotype image grids
        static_viz_config_enabled.static_viz.create_genotype_image_grids = False

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check no image grid files exist
        static_dir = tmp_path / "static_figures"
        grid_files = list(static_dir.glob("genotype_grid_*.png"))
        assert len(grid_files) == 0, (
            "Should not generate genotype image grids when disabled in config"
        )
