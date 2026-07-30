"""Tests for GenerateStaticFiguresStep (Step 9)."""

from __future__ import annotations

import logging
import matplotlib

from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import GenerateStaticFiguresStep
from tests.fixtures_visualization import (
    count_files_by_extension,
    setup_matplotlib_backend,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestFigureOrganization:
    """Test that figures are organized into logical subdirectories.

    Per VIZ-OUTPUT-001: All static figures MUST be saved to subdirectories
    within a single `figures/` directory, organized by plot type.
    """

    def test_no_static_figures_directory_created(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that NO static_figures/ directory is created (legacy structure).

        Per VIZ-LEGACY-001: static_figures/ flat directory is removed.
        """
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Verify static_figures/ does NOT exist
        static_figures_dir = tmp_path / "static_figures"
        assert not static_figures_dir.exists(), (
            "static_figures/ directory should NOT be created. "
            "Figures should go to figures/ with subdirectories."
        )

    def test_pca_plots_saved_to_figures_pca_subdirectory(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
    ):
        """Test that PCA plots are saved to figures/pca/ subdirectory.

        Per VIZ-OUTPUT-001 Scenario: PCA figures saved to figures/pca/
        """
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_pca,
        )

        # Check PCA plots exist in figures/pca/
        pca_dir = tmp_path / "figures" / "pca"
        assert pca_dir.exists(), "figures/pca/ directory should be created"

        expected_pca_files = [
            "pca_scree_plot.png",
            "pca_biplot.png",
            "pca_feature_variance.png",
            "pca_feature_loadings.png",
            "pca_pc_boxplots.png",
        ]
        for filename in expected_pca_files:
            assert (pca_dir / filename).exists(), f"Missing {filename} in figures/pca/"

        # Verify NO PCA plots in root figures/ directory
        figures_dir = tmp_path / "figures"
        root_pca_files = list(figures_dir.glob("pca_*.png"))
        assert (
            len(root_pca_files) == 0
        ), f"No PCA plots should exist in root figures/ directory. Found: {root_pca_files}"

    def test_heritability_plots_saved_to_figures_heritability_subdirectory(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_heritability,
        tmp_path,
    ):
        """Test that heritability plots are saved to figures/heritability/ subdirectory.

        Per VIZ-OUTPUT-001 Scenario: Heritability figures saved to figures/heritability/
        """
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_with_heritability,
        )

        # Check heritability plots exist in figures/heritability/
        heritability_dir = tmp_path / "figures" / "heritability"
        assert (
            heritability_dir.exists()
        ), "figures/heritability/ directory should be created"

        heritability_files = list(heritability_dir.glob("heritability_*.png"))
        assert (
            len(heritability_files) > 0
        ), "Should have heritability plots in figures/heritability/"

    def test_trait_histograms_saved_to_figures_trait_histograms_subdirectory(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that trait histograms are saved to figures/trait_histograms/ subdirectory.

        Per VIZ-OUTPUT-001 Scenario: Batched trait plots saved to dedicated subdirectories
        """
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check histograms exist in figures/trait_histograms/
        histograms_dir = tmp_path / "figures" / "trait_histograms"
        assert (
            histograms_dir.exists()
        ), "figures/trait_histograms/ directory should be created"

        histogram_files = list(histograms_dir.glob("trait_histograms_*.png"))
        assert (
            len(histogram_files) > 0
        ), "Should have histogram plots in figures/trait_histograms/"

    def test_trait_boxplots_saved_to_figures_trait_boxplots_subdirectory(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that trait boxplots are saved to figures/trait_boxplots/ subdirectory.

        Per VIZ-OUTPUT-001 Scenario: Batched trait plots saved to dedicated subdirectories
        """
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check boxplots exist in figures/trait_boxplots/
        boxplots_dir = tmp_path / "figures" / "trait_boxplots"
        assert (
            boxplots_dir.exists()
        ), "figures/trait_boxplots/ directory should be created"

        boxplot_files = list(boxplots_dir.glob("trait_boxplots_*.png"))
        assert (
            len(boxplot_files) > 0
        ), "Should have boxplot plots in figures/trait_boxplots/"

    def test_correlation_heatmap_saved_to_figures_overview_subdirectory(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that correlation heatmap is saved to figures/overview/ subdirectory.

        Per VIZ-OUTPUT-001: Overview plots saved to figures/overview/
        """
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check correlation heatmap exists in figures/overview/
        overview_dir = tmp_path / "figures" / "overview"
        assert overview_dir.exists(), "figures/overview/ directory should be created"
        assert (
            overview_dir / "trait_correlations.png"
        ).exists(), "trait_correlations.png should be in figures/overview/"

    def test_phenotype_variation_saved_to_figures_phenotype_variation_subdirectory(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_heritability,
        tmp_path,
    ):
        """Test that phenotype variation plots are saved to figures/phenotype_variation/ subdirectory.

        Per VIZ-OUTPUT-001: Phenotype variation plots saved to figures/phenotype_variation/
        """
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

        # Check phenotype variation plots exist in figures/phenotype_variation/
        variation_dir = tmp_path / "figures" / "phenotype_variation"
        assert (
            variation_dir.exists()
        ), "figures/phenotype_variation/ directory should be created"

        variation_files = list(variation_dir.glob("phenotype_variation_*.png"))
        assert (
            len(variation_files) >= 1
        ), "Should have phenotype variation plots in figures/phenotype_variation/"

    def test_umap_plots_saved_to_figures_umap_subdirectory(
        self,
        static_viz_config_with_umap,
        sample_trait_data,
        prev_result_with_all_viz_data,
        tmp_path,
    ):
        """Test that UMAP plots are saved to figures/umap/ subdirectory.

        Per VIZ-OUTPUT-001: UMAP figures saved to figures/umap/
        """
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_with_umap,
            run_dir=tmp_path,
            prev_result=prev_result_with_all_viz_data,
        )

        # Check UMAP plots exist in figures/umap/
        umap_dir = tmp_path / "figures" / "umap"
        assert umap_dir.exists(), "figures/umap/ directory should be created"
        assert (
            umap_dir / "umap_top_traits.png"
        ).exists(), "umap_top_traits.png should be in figures/umap/"


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

        # Check figures/ directory was created with subdirectories
        figures_dir = tmp_path / "figures"
        assert figures_dir.exists()
        assert figures_dir.is_dir()

        # Check at least one subdirectory exists and has files
        subdirs = [d for d in figures_dir.iterdir() if d.is_dir()]
        assert len(subdirs) > 0, "Should have at least one subdirectory in figures/"
        total_files = sum(len(list(sd.iterdir())) for sd in subdirs)
        assert total_files > 0, "Should have files in figures/ subdirectories"


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

        # Check PCA plots exist in figures/pca/
        pca_dir = tmp_path / "figures" / "pca"
        pca_files = [
            "pca_scree_plot.png",
            "pca_biplot.png",
            "pca_feature_variance.png",
            "pca_feature_loadings.png",
            "pca_pc_boxplots.png",
        ]

        for pca_file in pca_files:
            assert (pca_dir / pca_file).exists(), f"Missing {pca_file} in figures/pca/"

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

        # Check no PCA-specific plots exist (figures/pca/ shouldn't exist or be empty)
        pca_dir = tmp_path / "figures" / "pca"
        if pca_dir.exists():
            pca_files = list(pca_dir.glob("pca_*.png"))
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
        pca_dir = tmp_path / "figures" / "pca"
        if pca_dir.exists():
            pca_files = list(pca_dir.glob("pca_*.png"))
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

        # Check distribution plots exist in figures/trait_histograms/
        histograms_dir = tmp_path / "figures" / "trait_histograms"
        histogram_files = list(histograms_dir.glob("trait_histograms_*.png"))
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

        # Check boxplot files exist in figures/trait_boxplots/
        boxplots_dir = tmp_path / "figures" / "trait_boxplots"
        boxplot_files = list(boxplots_dir.glob("trait_boxplots_*.png"))
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
        histograms_dir = tmp_path / "figures" / "trait_histograms"
        boxplots_dir = tmp_path / "figures" / "trait_boxplots"
        if histograms_dir.exists():
            histogram_files = list(histograms_dir.glob("trait_histograms_*.png"))
            assert len(histogram_files) == 0
        if boxplots_dir.exists():
            boxplot_files = list(boxplots_dir.glob("trait_boxplots_*.png"))
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

        # Check correlation heatmap exists in figures/overview/
        overview_dir = tmp_path / "figures" / "overview"
        assert (overview_dir / "trait_correlations.png").exists()

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

        # Check no correlation plot exists in figures/overview/
        overview_dir = tmp_path / "figures" / "overview"
        if overview_dir.exists():
            assert not (overview_dir / "trait_correlations.png").exists()


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

        # Check heritability plot exists in figures/heritability/
        heritability_dir = tmp_path / "figures" / "heritability"
        assert (heritability_dir / "heritability_estimates.png").exists()

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

        # Check no heritability plot exists in figures/heritability/
        heritability_dir = tmp_path / "figures" / "heritability"
        if heritability_dir.exists():
            assert not (heritability_dir / "heritability_estimates.png").exists()

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

        # Check no heritability plot exists in figures/heritability/
        heritability_dir = tmp_path / "figures" / "heritability"
        if heritability_dir.exists():
            assert not (heritability_dir / "heritability_estimates.png").exists()


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

        # Check multiple format files exist across all subdirectories in figures/
        figures_dir = tmp_path / "figures"

        # Count files by extension across all subdirectories
        png_count = 0
        pdf_count = 0
        svg_count = 0
        for subdir in figures_dir.rglob("*"):
            if subdir.is_file():
                if subdir.suffix == ".png":
                    png_count += 1
                elif subdir.suffix == ".pdf":
                    pdf_count += 1
                elif subdir.suffix == ".svg":
                    svg_count += 1

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
        self,
        sample_trait_data,
        static_viz_config_enabled,
        prev_result_minimal,
        tmp_path,
        monkeypatch,
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

    def test_gc_collect_called_periodically_in_batch_generation(self, monkeypatch):
        """Verify gc.collect() is called periodically during batch generation."""
        import gc
        from sleap_roots_analyze.pipeline.steps import generate_static_figures

        # Check that gc module is imported in generate_static_figures
        assert hasattr(generate_static_figures, "gc"), "gc module should be imported"

        # Verify the code has gc.collect() calls by inspecting the module
        import inspect

        source = inspect.getsource(generate_static_figures.GenerateStaticFiguresStep)
        assert (
            "gc.collect()" in source
        ), "gc.collect() should be called in batch generation loops"

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
            data = pd.DataFrame(
                {
                    "trait1": np.random.randn(50),
                    "trait2": np.random.randn(50),
                }
            )
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
        self,
        sample_trait_data,
        static_viz_config_enabled,
        prev_result_minimal,
        tmp_path,
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

    def test_peak_concurrent_figures_bounded_during_static_figures(
        self,
        static_viz_config_enabled,
        tmp_path,
    ):
        """Peak concurrently-open figures stays small even with many genotypes.

        GenerateStaticFiguresStep previously materialized the full batch list
        (via create_trait_histograms_batched/create_trait_boxplots_by_genotype_batched)
        before closing any figure -- peak memory was the sum of every batch
        figure, not the size of the single largest one (Issue #110). Sampled
        at both figure-creation time (plt.subplots) and close time (plt.close)
        -- sampling only at close time could miss a figure that's created but
        never explicitly closed, silently undercounting a leak.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from unittest.mock import patch

        setup_matplotlib_backend()

        np.random.seed(42)
        n_genotypes = 480
        n_traits = 30
        samples_per_geno = 2
        n_samples = n_genotypes * samples_per_geno
        data = {
            "Barcode": [f"sample_{i}" for i in range(n_samples)],
            "Genotype": [f"geno_{i:03d}" for i in range(n_genotypes)]
            * samples_per_geno,
        }
        for i in range(n_traits):
            data[f"trait_{i}"] = np.random.randn(n_samples)
        df = pd.DataFrame(data)
        trait_cols = [f"trait_{i}" for i in range(n_traits)]

        prev_result = StepResult(
            data=df,
            metadata={"valid_trait_names": trait_cols},
        )

        config = static_viz_config_enabled
        config.static_viz.create_pca_plots = False
        config.static_viz.create_heritability_plots = False
        config.static_viz.create_trait_correlations = False
        config.static_viz.create_genotype_comparisons = False

        baseline_fignums = len(plt.get_fignums())
        peak_fignums = baseline_fignums
        original_close = plt.close
        original_subplots = plt.subplots

        def tracking_close(*args, **kwargs):
            nonlocal peak_fignums
            peak_fignums = max(peak_fignums, len(plt.get_fignums()))
            return original_close(*args, **kwargs)

        def tracking_subplots(*args, **kwargs):
            result = original_subplots(*args, **kwargs)
            nonlocal peak_fignums
            peak_fignums = max(peak_fignums, len(plt.get_fignums()))
            return result

        step = GenerateStaticFiguresStep()

        with (
            patch("matplotlib.pyplot.close", side_effect=tracking_close),
            patch("matplotlib.pyplot.subplots", side_effect=tracking_subplots),
        ):
            step.execute(
                data=df,
                config=config,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        # Compared against a baseline captured just before execute() runs
        # (not an absolute count), so a figure left open by an unrelated test
        # earlier in the same pytest session can't inflate this result.
        peak_delta = peak_fignums - baseline_fignums
        assert peak_delta <= 5, (
            f"Peak concurrently-open figures above baseline was {peak_delta}, "
            "expected a small constant, not scaling with the total number of "
            "figures generated"
        )


class TestBatchFileReduction:
    """Tests for Section 8: Batch File Reduction."""

    def test_config_accepts_save_pdf_option(self, static_viz_config_enabled):
        """Test that StaticVisualizationConfig accepts save_pdf option."""
        assert hasattr(static_viz_config_enabled.static_viz, "save_pdf")
        # Default should be True (generate PDF alongside PNG)
        assert static_viz_config_enabled.static_viz.save_pdf is True

    def test_no_pdf_files_when_save_pdf_disabled(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
    ):
        """Test that no PDF files are generated when save_pdf is False, even if pdf in formats."""
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        # Disable PDF generation but include pdf in formats
        # The pipeline should respect save_pdf=False and skip PDF generation
        static_viz_config_enabled.static_viz.save_pdf = False
        static_viz_config_enabled.static_viz.formats = [
            "png",
            "pdf",
        ]  # PDF is in formats

        result = step.execute(
            data=sample_trait_data,
            config=static_viz_config_enabled,
            run_dir=tmp_path,
            prev_result=prev_result_minimal,
        )

        # Check no PDF files exist in any figures/ subdirectory
        figures_dir = tmp_path / "figures"
        pdf_files = list(figures_dir.rglob("*.pdf"))
        assert (
            len(pdf_files) == 0
        ), f"Should not generate PDF files when save_pdf=False, found: {pdf_files}"

    def test_pdf_files_generated_when_save_pdf_enabled(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_minimal,
        tmp_path,
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

        # Check PDF files exist in figures/ subdirectories
        figures_dir = tmp_path / "figures"
        pdf_files = list(figures_dir.rglob("*.pdf"))
        assert (
            len(pdf_files) > 0
        ), "Should generate PDF files when save_pdf=True and pdf in formats"


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

        # Check feature contributions plot exists in figures/pca/
        pca_dir = tmp_path / "figures" / "pca"
        assert (
            pca_dir / "pca_feature_contributions.png"
        ).exists(), "Missing pca_feature_contributions.png in figures/pca/ when PCA results exist"

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

        # Check feature contributions plot does NOT exist in figures/pca/
        pca_dir = tmp_path / "figures" / "pca"
        if pca_dir.exists():
            assert not (
                pca_dir / "pca_feature_contributions.png"
            ).exists(), (
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
            static_viz_config_enabled.static_viz,
            "feature_contribution_variance_threshold",
        )
        assert hasattr(
            static_viz_config_enabled.static_viz, "feature_contribution_top_n"
        )
        # Check defaults
        assert (
            static_viz_config_enabled.static_viz.feature_contribution_variance_threshold
            is None
        )
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
        static_viz_config_enabled.static_viz.feature_contribution_variance_threshold = (
            None
        )
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
        static_viz_config_enabled.static_viz.feature_contribution_variance_threshold = (
            0.90
        )
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
        assert hasattr(
            static_viz_config_enabled.static_viz, "create_phenotype_variation_plots"
        )
        assert hasattr(
            static_viz_config_enabled.static_viz, "phenotype_variation_top_n"
        )
        assert (
            static_viz_config_enabled.static_viz.create_phenotype_variation_plots
            is True
        )
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

        # Check phenotype variation plots exist in figures/phenotype_variation/
        variation_dir = tmp_path / "figures" / "phenotype_variation"
        variation_files = list(variation_dir.glob("phenotype_variation_*.png"))
        assert (
            len(variation_files) >= 1
        ), "Should generate at least 1 phenotype variation plot when heritability exists"

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

        # Check no phenotype variation plots exist in figures/phenotype_variation/
        variation_dir = tmp_path / "figures" / "phenotype_variation"
        if variation_dir.exists():
            variation_files = list(variation_dir.glob("phenotype_variation_*.png"))
            assert (
                len(variation_files) == 0
            ), "Should not generate phenotype variation plots without heritability results"

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

        # Check no phenotype variation plots exist in figures/phenotype_variation/
        variation_dir = tmp_path / "figures" / "phenotype_variation"
        if variation_dir.exists():
            variation_files = list(variation_dir.glob("phenotype_variation_*.png"))
            assert (
                len(variation_files) == 0
            ), "Should not generate phenotype variation plots when disabled in config"

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

        # Check regression plots exist in figures/overview/
        overview_dir = tmp_path / "figures" / "overview"
        regression_files = list(overview_dir.glob("regression_*.png"))
        assert (
            len(regression_files) == 2
        ), f"Expected 2 regression plots in figures/overview/, got {len(regression_files)}"

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

        # Check no regression plots exist in figures/overview/
        overview_dir = tmp_path / "figures" / "overview"
        if overview_dir.exists():
            regression_files = list(overview_dir.glob("regression_*.png"))
            assert (
                len(regression_files) == 0
            ), "Should not generate regression plots when pairs list is empty"

    # --- 10d: Genotype Image Grids ---

    def test_config_accepts_genotype_image_grids(
        self,
        static_viz_config_enabled,
    ):
        """Test that config accepts create_genotype_image_grids field."""
        assert hasattr(
            static_viz_config_enabled.static_viz, "create_genotype_image_grids"
        )
        # Default should be True
        assert static_viz_config_enabled.static_viz.create_genotype_image_grids is True

    def test_genotype_image_grids_skipped_without_image_paths(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        prev_result_with_pca,
        tmp_path,
        caplog,
    ):
        """Task 10.18: Test that genotype image grids are skipped with log message.

        This test MUST verify that:
        1. No grid files are generated when image paths not available
        2. A log message explains WHY grids were skipped

        The test should FAIL until task 10.20 is implemented, because currently
        the function isn't wired in at all and there's no skip logic with logging.
        """
        setup_matplotlib_backend()
        step = GenerateStaticFiguresStep()

        # Ensure create_genotype_image_grids is enabled
        static_viz_config_enabled.static_viz.create_genotype_image_grids = True

        with caplog.at_level(logging.INFO):
            result = step.execute(
                data=sample_trait_data,
                config=static_viz_config_enabled,
                run_dir=tmp_path,
                prev_result=prev_result_with_pca,  # Has PCA but no image paths
            )

        # Check no image grid files exist in figures/ subdirectories
        figures_dir = tmp_path / "figures"
        grid_files = list(figures_dir.rglob("genotype_grid_*.png"))
        assert (
            len(grid_files) == 0
        ), "Should skip genotype image grids when image paths not available"

        # CRITICAL: Check that a skip log message was emitted
        # This ensures the code has actual skip logic, not just missing implementation
        log_messages = caplog.text.lower()
        assert "image" in log_messages and "skip" in log_messages, (
            "Should log a message explaining that image grids were skipped "
            "due to missing image paths. Task 10.20 must include skip logging. "
            f"Actual log: {caplog.text}"
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

        # Check no image grid files exist in figures/ subdirectories
        figures_dir = tmp_path / "figures"
        grid_files = list(figures_dir.rglob("genotype_grid_*.png"))
        assert (
            len(grid_files) == 0
        ), "Should not generate genotype image grids when disabled in config"

    def test_genotype_image_grids_generated_when_image_paths_available(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """Test that genotype image grids ARE generated when image paths and PCA results available.

        Task 10.17: This test MUST FAIL until task 10.20 is implemented.
        The step should call identify_extreme_genotypes_by_pc() and create_genotype_image_grid()
        when image_paths metadata is present.
        """
        setup_matplotlib_backend()
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from unittest.mock import patch, MagicMock
        from sleap_roots_analyze.pipeline.core import StepResult

        step = GenerateStaticFiguresStep()

        # Enable genotype image grids
        static_viz_config_enabled.static_viz.create_genotype_image_grids = True

        n_samples = len(sample_trait_data)
        n_features = 3
        n_components = 3

        # Create complete mock PCA results
        mock_pca = {
            "transformed_data": np.random.randn(n_samples, n_components),
            "cumulative_variance_ratio": np.array([0.50, 0.75, 0.95]),
            "explained_variance_ratio": np.array([0.50, 0.25, 0.20]),
            "loadings": np.random.randn(n_features, n_components),
            "eigenvalues": np.array([3.0, 1.5, 0.8]),
            "feature_names": ["trait1", "trait2", "trait3"],
            "n_components": n_components,
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
                "valid_trait_names": ["trait1", "trait2", "trait3"],
                "pca_results": mock_pca,
                "image_paths": image_paths,  # Key addition - image paths available
            },
        )

        # Mock all PCA plotting functions to avoid dimension errors
        # Also mock create_genotype_image_grid since test images don't exist
        with (
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_scree_plot"
            ) as mock_scree,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_biplot"
            ) as mock_biplot,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_heatmap"
            ) as mock_heatmap,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_plot"
            ) as mock_contrib,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pc_genotype_boxplots"
            ) as mock_boxplots,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_genotype_image_grid"
            ) as mock_image_grid,
        ):
            mock_scree.return_value = plt.figure()
            mock_biplot.return_value = plt.figure()
            mock_heatmap.return_value = (plt.figure(), plt.figure())
            mock_contrib.return_value = plt.figure()
            mock_boxplots.return_value = plt.figure()
            mock_image_grid.return_value = plt.figure()

            result = step.execute(
                data=sample_trait_data,
                config=static_viz_config_enabled,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            # Verify create_genotype_image_grid was called (at least once for extreme genotypes)
            assert mock_image_grid.called, (
                "Should call create_genotype_image_grid() when image paths AND PCA results are available. "
                "Task 10.20: create_genotype_image_grid() must be wired into generate_static_figures.py"
            )

        plt.close("all")

    def test_genotype_image_grid_figures_closed_after_saving(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """Task 10.19: Verify plt.close is called for each image grid figure.

        Image grid figures should be closed immediately after saving to prevent
        memory accumulation when generating many grids.
        """
        setup_matplotlib_backend()
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from unittest.mock import patch, MagicMock, call

        from sleap_roots_analyze.pipeline.core import StepResult

        step = GenerateStaticFiguresStep()

        # Enable genotype image grids
        static_viz_config_enabled.static_viz.create_genotype_image_grids = True

        n_samples = len(sample_trait_data)
        n_features = 3
        n_components = 3

        # Create complete mock PCA results
        mock_pca = {
            "transformed_data": np.random.randn(n_samples, n_components),
            "cumulative_variance_ratio": np.array([0.50, 0.75, 0.95]),
            "explained_variance_ratio": np.array([0.50, 0.25, 0.20]),
            "loadings": np.random.randn(n_features, n_components),
            "feature_names": ["trait1", "trait2", "trait3"],
            "n_components": n_components,
        }

        # Mock image paths (one per sample)
        image_paths = pd.Series(
            [f"path/to/image_{i}.png" for i in range(n_samples)],
            index=sample_trait_data.index,
        )

        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2", "trait3"],
                "pca_results": mock_pca,
                "image_paths": image_paths,
            },
        )

        # Track which figures are created and closed
        created_figures = []

        def track_figure_creation():
            fig = plt.figure()
            created_figures.append(fig)
            return fig

        # Mock all plotting functions
        with (
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_scree_plot"
            ) as mock_scree,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_biplot"
            ) as mock_biplot,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_heatmap"
            ) as mock_heatmap,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_plot"
            ) as mock_contrib,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pc_genotype_boxplots"
            ) as mock_boxplots,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_genotype_image_grid"
            ) as mock_image_grid,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.plt.close"
            ) as mock_plt_close,
        ):
            mock_scree.return_value = plt.figure()
            mock_biplot.return_value = plt.figure()
            mock_heatmap.return_value = (plt.figure(), plt.figure())
            mock_contrib.return_value = plt.figure()
            mock_boxplots.return_value = plt.figure()
            mock_image_grid.side_effect = track_figure_creation

            result = step.execute(
                data=sample_trait_data,
                config=static_viz_config_enabled,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            # Verify that plt.close was called for each image grid figure
            if mock_image_grid.call_count > 0:
                # Count plt.close calls that match the figures created by image_grid
                close_calls = mock_plt_close.call_args_list
                figures_closed = [c[0][0] for c in close_calls if c[0]]

                # At minimum, plt.close should be called once per image grid created
                assert mock_plt_close.call_count >= mock_image_grid.call_count, (
                    f"plt.close should be called at least {mock_image_grid.call_count} times "
                    f"(once per image grid), but was called {mock_plt_close.call_count} times. "
                    "Task 10.19: Each image grid figure must be closed after saving."
                )

        plt.close("all")

    # --- 10e: Genotype Image Grid Configuration (add-cylinder-image-grid-config) ---

    def test_config_accepts_genotype_image_grid_image_type(
        self,
        static_viz_config_enabled,
    ):
        """Test that config accepts genotype_image_grid_image_type field.

        Task 1.1: StaticVizConfig accepts genotype_image_grid_image_type field.
        Default should be "features.png" for RhizoVision compatibility.
        """
        assert hasattr(
            static_viz_config_enabled.static_viz, "genotype_image_grid_image_type"
        )
        # Default should be "features.png"
        assert (
            static_viz_config_enabled.static_viz.genotype_image_grid_image_type
            == "features.png"
        )

    def test_config_accepts_genotype_image_grid_trait_cols(
        self,
        static_viz_config_enabled,
    ):
        """Test that config accepts genotype_image_grid_trait_cols field.

        Task 1.2: StaticVizConfig accepts genotype_image_grid_trait_cols field.
        Default should be None (no statistics shown).
        """
        assert hasattr(
            static_viz_config_enabled.static_viz, "genotype_image_grid_trait_cols"
        )
        # Default should be None
        assert (
            static_viz_config_enabled.static_viz.genotype_image_grid_trait_cols is None
        )

    def test_genotype_image_grid_uses_configured_image_type(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """Test that _create_genotype_image_grids uses image_type from config.

        Task 1.3: _create_genotype_image_grids() uses image_type from config.
        """
        setup_matplotlib_backend()
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from unittest.mock import patch

        from sleap_roots_analyze.pipeline.core import StepResult
        from sleap_roots_analyze.pipeline.steps import GenerateStaticFiguresStep

        step = GenerateStaticFiguresStep()

        # Set cylinder-style image type
        static_viz_config_enabled.static_viz.create_genotype_image_grids = True
        static_viz_config_enabled.static_viz.genotype_image_grid_image_type = "1.jpg"

        n_samples = len(sample_trait_data)
        n_features = 3
        n_components = 3

        # Create image paths for each sample
        image_paths = pd.Series(
            {i: f"/path/to/sample_{i}/1.jpg" for i in range(n_samples)},
            name="image_path",
        )

        # Create mock PCA results
        pca_results = {
            "pc_scores": pd.DataFrame(
                np.random.randn(n_samples, n_components),
                columns=[f"PC{i + 1}" for i in range(n_components)],
            ),
            "transformed_data": np.random.randn(n_samples, n_components),
            "loadings": np.random.randn(n_features, n_components),
            "explained_variance": np.array([3.5, 1.8, 0.9]),
            "explained_variance_ratio": np.array([0.45, 0.30, 0.15]),
            "cumulative_variance_ratio": np.array([0.45, 0.75, 0.90]),
            "eigenvalues": np.array([3.5, 1.8, 0.9]),
            "feature_names": ["trait1", "trait2", "trait3"],
            "n_components": n_components,
            "total_variance_explained": 0.90,
        }

        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "pca_results": pca_results,
                "image_paths": image_paths,
            },
        )

        with (
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_scree_plot"
            ) as mock_scree,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_biplot"
            ) as mock_biplot,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_heatmap"
            ) as mock_heatmap,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_plot"
            ) as mock_contrib,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pc_genotype_boxplots"
            ) as mock_boxplots,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_genotype_image_grid"
            ) as mock_image_grid,
        ):
            mock_scree.return_value = plt.figure()
            mock_biplot.return_value = plt.figure()
            mock_heatmap.return_value = (plt.figure(), plt.figure())
            mock_contrib.return_value = plt.figure()
            mock_boxplots.return_value = plt.figure()
            mock_image_grid.return_value = plt.figure()

            result = step.execute(
                data=sample_trait_data,
                config=static_viz_config_enabled,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            # Verify create_genotype_image_grid was called with the configured image_type
            if mock_image_grid.called:
                call_kwargs = mock_image_grid.call_args.kwargs
                assert call_kwargs.get("image_type") == "1.jpg", (
                    "Should pass configured image_type='1.jpg' to create_genotype_image_grid(). "
                    "Task 2.3: Use config.static_viz.genotype_image_grid_image_type"
                )

        plt.close("all")

    def test_genotype_image_grid_passes_configured_trait_cols(
        self,
        static_viz_config_enabled,
        sample_trait_data,
        tmp_path,
    ):
        """Test that _create_genotype_image_grids passes trait_cols from config.

        Task 1.4: _create_genotype_image_grids() passes trait_cols to create_genotype_image_grid().
        """
        setup_matplotlib_backend()
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from unittest.mock import patch

        from sleap_roots_analyze.pipeline.core import StepResult
        from sleap_roots_analyze.pipeline.steps import GenerateStaticFiguresStep

        step = GenerateStaticFiguresStep()

        # Set trait columns to show statistics for
        static_viz_config_enabled.static_viz.create_genotype_image_grids = True
        static_viz_config_enabled.static_viz.genotype_image_grid_trait_cols = [
            "trait1",
            "trait2",
        ]

        n_samples = len(sample_trait_data)
        n_features = 3
        n_components = 3

        # Create image paths for each sample
        image_paths = pd.Series(
            {i: f"/path/to/sample_{i}/features.png" for i in range(n_samples)},
            name="image_path",
        )

        # Create mock PCA results
        pca_results = {
            "pc_scores": pd.DataFrame(
                np.random.randn(n_samples, n_components),
                columns=[f"PC{i + 1}" for i in range(n_components)],
            ),
            "transformed_data": np.random.randn(n_samples, n_components),
            "loadings": np.random.randn(n_features, n_components),
            "explained_variance": np.array([3.5, 1.8, 0.9]),
            "explained_variance_ratio": np.array([0.45, 0.30, 0.15]),
            "cumulative_variance_ratio": np.array([0.45, 0.75, 0.90]),
            "eigenvalues": np.array([3.5, 1.8, 0.9]),
            "feature_names": ["trait1", "trait2", "trait3"],
            "n_components": n_components,
            "total_variance_explained": 0.90,
        }

        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "pca_results": pca_results,
                "image_paths": image_paths,
            },
        )

        with (
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_scree_plot"
            ) as mock_scree,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_biplot"
            ) as mock_biplot,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_heatmap"
            ) as mock_heatmap,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_plot"
            ) as mock_contrib,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pc_genotype_boxplots"
            ) as mock_boxplots,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_genotype_image_grid"
            ) as mock_image_grid,
        ):
            mock_scree.return_value = plt.figure()
            mock_biplot.return_value = plt.figure()
            mock_heatmap.return_value = (plt.figure(), plt.figure())
            mock_contrib.return_value = plt.figure()
            mock_boxplots.return_value = plt.figure()
            mock_image_grid.return_value = plt.figure()

            result = step.execute(
                data=sample_trait_data,
                config=static_viz_config_enabled,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            # Verify create_genotype_image_grid was called with the configured trait_cols
            if mock_image_grid.called:
                call_kwargs = mock_image_grid.call_args.kwargs
                assert call_kwargs.get("trait_cols") == ["trait1", "trait2"], (
                    "Should pass configured trait_cols to create_genotype_image_grid(). "
                    "Task 2.4: Use config.static_viz.genotype_image_grid_trait_cols"
                )

        plt.close("all")


class TestPCABoxplotsAdaptiveSizing:
    """Tests for Section 6b: PCA PC Boxplots Adaptive Sizing.

    Per fix-plot-scalability tasks 6b.3 and 6b.5:
    - PCA PC boxplots should scale width based on genotype count
    - PCA PC boxplots should scale height based on number of PCs
    """

    def test_pca_boxplots_with_many_genotypes_has_adequate_width(
        self,
        static_viz_config_enabled,
        tmp_path,
    ):
        """Test that PCA boxplots with 150 genotypes have adequate width.

        Per task 6b.3: PCA PC boxplots with 150 genotypes have width scaled
        appropriately using adaptive_sizing config.
        """
        setup_matplotlib_backend()
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from unittest.mock import patch

        from sleap_roots_analyze.pipeline.core import StepResult
        from sleap_roots_analyze.pipeline.steps import GenerateStaticFiguresStep

        # Create data with 150 genotypes (many genotypes scenario)
        n_samples = 450  # 3 samples per genotype
        n_genotypes = 150
        n_components = 3
        n_features = 2

        large_data = pd.DataFrame(
            {
                "Barcode": [f"sample_{i}" for i in range(n_samples)],
                "Genotype": [f"geno_{i % n_genotypes}" for i in range(n_samples)],
                "Replicate": [i % 3 + 1 for i in range(n_samples)],
                "trait1": np.random.randn(n_samples),
                "trait2": np.random.randn(n_samples),
            }
        )

        # Create mock PCA results with ALL required keys
        mock_pca = {
            "transformed_data": np.random.randn(n_samples, n_components),
            "cumulative_variance_ratio": np.array([0.50, 0.75, 0.95]),
            "explained_variance_ratio": np.array([0.50, 0.25, 0.20]),
            "loadings": np.random.randn(n_features, n_components),
            "eigenvalues": np.array([3.0, 1.5, 0.8]),
            "feature_names": ["trait1", "trait2"],
            "n_components": n_components,
        }

        prev_result = StepResult(
            data=large_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2"],
                "pca_results": mock_pca,
            },
        )

        step = GenerateStaticFiguresStep()

        # Mock all PCA plotting functions to avoid errors, but capture boxplot call
        with (
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_scree_plot"
            ) as mock_scree,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_biplot"
            ) as mock_biplot,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_heatmap"
            ) as mock_heatmap,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_plot"
            ) as mock_contrib,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pc_genotype_boxplots"
            ) as mock_boxplots,
        ):
            # Return mock figures for all
            mock_scree.return_value = plt.figure()
            mock_biplot.return_value = plt.figure()
            mock_heatmap.return_value = (plt.figure(), plt.figure())
            mock_contrib.return_value = plt.figure()
            mock_boxplots.return_value = plt.figure()

            result = step.execute(
                data=large_data,
                config=static_viz_config_enabled,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            # Verify boxplots function was called
            assert mock_boxplots.called, "create_pc_genotype_boxplots should be called"

            # Get the figsize that was passed
            call_kwargs = mock_boxplots.call_args.kwargs
            figsize = call_kwargs.get("figsize")

            # With 150 genotypes, width should be scaled up significantly
            # adaptive_sizing.min_width = 6.0, max_width = 20.0
            # 150 genotypes * 0.25 = 37.5, so should hit max_width
            assert (
                figsize is not None
            ), "figsize should be passed to create_pc_genotype_boxplots"
            assert (
                figsize[0] >= 15
            ), f"With 150 genotypes, figure width should be at least 15 inches, got {figsize[0]}"

            plt.close("all")

    def test_pca_boxplots_height_scales_with_pc_count(
        self,
        static_viz_config_enabled,
        tmp_path,
    ):
        """Test that PCA boxplots height scales with number of PCs.

        Each PC subplot should have adequate vertical space (~3 inches minimum).
        """
        setup_matplotlib_backend()
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from unittest.mock import patch

        from sleap_roots_analyze.pipeline.core import StepResult
        from sleap_roots_analyze.pipeline.steps import GenerateStaticFiguresStep

        # Create data with 5 PCs (should need ~15 inches height)
        n_samples = 50
        n_components = 5
        n_features = 1

        sample_data = pd.DataFrame(
            {
                "Barcode": [f"sample_{i}" for i in range(n_samples)],
                "Genotype": [f"geno_{i % 10}" for i in range(n_samples)],
                "Replicate": [i % 3 + 1 for i in range(n_samples)],
                "trait1": np.random.randn(n_samples),
            }
        )

        mock_pca = {
            "transformed_data": np.random.randn(n_samples, n_components),
            "cumulative_variance_ratio": np.array([0.30, 0.50, 0.70, 0.85, 0.95]),
            "explained_variance_ratio": np.array([0.30, 0.20, 0.20, 0.15, 0.10]),
            "loadings": np.random.randn(n_features, n_components),
            "eigenvalues": np.array([3.0, 2.0, 1.5, 1.0, 0.5]),
            "feature_names": ["trait1"],
            "n_components": n_components,
        }

        prev_result = StepResult(
            data=sample_data,
            metadata={
                "valid_trait_names": ["trait1"],
                "pca_results": mock_pca,
            },
        )

        step = GenerateStaticFiguresStep()

        with (
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_scree_plot"
            ) as mock_scree,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_biplot"
            ) as mock_biplot,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_heatmap"
            ) as mock_heatmap,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_plot"
            ) as mock_contrib,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pc_genotype_boxplots"
            ) as mock_boxplots,
        ):
            mock_scree.return_value = plt.figure()
            mock_biplot.return_value = plt.figure()
            mock_heatmap.return_value = (plt.figure(), plt.figure())
            mock_contrib.return_value = plt.figure()
            mock_boxplots.return_value = plt.figure()

            result = step.execute(
                data=sample_data,
                config=static_viz_config_enabled,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            assert mock_boxplots.called, "create_pc_genotype_boxplots should be called"

            call_kwargs = mock_boxplots.call_args.kwargs
            figsize = call_kwargs.get("figsize")

            # With 5 PCs, height should be at least 5 * 3 = 15 inches
            # (capped at adaptive_sizing.max_height = 16)
            assert figsize is not None, "figsize should be passed"
            assert (
                figsize[1] >= 12
            ), f"With 5 PCs, figure height should be at least 12 inches, got {figsize[1]}"

            plt.close("all")

    def test_pca_boxplots_uses_adaptive_sizing_config(
        self,
        static_viz_config_enabled,
        mock_pca_results,
        sample_trait_data,
        tmp_path,
    ):
        """Test that PCA boxplots uses adaptive_sizing config when enabled."""
        setup_matplotlib_backend()
        import matplotlib.pyplot as plt
        from unittest.mock import patch

        from sleap_roots_analyze.pipeline.core import StepResult
        from sleap_roots_analyze.pipeline.steps import GenerateStaticFiguresStep

        # Ensure adaptive_sizing is enabled (default)
        assert static_viz_config_enabled.adaptive_sizing.enabled is True

        prev_result = StepResult(
            data=sample_trait_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2"],
                "pca_results": mock_pca_results,
            },
        )

        step = GenerateStaticFiguresStep()

        with (
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_scree_plot"
            ) as mock_scree,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pca_biplot"
            ) as mock_biplot,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_heatmap"
            ) as mock_heatmap,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_feature_contribution_plot"
            ) as mock_contrib,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures.create_pc_genotype_boxplots"
            ) as mock_boxplots,
        ):
            mock_scree.return_value = plt.figure()
            mock_biplot.return_value = plt.figure()
            mock_heatmap.return_value = (plt.figure(), plt.figure())
            mock_contrib.return_value = plt.figure()
            mock_boxplots.return_value = plt.figure()

            result = step.execute(
                data=sample_trait_data,
                config=static_viz_config_enabled,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            assert mock_boxplots.called
            call_kwargs = mock_boxplots.call_args.kwargs

            # figsize should be passed when adaptive_sizing is enabled
            assert "figsize" in call_kwargs, (
                "figsize should be passed to create_pc_genotype_boxplots "
                "when adaptive_sizing is enabled"
            )

            plt.close("all")


class TestAdaptiveBatchSize:
    """Tests for Section 8: Adaptive batch sizing for large trait counts.

    When there are many traits (100+), batch sizes should increase automatically
    to reduce the number of output files. For example:
    - Default: 9 histograms per page, 6 boxplots per page
    - With 300 traits at defaults: 34 histogram pages, 50 boxplot pages
    - Adaptive (36 per page): only 9 histogram pages, 9 boxplot pages
    """

    def test_adaptive_batch_size_increases_for_many_traits(
        self,
        static_viz_config_enabled,
        tmp_path,
    ):
        """Test 8.1: adaptive batch size increases when trait count > 100.

        When there are many traits, the batch size should automatically increase
        to reduce the number of output batch files.
        """
        setup_matplotlib_backend()
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from unittest.mock import patch

        from sleap_roots_analyze.pipeline.core import StepResult
        from sleap_roots_analyze.pipeline.steps import GenerateStaticFiguresStep

        # Create data with 150 traits (well above the 100 threshold)
        n_samples = 50
        n_traits = 150
        data = {
            "Barcode": [f"sample_{i}" for i in range(n_samples)],
            "Genotype": [f"geno_{i % 5}" for i in range(n_samples)],
        }
        for i in range(n_traits):
            data[f"trait_{i}"] = np.random.randn(n_samples)
        df = pd.DataFrame(data)
        trait_cols = [f"trait_{i}" for i in range(n_traits)]

        prev_result = StepResult(
            data=df,
            metadata={
                "valid_trait_names": trait_cols,
            },
        )

        # Enable adaptive batch sizing
        static_viz_config_enabled.adaptive_sizing.enabled = True
        static_viz_config_enabled.adaptive_sizing.adaptive_batch_size = True

        step = GenerateStaticFiguresStep()

        with (
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures._generate_trait_histogram_batches"
            ) as mock_histograms,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures._generate_trait_boxplot_batches"
            ) as mock_boxplots,
        ):
            mock_histograms.return_value = [plt.figure() for _ in range(5)]
            mock_boxplots.return_value = [plt.figure() for _ in range(5)]

            result = step.execute(
                data=df,
                config=static_viz_config_enabled,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            # Verify histograms were called with increased batch_size
            assert mock_histograms.called
            hist_kwargs = mock_histograms.call_args.kwargs
            hist_batch_size = hist_kwargs.get("batch_size", 9)

            # With 150 traits, batch size should increase from default (9) to 36+
            assert hist_batch_size >= 36, (
                f"With 150 traits and adaptive_batch_size=True, "
                f"histogram batch_size should be >= 36 to reduce file count, "
                f"but got {hist_batch_size}"
            )

            plt.close("all")

    def test_cylinder_scale_generates_reasonable_batch_count(
        self,
        static_viz_config_enabled,
        tmp_path,
    ):
        """Test 8.3: cylinder-scale experiment (300+ traits) generates < 30 batch files.

        Large experiments should not create hundreds of batch files.
        """
        setup_matplotlib_backend()
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from unittest.mock import patch

        from sleap_roots_analyze.pipeline.core import StepResult
        from sleap_roots_analyze.pipeline.steps import GenerateStaticFiguresStep

        # Create cylinder-scale data with 300+ traits
        n_samples = 100
        n_traits = 350
        data = {
            "Barcode": [f"sample_{i}" for i in range(n_samples)],
            "Genotype": [f"geno_{i % 10}" for i in range(n_samples)],
        }
        for i in range(n_traits):
            data[f"trait_{i}"] = np.random.randn(n_samples)
        df = pd.DataFrame(data)
        trait_cols = [f"trait_{i}" for i in range(n_traits)]

        prev_result = StepResult(
            data=df,
            metadata={
                "valid_trait_names": trait_cols,
            },
        )

        # Enable adaptive batch sizing
        static_viz_config_enabled.adaptive_sizing.enabled = True
        static_viz_config_enabled.adaptive_sizing.adaptive_batch_size = True

        step = GenerateStaticFiguresStep()

        with (
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures._generate_trait_histogram_batches"
            ) as mock_histograms,
            patch(
                "sleap_roots_analyze.pipeline.steps.generate_static_figures._generate_trait_boxplot_batches"
            ) as mock_boxplots,
        ):
            # Calculate expected batch counts based on adaptive sizing
            # With adaptive=True and 350 traits, batch_size should be ~49 (7x7 grid)
            # So: ceil(350/49) = 8 batches max
            expected_max_batches = 30  # Per task requirement

            # Mock return values - simulate fewer batch files with adaptive sizing
            mock_histograms.return_value = [plt.figure() for _ in range(8)]
            mock_boxplots.return_value = [plt.figure() for _ in range(8)]

            result = step.execute(
                data=df,
                config=static_viz_config_enabled,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            # Check that batch_size was increased appropriately
            assert mock_histograms.called
            hist_kwargs = mock_histograms.call_args.kwargs
            hist_batch_size = hist_kwargs.get("batch_size", 9)

            # Calculate expected number of histogram batches
            expected_batches = (n_traits + hist_batch_size - 1) // hist_batch_size

            assert expected_batches < expected_max_batches, (
                f"With 350 traits and adaptive_batch_size=True, "
                f"should generate < {expected_max_batches} histogram batches, "
                f"but batch_size={hist_batch_size} would generate {expected_batches} batches"
            )

            plt.close("all")
