"""Tests for cross-platform summary generator.

TDD tests for the CrossPlatformSummaryGenerator that generates detailed
markdown summaries from cross-platform correlation pipeline outputs.
"""

from __future__ import annotations

import json

import pytest
import pandas as pd


# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def mock_cross_platform_run(tmp_path):
    """Create a mock cross-platform pipeline run directory with all outputs.

    Returns:
        Path: Path to the mock cross-platform run directory
    """
    run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
    run_dir.mkdir(parents=True)

    # Create correlation CSV
    corr_data = pd.DataFrame(
        {
            "exp1_trait": ["TraitA", "TraitB", "TraitC", "TraitD", "TraitE"],
            "exp2_trait": ["TraitX", "TraitY", "TraitZ", "TraitW", "TraitV"],
            "spearman_r": [0.85, 0.72, -0.65, 0.45, 0.30],
            "spearman_p": [0.001, 0.01, 0.02, 0.1, 0.2],
            "spearman_p_adjusted": [0.005, 0.04, 0.05, 0.2, 0.3],
            "pearson_r": [0.82, 0.70, -0.62, 0.42, 0.28],
            "pearson_p": [0.002, 0.015, 0.025, 0.12, 0.22],
            "pearson_p_adjusted": [0.008, 0.05, 0.06, 0.22, 0.35],
            "spearman_r_ci_low": [0.65, 0.52, -0.85, 0.15, 0.0],
            "spearman_r_ci_high": [0.95, 0.88, -0.35, 0.70, 0.55],
            "pearson_r_ci_low": [0.62, 0.50, -0.82, 0.12, -0.05],
            "pearson_r_ci_high": [0.92, 0.85, -0.32, 0.68, 0.52],
            "n_genotypes": [19, 19, 19, 19, 19],
            "achieved_power": [0.95, 0.82, 0.75, 0.45, 0.25],
            "significant_fdr": [True, True, False, False, False],
        }
    )
    corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

    # Create exp1 trait clusters CSV
    exp1_clusters = pd.DataFrame(
        {
            "trait": ["TraitA", "TraitB", "TraitC", "TraitD", "TraitE"],
            "cluster_id": [0, 0, 1, 2, 2],
            "is_representative": [True, False, True, True, False],
            "variance": [1.5, 1.2, 0.8, 0.6, 0.5],
        }
    )
    exp1_clusters.to_csv(run_dir / "exp1_trait_clusters.csv", index=False)

    # Create exp2 trait clusters CSV
    exp2_clusters = pd.DataFrame(
        {
            "trait": ["TraitX", "TraitY", "TraitZ", "TraitW", "TraitV"],
            "cluster_id": [0, 0, 1, 1, 2],
            "is_representative": [True, False, True, False, True],
            "variance": [2.0, 1.8, 1.0, 0.9, 0.4],
        }
    )
    exp2_clusters.to_csv(run_dir / "exp2_trait_clusters.csv", index=False)

    # Create pipeline summary JSON
    pipeline_summary = {
        "pipeline_name": "cross_platform_Exp1_vs_Exp2",
        "version": "1.0",
        "start_time": "2026-02-02T12:00:00",
        "end_time": "2026-02-02T12:01:00",
        "total_elapsed_time": 60.0,
        "status": "success",
        "steps": [
            {
                "name": "01_load_cross_platform_data",
                "status": "success",
                "metadata": {
                    "exp1_name": "Experiment 1",
                    "exp2_name": "Experiment 2",
                    "exp1_samples": 100,
                    "exp2_samples": 150,
                    "exp1_traits": 5,
                    "exp2_traits": 5,
                    "common_genotypes": 19,
                },
            },
            {
                "name": "02_reduce_trait_redundancy",
                "status": "success",
                "metadata": {
                    "trait_reduction_method": "clustering",
                    "trait_reduction_target": "both",
                    "trait_clustering_threshold": 0.8,
                    "trait_clustering_linkage": "complete",
                    "exp1_original_traits": 5,
                    "exp1_reduced_traits": 3,
                    "exp1_n_clusters": 3,
                    "exp2_original_traits": 5,
                    "exp2_reduced_traits": 3,
                    "exp2_n_clusters": 3,
                },
            },
            {
                "name": "03_calculate_correlations",
                "status": "success",
                "metadata": {
                    "total_correlations": 5,
                    "correlation_method": "spearman",
                    "fdr_correction_method": "fdr_bh",
                    "significance_level": 0.05,
                    "significant_correlations": 2,
                    "power_analysis_alpha": 0.05,
                    "power_analysis_power": 0.8,
                    "minimum_detectable_r": 0.60,
                    "modal_n_genotypes": 19,
                },
            },
        ],
        "config": {
            "exp1_name": "Experiment 1",
            "exp2_name": "Experiment 2",
            "correlation_method": "spearman",
            "trait_reduction_method": "clustering",
            "trait_reduction_target": "both",
            "trait_clustering_threshold": 0.8,
        },
    }
    with open(run_dir / "pipeline_summary.json", "w") as f:
        json.dump(pipeline_summary, f, indent=2)

    # Create mock visualization files (empty PNGs for testing)
    for viz_file in [
        "exp1_trait_clustering_dendrogram.png",
        "exp1_trait_cluster_heatmap.png",
        "exp2_trait_clustering_dendrogram.png",
        "exp2_trait_cluster_heatmap.png",
        "correlation_summary.png",
        "joint_plot_TraitA_vs_TraitX.png",
        "joint_plot_TraitB_vs_TraitY.png",
        "joint_plot_TraitC_vs_TraitZ.png",
    ]:
        (run_dir / viz_file).touch()

    return run_dir


@pytest.fixture
def mock_run_with_multiple_comparisons(tmp_path):
    """Create a mock pipeline run directory with multiple cross-platform comparisons.

    Returns:
        Path: Path to the main pipeline run directory
    """
    main_run_dir = tmp_path / "2026-02-02_120000"
    cross_platform_dir = main_run_dir / "cross_platform"
    cross_platform_dir.mkdir(parents=True)

    # Create two comparison subdirectories
    for comparison_name in ["Exp1_vs_Exp2", "Exp3_vs_Exp4"]:
        run_dir = (
            cross_platform_dir / f"cross_platform_{comparison_name}_20260202_120000"
        )
        run_dir.mkdir(parents=True)

        # Create minimal correlation CSV
        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA", "TraitB"],
                "exp2_trait": ["TraitX", "TraitY"],
                "spearman_r": [0.75, 0.60],
                "spearman_p": [0.005, 0.02],
                "spearman_p_adjusted": [0.01, 0.04],
                "n_genotypes": [15, 15],
                "achieved_power": [0.85, 0.70],
                "significant_fdr": [True, False],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        # Create pipeline summary JSON
        pipeline_summary = {
            "pipeline_name": f"cross_platform_{comparison_name}",
            "status": "success",
            "config": {
                "exp1_name": comparison_name.split("_vs_")[0],
                "exp2_name": comparison_name.split("_vs_")[1],
                "trait_reduction_method": "none",
            },
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f, indent=2)

    return main_run_dir


# ============================================================================
# TESTS: CrossPlatformSummaryGenerator
# ============================================================================


class TestCrossPlatformSummaryGenerator:
    """Tests for CrossPlatformSummaryGenerator class."""

    def test_generate_summary_from_single_run(self, mock_cross_platform_run):
        """Test summary generation from a single cross-platform run.

        WHEN generate() is called on a single run directory
        THEN a CrossPlatformSummary object is returned with correct statistics
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        assert summary is not None
        assert len(summary.run_summaries) == 1
        run_summary = summary.run_summaries[0]
        assert run_summary.exp1_name == "Experiment 1"
        assert run_summary.exp2_name == "Experiment 2"

    def test_generate_summary_from_multiple_runs(
        self, mock_run_with_multiple_comparisons
    ):
        """Test aggregation across multiple cross-platform comparisons.

        WHEN generate() is called on a directory with multiple comparisons
        THEN all comparisons are included in the summary
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_run_with_multiple_comparisons)
        summary = generator.generate()

        assert len(summary.run_summaries) == 2

    def test_trait_reduction_statistics_accuracy(self, mock_cross_platform_run):
        """Test that reported reduction stats match trait_clusters.csv.

        WHEN summary is generated from a run with trait clustering
        THEN trait reduction statistics match the cluster membership files
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        run_summary = summary.run_summaries[0]

        # exp1: 5 original traits, 3 clusters (3 representatives)
        assert run_summary.exp1_trait_reduction is not None
        assert run_summary.exp1_trait_reduction.original_traits == 5
        assert run_summary.exp1_trait_reduction.n_clusters == 3
        assert run_summary.exp1_trait_reduction.representative_traits == 3

        # exp2: 5 original traits, 3 clusters (3 representatives)
        assert run_summary.exp2_trait_reduction is not None
        assert run_summary.exp2_trait_reduction.original_traits == 5
        assert run_summary.exp2_trait_reduction.n_clusters == 3

    def test_correlation_counts_match_csv(self, mock_cross_platform_run):
        """Test that correlation counts match the source CSV.

        WHEN summary is generated
        THEN total, nominal, and FDR significant counts match CSV data
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        run_summary = summary.run_summaries[0]

        # Check correlation counts
        assert run_summary.correlation_stats.total_correlations == 5
        # FDR significant: first 2 rows have significant_fdr=True
        assert run_summary.correlation_stats.fdr_significant == 2

    def test_top_correlations_match_csv(self, mock_cross_platform_run):
        """Test that top N correlations match sorted CSV values.

        WHEN summary is generated
        THEN top correlations are sorted by |r| descending and match CSV
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        run_summary = summary.run_summaries[0]
        top_corrs = run_summary.correlation_stats.top_correlations

        # Top correlation should be TraitA vs TraitX with r=0.85
        assert len(top_corrs) >= 1
        assert top_corrs[0].exp1_trait == "TraitA"
        assert top_corrs[0].exp2_trait == "TraitX"
        assert abs(top_corrs[0].r_value - 0.85) < 0.01

    def test_power_statistics_accuracy(self, mock_cross_platform_run):
        """Test that power statistics match achieved_power column.

        WHEN summary is generated
        THEN power statistics (min, median, max, pct_above_80) are correct
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        run_summary = summary.run_summaries[0]
        power_stats = run_summary.power_stats

        # Power values: [0.95, 0.82, 0.75, 0.45, 0.25]
        assert abs(power_stats.min_power - 0.25) < 0.01
        assert abs(power_stats.max_power - 0.95) < 0.01
        # Median of [0.25, 0.45, 0.75, 0.82, 0.95] = 0.75
        assert abs(power_stats.median_power - 0.75) < 0.01
        # 2 out of 5 above 80%
        assert abs(power_stats.pct_above_80 - 40.0) < 0.01

    def test_metadata_extraction(self, mock_cross_platform_run):
        """Test that config parameters are extracted correctly.

        WHEN summary is generated
        THEN metadata contains all config parameters from pipeline_summary.json
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        run_summary = summary.run_summaries[0]

        assert run_summary.config.correlation_method == "spearman"
        assert run_summary.config.trait_reduction_method == "clustering"
        assert run_summary.config.trait_reduction_target == "both"
        assert run_summary.config.trait_clustering_threshold == 0.8

    def test_validation_guardrails_pass(self, mock_cross_platform_run):
        """Test that validation passes for consistent data.

        WHEN summary is generated from valid data
        THEN validation passes with no errors
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        assert summary.validation_result.passed is True
        assert len(summary.validation_result.errors) == 0

    def test_missing_files_handled_gracefully(self, tmp_path):
        """Test graceful handling of missing CSV files.

        WHEN summary generator encounters missing files
        THEN it logs warnings and continues without crashing
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        # Create run dir with only pipeline_summary.json (no correlation CSV)
        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        pipeline_summary = {
            "pipeline_name": "cross_platform_Exp1_vs_Exp2",
            "status": "success",
            "config": {"trait_reduction_method": "none"},
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        # Should generate with warnings, not crash
        assert summary is not None
        assert len(summary.validation_result.warnings) > 0

    def test_empty_correlations_handled(self, tmp_path):
        """Test handling of empty correlation results.

        WHEN correlation CSV is empty
        THEN summary handles gracefully with zero counts
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        # Create empty correlation CSV with headers only
        corr_data = pd.DataFrame(
            columns=[
                "exp1_trait",
                "exp2_trait",
                "spearman_r",
                "spearman_p",
                "spearman_p_adjusted",
                "n_genotypes",
                "achieved_power",
                "significant_fdr",
            ]
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        pipeline_summary = {
            "pipeline_name": "cross_platform_Exp1_vs_Exp2",
            "status": "success",
            "config": {"trait_reduction_method": "none"},
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        run_summary = summary.run_summaries[0]
        assert run_summary.correlation_stats.total_correlations == 0
        assert run_summary.correlation_stats.fdr_significant == 0


class TestTraitReductionStats:
    """Tests for trait reduction statistics."""

    def test_exp1_dendrogram_generated_when_exp1_clustered(
        self, mock_cross_platform_run
    ):
        """Test that exp1 dendrogram PNG is referenced when exp1 is clustered.

        WHEN exp1 is clustered (trait_reduction_target includes exp1)
        THEN summary includes reference to exp1_trait_clustering_dendrogram.png
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        run_summary = summary.run_summaries[0]
        assert run_summary.exp1_dendrogram_path is not None
        assert "exp1_trait_clustering_dendrogram.png" in str(
            run_summary.exp1_dendrogram_path
        )

    def test_exp2_dendrogram_generated_when_exp2_clustered(
        self, mock_cross_platform_run
    ):
        """Test that exp2 dendrogram PNG is referenced when exp2 is clustered.

        WHEN exp2 is clustered (trait_reduction_target includes exp2)
        THEN summary includes reference to exp2_trait_clustering_dendrogram.png
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        run_summary = summary.run_summaries[0]
        assert run_summary.exp2_dendrogram_path is not None
        assert "exp2_trait_clustering_dendrogram.png" in str(
            run_summary.exp2_dendrogram_path
        )

    def test_exp1_heatmap_generated_when_exp1_clustered(self, mock_cross_platform_run):
        """Test that exp1 heatmap PNG is referenced when exp1 is clustered.

        WHEN exp1 is clustered
        THEN summary includes reference to exp1_trait_cluster_heatmap.png
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        run_summary = summary.run_summaries[0]
        assert run_summary.exp1_heatmap_path is not None
        assert "exp1_trait_cluster_heatmap.png" in str(run_summary.exp1_heatmap_path)

    def test_exp2_heatmap_generated_when_exp2_clustered(self, mock_cross_platform_run):
        """Test that exp2 heatmap PNG is referenced when exp2 is clustered.

        WHEN exp2 is clustered
        THEN summary includes reference to exp2_trait_cluster_heatmap.png
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        run_summary = summary.run_summaries[0]
        assert run_summary.exp2_heatmap_path is not None
        assert "exp2_trait_cluster_heatmap.png" in str(run_summary.exp2_heatmap_path)

    def test_visualizations_skipped_when_clustering_disabled(self, tmp_path):
        """Test that no visualization files are expected when clustering is disabled.

        WHEN trait_reduction_method is "none"
        THEN no dendrogram or heatmap paths are set
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        # Create minimal correlation CSV
        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA"],
                "exp2_trait": ["TraitX"],
                "spearman_r": [0.75],
                "spearman_p": [0.005],
                "spearman_p_adjusted": [0.01],
                "n_genotypes": [15],
                "achieved_power": [0.85],
                "significant_fdr": [True],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        # Pipeline with no clustering
        pipeline_summary = {
            "pipeline_name": "cross_platform_Exp1_vs_Exp2",
            "status": "success",
            "config": {
                "exp1_name": "Exp1",
                "exp2_name": "Exp2",
                "trait_reduction_method": "none",
            },
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        run_summary = summary.run_summaries[0]
        assert run_summary.exp1_dendrogram_path is None
        assert run_summary.exp2_dendrogram_path is None
        assert run_summary.exp1_heatmap_path is None
        assert run_summary.exp2_heatmap_path is None


class TestMarkdownGeneration:
    """Tests for markdown output generation."""

    def test_summary_embeds_visualizations(self, mock_cross_platform_run):
        """Test that markdown includes image references.

        WHEN to_markdown() is called
        THEN output contains markdown image syntax for visualizations
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should contain image references
        assert "![" in markdown
        assert ".png" in markdown

    def test_top_3_joint_plots_included(self, mock_cross_platform_run):
        """Test that top 3 joint plots are embedded in the summary.

        WHEN to_markdown() is called
        THEN output includes references to top 3 joint plot images
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should reference joint plots
        assert "joint_plot" in markdown

    def test_markdown_includes_statistics_table(self, mock_cross_platform_run):
        """Test that markdown includes statistics in table format.

        WHEN to_markdown() is called
        THEN output contains markdown table with key statistics
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should contain table formatting
        assert "|" in markdown
        # Should contain key statistics
        assert "Correlation" in markdown or "correlation" in markdown

    def test_markdown_includes_trait_reduction_section(self, mock_cross_platform_run):
        """Test that markdown includes trait reduction section when clustering used.

        WHEN to_markdown() is called for a run with clustering
        THEN output contains trait reduction section with cluster info
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should contain clustering information
        assert "cluster" in markdown.lower() or "reduction" in markdown.lower()


# ============================================================================
# TESTS: Summary Clarity Improvements
# ============================================================================


class TestImageEmbedding:
    """Tests for base64 image embedding."""

    def test_images_embedded_as_base64(self, mock_cross_platform_run):
        """Test that PNG images are converted to base64 data URIs.

        WHEN to_markdown() is called with embed_images=True
        THEN output contains data:image/png;base64 URIs instead of file paths
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        # Create a minimal PNG file (1x1 pixel red PNG)
        import base64

        # Minimal valid 1x1 red PNG
        minimal_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        (mock_cross_platform_run / "exp1_trait_clustering_dendrogram.png").write_bytes(
            minimal_png
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown(embed_images=True)

        # Should contain base64 data URI
        assert "data:image/png;base64," in markdown

    def test_missing_image_handled_gracefully(self, tmp_path):
        """Test that missing images don't cause crashes.

        WHEN to_markdown() is called with embed_images=True but image is missing
        THEN output is generated without the image, no crash
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        # Create minimal correlation CSV
        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA"],
                "exp2_trait": ["TraitX"],
                "spearman_r": [0.75],
                "spearman_p": [0.005],
                "spearman_p_adjusted": [0.01],
                "n_genotypes": [15],
                "achieved_power": [0.85],
                "significant_fdr": [True],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        pipeline_summary = {
            "pipeline_name": "cross_platform_test",
            "status": "success",
            "config": {
                "exp1_name": "Exp1",
                "exp2_name": "Exp2",
                "trait_reduction_method": "clustering",
                "trait_reduction_target": "both",
            },
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        # Note: dendrogram files don't exist

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        # Should not crash
        markdown = summary.to_markdown(embed_images=True)
        assert markdown is not None

    def test_base64_image_valid_format(self, mock_cross_platform_run):
        """Test that base64 images use correct data URI format.

        WHEN images are embedded
        THEN they use format: data:image/png;base64,<base64data>
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )
        import base64
        import re

        # Create a minimal PNG file
        minimal_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        (mock_cross_platform_run / "exp1_trait_clustering_dendrogram.png").write_bytes(
            minimal_png
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown(embed_images=True)

        # Should match data URI pattern
        pattern = r"!\[.*?\]\(data:image/png;base64,[A-Za-z0-9+/=]+\)"
        assert re.search(pattern, markdown) is not None


class TestVariableDefinitions:
    """Tests for inline variable definitions."""

    def test_table_headers_include_definitions(self, mock_cross_platform_run):
        """Test that correlation table headers include variable definitions.

        WHEN to_markdown() is called
        THEN table headers include inline definitions (e.g., "ρ (Spearman)")
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should contain defined headers
        assert "ρ" in markdown or "Spearman" in markdown
        assert "FDR" in markdown

    def test_legend_includes_all_abbreviations(self, mock_cross_platform_run):
        """Test that a legend defines all abbreviations used.

        WHEN to_markdown() is called
        THEN output includes a legend explaining r, p, q, Power, n
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should contain definitions for key abbreviations
        markdown_lower = markdown.lower()
        assert "correlation" in markdown_lower
        assert "p-value" in markdown_lower or "p value" in markdown_lower
        assert "genotype" in markdown_lower


class TestConfidenceIntervals:
    """Tests for confidence interval display."""

    def test_correlation_table_includes_ci(self, mock_cross_platform_run):
        """Test that correlation table includes 95% confidence intervals.

        WHEN to_markdown() is called with correlation data containing CIs
        THEN output includes CI values in the table
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should contain CI values (the fixture has ci_low and ci_high)
        # Check for bracket notation or CI mention
        assert "[" in markdown or "CI" in markdown

    def test_ci_format_brackets(self, mock_cross_platform_run):
        """Test that CIs are formatted with brackets.

        WHEN to_markdown() is called
        THEN CIs are displayed as [ci_low, ci_high] or similar format
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        # Check that top correlation has CI data
        run_summary = summary.run_summaries[0]
        top_corr = run_summary.correlation_stats.top_correlations[0]

        # TopCorrelation should have CI fields
        assert hasattr(top_corr, "ci_low") or hasattr(top_corr, "r_ci_low")

    def test_missing_ci_handled(self, tmp_path):
        """Test graceful handling when CI columns are missing.

        WHEN correlation CSV lacks CI columns
        THEN summary generates without errors, CIs shown as N/A or omitted
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        # Create correlation CSV without CI columns
        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA"],
                "exp2_trait": ["TraitX"],
                "spearman_r": [0.75],
                "spearman_p": [0.005],
                "spearman_p_adjusted": [0.01],
                "n_genotypes": [15],
                "achieved_power": [0.85],
                "significant_fdr": [True],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        pipeline_summary = {
            "pipeline_name": "cross_platform_test",
            "status": "success",
            "config": {"trait_reduction_method": "none"},
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        # Should not crash
        markdown = summary.to_markdown()
        assert markdown is not None


class TestPowerAnalysisEnhancements:
    """Tests for enhanced power analysis display."""

    def test_power_section_includes_parameters(self, mock_cross_platform_run):
        """Test that power section shows analysis parameters.

        WHEN to_markdown() is called
        THEN power section includes α (alpha) and n (sample size)
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should contain power parameters
        markdown_lower = markdown.lower()
        assert "α" in markdown or "alpha" in markdown_lower
        assert "n" in markdown or "genotype" in markdown_lower

    def test_power_section_includes_mde(self, mock_cross_platform_run):
        """Test that power section shows minimum detectable effect.

        WHEN to_markdown() is called
        THEN power section includes minimum detectable |r| at 80% power
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should mention minimum detectable effect
        markdown_lower = markdown.lower()
        assert "minimum" in markdown_lower or "detectable" in markdown_lower

    def test_power_warning_when_underpowered(self, tmp_path):
        """Test that warning appears when most correlations are underpowered.

        WHEN >50% of correlations have power < 80%
        THEN markdown includes a warning about underpowered analysis
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        # Create correlation CSV with low power values (all < 0.8)
        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA", "TraitB", "TraitC"],
                "exp2_trait": ["TraitX", "TraitY", "TraitZ"],
                "spearman_r": [0.45, 0.35, 0.25],
                "spearman_p": [0.05, 0.1, 0.2],
                "spearman_p_adjusted": [0.1, 0.2, 0.3],
                "n_genotypes": [19, 19, 19],
                "achieved_power": [0.45, 0.30, 0.15],  # All below 80%
                "significant_fdr": [False, False, False],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        pipeline_summary = {
            "pipeline_name": "cross_platform_test",
            "status": "success",
            "config": {"trait_reduction_method": "none"},
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should contain warning
        markdown_lower = markdown.lower()
        assert (
            "warning" in markdown_lower
            or "⚠" in markdown
            or "underpowered" in markdown_lower
        )

    def test_power_section_includes_sample_size_recommendation(
        self, mock_cross_platform_run
    ):
        """Test that power section includes sample size recommendation.

        WHEN to_markdown() is called
        THEN power section includes recommendation for required n at target effect size
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should mention sample size recommendation
        markdown_lower = markdown.lower()
        assert (
            "require" in markdown_lower
            or "recommend" in markdown_lower
            or "sample" in markdown_lower
        )

    def test_no_power_warning_when_adequately_powered(self, tmp_path):
        """Test that no warning appears when most correlations are well-powered.

        WHEN >50% of correlations have power >= 80%
        THEN no underpowered warning is shown
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        # Create correlation CSV with high power values (most >= 0.8)
        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA", "TraitB", "TraitC"],
                "exp2_trait": ["TraitX", "TraitY", "TraitZ"],
                "spearman_r": [0.85, 0.80, 0.75],
                "spearman_p": [0.001, 0.002, 0.003],
                "spearman_p_adjusted": [0.003, 0.006, 0.009],
                "n_genotypes": [50, 50, 50],  # Large n for high power
                "achieved_power": [0.95, 0.90, 0.85],  # All above 80%
                "significant_fdr": [True, True, True],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        pipeline_summary = {
            "pipeline_name": "cross_platform_test",
            "status": "success",
            "config": {"trait_reduction_method": "none"},
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should NOT contain underpowered warning
        markdown_lower = markdown.lower()
        assert "underpowered" not in markdown_lower

    def test_power_stats_uses_config_significance_level(self, tmp_path):
        """Test that alpha is read from config.significance_level when available.

        WHEN pipeline_summary.json contains config.significance_level = 0.01
        THEN PowerStats.alpha should be 0.01, not the default 0.05
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA", "TraitB"],
                "exp2_trait": ["TraitX", "TraitY"],
                "spearman_r": [0.65, 0.55],
                "spearman_p": [0.003, 0.02],
                "spearman_p_adjusted": [0.01, 0.04],
                "n_genotypes": [25, 25],
                "achieved_power": [0.75, 0.60],
                "significant_fdr": [True, False],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        # Config specifies significance_level = 0.01 (not default 0.05)
        pipeline_summary = {
            "pipeline_name": "cross_platform_test",
            "status": "success",
            "config": {
                "trait_reduction_method": "none",
                "significance_level": 0.01,  # Custom alpha
            },
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        # PowerStats should use the config value, not default
        power_stats = summary.run_summaries[0].power_stats
        assert power_stats is not None
        assert (
            power_stats.alpha == 0.01
        ), f"Expected alpha=0.01 from config, got {power_stats.alpha}"

    def test_power_stats_defaults_alpha_when_not_in_config(self, tmp_path):
        """Test that alpha defaults to 0.05 when not specified in config.

        WHEN pipeline_summary.json does not contain config.significance_level
        THEN PowerStats.alpha should default to 0.05
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA"],
                "exp2_trait": ["TraitX"],
                "spearman_r": [0.65],
                "spearman_p": [0.003],
                "spearman_p_adjusted": [0.01],
                "n_genotypes": [25],
                "achieved_power": [0.75],
                "significant_fdr": [True],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        # Config does NOT specify significance_level
        pipeline_summary = {
            "pipeline_name": "cross_platform_test",
            "status": "success",
            "config": {"trait_reduction_method": "none"},
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        # PowerStats should use default alpha = 0.05
        power_stats = summary.run_summaries[0].power_stats
        assert power_stats is not None
        assert (
            power_stats.alpha == 0.05
        ), f"Expected default alpha=0.05, got {power_stats.alpha}"


class TestFDRInterpretation:
    """Tests for FDR=0 interpretation section."""

    def test_fdr_zero_shows_interpretation(self, tmp_path):
        """Test that interpretation section appears when FDR significant count = 0.

        WHEN no correlations survive FDR correction
        THEN markdown includes an interpretation section explaining this
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        # Create correlation CSV with no FDR-significant results
        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA", "TraitB", "TraitC"],
                "exp2_trait": ["TraitX", "TraitY", "TraitZ"],
                "spearman_r": [0.45, 0.35, 0.25],
                "spearman_p": [0.05, 0.1, 0.2],
                "spearman_p_adjusted": [0.4, 0.5, 0.6],  # All > 0.05 after FDR
                "n_genotypes": [19, 19, 19],
                "achieved_power": [0.45, 0.30, 0.15],
                "significant_fdr": [False, False, False],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        pipeline_summary = {
            "pipeline_name": "cross_platform_test",
            "status": "success",
            "config": {"trait_reduction_method": "none"},
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should contain interpretation
        markdown_lower = markdown.lower()
        assert "interpretation" in markdown_lower or "note" in markdown_lower

    def test_fdr_zero_shows_nominal_count(self, tmp_path):
        """Test that FDR=0 interpretation shows nominal significant count.

        WHEN no FDR-significant correlations but some nominally significant
        THEN markdown mentions the number of nominally significant correlations
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        # 2 nominally significant (p < 0.05), but none FDR-significant
        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA", "TraitB", "TraitC"],
                "exp2_trait": ["TraitX", "TraitY", "TraitZ"],
                "spearman_r": [0.65, 0.55, 0.25],
                "spearman_p": [0.003, 0.02, 0.2],  # First 2 are p < 0.05
                "spearman_p_adjusted": [0.10, 0.20, 0.60],  # None FDR significant
                "n_genotypes": [19, 19, 19],
                "achieved_power": [0.75, 0.60, 0.15],
                "significant_fdr": [False, False, False],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        pipeline_summary = {
            "pipeline_name": "cross_platform_test",
            "status": "success",
            "config": {"trait_reduction_method": "none"},
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should mention nominal significance
        markdown_lower = markdown.lower()
        assert (
            "nominal" in markdown_lower
            or "raw" in markdown_lower
            or "p < 0.05" in markdown_lower
        )

    def test_fdr_zero_shows_recommendations(self, tmp_path):
        """Test that FDR=0 interpretation includes recommendations.

        WHEN no FDR-significant correlations
        THEN markdown includes recommendations for next steps
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_Exp1_vs_Exp2_20260202_120000"
        run_dir.mkdir(parents=True)

        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA", "TraitB"],
                "exp2_trait": ["TraitX", "TraitY"],
                "spearman_r": [0.45, 0.35],
                "spearman_p": [0.05, 0.1],
                "spearman_p_adjusted": [0.4, 0.5],
                "n_genotypes": [19, 19],
                "achieved_power": [0.45, 0.30],
                "significant_fdr": [False, False],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        pipeline_summary = {
            "pipeline_name": "cross_platform_test",
            "status": "success",
            "config": {"trait_reduction_method": "none"},
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # Should contain recommendations
        markdown_lower = markdown.lower()
        assert (
            "recommend" in markdown_lower
            or "suggest" in markdown_lower
            or "consider" in markdown_lower
            or "increase" in markdown_lower
        )

    def test_fdr_nonzero_no_interpretation(self, mock_cross_platform_run):
        """Test that no extra interpretation when FDR-significant correlations exist.

        WHEN some correlations survive FDR correction
        THEN no special interpretation section is added
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown()

        # The fixture has FDR-significant correlations
        assert summary.run_summaries[0].correlation_stats.fdr_significant > 0

        # Should NOT contain the FDR=0 specific interpretation
        assert "no correlations remained significant after FDR" not in markdown.lower()


# ============================================================================
# TESTS: Memory-Aware Summary Output (add-memory-aware-summary-output)
# ============================================================================


class TestImageSizeCalculation:
    """Tests for total image size calculation."""

    def test_calculate_total_image_size_empty(self, tmp_path):
        """Test that empty image list returns 0 bytes.

        WHEN _calculate_total_image_size is called with empty list
        THEN it returns 0
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            _calculate_total_image_size,
        )

        total = _calculate_total_image_size([])
        assert total == 0

    def test_calculate_total_image_size_single_image(self, tmp_path):
        """Test accurate size calculation for one image.

        WHEN _calculate_total_image_size is called with one file
        THEN it returns that file's size
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            _calculate_total_image_size,
        )

        # Create a file with known size
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"x" * 1000)  # 1000 bytes

        total = _calculate_total_image_size([test_file])
        assert total == 1000

    def test_calculate_total_image_size_multiple_images(self, tmp_path):
        """Test that multiple image sizes are summed.

        WHEN _calculate_total_image_size is called with multiple files
        THEN it returns the sum of all file sizes
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            _calculate_total_image_size,
        )

        # Create files with known sizes
        file1 = tmp_path / "img1.png"
        file1.write_bytes(b"x" * 1000)
        file2 = tmp_path / "img2.png"
        file2.write_bytes(b"y" * 2000)
        file3 = tmp_path / "img3.png"
        file3.write_bytes(b"z" * 500)

        total = _calculate_total_image_size([file1, file2, file3])
        assert total == 3500

    def test_calculate_total_image_size_missing_file(self, tmp_path):
        """Test that missing files are handled gracefully.

        WHEN _calculate_total_image_size encounters missing files
        THEN it skips them and continues without error
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            _calculate_total_image_size,
        )

        existing = tmp_path / "exists.png"
        existing.write_bytes(b"x" * 500)
        missing = tmp_path / "missing.png"  # Does not exist

        total = _calculate_total_image_size([existing, missing])
        assert total == 500


class TestImageModeSelection:
    """Tests for image_mode parameter in to_markdown()."""

    def test_file_path_mode_uses_relative_paths(self, mock_cross_platform_run):
        """Test that file_path mode generates relative paths, not base64.

        WHEN to_markdown(image_mode="file_path") is called
        THEN markdown contains relative file paths, not data URIs
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown(image_mode="file_path")

        # Should contain file paths
        assert ".png)" in markdown or ".png]" in markdown
        # Should NOT contain base64 data URIs
        assert "data:image/png;base64," not in markdown

    def test_embed_mode_under_threshold_embeds(self, mock_cross_platform_run):
        """Test that embed mode embeds images when under threshold.

        WHEN to_markdown(image_mode="embed") with small images
        THEN images are embedded as base64
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )
        import base64

        # Create small test images (well under 10MB threshold)
        minimal_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        (mock_cross_platform_run / "exp1_trait_clustering_dendrogram.png").write_bytes(
            minimal_png
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown(image_mode="embed")

        # Should contain base64 data URI
        assert "data:image/png;base64," in markdown

    def test_embed_mode_over_threshold_warns_and_fallback(self, tmp_path, caplog):
        """Test that embed mode falls back when over threshold.

        WHEN to_markdown(image_mode="embed") with images over threshold
        THEN a warning is logged and file paths are used instead
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )
        import logging

        run_dir = tmp_path / "cross_platform_test"
        run_dir.mkdir()

        # Create correlation CSV
        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA"],
                "exp2_trait": ["TraitX"],
                "spearman_r": [0.75],
                "spearman_p": [0.005],
                "spearman_p_adjusted": [0.01],
                "n_genotypes": [15],
                "achieved_power": [0.85],
                "significant_fdr": [True],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        # Create large image (over a small test threshold)
        large_image = run_dir / "exp1_trait_clustering_dendrogram.png"
        large_image.write_bytes(b"x" * 2000)  # 2KB

        pipeline_summary = {
            "pipeline_name": "test",
            "status": "success",
            "config": {
                "trait_reduction_method": "clustering",
                "trait_reduction_target": "exp1",
            },
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        # Use tiny threshold to trigger fallback
        with caplog.at_level(logging.WARNING):
            markdown = summary.to_markdown(
                image_mode="embed", embed_threshold_bytes=100
            )

        # Should have warning about size
        assert (
            any(
                "threshold" in r.message.lower() or "size" in r.message.lower()
                for r in caplog.records
            )
            or "data:image" not in markdown
        )

    def test_auto_mode_under_threshold_embeds(self, mock_cross_platform_run):
        """Test that auto mode embeds when images are small.

        WHEN to_markdown(image_mode="auto") with small total size
        THEN images are embedded
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )
        import base64

        # Create small images
        minimal_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        (mock_cross_platform_run / "exp1_trait_clustering_dendrogram.png").write_bytes(
            minimal_png
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown(image_mode="auto")

        # Small images should be embedded
        assert "data:image/png;base64," in markdown

    def test_auto_mode_over_threshold_uses_file_path(self, tmp_path):
        """Test that auto mode uses file paths when images are large.

        WHEN to_markdown(image_mode="auto") with large total size
        THEN file paths are used instead of embedding
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_test"
        run_dir.mkdir()

        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA"],
                "exp2_trait": ["TraitX"],
                "spearman_r": [0.75],
                "spearman_p": [0.005],
                "spearman_p_adjusted": [0.01],
                "n_genotypes": [15],
                "achieved_power": [0.85],
                "significant_fdr": [True],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        # Create "large" image for small threshold test
        (run_dir / "exp1_trait_clustering_dendrogram.png").write_bytes(b"x" * 5000)

        pipeline_summary = {
            "pipeline_name": "test",
            "status": "success",
            "config": {
                "trait_reduction_method": "clustering",
                "trait_reduction_target": "exp1",
            },
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        # With tiny threshold, should use file paths
        markdown = summary.to_markdown(image_mode="auto", embed_threshold_bytes=100)

        # Should NOT contain base64
        assert "data:image/png;base64," not in markdown

    def test_threshold_configurable(self, mock_cross_platform_run):
        """Test that embed_threshold_bytes parameter is respected.

        WHEN to_markdown() is called with custom threshold
        THEN that threshold determines embed vs file_path behavior
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )
        import base64

        # Create image of known size
        minimal_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )  # ~70 bytes
        (mock_cross_platform_run / "exp1_trait_clustering_dendrogram.png").write_bytes(
            minimal_png
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()

        # With threshold of 1MB, should embed
        markdown_large_threshold = summary.to_markdown(
            image_mode="auto", embed_threshold_bytes=1024 * 1024
        )
        assert "data:image/png;base64," in markdown_large_threshold

        # With threshold of 10 bytes, should use file paths
        markdown_small_threshold = summary.to_markdown(
            image_mode="auto", embed_threshold_bytes=10
        )
        assert "data:image/png;base64," not in markdown_small_threshold


class TestOutputFormatOptions:
    """Tests for output format options (markdown, html, both)."""

    def test_to_html_generates_valid_html(self, mock_cross_platform_run):
        """Test that to_html() generates valid HTML.

        WHEN to_html() is called
        THEN output contains proper HTML structure
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        html = summary.to_html()

        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "</html>" in html

    def test_html_output_renders_tables(self, mock_cross_platform_run):
        """Test that HTML contains properly formatted tables.

        WHEN to_html() is called
        THEN tables are rendered with <table> tags
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        html = summary.to_html()

        assert "<table" in html
        assert "<th>" in html or "<td>" in html

    def test_html_includes_styling(self, mock_cross_platform_run):
        """Test that HTML has embedded CSS.

        WHEN to_html() is called
        THEN output includes <style> block
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        html = summary.to_html()

        assert "<style>" in html or 'style="' in html

    def test_html_output_renders_images(self, mock_cross_platform_run):
        """Test that HTML img tags reference images correctly.

        WHEN to_html() is called
        THEN images are referenced with <img> tags
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )
        import base64

        # Create test image
        minimal_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        (mock_cross_platform_run / "exp1_trait_clustering_dendrogram.png").write_bytes(
            minimal_png
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        html = summary.to_html()

        assert "<img" in html


class TestFilePathReferences:
    """Tests for file path reference correctness."""

    def test_file_path_references_are_relative(self, mock_cross_platform_run):
        """Test that file paths are relative to run directory.

        WHEN to_markdown(image_mode="file_path") is called
        THEN paths are relative (not absolute)
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown(image_mode="file_path")

        # Should NOT contain absolute paths
        assert "C:\\" not in markdown
        assert "/home/" not in markdown
        assert "/Users/" not in markdown

    def test_file_path_handles_special_characters(self, tmp_path):
        """Test that file paths handle traits with special characters.

        WHEN trait names contain spaces or slashes
        THEN file paths are still valid
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        run_dir = tmp_path / "cross_platform_test"
        run_dir.mkdir()

        # Create correlation with special character trait names
        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["Trait A/B"],  # Contains space and slash
                "exp2_trait": ["Trait X"],
                "spearman_r": [0.75],
                "spearman_p": [0.005],
                "spearman_p_adjusted": [0.01],
                "n_genotypes": [15],
                "achieved_power": [0.85],
                "significant_fdr": [True],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        pipeline_summary = {
            "pipeline_name": "test",
            "status": "success",
            "config": {"trait_reduction_method": "none"},
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        # Should not crash
        markdown = summary.to_markdown(image_mode="file_path")
        assert markdown is not None


class TestSummaryFileSize:
    """Tests for summary file size constraints."""

    def test_file_path_mode_produces_small_file(self, mock_cross_platform_run):
        """Test that file_path mode produces a file under 1MB.

        WHEN to_markdown(image_mode="file_path") is called
        THEN the resulting markdown is under 1MB
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(mock_cross_platform_run)
        summary = generator.generate()
        markdown = summary.to_markdown(image_mode="file_path")

        # File path mode should be small (no embedded images)
        size_bytes = len(markdown.encode("utf-8"))
        assert size_bytes < 1024 * 1024  # Under 1MB

    def test_embed_mode_can_produce_large_file(self, tmp_path):
        """Test that embed mode produces larger files with images.

        WHEN to_markdown(image_mode="embed") with images
        THEN the resulting file is larger than file_path mode
        """
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )
        import base64

        run_dir = tmp_path / "cross_platform_test"
        run_dir.mkdir()

        corr_data = pd.DataFrame(
            {
                "exp1_trait": ["TraitA"],
                "exp2_trait": ["TraitX"],
                "spearman_r": [0.75],
                "spearman_p": [0.005],
                "spearman_p_adjusted": [0.01],
                "n_genotypes": [15],
                "achieved_power": [0.85],
                "significant_fdr": [True],
            }
        )
        corr_data.to_csv(run_dir / "cross_platform_correlations.csv", index=False)

        # Create trait clusters file (required for dendrogram to be picked up)
        exp1_clusters = pd.DataFrame(
            {
                "trait": ["TraitA", "TraitB"],
                "cluster_id": [0, 0],
                "is_representative": [True, False],
            }
        )
        exp1_clusters.to_csv(run_dir / "exp1_trait_clusters.csv", index=False)

        # Create a test image
        minimal_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        (run_dir / "exp1_trait_clustering_dendrogram.png").write_bytes(minimal_png)

        pipeline_summary = {
            "pipeline_name": "test",
            "status": "success",
            "config": {
                "trait_reduction_method": "clustering",
                "trait_reduction_target": "exp1",
            },
        }
        with open(run_dir / "pipeline_summary.json", "w") as f:
            json.dump(pipeline_summary, f)

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        md_file_path = summary.to_markdown(image_mode="file_path")
        md_embed = summary.to_markdown(image_mode="embed")

        # Embed mode should be larger (base64 images are embedded)
        assert len(md_embed) > len(md_file_path)
