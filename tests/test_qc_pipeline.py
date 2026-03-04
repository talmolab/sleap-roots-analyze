"""Tests for QC Pipeline orchestrator."""

from __future__ import annotations

import pytest
from pathlib import Path
import pandas as pd

from sleap_roots_analyze.pipeline import (
    QCPipeline,
    QCPipelineConfig,
    get_default_qc_config,
)


class TestQCPipelineCreation:
    """Test QCPipeline initialization and basic functionality."""

    def test_qc_pipeline_creation(self, tmp_path):
        """Test creating a QCPipeline instance."""
        config = get_default_qc_config()
        config.data.csv_path = "test.csv"
        config.outlier_detection.traditional_methods = ["mahalanobis"]

        pipeline = QCPipeline(config, output_dir=tmp_path)

        assert pipeline.config == config
        assert pipeline.pipeline_name == "qc_pipeline"
        assert pipeline.version == "1.0"
        assert pipeline.output_dir == tmp_path

    def test_qc_pipeline_config_validation(self, tmp_path):
        """Test that invalid config raises ValueError."""
        from omegaconf import MISSING

        config = get_default_qc_config()
        # Set csv_path to MISSING (required field)
        config.data.csv_path = MISSING

        with pytest.raises(ValueError, match="data.csv_path is required"):
            QCPipeline(config, output_dir=tmp_path)

    def test_qc_pipeline_skip_validation(self, tmp_path):
        """Test skipping validation allows invalid config."""
        config = get_default_qc_config()
        # Missing csv_path but validation disabled
        config.data.csv_path = None

        # Should not raise when validate=False
        pipeline = QCPipeline(config, output_dir=tmp_path, validate=False)
        assert pipeline.config.data.csv_path is None

    def test_qc_pipeline_creates_tasks(self, tmp_path):
        """Test that create_tasks returns 10 tasks."""
        config = get_default_qc_config()
        config.data.csv_path = "test.csv"
        config.outlier_detection.traditional_methods = ["mahalanobis"]

        pipeline = QCPipeline(config, output_dir=tmp_path)
        tasks = pipeline.create_tasks()

        assert len(tasks) == 10
        assert tasks[0].name == "01_load_data"
        assert tasks[1].name == "02_cleanup_traits"
        assert tasks[2].name == "03_validate_clean"
        assert tasks[3].name == "04_exploratory_analysis"
        assert tasks[4].name == "05_detect_outliers"
        assert tasks[5].name == "06_visualize_outliers"
        assert tasks[6].name == "07_remove_outliers"
        assert tasks[7].name == "08_statistical_analysis"
        assert tasks[8].name == "09_filter_heritability"
        assert tasks[9].name == "10_generate_summary"

    def test_qc_pipeline_task_dependencies(self, tmp_path):
        """Test that tasks have correct dependencies (linear chain)."""
        config = get_default_qc_config()
        config.data.csv_path = "test.csv"
        config.outlier_detection.traditional_methods = ["mahalanobis"]

        pipeline = QCPipeline(config, output_dir=tmp_path)
        tasks = pipeline.create_tasks()

        # Step 1 has no dependencies
        assert tasks[0].depends_on == []

        # Steps 2-10 each depend on the previous step
        assert tasks[1].depends_on == ["01_load_data"]
        assert tasks[2].depends_on == ["02_cleanup_traits"]
        assert tasks[3].depends_on == ["03_validate_clean"]
        assert tasks[4].depends_on == ["04_exploratory_analysis"]
        assert tasks[5].depends_on == ["05_detect_outliers"]
        assert tasks[6].depends_on == ["06_visualize_outliers"]
        assert tasks[7].depends_on == ["07_remove_outliers"]
        assert tasks[8].depends_on == ["08_statistical_analysis"]
        assert tasks[9].depends_on == ["09_filter_heritability"]

    def test_qc_pipeline_all_steps_initialized(self, tmp_path):
        """Test that all 10 step instances are created."""
        config = get_default_qc_config()
        config.data.csv_path = "test.csv"
        config.outlier_detection.traditional_methods = ["mahalanobis"]

        pipeline = QCPipeline(config, output_dir=tmp_path)

        # Check all step instances exist
        assert hasattr(pipeline, "step_1_load_data")
        assert hasattr(pipeline, "step_2_cleanup_traits")
        assert hasattr(pipeline, "step_3_validate_clean")
        assert hasattr(pipeline, "step_4_exploratory_analysis")
        assert hasattr(pipeline, "step_5_detect_outliers")
        assert hasattr(pipeline, "step_6_visualize_outliers")
        assert hasattr(pipeline, "step_7_remove_outliers")
        assert hasattr(pipeline, "step_8_statistical_analysis")
        assert hasattr(pipeline, "step_9_filter_heritability")
        assert hasattr(pipeline, "step_10_generate_summary")

    def test_qc_pipeline_custom_name_and_version(self, tmp_path):
        """Test QC pipeline with custom name from config."""
        config = get_default_qc_config()
        config.pipeline_name = "my_qc_pipeline"
        config.version = "2.0"
        config.data.csv_path = "test.csv"
        config.outlier_detection.traditional_methods = ["mahalanobis"]

        pipeline = QCPipeline(config, output_dir=tmp_path)

        assert pipeline.pipeline_name == "my_qc_pipeline"
        assert pipeline.version == "2.0"


class TestQCPipelineIntegration:
    """Integration tests for full QC pipeline execution."""

    @pytest.fixture
    def test_data(self, tmp_path):
        """Create test CSV data."""
        import numpy as np

        # Create larger test dataset with enough samples for outlier detection
        np.random.seed(42)
        n_samples = 30

        data = {
            "Barcode": [f"S{i:03d}" for i in range(n_samples)],
            "geno": ["WT"] * 15 + ["MUT"] * 15,
            "rep": [i % 3 + 1 for i in range(n_samples)],
            # Add 5 traits with realistic variation
            "trait1": np.random.normal(10, 2, n_samples),
            "trait2": np.random.normal(20, 3, n_samples),
            "trait3": np.random.normal(15, 2.5, n_samples),
            "trait4": np.random.normal(8, 1.5, n_samples),
            "trait5": np.random.normal(12, 2, n_samples),
        }

        # Add one clear outlier
        data["trait1"][0] = 50.0  # Extreme outlier

        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_qc_pipeline_full_run(self, test_data, tmp_path):
        """Test full QC pipeline execution end-to-end with synthetic data.

        This test uses synthetic data to verify the pipeline runs correctly
        with a simple dataset. For real-world data testing, see
        test_qc_pipeline_turface_integration.
        """
        config = get_default_qc_config()
        config.data.csv_path = str(test_data)
        config.outlier_detection.traditional_methods = ["mahalanobis"]
        config.heritability.enabled = False  # Disable for simple test

        pipeline = QCPipeline(config, output_dir=tmp_path / "runs")

        # Run the pipeline
        results = pipeline.run()

        # Check that all 10 steps completed
        assert len(results) == 10
        assert "01_load_data" in results
        assert "02_cleanup_traits" in results
        assert "03_validate_clean" in results
        assert "04_exploratory_analysis" in results
        assert "05_detect_outliers" in results
        assert "06_visualize_outliers" in results
        assert "07_remove_outliers" in results
        assert "08_statistical_analysis" in results
        assert "09_filter_heritability" in results
        assert "10_generate_summary" in results

        # Check that each result has data
        for step_name, result in results.items():
            assert result.data is not None, f"{step_name} has no data"

        # Check that run directory was created
        assert pipeline.run_dir.exists()

        # Check that summary was saved
        summary_path = pipeline.run_dir / "pipeline_summary.json"
        assert summary_path.exists()

        # Check pipeline status
        summary = pipeline.get_summary()
        assert summary.status == "success"
        assert len(summary.steps) == 10

    def test_qc_pipeline_turface_integration(self, turface_rsr_csv_path, tmp_path):
        """Integration test using real Turface wheat data.

        This test uses the actual Turface_all_traits_2024_RSR.csv dataset
        and verifies the pipeline produces expected results matching the
        trait_qc_turface_20251010.ipynb notebook workflow.

        Expected results from notebook:
        - Start: 187 samples, 35 traits, 19 genotypes
        - After NaN removal: 158 samples (29 removed)
        - After outlier removal (mahalanobis): 152 samples (6 removed)
        - After heritability filtering (H²≥0.62): 12 traits (23 removed)
        """
        # Use test data path
        assert (
            turface_rsr_csv_path.exists()
        ), f"Test data not found: {turface_rsr_csv_path}"

        # Load qc_mahalanobis preset and configure for Turface data
        config = get_default_qc_config()
        config.pipeline_name = "turface_integration_test"
        config.data.csv_path = str(turface_rsr_csv_path)
        config.columns.barcode = "Barcode"
        config.columns.genotype = "geno"
        config.columns.replicate = "rep"

        # Match notebook parameters
        config.cleanup.max_nan_fraction = (
            0.0  # Strict NaN removal (0.0 = any NaN removes sample)
        )
        config.cleanup.max_zeros_per_trait = 0.5
        config.cleanup.max_nans_per_trait = 0.2
        config.cleanup.min_samples_per_trait = 10

        # Outlier detection - use only Mahalanobis like notebook
        config.outlier_detection.traditional_methods = ["mahalanobis"]
        config.outlier_detection.clustering_methods = []
        config.outlier_detection.mahalanobis.variance_threshold = 0.95
        config.outlier_detection.mahalanobis.use_chi_squared = True
        config.outlier_detection.mahalanobis.chi2_percentile = 99.0

        # Outlier removal - single method
        config.outlier_removal.strategy = "single"
        config.outlier_removal.method = "mahalanobis"

        # Heritability filtering
        config.heritability.enabled = True
        config.heritability.threshold = 0.62

        # Visualization
        config.visualization.dpi = 100
        config.visualization.figsize = (10, 8)

        # Logging
        config.logging.level = "INFO"
        config.logging.log_to_file = True

        # Create pipeline
        pipeline = QCPipeline(config, output_dir=tmp_path / "turface_test")

        # Run the pipeline
        print("\nRunning Turface integration test...")
        results = pipeline.run()

        # Verify all 10 steps completed
        assert len(results) == 10, f"Expected 10 steps, got {len(results)}"

        # Verify each step returned results
        step_names = [
            "01_load_data",
            "02_cleanup_traits",
            "03_validate_clean",
            "04_exploratory_analysis",
            "05_detect_outliers",
            "06_visualize_outliers",
            "07_remove_outliers",
            "08_statistical_analysis",
            "09_filter_heritability",
            "10_generate_summary",
        ]

        for step_name in step_names:
            assert step_name in results, f"Missing step: {step_name}"
            step_result = results[
                step_name
            ].data  # TaskResult.data contains the StepResult
            assert step_result.data is not None, f"{step_name} has no data"

        # Step 1: Load Data
        load_result = results["01_load_data"].data
        df_loaded = load_result.data
        assert (
            len(df_loaded) == 187
        ), f"Expected 187 samples loaded, got {len(df_loaded)}"
        # Bug #75 fix: previously 35 traits — 3 width/solidity columns were
        # incorrectly excluded (Maximum.Width.mm, Width-to-Depth.Ratio, Solidity).
        assert len(load_result.metadata["trait_names"]) == 38, f"Expected 38 traits"
        print(
            f"OK Step 1: Loaded {len(df_loaded)} samples, {len(load_result.metadata['trait_names'])} traits"
        )

        # Step 2: Cleanup Traits (removes NaN samples)
        cleanup_result = results["02_cleanup_traits"].data
        df_cleaned = cleanup_result.data
        # Notebook: 158 samples after NaN removal (MAX_NAN_FRACTION=0.0 removes 29 samples)
        assert (
            len(df_cleaned) == 158
        ), f"Expected 158 samples after cleanup, got {len(df_cleaned)}"
        # Bug #75 fix: 38 traits (was 35 before width/solidity fix)
        assert (
            len(cleanup_result.metadata["trait_names"]) == 38
        ), f"Expected 38 traits retained"
        print(f"OK Step 2: {len(df_cleaned)} samples after NaN removal (29 removed)")

        # Step 3: Validate Clean
        validate_result = results["03_validate_clean"].data
        assert validate_result.data is not None
        assert validate_result.metadata["validation_passed"] == True
        print(f"OK Step 3: Data validation passed")

        # Step 4: Exploratory Analysis
        eda_result = results["04_exploratory_analysis"].data
        assert "figures_generated" in eda_result.metadata
        assert eda_result.metadata["figures_generated"] > 0, "No EDA figures created"
        print(
            f"OK Step 4: Created {eda_result.metadata['figures_generated']} EDA figures"
        )

        # Verify batched trait visualizations generated (Issue #27)
        # Turface has 38 traits, so should generate batched plots (>16 threshold)
        figures_dir = pipeline.run_dir / "figures"

        # Check for batched histograms (should have 3 batches: 16+16+6=38)
        batch_hist_1 = figures_dir / "04_trait_histograms_batch_1.png"
        batch_hist_2 = figures_dir / "04_trait_histograms_batch_2.png"
        batch_hist_3 = figures_dir / "04_trait_histograms_batch_3.png"
        assert batch_hist_1.exists(), "Batched histogram batch 1 not generated"
        assert batch_hist_2.exists(), "Batched histogram batch 2 not generated"
        assert batch_hist_3.exists(), "Batched histogram batch 3 not generated"

        # Check for batched boxplots (should have 3 batches)
        batch_box_1 = figures_dir / "04_trait_boxplots_batch_1.png"
        batch_box_2 = figures_dir / "04_trait_boxplots_batch_2.png"
        batch_box_3 = figures_dir / "04_trait_boxplots_batch_3.png"
        assert batch_box_1.exists(), "Batched boxplot batch 1 not generated"
        assert batch_box_2.exists(), "Batched boxplot batch 2 not generated"
        assert batch_box_3.exists(), "Batched boxplot batch 3 not generated"

        print("OK Step 4: Batched trait visualizations generated (3 batches each)")

        # Step 5: Detect Outliers
        detect_result = results["05_detect_outliers"].data
        assert "outlier_results" in detect_result.metadata
        outlier_results = detect_result.metadata["outlier_results"]
        assert "mahalanobis" in outlier_results, "Mahalanobis results missing"
        mahal_outliers = outlier_results["mahalanobis"]["outlier_indices"]
        # Bug #75 fix: 7 outliers with correct 38-trait space (was 6 with buggy 35)
        assert (
            len(mahal_outliers) == 7
        ), f"Expected 7 Mahalanobis outliers, got {len(mahal_outliers)}"
        print(f"OK Step 5: Detected {len(mahal_outliers)} outliers (Mahalanobis)")

        # Step 6: Visualize Outliers
        viz_result = results["06_visualize_outliers"].data
        assert "figures_generated" in viz_result.metadata
        assert (
            viz_result.metadata["figures_generated"] > 0
        ), "No outlier figures created"
        print(
            f"OK Step 6: Created {viz_result.metadata['figures_generated']} outlier visualizations"
        )

        # Step 7: Remove Outliers
        remove_result = results["07_remove_outliers"].data
        df_outliers_removed = remove_result.data
        # Bug #75 fix: 151 samples after removing 7 outliers (was 152/6 with buggy traits)
        assert (
            len(df_outliers_removed) == 151
        ), f"Expected 151 samples after outlier removal, got {len(df_outliers_removed)}"
        assert remove_result.metadata["outliers_removed"] == 7
        print(
            f"OK Step 7: {len(df_outliers_removed)} samples after outlier removal (7 removed)"
        )

        # Step 8: Statistical Analysis
        stats_result = results["08_statistical_analysis"].data
        assert "anova_results" in stats_result.metadata
        assert "heritability_results" in stats_result.metadata
        anova_results = stats_result.metadata["anova_results"]
        h2_results = stats_result.metadata["heritability_results"]
        # Bug #75 fix: ~29/38 traits with significant genotype effects (was ~26/35)
        n_significant = sum(
            1
            for r in anova_results.values()
            if isinstance(r, dict) and r.get("significant", False)
        )
        assert (
            n_significant >= 20
        ), f"Expected >=20 significant traits, got {n_significant}"
        print(f"OK Step 8: {n_significant} traits with significant genotype effects")

        # Step 9: Filter Heritability
        h2_filter_result = results["09_filter_heritability"].data
        df_h2_filtered = h2_filter_result.data
        final_traits = h2_filter_result.metadata["trait_names"]
        # Bug #75 fix: 16 traits after H²≥0.62 filtering on correct 38-trait set
        # (was 12 traits from buggy 35-trait set — 3 restored traits all pass H² threshold)
        assert (
            len(final_traits) == 16
        ), f"Expected 16 traits after H² filtering, got {len(final_traits)}"
        assert (
            len(df_h2_filtered) == 151
        ), "Sample count should not change in heritability filtering"
        print(
            f"OK Step 9: {len(final_traits)} traits after heritability filtering (22 removed)"
        )

        # Verify heritability threshold analysis plot was generated (Issue #26)
        threshold_plot = (
            pipeline.run_dir / "figures" / "09_heritability_threshold_analysis.png"
        )
        assert (
            threshold_plot.exists()
        ), f"Heritability threshold plot not found: {threshold_plot}"
        print("OK Step 9: Heritability threshold analysis plot generated")

        # Step 10: Generate Summary
        summary_result = results["10_generate_summary"].data
        assert summary_result.metadata is not None, "No metadata in summary"
        # Verify key metadata exists
        # Bug #75 fix: 151 samples (was 152), 16 traits (was 12)
        assert summary_result.metadata.get("final_n_samples") == 151
        assert summary_result.metadata.get("final_n_traits") == 16
        print(
            "OK Step 10: Generated summary report - "
            f"Final: {summary_result.metadata.get('final_n_samples')} samples, "
            f"{summary_result.metadata.get('final_n_traits')} traits"
        )

        # Verify run directory structure
        assert pipeline.run_dir.exists()
        assert (pipeline.run_dir / "pipeline_summary.json").exists()
        assert (pipeline.run_dir / "figures").exists()
        print(f"OK Run directory created: {pipeline.run_dir}")

        # Verify pipeline status
        summary = pipeline.get_summary()
        assert summary.status == "success"
        assert len(summary.steps) == 10

        print(f"\nPASSED Turface integration test passed!")
        print(f"   Final: 152 samples, 12 traits (from 187 samples, 35 traits)")

    def test_qc_pipeline_no_outlier_methods_warning(
        self, turface_rsr_csv_path, tmp_path
    ):
        """Test that warning is raised when no outlier detection methods are configured."""
        import warnings

        # Use test data path
        assert (
            turface_rsr_csv_path.exists()
        ), f"Test data not found: {turface_rsr_csv_path}"

        # Configure pipeline with NO outlier detection methods
        config = get_default_qc_config()
        config.pipeline_name = "no_outliers_test"
        config.data.csv_path = str(turface_rsr_csv_path)
        config.columns.barcode = "Barcode"
        config.columns.genotype = "geno"
        config.columns.replicate = "rep"

        # Disable all outlier detection methods
        config.outlier_detection.traditional_methods = []
        config.outlier_detection.clustering_methods = []

        # Since no methods configured, don't try to use a removal strategy
        # Skip config validation to test runtime warning behavior
        config.outlier_removal.method = None  # Won't be used anyway

        # Disable heritability filtering to speed up test
        config.heritability.enabled = False

        # Create pipeline (skip validation to test runtime warnings)
        pipeline = QCPipeline(
            config, output_dir=tmp_path / "no_outliers_test", validate=False
        )

        # Run pipeline and capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = pipeline.run()

            # Check that warnings were issued
            warning_messages = [str(warning.message) for warning in w]

            # Should have warning from DetectOutliersStep
            assert any(
                "No outlier detection methods configured" in msg
                for msg in warning_messages
            ), f"Expected warning about no outlier methods. Got warnings: {warning_messages}"

            # Should have warning from VisualizeOutliersStep
            assert any(
                "No outlier detection methods were run" in msg
                for msg in warning_messages
            ), f"Expected warning about no methods run. Got warnings: {warning_messages}"

        # Verify pipeline still completes successfully
        assert len(results) == 10  # QC pipeline has 10 steps
        assert results["05_detect_outliers"].data.metadata["total_methods"] == 0
        assert results["06_visualize_outliers"].data.metadata["figures_generated"] == 0

        print("\nPASSED No outlier methods warning test!")
