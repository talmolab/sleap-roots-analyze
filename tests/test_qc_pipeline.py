"""Tests for QC Pipeline orchestrator."""

from __future__ import annotations

import pytest
from pathlib import Path
import pandas as pd

from sleap_roots_analyze.pipeline import QCPipeline, PipelineConfig, get_default_config


class TestQCPipelineCreation:
    """Test QCPipeline initialization and basic functionality."""

    def test_qc_pipeline_creation(self, tmp_path):
        """Test creating a QCPipeline instance."""
        config = get_default_config()
        config.data.csv_path = "test.csv"
        config.outlier_detection.traditional_methods = ["mahalanobis"]

        pipeline = QCPipeline(config, output_dir=tmp_path)

        assert pipeline.config == config
        assert pipeline.pipeline_name == "pipeline"
        assert pipeline.version == "1.0"
        assert pipeline.output_dir == tmp_path

    def test_qc_pipeline_config_validation(self, tmp_path):
        """Test that invalid config raises ValueError."""
        from omegaconf import MISSING

        config = get_default_config()
        # Set csv_path to MISSING (required field)
        config.data.csv_path = MISSING

        with pytest.raises(ValueError, match="data.csv_path is required"):
            QCPipeline(config, output_dir=tmp_path)

    def test_qc_pipeline_skip_validation(self, tmp_path):
        """Test skipping validation allows invalid config."""
        config = get_default_config()
        # Missing csv_path but validation disabled
        config.data.csv_path = None

        # Should not raise when validate=False
        pipeline = QCPipeline(config, output_dir=tmp_path, validate=False)
        assert pipeline.config.data.csv_path is None

    def test_qc_pipeline_creates_tasks(self, tmp_path):
        """Test that create_tasks returns 10 tasks."""
        config = get_default_config()
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
        config = get_default_config()
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
        config = get_default_config()
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
        config = get_default_config()
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

    @pytest.mark.skip(
        reason="Integration test needs robust error handling in RemoveOutliersStep - see Issue #20"
    )
    def test_qc_pipeline_full_run(self, test_data, tmp_path):
        """Test full QC pipeline execution end-to-end.

        TODO: This test is currently skipped because RemoveOutliersStep doesn't handle
        detection errors gracefully (when outlier_mask is missing from results).
        This will be fixed as part of Issue #20 (comprehensive test coverage for Steps 3-10).
        """
        config = get_default_config()
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
