"""Tests for StatisticalAnalysisStep (Step 8)."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import (
    HeritabilityConfig,
    ColumnConfig,
    DataConfig,
    QCPipelineConfig,
)
from sleap_roots_analyze.pipeline.config.components import StatisticsConfig
from sleap_roots_analyze.pipeline.config.viz_config import VizPipelineConfig
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import StatisticalAnalysisStep


@pytest.fixture
def sample_data():
    """Create sample data with standardized column names (as after CleanupTraitsStep)."""
    np.random.seed(42)
    n_samples = 30
    return pd.DataFrame(
        {
            "Barcode": [f"plant{i}" for i in range(n_samples)],
            "Genotype": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "Replicate": [1, 2, 3] * 10,
            "trait1": np.random.randn(n_samples) * 10 + 50,
            "trait2": np.random.randn(n_samples) * 5 + 25,
            "trait3": np.random.randn(n_samples) * 3 + 15,
        }
    )


@pytest.fixture
def config():
    """Create config with standardized column names."""
    return QCPipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(
            barcode="Barcode", genotype="Genotype", replicate="Replicate"
        ),
        data=DataConfig(csv_path="dummy.csv"),
    )


@pytest.fixture
def prev_result(sample_data):
    """Previous result."""
    return StepResult(
        data=sample_data,
        metadata={
            "valid_trait_names": ["trait1", "trait2", "trait3"],
            "samples": 30,
        },
        files_generated=[],
    )


class TestStatisticalAnalysisStepBasic:
    """Test basic functionality."""

    def test_step_initialization(self):
        """Test initialization."""
        step = StatisticalAnalysisStep()
        assert step.step_name == "StatisticalAnalysis"
        assert "statistic" in step.description.lower()

    def test_basic_execution(self, sample_data, config, prev_result, tmp_path):
        """Test basic execution."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        assert isinstance(result, StepResult)
        assert result.data.equals(sample_data)
        assert "heritability_results" in result.metadata
        assert len(result.files_generated) > 0

    def test_heritability_file_generated(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test heritability results file is created in data/ directory."""
        step = StatisticalAnalysisStep()
        step.execute(sample_data, config, tmp_path, prev_result)

        assert (tmp_path / "data" / "08_heritability_results.csv").exists()

    def test_generate_blup_table_default_true(self):
        """StatisticsConfig.generate_blup_table defaults to True (#109)."""
        assert StatisticsConfig().generate_blup_table is True

    def test_data_unchanged(self, sample_data, config, prev_result, tmp_path):
        """Test data unchanged."""
        step = StatisticalAnalysisStep()
        original = sample_data.copy()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        pd.testing.assert_frame_equal(result.data, original)


class TestStatisticalAnalysisStepNoReplicate:
    """Cylinder-shaped data with no replicate column (issue #142)."""

    @pytest.fixture
    def cylinder_data(self):
        """Genotype -> multiple plants, no replicate column."""
        np.random.seed(42)
        n = 30
        df = pd.DataFrame(
            {
                "Barcode": [f"plant{i}" for i in range(n)],
                "Genotype": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
                "trait1": np.random.randn(n) * 10 + 50,
                "trait2": np.random.randn(n) * 5 + 25,
            }
        )
        df.loc[df["Genotype"] == "A", "trait1"] += 15
        return df

    @pytest.fixture
    def cylinder_config(self):
        """Config with replicate unset (None)."""
        return QCPipelineConfig(
            pipeline_name="test_cylinder",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate=None
            ),
            data=DataConfig(csv_path="dummy.csv"),
        )

    @pytest.fixture
    def cylinder_prev_result(self, cylinder_data):
        return StepResult(
            data=cylinder_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2"],
                "samples": len(cylinder_data),
            },
            files_generated=[],
        )

    def test_full_path_runs_and_produces_heritability(
        self, cylinder_data, cylinder_config, cylinder_prev_result, tmp_path
    ):
        """The QC heritability step runs and returns real H², not an error dict."""
        step = StatisticalAnalysisStep()
        result = step.execute(
            cylinder_data, cylinder_config, tmp_path, cylinder_prev_result
        )

        assert isinstance(result, StepResult)
        h2 = result.metadata["heritability_results"]
        for trait in ["trait1", "trait2"]:
            assert trait in h2
            assert "error" not in h2[trait]
            assert 0 <= h2[trait]["heritability"] <= 1


class TestStatisticalAnalysisStepMetadata:
    """Test metadata."""

    def test_heritability_metadata(self, sample_data, config, prev_result, tmp_path):
        """Test heritability in metadata."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        h2 = result.metadata["heritability_results"]
        assert "trait1" in h2
        assert "trait2" in h2
        assert "trait3" in h2

    def test_trait_names_propagation(self, sample_data, config, prev_result, tmp_path):
        """Test trait names propagated."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        assert result.metadata["valid_trait_names"] == ["trait1", "trait2", "trait3"]

    def test_image_paths_metadata_preserved(self, sample_data, config, tmp_path):
        """TDD Test: image_paths from previous step must be preserved in output metadata.

        Bug: StatisticalAnalysisStep creates a new metadata dict without **prev_result.metadata,
        causing image_paths (set by LoadDataAndImagesStep) to be lost.

        This breaks the image-dependent interactive plots in the Viz pipeline because
        GenerateInteractiveStep receives image_paths=None from PCA step metadata.

        Flow:
            LoadDataAndImagesStep -> sets image_paths in metadata
            StatisticalAnalysisStep -> MUST preserve image_paths
            PCAAnalysisStep -> preserves with **prev_result.metadata
            GenerateInteractiveStep -> reads image_paths from metadata

        Fix: Add **prev_result.metadata to metadata dict in StatisticalAnalysisStep.
        """
        from pathlib import Path

        # Create mock image_paths dict (as returned by link_rhizovision_images_to_samples)
        mock_image_paths = {
            "plant0": {"features.png": Path("/mock/plant0_features.png")},
            "plant1": {"features.png": Path("/mock/plant1_features.png")},
        }

        # Previous result with image_paths in metadata
        prev_result_with_images = StepResult(
            data=sample_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2", "trait3"],
                "image_paths": mock_image_paths,  # This MUST be preserved
                "n_samples": 30,
                "other_metadata": "should_also_be_preserved",
            },
            files_generated=[],
        )

        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result_with_images)

        # CRITICAL: image_paths must be preserved in output metadata
        assert "image_paths" in result.metadata, (
            "StatisticalAnalysisStep must preserve image_paths from previous step metadata. "
            "Fix: Add **prev_result.metadata to the metadata dict creation. "
            "Without this, GenerateInteractiveStep receives image_paths=None and "
            "skips all image-dependent plots (scatter_with_images, image_gallery, etc.)"
        )
        assert result.metadata["image_paths"] == mock_image_paths

    def test_all_previous_metadata_preserved(self, sample_data, config, tmp_path):
        """TDD Test: All metadata from previous step should be preserved.

        Ensures **prev_result.metadata is used in the metadata dict.
        """
        # Create previous result with various metadata keys
        prev_result_with_metadata = StepResult(
            data=sample_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2", "trait3"],
                "n_samples": 30,
                "n_traits": 3,
                "barcode_col": "Barcode",
                "genotype_col": "Genotype",
                "custom_key": "custom_value",
            },
            files_generated=[],
        )

        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result_with_metadata)

        # All previous metadata keys should be preserved
        assert result.metadata.get("n_samples") == 30
        assert result.metadata.get("n_traits") == 3
        assert result.metadata.get("barcode_col") == "Barcode"
        assert result.metadata.get("genotype_col") == "Genotype"
        assert result.metadata.get("custom_key") == "custom_value"


class TestStatisticalAnalysisNoHeritabilityPlots:
    """Test that statistical_analysis step does NOT generate heritability plots.

    Per VIZ-OUTPUT-002: Each figure type MUST be generated by exactly one pipeline step.
    Heritability plots should only be generated by generate_static_figures step,
    NOT by statistical_analysis step.
    """

    def test_no_heritability_plots_in_figures_dir(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that NO heritability plots are generated in figures/ directory.

        The statistical_analysis step should only generate data files (CSVs, JSONs),
        not visualization plots. Heritability plots are generated by the
        generate_static_figures step to avoid duplication.
        """
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Check that no figures/ directory is created by this step
        figures_dir = tmp_path / "figures"
        if figures_dir.exists():
            # If figures/ exists, verify no heritability plots are in it
            heritability_plots = list(figures_dir.glob("*heritability*"))
            assert len(heritability_plots) == 0, (
                f"statistical_analysis step should NOT generate heritability plots. "
                f"Found: {[p.name for p in heritability_plots]}"
            )

    def test_files_generated_are_data_only(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that files_generated contains only data files, no figures."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # All files should be data files (CSVs, JSONs), not figures (PNGs, PDFs)
        figure_extensions = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
        for file_path in result.files_generated:
            suffix = file_path.suffix.lower()
            assert suffix not in figure_extensions, (
                f"statistical_analysis step should not generate figure files. "
                f"Found: {file_path.name}"
            )


class TestStatisticalAnalysisDataOrganization:
    """Test that statistical analysis outputs are saved to data/ subdirectory (VIZ-OUTPUT-001)."""

    def test_all_outputs_saved_to_data_directory(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that all statistical analysis outputs are saved to data/ directory."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        data_dir = tmp_path / "data"
        assert data_dir.exists(), "data/ directory should exist"

        # All expected files should be in data/
        assert (data_dir / "08_trait_statistics.json").exists()
        assert (data_dir / "08_anova_results.csv").exists()
        assert (data_dir / "08_heritability_results.csv").exists()
        assert (data_dir / "08_statistical_analysis_summary.json").exists()

    def test_outputs_not_in_root_directory(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that statistical outputs are NOT saved directly to run_dir root."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Files should NOT be at root level
        assert not (tmp_path / "08_trait_statistics.json").exists()
        assert not (tmp_path / "08_anova_results.csv").exists()
        assert not (tmp_path / "08_heritability_results.csv").exists()
        assert not (tmp_path / "08_statistical_analysis_summary.json").exists()

    def test_files_generated_paths_correct(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that files_generated paths point to data/ directory."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        data_dir = tmp_path / "data"
        for file_path in result.files_generated:
            # All files should be under data/ directory
            assert file_path.parent == data_dir, (
                f"File {file_path.name} should be in data/ directory, "
                f"but is in {file_path.parent}"
            )


# --- Task 1: Tests for heritability flag respected ---


@pytest.fixture
def viz_config_heritability_disabled():
    """Create VizPipelineConfig with heritability calculation disabled."""
    return VizPipelineConfig(
        pipeline_name="test_viz",
        columns=ColumnConfig(
            barcode="Barcode", genotype="Genotype", replicate="Replicate"
        ),
        data=DataConfig(csv_path="dummy.csv"),
        statistics=StatisticsConfig(calculate_heritability=False),
        heritability=HeritabilityConfig(enabled=False),
    )


@pytest.fixture
def viz_config_heritability_enabled():
    """Create VizPipelineConfig with heritability calculation enabled (default)."""
    return VizPipelineConfig(
        pipeline_name="test_viz",
        columns=ColumnConfig(
            barcode="Barcode", genotype="Genotype", replicate="Replicate"
        ),
        data=DataConfig(csv_path="dummy.csv"),
        statistics=StatisticsConfig(calculate_heritability=True),
    )


class TestHeritabilityFlagRespected:
    """Test that the calculate_heritability config flag is respected."""

    def test_heritability_skipped_when_disabled(
        self, sample_data, viz_config_heritability_disabled, prev_result, tmp_path
    ):
        """Test heritability_estimates NOT called when flag is False.

        Verifies that config.statistics.calculate_heritability is False
        causes the step to skip heritability and return empty dict.
        """
        # GIVEN a config with calculate_heritability=False
        step = StatisticalAnalysisStep()

        # WHEN we execute the step
        with patch(
            "sleap_roots_analyze.pipeline.steps.statistical_analysis"
            ".calculate_heritability_estimates"
        ) as mock_h2:
            result = step.execute(
                sample_data,
                viz_config_heritability_disabled,
                tmp_path,
                prev_result,
            )

        # THEN calculate_heritability_estimates should NOT be called
        mock_h2.assert_not_called()

        # AND heritability_results should be an empty dict
        assert result.metadata["heritability_results"] == {}

    def test_heritability_calculated_when_enabled(
        self, sample_data, viz_config_heritability_enabled, prev_result, tmp_path
    ):
        """Test heritability_estimates IS called when flag is True.

        Verifies that config.statistics.calculate_heritability is True
        causes the step to calculate heritability and populate results.
        """
        # GIVEN a config with calculate_heritability=True
        step = StatisticalAnalysisStep()

        # WHEN we execute the step
        with patch(
            "sleap_roots_analyze.pipeline.steps.statistical_analysis"
            ".calculate_heritability_estimates"
        ) as mock_h2:
            mock_h2.return_value = {
                "trait1": {"heritability": 0.5},
                "trait2": {"heritability": 0.6},
                "trait3": {"heritability": 0.7},
            }
            result = step.execute(
                sample_data,
                viz_config_heritability_enabled,
                tmp_path,
                prev_result,
            )

        # THEN calculate_heritability_estimates should be called
        mock_h2.assert_called_once()

        # AND heritability_results should be populated
        assert result.metadata["heritability_results"] != {}
        assert "trait1" in result.metadata["heritability_results"]

    def test_heritability_csv_not_generated_when_disabled(
        self, sample_data, viz_config_heritability_disabled, prev_result, tmp_path
    ):
        """Test heritability CSV is NOT created when disabled."""
        # GIVEN a config with calculate_heritability=False
        step = StatisticalAnalysisStep()

        # WHEN we execute the step
        result = step.execute(
            sample_data,
            viz_config_heritability_disabled,
            tmp_path,
            prev_result,
        )

        # THEN the heritability CSV should NOT exist
        assert not (tmp_path / "data" / "08_heritability_results.csv").exists()

    def test_heritability_works_with_qc_config_no_statistics(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test heritability works with QCPipelineConfig (no statistics attr).

        When config lacks a statistics attribute, heritability should
        default to True (backward compatible).
        """
        # GIVEN a QCPipelineConfig (no statistics attribute)
        step = StatisticalAnalysisStep()

        # WHEN we execute the step
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # THEN heritability should be calculated (default True)
        assert result.metadata["heritability_results"] != {}


class TestBLUPTableOutput:
    """Test the BLUP-adjusted-means CSV output (#109)."""

    def test_blup_csv_written_when_both_enabled(
        self, sample_data, viz_config_heritability_enabled, prev_result, tmp_path
    ):
        """08_blup_adjusted_means.csv is written when both flags are enabled."""
        step = StatisticalAnalysisStep()
        step.execute(
            sample_data, viz_config_heritability_enabled, tmp_path, prev_result
        )

        blup_path = tmp_path / "data" / "08_blup_adjusted_means.csv"
        assert blup_path.exists()
        blup_df = pd.read_csv(blup_path, index_col=0)
        assert len(blup_df) == sample_data["Genotype"].nunique()

    def test_blup_csv_absent_when_generate_blup_table_false(
        self, sample_data, prev_result, tmp_path
    ):
        """No BLUP CSV when generate_blup_table=False; heritability CSV unaffected."""
        config = VizPipelineConfig(
            pipeline_name="test_viz",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(csv_path="dummy.csv"),
            statistics=StatisticsConfig(
                calculate_heritability=True, generate_blup_table=False
            ),
        )
        step = StatisticalAnalysisStep()
        step.execute(sample_data, config, tmp_path, prev_result)

        assert not (tmp_path / "data" / "08_blup_adjusted_means.csv").exists()
        assert (tmp_path / "data" / "08_heritability_results.csv").exists()

    def test_blup_csv_absent_when_heritability_disabled(
        self, sample_data, viz_config_heritability_disabled, prev_result, tmp_path
    ):
        """No CSV, no exception, no warning when heritability is disabled.

        This is an ordinary, legitimate configuration; a warning here would be
        noise, not a useful signal (see design.md Decision 5).
        """
        step = StatisticalAnalysisStep()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step.execute(
                sample_data, viz_config_heritability_disabled, tmp_path, prev_result
            )

        assert not (tmp_path / "data" / "08_blup_adjusted_means.csv").exists()
        assert caught == []

    def test_blup_table_works_with_qc_config_no_statistics(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Works with a bare QCPipelineConfig (no statistics attribute), no crash.

        Regression guard for the QC-pipeline AttributeError risk found in
        review: generate_blup_table must be resolved via the same
        getattr(config, "statistics", None) fallback calculate_heritability
        already uses.
        """
        step = StatisticalAnalysisStep()
        step.execute(sample_data, config, tmp_path, prev_result)

        assert (tmp_path / "data" / "08_blup_adjusted_means.csv").exists()


# --- Task 2: Tests for metadata and summary ---


class TestHeritabilityMetadataAndSummary:
    """Test metadata and summary JSON when heritability is disabled."""

    def test_summary_reflects_skipped_heritability(
        self, sample_data, viz_config_heritability_disabled, prev_result, tmp_path
    ):
        """Test summary shows skipped heritability when disabled."""
        # GIVEN a config with calculate_heritability=False
        step = StatisticalAnalysisStep()

        # WHEN we execute the step
        result = step.execute(
            sample_data,
            viz_config_heritability_disabled,
            tmp_path,
            prev_result,
        )

        # THEN the summary should indicate heritability was skipped
        summary = result.metadata["summary"]
        assert summary["heritability_summary"] == {"skipped": True}

    def test_metadata_heritability_results_empty_when_disabled(
        self, sample_data, viz_config_heritability_disabled, prev_result, tmp_path
    ):
        """Test that metadata['heritability_results'] is {} when disabled."""
        # GIVEN a config with calculate_heritability=False
        step = StatisticalAnalysisStep()

        # WHEN we execute the step
        result = step.execute(
            sample_data,
            viz_config_heritability_disabled,
            tmp_path,
            prev_result,
        )

        # THEN heritability_results should be empty dict
        assert result.metadata["heritability_results"] == {}


# --- Task 3: Tests for downstream compatibility ---


class TestHeritabilityDownstreamCompatibility:
    """Test downstream compatibility when heritability is disabled."""

    def test_pipeline_completes_with_heritability_disabled(
        self, sample_data, viz_config_heritability_disabled, prev_result, tmp_path
    ):
        """Integration test: step completes with heritability disabled.

        Verifies that StatisticalAnalysisStep completes successfully when
        calculate_heritability=False, no heritability CSV is generated, and
        heritability_results is {}.
        """
        # GIVEN a config with calculate_heritability=False
        step = StatisticalAnalysisStep()

        # WHEN we execute the step
        result = step.execute(
            sample_data,
            viz_config_heritability_disabled,
            tmp_path,
            prev_result,
        )

        # THEN the step should complete successfully
        assert isinstance(result, StepResult)
        assert result.data is not None
        assert len(result.data) == len(sample_data)

        # AND no heritability CSV should be generated
        assert not (tmp_path / "data" / "08_heritability_results.csv").exists()

        # AND heritability_results should be empty dict
        assert result.metadata["heritability_results"] == {}

        # AND other outputs should still be generated
        assert (tmp_path / "data" / "08_trait_statistics.json").exists()
        assert (tmp_path / "data" / "08_anova_results.csv").exists()
        assert (tmp_path / "data" / "08_statistical_analysis_summary.json").exists()

    def test_filter_heritability_handles_empty_results(self, sample_data, tmp_path):
        """Test FilterHeritabilityStep handles empty heritability_results."""
        from sleap_roots_analyze.pipeline import HeritabilityConfig
        from sleap_roots_analyze.pipeline.steps import FilterHeritabilityStep

        # GIVEN a config with heritability filtering disabled
        filter_config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(
                barcode="Barcode",
                genotype="Genotype",
                replicate="Replicate",
            ),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(enabled=False, threshold=0.3),
        )

        # AND a previous result with empty heritability_results
        prev = StepResult(
            data=sample_data,
            metadata={
                "trait_names": ["trait1", "trait2", "trait3"],
                "valid_trait_names": ["trait1", "trait2", "trait3"],
                "heritability_results": {},
            },
            files_generated=[],
        )

        # WHEN we execute FilterHeritabilityStep
        step = FilterHeritabilityStep()
        result = step.execute(sample_data, filter_config, tmp_path, prev)

        # THEN it should complete without error
        assert isinstance(result, StepResult)
        # AND all traits should be retained (filtering disabled)
        assert result.metadata["valid_trait_names"] == [
            "trait1",
            "trait2",
            "trait3",
        ]
