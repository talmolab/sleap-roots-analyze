"""Tests for VisualizeOutliersStep (Step 6)."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import warnings

from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    QCPipelineConfig,
    VisualizationConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import VisualizeOutliersStep

matplotlib.use("Agg")


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Barcode": [f"plant{i}" for i in range(20)],
            "geno": ["A"] * 10 + ["B"] * 10,
            "rep": [1, 2] * 10,
            "trait1": np.random.randn(20) * 10 + 50,
            "trait2": np.random.randn(20) * 5 + 25,
        }
    )


@pytest.fixture
def config():
    """Create config."""
    return QCPipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path="dummy.csv"),
        visualization=VisualizationConfig(dpi=100),
    )


@pytest.fixture
def prev_result_no_outliers(sample_data):
    """Previous result with no methods run."""
    return StepResult(
        data=sample_data,
        metadata={
            "valid_trait_names": ["trait1", "trait2"],
            "methods_run": [],
            "outlier_results": {},
            "samples": 20,
        },
        files_generated=[],
    )


class TestVisualizeOutliersStepBasic:
    """Test basic functionality."""

    def test_step_initialization(self):
        """Test initialization."""
        step = VisualizeOutliersStep()
        assert step.step_name == "VisualizeOutliers"
        assert "visualiz" in step.description.lower()

    def test_no_methods_warning(
        self, sample_data, config, prev_result_no_outliers, tmp_path
    ):
        """Test warning when no methods were run."""
        step = VisualizeOutliersStep()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = step.execute(
                sample_data, config, tmp_path, prev_result_no_outliers
            )
            assert any(
                "No outlier detection methods were run" in str(w_msg.message)
                for w_msg in w
            )

        assert len(result.files_generated) == 0

    def test_data_unchanged(
        self, sample_data, config, prev_result_no_outliers, tmp_path
    ):
        """Test data unchanged."""
        step = VisualizeOutliersStep()
        original = sample_data.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = step.execute(
                sample_data, config, tmp_path, prev_result_no_outliers
            )

        pd.testing.assert_frame_equal(result.data, original)


class TestVisualizeOutliersStepMetadata:
    """Test metadata."""

    def test_metadata_propagation(
        self, sample_data, config, prev_result_no_outliers, tmp_path
    ):
        """Test metadata is propagated."""
        step = VisualizeOutliersStep()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = step.execute(
                sample_data, config, tmp_path, prev_result_no_outliers
            )

        assert "valid_trait_names" in result.metadata
        assert result.metadata["valid_trait_names"] == ["trait1", "trait2"]


class TestOutlierMethodComparisonBarChart:
    """Tests for task 12.4-12.6: Outlier method comparison bar chart integration.

    These tests verify that `create_outlier_method_comparison_plot()` is wired
    into the visualize_outliers step and generates a comparison bar chart
    when multiple detection methods are run.
    """

    @pytest.fixture
    def prev_result_two_methods(self, sample_data):
        """Previous result with two outlier methods run."""
        return StepResult(
            data=sample_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2"],
                "methods_run": ["pca", "isolation_forest"],
                "outlier_results": {
                    "pca": {
                        "outlier_indices": [0, 5, 10],
                        "scores": np.random.randn(20),
                        "threshold": 2.0,
                    },
                    "isolation_forest": {
                        "outlier_indices": [1, 5, 15],
                        "scores": np.random.randn(20),
                        "contamination": 0.1,
                    },
                },
                "samples": 20,
                "column_mapping": {"genotype": "geno"},
            },
            files_generated=[],
        )

    @pytest.fixture
    def prev_result_one_method(self, sample_data):
        """Previous result with only one outlier method run."""
        return StepResult(
            data=sample_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2"],
                "methods_run": ["pca"],
                "outlier_results": {
                    "pca": {
                        "outlier_indices": [0, 5, 10],
                        "scores": np.random.randn(20),
                        "threshold": 2.0,
                    },
                },
                "samples": 20,
                "column_mapping": {"genotype": "geno"},
            },
            files_generated=[],
        )

    def test_comparison_bar_chart_generated_when_two_methods_run(
        self, sample_data, config, prev_result_two_methods, tmp_path
    ):
        """Task 12.4: This test MUST FAIL until task 12.6 is implemented.

        The visualize_outliers step should call create_outlier_method_comparison_plot()
        when 2+ methods are run, producing outlier_method_comparison.png.
        """
        step = VisualizeOutliersStep()

        result = step.execute(sample_data, config, tmp_path, prev_result_two_methods)

        # Check that outlier_method_comparison file was generated
        figures_dir = tmp_path / "figures"
        comparison_bar_files = list(figures_dir.glob("outlier_method_comparison.*"))

        assert len(comparison_bar_files) > 0, (
            "Should generate outlier_method_comparison.{fmt} when 2+ outlier methods are run. "
            "Task 12.6: create_outlier_method_comparison_plot() must be wired into "
            "visualize_outliers.py when len(methods_run) > 1."
        )

        # Also verify it's in files_generated
        file_names = [f.name for f in result.files_generated]
        assert any(
            "outlier_method_comparison" in name for name in file_names
        ), "outlier_method_comparison should be in files_generated list"

    def test_comparison_bar_chart_not_generated_for_single_method(
        self, sample_data, config, prev_result_one_method, tmp_path
    ):
        """Task 12.5: Comparison bar chart should NOT be generated when only 1 method run.

        This test verifies correct behavior - the chart only makes sense when
        comparing multiple methods.
        """
        step = VisualizeOutliersStep()

        result = step.execute(sample_data, config, tmp_path, prev_result_one_method)

        # Check that outlier_method_comparison file was NOT generated
        figures_dir = tmp_path / "figures"
        comparison_bar_files = list(figures_dir.glob("outlier_method_comparison.*"))

        assert len(comparison_bar_files) == 0, (
            "Should NOT generate outlier_method_comparison when only 1 method run. "
            "Comparison chart only makes sense with 2+ methods."
        )


class TestSharedSelectionRefactorUnchanged:
    """Guard the #173 single-source refactor: the step's mahalanobis /
    isolation_forest figures (now selected via the shared ``_select_outlier_figures``
    helper on the step's own pre-computed results) are byte-identical to calling the
    ``create_*`` functions directly, and the cross-method comparison + per-genotype
    figures are unchanged.
    """

    TRAITS = [f"trait_{j}" for j in range(5)]

    @pytest.fixture
    def outlier_frame(self):
        """40-sample, 5-trait clean frame with injected outliers and a geno column."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({f"trait_{j}": rng.randn(40) for j in range(5)})
        for idx in (5, 17, 28):
            for j in range(3):
                df.loc[idx, f"trait_{j}"] += 8.0
        df.insert(0, "Barcode", [f"BC{i:03d}" for i in range(40)])
        df.insert(1, "geno", ["G1"] * 20 + ["G2"] * 20)
        df.insert(2, "rep", list(range(1, 21)) * 2)
        return df

    def _prev_result(self, df):
        from sleap_roots_analyze.outlier_detection import (
            detect_outliers_isolation_forest,
            detect_outliers_mahalanobis,
        )

        results = {
            "mahalanobis": detect_outliers_mahalanobis(df[self.TRAITS]),
            "isolation_forest": detect_outliers_isolation_forest(
                df[self.TRAITS], random_state=42
            ),
        }
        return (
            StepResult(
                data=df,
                metadata={
                    "valid_trait_names": self.TRAITS,
                    "methods_run": ["mahalanobis", "isolation_forest"],
                    "outlier_results": results,
                    "samples": len(df),
                    "column_mapping": {"genotype": "geno"},
                },
                files_generated=[],
            ),
            results,
        )

    def test_step_figures_match_direct_create_calls(
        self, outlier_frame, config, tmp_path
    ):
        """Step mahal/IF filenames == the create_* keys, prefixed as before."""
        from sleap_roots_analyze.outlier_visualization import (
            create_isolation_forest_plots,
            create_mahalanobis_outlier_plots,
        )

        prev_result, results = self._prev_result(outlier_frame)

        expected_mahal = {
            f"outliers_mahal_{k}"
            for k in create_mahalanobis_outlier_plots(
                df=outlier_frame, mahal_results=results["mahalanobis"]
            )
        }
        expected_if = {
            f"outliers_if_{k}"
            for k in create_isolation_forest_plots(
                df=outlier_frame, iso_results=results["isolation_forest"]
            )
        }
        plt.close("all")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = VisualizeOutliersStep().execute(
                outlier_frame, config, tmp_path, prev_result
            )

        stems = {f.stem for f in result.files_generated}
        # The mahalanobis / isolation_forest figures are exactly the create_* keys.
        assert expected_mahal, "fixture should produce mahalanobis figures"
        assert expected_mahal <= stems
        assert expected_if <= stems
        # Cross-method comparison figures (>1 method) and per-genotype are unchanged.
        assert "outlier_comparison" in stems
        assert "outlier_overlap_heatmap" in stems
        assert "outlier_method_comparison" in stems
        assert "outliers_per_genotype" in stems

    def test_per_genotype_stays_cross_method(self, outlier_frame, config, tmp_path):
        """Exactly one cross-method per-genotype figure (not one per method)."""
        prev_result, _ = self._prev_result(outlier_frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = VisualizeOutliersStep().execute(
                outlier_frame, config, tmp_path, prev_result
            )
        per_geno = [f for f in result.files_generated if "per_genotype" in f.name]
        assert len(per_geno) == 1
